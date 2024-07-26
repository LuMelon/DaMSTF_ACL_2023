import sys
sys.path.append('../..')
from TrainingEnv import VirtualModel
import torch.nn as nn
import torch, os, random, copy
from typing import Tuple, Optional, AnyStr, Any, List
import BaseModel.CV_Utils as CV_Utils
from Data.OfficeHomeLoader import OfficeHome, TransformFixMatch, ResizeImage
import torchvision.transforms as T, numpy as np
from DomainAdaptationTrainer.MME import MMETrainer
from TrainingEnv import GradientReversal

class OfficeHomeBatch:
    device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    def __init__(self, X:torch.Tensor, Y:torch.Tensor, domain:torch.Tensor):
        self.input = X.to(device=self.device)
        self.label = Y.to(device=self.device)
        self.domain = domain.to(device=self.device)

    def __getitem__(self, index):
        if index == -2 or index == 2:
            return self.label
        elif index == -1 or index == 3:
            return self.domain
        elif index == 0:
           return self.input
        else:
            raise Exception('index is out of range!')

class cls_head(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cls_head, self).__init__()
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, feature:torch.Tensor):
        return self.classifier(feature)

    def grouped_parameters(self, learning_rate=5e-3):
        base_lr = learning_rate
        params = [
            {"params": self.classifier.parameters(), "lr": 1.0 * base_lr},
        ]
        return params

    def obtain_optim(self, learning_rate=5e-3, optimizer=None):
        paras = self.grouped_parameters(learning_rate)
        if optimizer is None:
            return torch.optim.Adam(paras)
        else:
            return optimizer(paras)

class ImageClassifier(VirtualModel, nn.Module):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None, finetune=True):
        super(ImageClassifier, self).__init__()
        self.backbone = nn.Sequential(backbone,nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten())
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        self.num_classes = num_classes
        if bottleneck is None:
            self.bottleneck = nn.Sequential(
            )
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim

        self.head = cls_head(bottleneck_dim, self.num_classes)
        self.finetune = finetune
        self.dropout = nn.Dropout(0.2)

    def Batch2Vecs(self, batch:OfficeHomeBatch):
        x = batch.input
        f = self.backbone(x)
        return f

    def lossAndAcc(self, batch:OfficeHomeBatch, temperature=1.0, label_weight:torch.Tensor=None, reduction='mean'):
        preds = self.predict(batch, temperature=temperature)
        epsilon = torch.ones_like(preds) * 1e-8
        preds = (preds - epsilon).abs()  # to avoid the prediction [1.0, 0.0], which leads to the 'nan' value in log operation
        labels = batch.label
        loss, acc = self.loss_func(preds, labels, label_weight=label_weight, reduction=reduction)
        return loss, acc

    def predict(self, batch:OfficeHomeBatch, temperature=1.0):
        f = self.Batch2Vecs(batch)
        f1 = self.bottleneck(f)
        predictions = self.head(self.dropout(f1)).softmax(dim=1)
        return predictions

    def AdvPredict(self, batch):
        f = self.Batch2Vecs(batch)
        f = GradientReversal.apply(f)
        f1 = self.bottleneck(f)
        predictions = self.head(self.dropout(f1)).softmax(dim=1)
        return predictions

    def grouped_parameters(self, learning_rate=5e-3):
        base_lr = learning_rate
        if self.bottleneck is None:
            params = [
                {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
                {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
                {"params": self.head.parameters(), "lr": 1.0 * base_lr},
            ]
        else:
            params = [
                {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
                {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
                {"params": self.head.parameters(), "lr": 1.0 * base_lr},
            ]
        return params

    def obtain_optim(self, learning_rate=5e-3, optimizer=None):
        paras = self.grouped_parameters(learning_rate)
        if optimizer is None:
            return torch.optim.Adam(paras)
        else:
            return optimizer(paras)

class OfficeDataset(OfficeHome):
    items, _label, _confidence, _entropy, _domain = [], None, None, None, None
    read_indexs:np.array = None
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    def __init__(self, root: str, task: str, download: Optional[bool] = False, **kwargs):
        super(OfficeDataset, self).__init__(root, task, download, **kwargs)
        self.task = task
        self.init_dataset()

    def init_dataset(self):
        self._label = np.array([y for _, y in self.samples], dtype=np.int64)
        self._domain = np.array([idx for idx, line in enumerate(self.image_list) if self.task == line]*len(self.samples), dtype=np.int64)
        self.read_indexs = np.arange(len(self.samples), dtype=np.int64)
        self._confidence = np.zeros(len(self.samples))
        self._entropy = np.zeros(len(self.samples))

    def __len__(self):
        return len(self.read_indexs)

    def __getitem__(self, index: int) -> Tuple[Any, int, int]:
        """
         Args:
            index (int): Index
            return (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, self._label[index], self._domain[index]

    def collate_fn(self, items):
        return self.collate_raw_batch(items)

    def collate_raw_batch(self, items):
        ipt = torch.stack([x for (x,_, _) in items])
        label = torch.tensor([y for (_,y, _) in items])
        domain = torch.tensor([d for (_, _, d) in items])
        return OfficeHomeBatch(ipt, label, domain)

    def reset(self, label:np.array, domain:np.array, confidence:np.array, entropy:np.array):
        assert len(label) == len(domain) and len(domain)==len(confidence) and len(confidence)==len(entropy)
        self.read_indexs = np.arange(len(label))
        self._label = label
        self._domain = domain
        self._confidence = confidence
        self._entropy = entropy

    @property
    def label(self):
        return self._label[self.read_indexs]

    def setLabel(self, label:np.array, idxs=None):
        indexs = self.read_indexs[idxs] if idxs is not None else self.read_indexs
        self._label[indexs] = label

    def labelTensor(self, device=None):
        if device is None:
            device = self.device
        dim = len(self._label.shape)
        return torch.tensor(self._label[self.read_indexs], device=device, dtype=torch.float32 if dim>1 else None)

    @property
    def confidence(self):
        return self._confidence[self.read_indexs]

    def setConfidence(self, confidence, idxs):
        indexs = self.read_indexs[idxs]
        self._confidence[indexs] = confidence

    @property
    def entropy(self):
        return self._entropy[self.read_indexs]

    def setEntrophy(self, entropy, idxs):
        indexs = self.read_indexs[idxs]
        self._entropy[indexs] = entropy

    @property
    def domain(self):
        return self._domain[self.read_indexs]

    def domain_tensor(self, device=None):
        if device is None:
            device = self.device
        dim = len(self._domain[0])
        return torch.tensor(self._domain[self.read_indexs], device=device, dtype=torch.float32 if dim>1 else None)

    def initLabel(self, label):
        assert len(self.read_indexs) == len(label)
        self._label = label

    def initConfidence(self, confidence):
        assert len(self.read_indexs) == len(confidence)
        self._confidence = confidence

    def initEntrophy(self, entropy):
        assert len(self.read_indexs) == len(entropy)
        self._entropy = entropy

    def initDomain(self, d_arr):
        assert len(self.read_indexs) == len(d_arr)
        self._domain = d_arr

    def Derive(self, idxs: List=None):
        indexs = self.read_indexs[idxs]
        new_set = copy.deepcopy(self)
        new_set.samples = [self.samples[idx] for idx in indexs]
        new_set.init_dataset()

        tmp_samples = [self.samples[idx] for idx in self.read_indexs if not idx in indexs]
        self.samples = tmp_samples
        self.init_dataset()
        return new_set

    def Split(self, percent=[0.5, 1.0]):
        length = len(self.samples)
        sizes = [int(item*length) for item in percent]
        data_list = [copy.deepcopy(self) for _ in sizes]
        np.random.shuffle(self.read_indexs)
        start = 0
        for idx, end in enumerate(sizes):
            data_list[idx].samples = [self.samples[idx] for idx in self.read_indexs[start:end]]
            data_list[idx].init_dataset()
            start=end
        return tuple(data_list)

    @staticmethod
    def Merge(set_list:List):
        if len(set_list) == 1:
            return set_list[0]

        for set in set_list:
            set.init_dataset()
        rst_set:OfficeDataset = copy.deepcopy(set_list[0])
        label = np.concatenate([set.label for set in set_list])
        domain = np.concatenate([set.domain for set in set_list])
        confidence = np.concatenate([set.confidence for set in set_list])
        entropy = np.concatenate([set.entropy for set in set_list])
        rst_set.reset(label, domain, confidence, entropy)
        rst_set.samples = [item for set in set_list for item in set.samples]
        return rst_set

    def tensor_dataset(self, device=None):
        """
        :param device:
        :return: TensorDataset, collate_fn
        """
        raise NotImplementedError("'tensor_dataset' is not impleted")


def load_model(arch='resnet50', model_path='./resnet50-19c8e357.pth', bottleneck_dim=2048, num_cls=64, device=None):
    print("=> using pre-trained model '{}'".format(arch))
    backbone = CV_Utils.__dict__[arch](pretrained=True, model_path=model_path)
    classifier = ImageClassifier(backbone, num_cls, bottleneck_dim=bottleneck_dim).to(device)
    return classifier

def obtain_data(data_dir:AnyStr='/mnt/VMSTORE/OfficeHome/', source: AnyStr="Cl", target:AnyStr="Pr", transforms:Tuple=None, few_shot_cnt=20):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if transforms is None:
        train_transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])

        # unlabeled_transform = TransformFixMatch()
        unlabeled_transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
        ])

        val_transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
        ])
    else:
        train_transform, unlabeled_transform, val_transform = transforms
    source = OfficeDataset(root=data_dir, task=source, download=False, transform=train_transform)
    unlabeled_target = OfficeDataset(root=data_dir, task=target, download=False, transform=unlabeled_transform)
    target = OfficeDataset(root=data_dir, task=target, download=False, transform=val_transform)
    lt_idxs = random.sample(range(len(target)), few_shot_cnt)
    labeled_target = target.Derive(lt_idxs)
    valid_target, test_target = target.Split([0.4, 1.0])
    return source, valid_target, test_target, labeled_target, unlabeled_target

def Mul2One_data(data_dir:AnyStr='/mnt/VMSTORE/OfficeHome/', source_list:List=['Cl'], target:AnyStr="Pr",
                    transforms:Tuple=None, few_shot_cnt=20):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if transforms is None:
        train_transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])

        # unlabeled_transform = TransformFixMatch()
        unlabeled_transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
        ])

        val_transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
        ])
    else:
        train_transform, unlabeled_transform, val_transform = transforms
    source_set_list =[
        OfficeDataset(root=data_dir, task=s_name, download=False, transform=train_transform)
        for s_name in source_list
    ]
    source = OfficeDataset.Merge(source_set_list)
    unlabeled_target = OfficeDataset(root=data_dir, task=target, download=False, transform=unlabeled_transform)
    target = OfficeDataset(root=data_dir, task=target, download=False, transform=val_transform)
    lt_idxs = random.sample(range(len(target)), few_shot_cnt)
    labeled_target = target.Derive(lt_idxs)
    valid_target, test_target = target.Split([0.4, 1.0])
    return source, valid_target, test_target, labeled_target, unlabeled_target


def RunTask(source=["Cl"], target="Pr", fewShotCnt=20, data_dir='/mnt/VMSTORE/OfficeHome/', model_path='../../resnet50-19c8e357.pth'):
    newDomainName = target
    logDir = f'MME-{source}-{target}'
    # source_domain, val_set, test_target, labeled_target, unlabeled_target = obtain_data(
    #                       data_dir=data_dir,
    #                       source=source,
    #                       target=target,
    #                       transforms=None,
    #                       few_shot_cnt=20
    # )
    domains = [
        "Ar" ,
        "Cl" ,
        "Pr" ,
        "Rw" ,
    ]
    source_names = [dname for dname in domains if dname != target]
    source_domain, val_set, test_target, labeled_target, unlabeled_target = Mul2One_data(
                          data_dir=data_dir,
                          source_list=source_names,
                          target=target,
                          transforms=None,
                          few_shot_cnt=20
    )
    model1: ImageClassifier = load_model(model_path=model_path, num_cls=val_set.num_classes)
    model1 = model1.cuda()
    TestLabel = test_target.labelTensor()
    print("TestLabel : ", TestLabel.tolist())
    unlabeled_target.setLabel(np.zeros_like(unlabeled_target.label),
                              list(range(len(unlabeled_target))))
    print("Zero ULabel : ", unlabeled_target.label.tolist())

    trainer = MMETrainer(seed=10086, log_dir=logDir, suffix=f"{newDomainName}_FS{fewShotCnt}",
                            model_file=f"./MME_{newDomainName}_FS{fewShotCnt}.pkl",
                                class_num=model1.num_classes, temperature=0.05,
                                    batch_size=28, Lambda=0.1)
    trainer.collate_fn = val_set.collate_raw_batch

    # Pretrain Task-specific Model
    if os.path.exists(f"./PreTrainClassifier_T{newDomainName}.pkl"):
        model1.load_state_dict(
            torch.load(f"./PreTrainClassifier_T{newDomainName}.pkl")
        )
    else:
        trainer.training(model1, source_domain, batch_size=28, max_epoch=20, lr4model=5e-3,
                    dev_evaluator=None, test_evaluator=None,
                        grad_accum_cnt=1, valid_every=100, model_path="./PreTrainClassifier_T{newDomainName}.pkl")
        if os.path.exists(f"./PreTrainClassifier_T{newDomainName}.pkl"):
            model1.load_state_dict(
                torch.load(f"./PreTrainClassifier_T{newDomainName}.pkl")
            )
        else:
            torch.save(model1.state_dict(),
                       f"./PreTrainClassifier_T{newDomainName}.pkl")

    trainer.valid(model1, val_set, val_set.labelTensor(), f"{trainer.suffix}_valid", 0)
    trainer.valid(model1, test_target, TestLabel, f"{trainer.suffix}_test", 0)
    trainer.ModelTrain(model1, source_domain, labeled_target, unlabeled_target, val_set, test_target,
                         maxEpoch=40, validEvery=100)

if __name__ == '__main__':
    RunTask()