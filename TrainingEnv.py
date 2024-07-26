import sys, random
from sklearn.metrics import precision_recall_fscore_support
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from typing import List

"""
1. dataset has to inherit the 'CustomDatset 'class.

2. model has to inherit the VirtualModel class, thereby containing:
    the 'predict' function, where the input is a data batch and the output is a probablity distribution. 
    Also, the 'predict' function should have an attribute 'temperature', which can be used to smooth the
    probability distribution. 
    
    
    
3. the final trainer has to contain:
    the 'collate_fn' function to construct a data batch, where the final element of the batch (index -1) indicates 
    the domain label (Long Tensor), and index -2 indicates the task label. Considering the exsitence of the unlabeled data,
    we suggest the task label is a C-Dimension float Tensor, so the unlabeled data can be represented by a zero-like tensor.
    Accordingly, the risk loss in the trainer is computed by expanding formula of the 'CrossEntropy' function. 
    Other elements only needs to be compatable with the model.
"""


from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from tqdm import tqdm
import re
from nltk.stem import WordNetLemmatizer
import nltk

def Lemma_Factory():
    lemmatizer = WordNetLemmatizer()
    def lemma(word_tokens):
        tags = nltk.pos_tag(word_tokens)
        new_words = []
        for pair in tags:
            if pair[1].startswith('J'):
                new_words.append(lemmatizer.lemmatize(pair[0], 'a'))
            elif pair[1].startswith('V'):
                new_words.append(lemmatizer.lemmatize(pair[0], 'v'))
            elif pair[1].startswith('N'):
                new_words.append(lemmatizer.lemmatize(pair[0], 'n'))
            elif pair[1].startswith('R'):
                new_words.append(lemmatizer.lemmatize(pair[0], 'r'))
            else:
                new_words.append(pair[0])
        return new_words
    return lemma

def obtainLDAModel(sent_list):
    words = stopwords.words('english')
    lemma = Lemma_Factory()
    func = lambda sent : lemma([w.strip() for w in sent.replace('[pad]', "'").split(' ') \
                if w.strip() not in words and re.search(r'([\W0-9_]+)', w.strip()) is None])
    train = [func(sent) for sent in tqdm(sent_list)]
    dictionary = Dictionary(train)
    corpus = [ dictionary.doc2bow(text) for text in train ]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=100)
    return lda

def reorganize_labels(labels: torch.Tensor):
    labels = torch.sparse_coo_tensor(
        indices=torch.stack([
            torch.arange(len(labels), device=labels.device),
            labels
        ]),
        values=torch.ones_like(labels, device=labels.device, dtype=torch.float32),
        size=(len(labels), max(labels) + 1)
    ).to_dense()
    return labels

class GradientReversal(torch.autograd.Function):
    """
    Basic layer for doing gradient reversal
    """
    lambd = 1.0

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReversal.lambd * grad_output.neg()

class CustomBatch:
    domain_label=None
    label = None
    def __init__(self, **kwargs):
        for attr, value in kwargs.items():
            if hasattr(self, attr):
                setattr(self, attr, value)
            else:
                print("Warning: CustomBatch has no attribute {}".format(attr))
    def __getitem__(self, index):
        if index == -1:
            return self.domain_label
        elif index == -2:
            return self.label
        else:
            raise Exception("Error: index = {} is not supported!".format(index))

class CustomDataset(Dataset):
    items, _label, _confidence, _entrophy, _domain = [], None, None, None, None
    read_indexs:np.array = None # indicating whether a data is exposed to users.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def collate_fn(self, items):
        raise NotImplementedError("'collate_fn' is not impleted")

    def __len__(self):
        return len(self.read_indexs)

    @property
    def label(self):
        """
        :return: labels indexed by 'read_indexs'
        """
        return self._label[self.read_indexs]

    def setLabel(self, label, idxs=None):
        if idxs is None:
            assert len(self.read_indexs) == len(label)
            indexs = self.read_indexs
        else:
            indexs = self.read_indexs[idxs]
        self._label[indexs] = label

    def labelTensor(self, device=None):
        if device is None:
            device = self.device
        dim = len(self._label[0])
        return torch.tensor(self._label[self.read_indexs], device=device, dtype=torch.float32 if dim>1 else None)

    @property
    def confidence(self):
        return self._confidence[self.read_indexs]

    def setConfidence(self, confidence, idxs):
        indexs = self.read_indexs[idxs]
        self._confidence[indexs] = confidence

    @property
    def entrophy(self):
        return self._entrophy[self.read_indexs]

    def setEntrophy(self, entrophy, idxs):
        indexs = self.read_indexs[idxs]
        self._entrophy[indexs] = entrophy

    @property
    def domain(self):
        return self._domain[self.read_indexs]

    def tensor_dataset(self, device=None):
        """
        :param device:
        :return: TensorDataset, collate_fn
        """
        raise NotImplementedError("'tensor_dataset' is not impleted")

    def domain_tensor(self, device=None):
        if device is None:
            device = self.device
        dim = len(self._domain[0])
        return torch.tensor(self._domain[self.read_indexs], device=device, dtype=torch.float32 if dim>1 else None)

    def initLabel(self, label):
        assert len(self.read_indexs) == len(label)
        if isinstance(label, list):
            label = np.array(label, dtype=np.int64)
        self._label = label

    def initConfidence(self, confidence):
        assert len(self.read_indexs) == len(confidence)
        if isinstance(confidence, list):
            confidence = np.array(confidence, dtype=np.float32)
        self._confidence = confidence

    def initEntrophy(self, entrophy):
        assert len(self.read_indexs) == len(entrophy)
        if isinstance(entrophy, list):
            entrophy = np.array(entrophy, dtype=np.float32)
        self._entrophy = entrophy

    def initDomain(self, d_arr):
        assert len(self.read_indexs) == len(d_arr)
        if isinstance(d_arr, list):
            d_arr = np.array(d_arr, dtype=np.int64)
        self._domain = d_arr

    def read_item(self, index):
        raise NotImplementedError("'read_item' has not been implemented!")

    def Derive(self, idxs: List):
        raise NotImplementedError("'Derive' has not been implemented!")

    def Split(self, percent=[0.5, 1.0]):
        raise NotImplementedError("'Split' has not been implemented!")

    def Merge(self, another_set):
        raise NotImplementedError("'Merge' has not been implemented!")

    def __getitem__(self, idx):
        if self.read_indexs is not None:
            index = self.read_indexs[idx]
            return self.read_item(index)
        else:
            return self.read_item(idx)

class VirtualModel(nn.Module):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    valid_counter = {}
    def loss_func(self, preds:torch.Tensor, labels:torch.Tensor, label_weight=None, reduction='none'):
        if labels.dim() == 2:
            loss, acc = self.expandCrossEntropy(preds, labels, label_weight, reduction)
        elif labels.dim() == 1:
            loss = F.nll_loss(preds.log(), labels, weight=label_weight, reduction=reduction)
            acc_t = ((preds.argmax(dim=1) - labels).__eq__(0).float().sum()) / len(labels)
            acc = acc_t.data.item()
        else:
            raise Exception("weird label tensor!")
        return loss, acc

    def lossList(self, batch, temperature=1.0, label_weight=None):
        """
        :param batch: batch[-1] is domain label. If the model is trained on a single domain, please add an extra zero
                        tensor to fill this place holder. batch[-2] is a label tensor. If the model is not designed
                        for the classification task, you should rewrite the function 'reorganize_labels'
        :param temperature:
        :param label_weight:
        :return:
        """
        preds = self.predict(batch, temperature=temperature)
        epsilon = torch.ones_like(preds) * 1e-8
        preds = (preds - epsilon).abs()  # to avoid the prediction [1.0, 0.0], which leads to the 'nan' value in log operation
        labels = batch[-2].to(preds.device)
        loss, _ = self.loss_func(preds, labels, label_weight=label_weight, reduction='none')
        return loss

    def Batch2Vecs(self, batch):
        raise NotImplementedError("'Batch2Vecs' is not impleted")

    def predict(self, batch, temperature=1.0):
        raise NotImplementedError("'predict' is not impleted")

    def grouped_parameters(self, learning_rate):
        raise NotImplementedError("'grouped_parameters' is not impleted")

    def obtain_optim(self, learning_rate=None, optimizer=None):
        paras = self.grouped_parameters(learning_rate)
        if optimizer is None:
            return torch.optim.Adam(paras)
        else:
            return optimizer(paras)

    def dataset_logits(self, dataset:CustomDataset, temperature=1.0, batch_size=32):
        try:
            tensor_set, collate_fn = dataset.tensor_dataset(self.device)
            data_loader = DataLoader(
                tensor_set,
                sampler=SequentialSampler(tensor_set),
                batch_size=batch_size,
                collate_fn=collate_fn
            )
        except NotImplementedError:
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=dataset.collate_fn
            )
        pred_list = []
        with torch.no_grad():
            for batch in tqdm(data_loader):
                pred_list.append(self.predict(batch, temperature=temperature))
        return torch.cat(pred_list, dim=0)

    def valid(self, valid_set, valid_label, p_r_f1=True, suffix="tmp"):
        if suffix in self.valid_counter:
            self.valid_counter[suffix] += 1
        else:
            self.valid_counter[suffix] = 0

        with torch.no_grad():
            logits_tensor = self.dataset_logits(valid_set)
            loss, acc_v = self.loss_func(logits_tensor, valid_label, reduction='mean')
        output_items = [(f"{suffix}_loss", loss.data.item()), (f"{suffix}_valid_acc", acc_v)]
        if p_r_f1:
            labelTensor = valid_label if valid_label.dim() == 1 else valid_label.argmax(dim=1)
            predTensor = logits_tensor.argmax(dim=1)
            perf = precision_recall_fscore_support(labelTensor.cpu(), predTensor.cpu())
            (p_v, r_v, f1_v, count) = perf
            class_num = len(p_v)
            output_items.extend( [(f'{suffix}_valid_prec_{i}', p_v[i]) for i in range(class_num)] + \
                                   [(f'{suffix}_valid_recall_{i}', r_v[i]) for i in range(class_num)] + \
                                    [(f'{suffix}_valid_f1_{i}', f1_v[i]) for i in range(class_num)] + \
                                      [(f'{suffix}_class_cnt_{i}', count[i]) for i in range(class_num)])
        print(f"{suffix} Performance : \n", "\n".join([f"\t {k} = {v}" for k, v in output_items]))
        return acc_v

    def expandCrossEntropy(self, preds, labels, label_weight:torch.Tensor=None, reduction='mean'):
        entropy = labels * (preds.log().neg())
        if label_weight is not None:
            if label_weight.dim() == 1:
                assert label_weight.size(0) == labels.size(1)
                label_weight = label_weight.unsqueeze(0)
            else:
                assert label_weight.size(1) == labels.size(1)

            if label_weight.sum(dim=1)[0] != 1.0:
                label_weight = label_weight / (label_weight.sum(dim=1)[0])
            loss_t = (label_weight * entropy).sum(dim=1)
        else:
            loss_t = entropy.sum(dim=1)

        if reduction == 'sum':
            loss = loss_t.sum()
        elif reduction == 'mean':
            loss = loss_t.mean()
        elif reduction == 'none':
            loss = loss_t
        else:
            raise Exception(f"'reduction'={reduction} is not a valid parameters, \
                                and the feasible choice is ['sum', 'mean', 'none']")
        acc_t = ((preds.argmax(dim=1) - labels.argmax(dim=1)).__eq__(0).float().sum()) / len(labels)
        acc = acc_t.data.item()
        return loss, acc

    def lossAndAcc(self, batch, temperature=1.0, label_weight:torch.Tensor=None, reduction='mean'):
        preds = self.predict(batch, temperature=temperature)
        epsilon = torch.ones_like(preds) * 1e-8
        preds = (preds - epsilon).abs()  # to avoid the prediction [1.0, 0.0], which leads to the 'nan' value in log operation
        labels = batch[-2].to(preds.device)
        loss, acc = self.loss_func(preds, labels, label_weight=label_weight, reduction=reduction)
        return loss, acc

    def auxiliaryLossAndAcc(self, extra_classifier, batch, labels:torch.Tensor=None,
                                temperature=1.0, label_weight=None, reduction='mean', grad_reverse=False):
        vecs = self.Batch2Vecs(batch)
        if grad_reverse:
            vecs = GradientReversal.apply(vecs)
        logits = extra_classifier(vecs)
        preds = F.softmax(logits /temperature, dim=1)
        epsilon = torch.ones_like(preds) * (1e-8)
        preds = (preds - epsilon).abs()
        if labels is None:
            labels = batch[-2].to(preds.device)
        else:
            labels = labels.to(preds.device)
        if labels.dim() == 2:
            loss, acc = self.expandCrossEntropy(preds, labels, label_weight, reduction)
        elif labels.dim() == 1:
            loss = F.nll_loss(preds.log(), labels, weight=label_weight, reduction=reduction)
            acc_t = ((preds.argmax(dim=1) - labels).__eq__(0).float().sum()) / len(labels)
            acc = acc_t.data.item()
        else:
            raise Exception("weird label tensor!")
        return loss, acc

class AdversarialModel(VirtualModel):
    def advLossAndAcc(self, classifier, batch, labels:torch.Tensor=None,
                                temperature=1.0, label_weight=None, reduction='mean'):
        return self.auxiliaryLossAndAcc(classifier, batch, labels, temperature, label_weight, reduction,
                                        grad_reverse=True)

class AugmentationModel(VirtualModel):

    def AugBatch2Vecs(self, batch, aug_type=None):
        raise NotImplementedError("'AugBatch2Vecs' is not impleted")

    def augLossAndAccV2(self, classifier, batch, aug_type, labels:torch.Tensor=None,
                        temperature=1.0, label_weight=None, reduction='mean'):
        vecs = self.AugBatch2Vecs(batch, aug_type=aug_type)
        logits = classifier(vecs)
        preds = F.softmax(logits / temperature, dim=1)
        epsilon = torch.ones_like(preds) * (1e-8)
        preds = (preds - epsilon).abs()
        if labels is None:
            labels = batch[-2].to(preds.device)
        else:
            labels = labels.to(preds.device)
        if labels.dim() == 2:
            loss, acc = self.expandCrossEntropy(preds, labels, label_weight, reduction)
        elif labels.dim() == 1:
            loss = F.nll_loss(preds.log(), labels, weight=label_weight, reduction=reduction)
            acc_t = ((preds.argmax(dim=1) - labels).__eq__(0).float().sum()) / len(labels)
            acc = acc_t.data.item()
        else:
            raise Exception("weird label tensor!")
        return loss, acc

class VirtualEvaluater:
    dataset:Dataset = None
    labelTensor:torch.Tensor = None
    collate_fn = None
    valid_counter = 0
    suffix = ""

    def logits(self, model, batch):
        raise NotImplementedError("'logits' is not impleted")

    def loss_func(self, preds:torch.Tensor, labels:torch.Tensor, label_weight=None, reduction='mean'):
        if labels.dim() == 2:
            labels = labels.argmax(dim=1)
        elif labels.dim() == 1:
            pass
        else:
            raise Exception("weird label tensor!")
        loss = F.nll_loss(preds.log(), labels, weight=label_weight, reduction=reduction)
        acc_t = ((preds.argmax(dim=1) - labels).__eq__(0).float().sum()) / len(labels)
        acc = acc_t.data.item()
        return loss, acc

    def dataset2dataloader(self):
        raise NotImplementedError("'dataset2dataloader' is not impleted")

    def P_R_F1(self, y_true, y_pred):
        return precision_recall_fscore_support(y_true, y_pred)

    def __call__(self, model:VirtualModel, p_r_f1=True):
        dataloader = self.dataset2dataloader()
        logits_list = []
        for batch in dataloader:
            with torch.no_grad():
                logits_list.append(
                    self.logits(model, batch)
                )
        logits_tensor = torch.cat(logits_list, dim=0)
        loss, acc_v = self.loss_func(logits_tensor, self.labelTensor, reduction='mean')
        output_items = [(f"{self.suffix}_loss", loss.data.item()), (f"{self.suffix}_valid_acc", acc_v)]
        if p_r_f1:
            labelTensor = self.labelTensor if self.labelTensor.dim() == 1 else self.labelTensor.argmax(dim=1)
            predTensor = logits_tensor.argmax(dim=1)
            perf = self.P_R_F1(labelTensor.cpu(), predTensor.cpu())
            (p_v, r_v, f1_v, count) = perf
            class_num = len(p_v)
            output_items.extend([(f'{self.suffix}_valid_prec_{i}', p_v[i]) for i in range(class_num)] + \
                                [(f'{self.suffix}_valid_recall_{i}', r_v[i]) for i in range(class_num)] + \
                                [(f'{self.suffix}_valid_f1_{i}', f1_v[i]) for i in range(class_num)] + \
                                [(f'{self.suffix}_class_cnt_{i}', count[i]) for i in range(class_num)])
        print(f"{self.suffix} Performance : \n", "\n".join([f"\t {k} = {v}" for k, v in output_items]))
        self.valid_counter += 1
        return acc_v

class BaseTrainer:
    running_dir = "./"
    valid_counter = {}
    def initTrainingEnv(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def valid(self, model:VirtualModel, valid_set, valid_label, p_r_f1=True, suffix="tmp"):
        return model.valid(valid_set, valid_label, p_r_f1, suffix)

    def trainset2trainloader(self, train_set:CustomDataset, batch_size):
        try:
            train, collate_fn = train_set.tensor_dataset()
            return  DataLoader(
                train,
                batch_size=batch_size,
                sampler=RandomSampler(train),
                collate_fn=collate_fn
            )
        except NotImplementedError:
            return DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=train_set.collate_fn
            )

    def training(self, trModel:VirtualModel, train_set:CustomDataset, batch_size, max_epoch, lr4model=5e-5,
                    dev_evaluator:VirtualEvaluater =None, test_evaluator:VirtualEvaluater=None,
                        grad_accum_cnt=1, valid_every=100, model_path="./tmp.pkl"):
        # the rule for early stop: when the variance of the recent 50 training loss is smaller than 0.05,
        # the training process will be stopped
        optim1 = trModel.obtain_optim(lr4model)
        optim1.zero_grad()
        lossList = []
        best_acc, step = 0.0, 0
        for epoch in range(max_epoch):
            for batch in self.trainset2trainloader(train_set, batch_size):
                step += 1
                DLoss, DAcc = trModel.lossAndAcc(batch, temperature=0.1)
                DLoss.backward()
                if step % grad_accum_cnt == 0:
                    optim1.step()
                    optim1.zero_grad()
                torch.cuda.empty_cache()
                print('####Pre Train Classifier %3d |  (%3d , %3d) ####, loss = %6.8f, Acc = %6.8f, mean_loss = %6.8f' % (
                    step, epoch, max_epoch, DLoss.data.item(), DAcc, np.mean(lossList).item()
                ))
                lossList.append(DLoss.data.item())

                if len(lossList) > 20:
                    lossList.pop(0)

                if dev_evaluator is not None and step%valid_every == 0:
                    acc_v = dev_evaluator(trModel)
                    if acc_v > best_acc:
                        best_acc = acc_v
                        torch.save(trModel.state_dict(), f"{model_path}")

        if test_evaluator is not None:
            trModel.load_state_dict(torch.load(f"{model_path}"))
            test_evaluator(trModel)