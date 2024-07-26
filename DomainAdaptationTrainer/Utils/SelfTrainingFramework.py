from TrainingEnv import CustomDataset, VirtualModel, BaseTrainer
import torch, os, numpy as np, torch.nn as nn, torch.nn.functional as F
from typing import List

class PseudoDataset(CustomDataset):

    def update_indexs(self, new_idxs:List):
        self.read_indexs = self.read_indexs[new_idxs]

    def reinit_indexs(self):
        self.read_indexs = np.arange(0, len(self._label), 1)

class SelfTraining(BaseTrainer):
    learning_rate = 5e-5
    suffix = "suffix"
    log_dir = "./tmp"
    threshold=0.7
    class_num = 2
    def dataset2dataloader(self, dataset_list:List, **kwargs):
        raise NotImplementedError("'dataset2dataloader' is not impleted")

    def training_loss_and_acc(self, model:VirtualModel, training_batch, temperature=1.0, label_weight:torch.Tensor=None,
                                reduction='mean'):
        return model.lossAndAcc(training_batch, temperature, label_weight, reduction)

    def PseudoLabeling(self, model:VirtualModel, data:PseudoDataset, batch_size=20):
        with torch.no_grad():
            pred_tensor = model.dataset_logits(data, batch_size=batch_size)
            weak_label = (pred_tensor > (1.0/self.class_num)).long().tolist()
            data.setLabel(weak_label, [*range(len(data))])

            confidence, _ = pred_tensor.max(dim=1)
            valid_indexs = torch.arange(len(confidence), device=confidence.device)[
                confidence.__gt__(self.threshold)
            ].cpu().tolist()
            data.update_indexs(valid_indexs)

    def ModelReTraining(self, max_epoch, model:VirtualModel, train_set:CustomDataset, valid_set:CustomDataset,
                            extra_train:CustomDataset=None, suffix='train', best_valid_acc= 0.0, batch_size=32,
                                grad_accum_cnt=1, valid_every=100, threshold=0.2):
        valid_acc = best_valid_acc if best_valid_acc != -1 else best_valid_acc
        loss_list = []
        tmp_model_file = "{}/{}_tmp.pkl".format(self.log_dir, suffix)
        optim = model.obtain_optim(self.learning_rate)
        optim.zero_grad()
        step = 0
        for epoch in range(max_epoch):
            for batch in self.dataset2dataloader([train_set, extra_train], batch_size=batch_size):
                loss, acc = self.training_loss_and_acc(model, batch)
                loss.backward()
                torch.cuda.empty_cache()
                if (step + 1)%grad_accum_cnt == 0:
                    optim.step()
                    optim.zero_grad()
                print('####Model Update#### step={} ({} | {}) ####, loss = {}'.format(
                    step, epoch, max_epoch, loss.data.item()
                ))
                loss_list.append(loss.data.item())
                step += 1
                if step % (valid_every*grad_accum_cnt) == 0:
                    acc_v = self.valid(model, valid_set, valid_set.labelTensor(), p_r_f1=True, suffix=f"{suffix}_valid")
                    if acc_v > valid_acc:
                        valid_acc = acc_v
                        torch.save(model.state_dict(), tmp_model_file)
            mean_loss = np.mean(loss_list)
            loss_list = []
            print("========> mean loss:", mean_loss)
            if mean_loss < threshold:  # early stop
                break
        if step < (valid_every*grad_accum_cnt):
            valid_acc = self.valid(model, valid_set, valid_set.labelTensor(), p_r_f1=True, suffix=f"{suffix}_valid")

        if os.path.exists(tmp_model_file):
            model.load_state_dict(torch.load(tmp_model_file))
            os.system("rm {}".format(tmp_model_file))
        return valid_acc, tmp_model_file

    def iterating(self, model:VirtualModel, labeled_source:CustomDataset, labeled_target:CustomDataset,
                    unlabeled_target:PseudoDataset, valid_set:CustomDataset, test_set:CustomDataset,
                        test_label=None, max_iterate=100, isWeightInited=False):
        if not isWeightInited:
            self.ModelReTraining(10, model, labeled_source, valid_set, best_valid_acc=0.0)
        for iterate in range(max_iterate):
            unlabeled_target.reinit_indexs()
            self.PseudoLabeling(model, unlabeled_target)
            val_acc, model_file = self.ModelReTraining(1, model, unlabeled_target, valid_set, suffix=self.suffix,
                                                    extra_train=labeled_target)
            if val_acc > self.best_valid_acc:
                self.best_valid_acc = val_acc
                self.valid(model, test_set, test_label, suffix=f"{self.suffix}_test")

class CRST_LRENT_MRKLD(SelfTraining):
    def __init__(self, random_seed, log_dir, suffix, model_file, class_num,
                    learning_rate=5e-5, alpha=0.1, beta=0.5, topK=0.2):
        self.seed = random_seed
        if not os.path.exists(log_dir):
            os.system("mkdir {}".format(log_dir))
        self.log_dir = log_dir
        self.suffix = suffix
        self.model_file = model_file
        self.best_valid_acc = 0.0
        self.min_valid_loss = 1e8
        self.class_num = class_num
        self.valid_counter, self.valid_step = 0, 0
        self.learning_rate = learning_rate

        self.alpha = alpha
        self.topK = topK
        self.beta = beta

    def PseudoLabeling(self, model:VirtualModel, data:CustomDataset, pseaudo_idxs=[], batch_size=20):
        c_idxs = list(set(range(len(data))) - set(pseaudo_idxs))
        with torch.no_grad():
            pred_tensor = model.dataset_logits(data, batch_size=batch_size)
            pred_tensor = pred_tensor[c_idxs]
        topk_vals, _ = pred_tensor.topk(int(self.topK*len(pred_tensor)), dim=0)
        self.lambda_k = topk_vals[-1]
        pseudo_label = (pred_tensor/self.lambda_k).pow(1.0/self.alpha)
        pseudo_label = (pseudo_label/(pseudo_label.sum(dim=1).unsqueeze(-1))).tolist()
        data.setLabel(pseudo_label, c_idxs)

        # data selection
        vals, _ = (pred_tensor - self.lambda_k).max(dim=1)
        valid_indexs = torch.arange(len(vals), device=vals.device)[vals.__gt__(0.0)].cpu().tolist()
        data.valid_indexs = [c_idxs[idx] for idx in valid_indexs]

    def MRKLD_LossAndAcc(self, model, batch, temperature=1.0, label_weight:torch.Tensor=None, reduction='mean'):
        preds = model.predict(batch, temperature=temperature)
        epsilon = torch.ones_like(preds) * 1e-8
        preds = (preds - epsilon).abs()  # to avoid the prediction [1.0, 0.0], which leads to the 'nan' value in log operation
        labels = batch[-2].to(preds.device)
        if labels.dim() == 2:
            loss, acc = model.expandCrossEntropy(preds, labels, label_weight, reduction)
        elif labels.dim() == 1:
            loss = F.nll_loss(preds.log(), labels, weight=label_weight, reduction=reduction)
            acc_t = ((preds.argmax(dim=1) - labels).__eq__(0).float().sum()) / len(labels)
            acc = acc_t.data.item()
        else:
            raise Exception("weird label tensor!")

        if reduction == 'sum':
            reg = preds.max(dim=1)[0].sum()
        else:
            reg = preds.max(dim=1)[0].mean()
        return loss + self.beta*reg, acc

    def training_loss_and_acc(self, model:VirtualModel, training_batch, temperature=1.0, label_weight:torch.Tensor=None,
                                reduction='mean'):
        return self.MRKLD_LossAndAcc(model, training_batch, temperature, label_weight, reduction)