import os, sys, math, time
sys.path.extend(["..", "../.."])
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from TrainingEnv import VirtualModel, VirtualEvaluater, BaseTrainer, CustomDataset

class SentenceModel(nn.Module):
    def tokens2vecs(self, sents, attn_mask=None):
        raise NotImplementedError("'tokens2vecs' is not impleted")

    def tokens2vecs_aug(self, sents, attn_mask=None):
        raise NotImplementedError("'tokens2vecs_aug' is not impleted")

    def AugForward(self, sents):
        raise NotImplementedError("'AugForward' is not impleted")

    def forward(self, sents):
        raise NotImplementedError("'forward' is not impleted")

    def save_model(self, model_file):
        raise NotImplementedError("'save_model' is not impleted")

    def load_model(self, model_file):
        raise NotImplementedError("'load_model' is not impleted")

class RumorDetection(VirtualModel):
    def __init__(self, sent2vec:SentenceModel, prop_model:nn.Module, rdm_cls:nn.Module, **kwargs):
        super(RumorDetection, self).__init__()
        self.sent2vec = sent2vec
        self.prop_model = prop_model
        self.rdm_cls = rdm_cls
        self.nll_loss_fn = nn.NLLLoss
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception(f"Attribute '{k}' is not a valid attribute of DaMSTF")
            self.__setattr__(k, v)

    def seq2sents(self, seqs):
        sent_list = [sent for seq in seqs for sent in seq]
        seq_len = [len(seq) for seq in seqs]
        return sent_list, seq_len

    def Batch2Vecs(self, batch):
        sents, seq_len = self.seq2sents(batch[0])
        sent_vecs = self.sent2vec(sents)
        seq_tensors = [sent_vecs[sum(seq_len[:idx]):sum(seq_len[:idx]) + seq_len[idx]] for idx, s_len in
                       enumerate(seq_len)]
        seq_outs = self.prop_model(seq_tensors)
        return seq_outs

    def predict(self, batch, temperature=1.0):
        seq_outs = self.Batch2Vecs(batch)
        logits = self.rdm_cls(seq_outs)
        t_logits = logits/temperature
        preds = t_logits.softmax(dim=1)
        return preds

    def dataset_logits(self, dataset:CustomDataset, temperature=1.0, batch_size=32):
        logits_list = []
        with torch.no_grad():
            for start in range(0, len(dataset), batch_size):
                idxs = [*range(start, min(start+batch_size, len(dataset)), 1)]
                batch = dataset.collate_fn(
                    [dataset[idx] for idx in idxs]
                )
                logits = self.predict(batch, temperature=temperature)
                logits_list.append(logits.data)
        return torch.cat(logits_list, dim=0)

    def grouped_parameters(self, learning_rate):
        return [
            {'params': self.sent2vec.parameters(), 'lr': learning_rate},
            {'params': self.prop_model.parameters(), 'lr': learning_rate*1.1},
            {'params': self.rdm_cls.parameters(), 'lr': learning_rate*1.21}
        ]

    def save_model(self, model_file):
        sent_model_file = "%s_sent.pkl"%(model_file.rstrip(".pkl"))
        self.sent2vec.save_model(sent_model_file)
        torch.save(
            {
                "prop_model": self.prop_model.state_dict(),
                "rdm_cls": self.rdm_cls.state_dict()
            },
            model_file
        )

    def load_model(self, model_file):
        if os.path.exists(model_file):
            sent_model_file = "%s_sent.pkl" % (model_file.rstrip(".pkl"))
            self.sent2vec.load_model(sent_model_file)
            checkpoint = torch.load(model_file)
            self.rdm_cls.load_state_dict(checkpoint["rdm_cls"])
            self.prop_model.load_state_dict(checkpoint['prop_model'])
        else:
            print("Error: pretrained file %s is not existed!" % model_file)
            sys.exit()

class RumorBaseTrainer(BaseTrainer):
    running_dir = "./"
    model_rename = False

    def trainset2trainloader(self, dataset:Dataset, shuffle=False, batch_size=32):
        raise NotImplementedError("'dataset2dataloader' is not impleted")

    def fit(self, model:RumorDetection, train_set, dev_eval=None, test_eval=None, batch_size=5, grad_accum_cnt=4,
                    valid_every=100, max_epochs=10, learning_rate=5e-3, model_file=""):
        best_valid_acc, counter = 0.0, 0
        sum_loss, sum_acc = 0.0, 0.0
        optim = model.obtain_optim(learning_rate*1.0/grad_accum_cnt)
        optim.zero_grad()
        loss_list = []
        for epoch in range(max_epochs):
            train_loader = self.trainset2trainloader(train_set, shuffle=True, batch_size=batch_size)
            for batch in train_loader:
                counter += 1
                loss, acc = model.lossAndAcc(batch)
                loss.backward()
                torch.cuda.empty_cache()
                sum_loss += loss.data.item()
                sum_acc += acc
                if counter % grad_accum_cnt == 0:
                    optim.step()
                    optim.zero_grad()
                    mean_loss, mean_acc = sum_loss /grad_accum_cnt, sum_acc /grad_accum_cnt
                    loss_list.append(mean_loss)
                    if len(loss_list) > 20:
                        loss_list.pop(0)
                    print('%6d  [%3d | %3d], loss/acc = %6.8f/%6.7f, loss_mean/std=%6.7f/%6.7f, best_valid_acc:%6.7f ' % (
                        counter, epoch, max_epochs,
                        mean_loss, mean_acc, np.mean(loss_list), np.std(loss_list),
                        best_valid_acc))
                    sum_loss, sum_acc = 0.0, 0.0

                if counter % (valid_every * grad_accum_cnt) == 0 and dev_eval is not None:
                    val_acc = dev_eval(model)
                    if val_acc > best_valid_acc:
                        best_valid_acc = val_acc
                        model.save_model(model_file)
        model.load_model(model_file)
        if test_eval is not None:
            test_acc = test_eval(model)
        else:
            test_acc = best_valid_acc
        if self.model_rename:
            self.RenameModel(model_file, test_acc)

    def RenameModel(self, model_file, best_valid_acc):
        new_model_file = f"{model_file.rstrip('.pkl')}_{best_valid_acc}.pkl"
        sent_model_file = "%s_sent.pkl" % (model_file.rstrip(".pkl"))
        new_sent_model_file = "%s_%2.2f_sent.pkl" % (model_file.rstrip(".pkl"), best_valid_acc)
        os.system("mv %s %s" % (model_file, new_model_file))
        os.system("mv %s %s" % (sent_model_file, new_sent_model_file))

class BaseEvaluator(VirtualEvaluater):
    def logits(self, model:RumorDetection, batch):
        return model.predict(batch)

    def lossfunc(self, logits:torch.Tensor, labels:torch.Tensor):
        assert labels.dim() == 1 and labels.dtype == torch.long
        return F.nll_loss(logits.log(), labels)