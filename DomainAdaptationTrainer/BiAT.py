import sys
sys.path.append("../../")
import time
import torch, pickle, copy, random, os
from tqdm import trange
import torch.nn as nn
from torch.utils.data import Dataset
from TrainingEnv import BaseTrainer, GradientReversal, VirtualModel, CustomDataset
import torch.nn.functional as F, numpy as np

class BiATTrainEnv(BaseTrainer):
    def __init__(self, random_seed, HClassifier):
        self.seed = random_seed
        self.HClassifier = HClassifier

    def model_predict(self, model, batch, temperature=1.0):
        vecs = model.Batch2Vecs(batch)
        logits = self.HClassifier(vecs)
        preds = F.softmax(logits / temperature, dim=1)
        return preds

    def pred_Logits(self, model: VirtualModel, data: CustomDataset, idxs=None, batch_size=20):
        preds = []
        if idxs is None:
            idxs = list(range(len(data)))
        with torch.no_grad():
            for i in trange(0, len(idxs), batch_size):
                batch_idxs = idxs[i:min(len(idxs), i + batch_size)]
                batch = data.collate_fn([data[idx] for idx in batch_idxs])
                pred = model.predict(batch)
                preds.append(pred)
        pred_tensor = torch.cat(preds)
        print("pred_tensor : ", pred_tensor)
        return pred_tensor

    def lossAndAcc(self, model:VirtualModel, batch, temperature=1.0, label_weight: torch.Tensor = None, reduction='mean'):
        return model.lossAndAcc(batch, temperature, label_weight, reduction)

    def dataIter(self, pseudo_set, labeled_target=None, batch_size=32):
        p_idxs = list(range(len(pseudo_set))) if not hasattr(pseudo_set, 'valid_indexs') else pseudo_set.valid_indexs
        p_len = len(p_idxs)
        if labeled_target is None:
            l_len = 0
            l_idxs = []
        else:
            l_idxs = list(range(len(labeled_target))) if not hasattr(labeled_target, 'valid_indexs') \
                                                        else labeled_target.valid_indexs
            l_len = len(l_idxs)
        data_size = p_len + l_len
        idxs = random.sample(range(data_size), data_size)*2
        for start_i in range(0, data_size, batch_size):
            batch_idxs = idxs[(start_i):(start_i+batch_size)]
            items = [pseudo_set[p_idxs[idx]] if idx < p_len else \
                        labeled_target[l_idxs[idx-p_len]] for idx in batch_idxs]
            yield pseudo_set.collate_raw_batch(items)

class BiATTrainer(BiATTrainEnv):
    def __init__(self, classifier, random_seed, log_dir, suffix, model_file, class_num, lambda1, lambda2, lambda3, lambda4,
                 temperature=1.0, learning_rate=5e-3, batch_size=32):
        super(BiATTrainer, self).__init__(random_seed=random_seed, HClassifier=classifier)
        self.HClassifier = classifier
        if not os.path.exists(log_dir):
            os.system("mkdir {}".format(log_dir))
        self.log_dir = log_dir
        self.suffix = suffix
        self.model_file = model_file
        self.best_valid_acc = 0.0
        self.min_valid_loss = 1e8
        self.class_num = class_num
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.valid_step = 0
        self.lambda1, self.lambda2, self.lambda3, self.lambda4 = lambda1, lambda2, lambda3, lambda4

    def augLossAndAccV2(self, model:VirtualModel, classifier, batch, aug_type="adver",
                                     temperature=1.0):
        raise NotImplementedError("'augLossAndAccV2' has not been implemented! ")

    def Entrophy(self, trModel:VirtualModel, classifier, batch):
        pooledOutput = trModel.Batch2Vecs(batch)
        pooledOutput = GradientReversal.apply(pooledOutput)
        normedPoolOut = pooledOutput / (pooledOutput.norm())
        logits = classifier(normedPoolOut)
        preds = F.softmax(logits / self.temperature, dim=1)
        epsilon = torch.ones_like(preds) * (1e-8)
        preds = (preds - epsilon).abs()
        loss = -1 * (preds * (preds.log())).sum()
        return loss

    def CELoss(self, model:VirtualModel, classifier:nn.Module, batch, label=None):
        if label is not None:
            batch = [batch[idx] if idx!=(len(batch)-2) else label for idx in range(len(batch))]
        return model.auxiliaryLossAndAcc(classifier, batch, temperature=self.temperature)

    def ATLoss(self, model:VirtualModel, classifier:nn.Module, batch, label=None, need_gradient=False):
        if label is not None:
            batch = [batch[idx] if idx!=(len(batch)-2) else label for idx in range(len(batch))]
        if need_gradient:
            model.zero_grad()
            loss, acc = model.auxiliaryLossAndAcc(classifier, batch, temperature=self.temperature)
            loss.backward()
        return self.augLossAndAccV2(model, classifier, batch, aug_type="adver",
                                     temperature=self.temperature)

    def AATLoss(self, model:VirtualModel, classifier1:nn.Module, classifier2:nn.Module, batch):
        loss, acc = model.auxiliaryLossAndAcc(classifier1, batch, temperature=self.temperature)
        model.zero_grad()
        loss.backward()
        aatLoss, aatAcc = self.ATLoss(model, classifier2, batch)
        return aatLoss, aatAcc

    def MMELoss(self, model:nn.Module, classifier, batch):
        pooledOutput = model.Batch2Vecs(batch)
        pooledOutput = GradientReversal.apply(pooledOutput)
        normedPoolOut = pooledOutput / (pooledOutput.norm())
        logits = classifier(normedPoolOut)
        preds = F.softmax(logits / self.temperature, dim=1)
        epsilon = torch.ones_like(preds) * (1e-8)
        preds = (preds - epsilon).abs()
        loss = -1 * (preds * (preds.log())).sum()
        return loss

    def predict(self, model, batch):
        assert hasattr(self, 'HClassifier')
        vecs = model.Batch2Vecs(batch)
        logits = self.HClassifier(vecs)
        preds = F.softmax(logits, dim=1)
        return preds

    def EVatLoss(self, model, classifier, batch):
        with torch.no_grad():
            preds = self.predict(model, batch)
            pseudoLable = preds.argmax(dim=1)
        loss, _ = self.CELoss(model, classifier, batch, pseudoLable)
        model.zero_grad()
        loss.backward()
        pooledOutput = model.AugBatch2Vecs(batch, aug_type="adver")
        logits = classifier(pooledOutput)
        preds = F.softmax(logits / self.temperature, dim=1)
        epsilon = torch.ones_like(preds) * (1e-8)
        preds = (preds - epsilon).abs()
        vat_loss = F.nll_loss(preds.log(), pseudoLable)
        entrophy = -1 * (preds * (preds.log())).sum()
        model.set_aug_type(None)
        return vat_loss+entrophy

    def TLoss(self, HNet:VirtualModel, TClassifier:nn.Module, batch):
        return HNet.lossAndAccV2(TClassifier, batch, temperature=self.temperature)

    def dataSampler(self, labeledSource, labeledTarget, unlabeledTarget):
        half_size = self.batch_size//2
        bs2 = len(labeledTarget) if len(labeledTarget) < half_size else half_size
        bs1 = self.batch_size - bs2
        def sample_fn(max_iter_step):
            for _ in range(max_iter_step):
                items1 = [labeledSource[idx] for idx in random.sample(range(len(labeledSource)),
                                                                      min(len(labeledSource), self.batch_size))]
                items2 = [labeledTarget[idx] for idx in random.sample(range(len(labeledTarget)),
                                                                      min(len(labeledTarget), self.batch_size))]
                items3 = [unlabeledTarget[idx] for idx in random.sample(range(len(unlabeledTarget)), self.batch_size)]
                batchCE = labeledSource.collate_raw_batch(items1[:bs1] + items2[:bs2])
                batchEVat = unlabeledTarget.collate_raw_batch(items3)
                batchTNet = labeledTarget.collate_raw_batch(items2)
                batchAAT = labeledSource.collate_raw_batch(items1)
                yield batchCE, batchAAT, batchEVat, batchTNet
        return sample_fn

    def PreTrainTNet(self, HNet, TClassifier, labeledTarget : Dataset, batchSize=32, max_epoch=1):
        # the rule for early stop: when the variance of the recent 50 training loss is smaller than 0.05, the training process will be stopped
        optim_H = HNet.obtain_optim()
        optim_T = TClassifier.obtain_optim()
        lossList = []
        for epoch in range(max_epoch):
            for step in range(200):
                batch = self.collate_fn(
                    [labeledTarget[idx] for idx in random.sample(range(len(labeledTarget)),
                                                                 min(len(labeledTarget), batchSize))]
                )
                TLoss, TAcc = self.TLoss(HNet, TClassifier, batch)
                optim_H.zero_grad()
                optim_T.zero_grad()
                TLoss.backward()
                optim_H.step()
                optim_T.step()
                print('####Pre Train Domain Classifier %3d (%3d | %3d) ####, loss = %6.8f, Acc = %6.8f' % (
                    step, epoch, max_epoch, TLoss.data.item(), TAcc
                ))
                lossList.append(TLoss.data.item())
                if len(lossList) > 20:
                    lossList.pop(0)
                    if np.std(lossList) < 0.05 and np.mean(lossList) < 0.2:
                        return

    def Training(self, HNet:nn.Module, HClassifier, TClassifier, labeledSource : Dataset, labeledTarget : Dataset,
                    unlabeledTarget : Dataset, validSet : Dataset, testSet:Dataset,
                        testLabel, maxStep=10000, validEvery=20):
        assert labeledTarget is not None
        print("labeled Source/labeled Target/unlabeled Target: {}/{}/{}".format(len(labeledSource),
                                                                                len(labeledTarget),
                                                                                len(unlabeledTarget)))
        optim_N = HNet.obtain_optim()
        optim_H = HClassifier.obtain_optim()
        optim_T = TClassifier.obtain_optim()
        validLabel = validSet.labelTensor()
        sampler = self.dataSampler(labeledSource, labeledTarget, unlabeledTarget)
        for step, (batchCE, batchAAT, batchEVat, batchTNet) in enumerate(sampler(maxStep)):
            # L_CE + L_AT
            loss, acc = self.CELoss(HNet, HClassifier, batchCE)
            HNet.zero_grad()
            HClassifier.zero_grad()
            loss.backward()
            HNet.set_aug_type('adver')
            ATLoss, ATAcc = self.ATLoss(HNet, HClassifier, batchCE)
            (self.lambda1*ATLoss).backward()
            optim_H.step()
            optim_N.step()

            # L_AT
            AATLoss, AATAcc = self.AATLoss(HNet, HClassifier, TClassifier, batchAAT)
            optim_T.zero_grad()
            optim_N.zero_grad()
            (self.lambda2*AATLoss).backward()
            optim_N.step()
            optim_T.step()

            #L_E-VaT
            EVaTLoss = self.EVatLoss(HNet, HClassifier, batchEVat)
            optim_N.zero_grad()
            optim_H.zero_grad()
            (self.lambda3*EVaTLoss).backward()
            optim_N.step()
            optim_H.step()

            TLoss, TAcc = self.TLoss(HNet,TClassifier, batchTNet)
            optim_N.zero_grad()
            optim_T.zero_grad()
            TLoss.backward()
            optim_T.step()
            optim_N.step()

            MMELoss = self.MMELoss(HNet, HClassifier, batchEVat)
            optim_N.zero_grad()
            optim_H.zero_grad()
            (-1*self.lambda4*MMELoss).backward()
            optim_N.step()
            optim_H.step()

            print('#Model Update (%3d | %3d) #, loss_CE/acc_CE = %6.8f/%6.8f  |  loss_AT/acc_AT = %6.8f/%6.8f | \
                        loss_AAT/acc_AAT = %6.8f/%6.8f  |  loss_EVaT = %6.8f  | loss_T/acc_T = %6.8f/%6.8f  | loss_MME = %6.8f ' % (
                            step, maxStep, loss.data.item(), acc, ATLoss.data.item(), ATAcc, AATLoss.data.item(), AATAcc,
                                EVaTLoss.data.item(), TLoss.data.item(), TAcc,MMELoss.data.item()
            ))
            if (step + 1) % validEvery == 0:
                acc = self.valid(HNet, validSet, validLabel, suffix=f"ValidPerf_{self.suffix}")
                self.valid_step += 1
                if acc > self.best_valid_acc:
                    self.valid(HNet, testSet, testLabel, f"TestPerf_{self.suffix}")
                    torch.save(HNet.state_dict(), self.model_file)
                    self.best_valid_acc = acc
                else:
                    self.valid(HNet, testSet, testLabel, f"Perf_{self.suffix}")
