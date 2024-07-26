import sys
sys.path.append("../../")
import os, pickle, random, numpy as np
from TrainingEnv import BaseTrainer, AdversarialModel, CustomDataset
import torch, torch.nn as nn, torch.nn.functional as F
from prefetch_generator import background
from SAM import SAM


def Generator1(labeledSource, unlabeledTarget, labeledTarget, batchSize):
    halfBS = batchSize // 2
    bs2 = halfBS if halfBS < len(labeledTarget) else len(labeledTarget)
    bs1 = batchSize - bs2
    bs3 = batchSize // 3 if batchSize < 3 * len(unlabeledTarget) else len(unlabeledTarget)
    iter1, iter2, iter3 = len(labeledSource) // bs1, \
                          len(labeledTarget) // bs2, \
                          len(unlabeledTarget) // bs3
    maxIters = max([iter1, iter2, iter3])

    @background(max_prefetch=5)
    def generator():
        idxsS, idxsLT, idxsUT = [], [], []
        for i in range(maxIters + 1):
            if i % iter1 == 0:
                idxsS = random.sample(range(len(labeledSource)), len(labeledSource)) * 2
            if i % iter2 == 0:
                idxsLT = random.sample(range(len(labeledTarget)), len(labeledTarget)) * 2
            if i % iter3 == 0:
                idxsUT = random.sample(range(len(unlabeledTarget)), len(unlabeledTarget)) * 2
            # i * bs1 could be larger than the len(labelSource), thus we need to start from the remainder
            start_LS, start_LT, start_UT = (i * bs1) % len(labeledSource), \
                                           (i * bs2) % len(labeledTarget), \
                                           (i * bs3) % len(unlabeledTarget)
            end_LS, end_LT, end_UT = start_LS + bs1, start_LT + bs2, start_UT + bs3

            items1 = [labeledSource[jj] for jj in idxsS[start_LS:end_LS]]
            items2 = [labeledTarget[jj] for jj in idxsLT[start_LT:end_LT]]
            items3 = [unlabeledTarget[jj] for jj in idxsUT[start_UT:end_UT]]
            batch1 = labeledTarget.collate_raw_batch(items1 + items2)
            batch2 = unlabeledTarget.collate_raw_batch(
                random.sample(items2, batchSize - bs1 - bs3) + items1 + items3
            )
            yield batch1, batch2
    return maxIters, generator

def Generator2(labeledSource, unlabeledTarget, batchSize):
    bs2 = batchSize // 2 if batchSize < 2 * len(unlabeledTarget) else len(unlabeledTarget)
    iter1, iter2 = len(labeledSource) // batchSize, \
                   len(unlabeledTarget) // bs2
    maxIters = max([iter1, iter2])

    @background(max_prefetch=5)
    def generator():
        idxsS, idxsUT = [], []
        for i in range(maxIters + 1):
            if i % iter1 == 0:
                idxsS = random.sample(range(len(labeledSource)), len(labeledSource)) * 2
            if i % iter2 == 0:
                idxsUT = random.sample(range(len(unlabeledTarget)), len(unlabeledTarget)) * 2
            # i * bs1 could be larger than the len(labelSource), thus we need to start from the remainder
            start_LS, start_UT = (i * batchSize) % len(labeledSource), \
                                    (i * bs2) % len(unlabeledTarget)
            end_LS, end_UT = start_LS + batchSize, start_UT + bs2

            items1 = [labeledSource[jj] for jj in idxsS[start_LS:end_LS]]
            items2 = [unlabeledTarget[jj] for jj in idxsUT[start_UT:end_UT]]
            batch1 = labeledSource.collate_raw_batch(items1)
            batch2 = unlabeledTarget.collate_raw_batch(
                random.sample(items1, batchSize - bs2) + items2
            )
            yield batch1, batch2 # batch1 for CE Loss, batch2 for DA Loss
    return maxIters, generator

def Generator3(labeledSource, unlabeledTarget, batchSize):
    bs2 = batchSize // 2 if batchSize < 2 * len(unlabeledTarget) else len(unlabeledTarget)
    bs1 = batchSize - bs2
    iter1, iter2 = len(labeledSource) // bs1, \
                   len(unlabeledTarget) // bs2
    maxIters = max([iter1, iter2])

    @background(max_prefetch=5)
    def generator():
        idxsS, idxsUT = [], []
        for i in range(maxIters + 1):
            if i % iter1 == 0:
                idxsS = random.sample(range(len(labeledSource)), len(labeledSource)) * 2
            if i % iter2 == 0:
                idxsUT = random.sample(range(len(unlabeledTarget)), len(unlabeledTarget)) * 2
            # i * bs1 could be larger than the len(labelSource), thus we need to start from the remainder

            start_LS, start_UT = (i * bs1) % len(labeledSource), \
                                    (i * bs2) % len(unlabeledTarget)
            end_LS, end_UT = start_LS + bs1, start_UT + bs2

            items1 = [labeledSource[jj] for jj in idxsS[start_LS:end_LS]]
            items2 = [unlabeledTarget[jj] for jj in idxsUT[start_UT:end_UT]]
            batch = unlabeledTarget.collate_raw_batch(
                items1 + items2
            )
            yield batch # batch1 for CE Loss, batch2 for DA Loss
    return maxIters, generator

def DataIter(labeledSource, unlabeledTarget, labeledTarget=None, batchSize=32):
    if labeledTarget is not None:
        assert len(labeledTarget) > 0
        return  Generator1(labeledSource, unlabeledTarget, labeledTarget, batchSize)
    else:
        return Generator2(labeledSource, unlabeledTarget, batchSize)


class DANNTrainer(BaseTrainer):
    def __init__(self, random_seed, log_dir, suffix, model_file, class_num, temperature=1.0,
                 learning_rate=5e-3, batch_size=32, Lambda=0.1):
        super(DANNTrainer, self).__init__()
        if not os.path.exists(log_dir):
            os.system("mkdir {}".format(log_dir))
        self.random_seed = random_seed
        self.log_dir = log_dir
        self.suffix = suffix
        self.model_file = model_file
        self.best_valid_acc = 0.0
        self.min_valid_loss = 1e8
        self.class_num = class_num
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.Lambda = Lambda
        self.valid_step = 0

    def PreTrainDomainClassifier(self, trModel:nn.Module, discriminator:nn.Module,
                                    labeledSource : CustomDataset, labeledTarget : CustomDataset,
                                    unlabeledTarget : CustomDataset, maxEpoch, learning_rate=5e-3):
        # the rule for early stop: when the variance of the recent 50 training loss is smaller than 0.05,
        # the training process will be stopped
        assert hasattr(trModel, 'AdvDLossAndAcc')
        paras =  trModel.grouped_parameters(self.learning_rate) + \
                 [{"params": discriminator.parameters(), "lr": self.learning_rate}]
        optim = torch.optim.Adam(paras, lr=self.learning_rate)
        lossList = []
        for epoch in range(maxEpoch):
            maxIters, trainLoader = DataIter(labeledSource, unlabeledTarget, labeledTarget, self.batch_size)
            for step, (_, batch2) in enumerate(trainLoader()):
                DLoss, DAcc = trModel.discriminatorLoss(discriminator, batch2, grad_reverse=False)
                optim.zero_grad()
                DLoss.backward()
                optim.step()
                torch.cuda.empty_cache()
                print('####Pre Train Domain Classifier (%3d | %3d) %3d | %3d ####, loss = %6.8f, Acc = %6.8f' % (
                    step, maxIters, epoch, maxEpoch, DLoss.data.item(), DAcc
                ))
                lossList.append(DLoss.data.item())
                if len(lossList) > 20:
                    lossList.pop(0)
                    if np.std(lossList) < 0.05 and np.mean(lossList) < 0.2:
                        return

    def ModelTrain(self, trModel : AdversarialModel, discriminator:nn.Module,
                    labeledSource : CustomDataset, labeledTarget : CustomDataset,
                    unlabeledTarget : CustomDataset, validSet : CustomDataset,
                    testSet: CustomDataset, maxEpoch, validEvery=20, test_label=None):
        assert hasattr(trModel, 'AdvDLossAndAcc')
        print("labeled Source/labeled Target/unlabeled Target: {}/{}/{}".format(len(labeledSource),
                                                                                len(labeledTarget) if labeledTarget is not None else 0,
                                                                                len(unlabeledTarget)))
        paras =  trModel.grouped_parameters(self.learning_rate) + \
                 [{"params": discriminator.parameters(), "lr": self.learning_rate}]
        optim = torch.optim.Adam(paras, lr=self.learning_rate)
        for epoch in range(maxEpoch):
            maxIters, trainLoader = DataIter(labeledSource, unlabeledTarget, labeledTarget, self.batch_size)
            for step, (batch1, batch2) in enumerate(trainLoader()):
                loss, acc = trModel.lossAndAcc(trModel, batch1)
                DLoss, DAcc = trModel.advLossAndAccV2(discriminator, batch2)
                trainLoss = loss + self.Lambda*DLoss
                optim.zero_grad()
                trainLoss.backward()
                optim.step()

                torch.cuda.empty_cache()
                print('####Model Update (%3d | %3d) %3d | %3d ####, loss/acc = %6.8f/%6.8f, DLoss/DAcc = %6.8f/%6.8f' % (
                    step, maxIters, epoch, maxEpoch, loss.data.item(), acc, DLoss.data.item(), DAcc
                ))
                if (step+1) % validEvery == 0:
                    acc_v = self.valid(trModel, validSet, validSet.labelTensor(), suffix=f"Valid_{self.suffix}")
                    if acc_v > self.best_valid_acc:
                        torch.save(trModel.state_dict(), self.model_file)
                        self.best_valid_acc = acc_v
                        self.valid(
                            trModel, testSet, testSet.labelTensor() if test_label is None else test_label,
                            suffix=f"BestTest_{self.suffix}"
                        )
                    else:
                        self.valid(
                            trModel, testSet, testSet.labelTensor() if test_label is None else test_label,
                            suffix=f"Test_{self.suffix}"
                        )

    def ModelTrainV2(self, trModel : AdversarialModel, discriminator:nn.Module,
                    labeledSource : CustomDataset, labeledTarget : CustomDataset,
                    unlabeledTarget : CustomDataset, validSet : CustomDataset, testSet: CustomDataset=None,
                    maxEpoch=50, validEvery=20, D_Step=5, T_Step=1):
        assert hasattr(trModel, 'AdvDLossAndAcc')
        print("labeled Source/labeled Target/unlabeled Target: {}/{}/{}".format(len(labeledSource),
                                                                                len(labeledTarget) if labeledTarget is not None else 0,
                                                                                len(unlabeledTarget)))
        paras =  trModel.grouped_parameters(self.learning_rate) + \
                 [{"params": discriminator.parameters(), "lr": self.learning_rate}]
        optim = torch.optim.Adam(paras, lr=self.learning_rate)

        for epoch in range(maxEpoch):
            maxIters, trainLoader = DataIter(labeledSource, unlabeledTarget, labeledTarget, self.batch_size)
            for step, (batch1, batch2) in enumerate(trainLoader()):
                optim.zero_grad()
                for da_idx in range(D_Step):
                    DLoss, DAcc = trModel.advLossAndAcc(discriminator, batch2)
                    optim.zero_grad()
                    (self.Lambda*DLoss).backward()
                    optim.step()
                    print('####Domain Adversarial [%3d] (%3d | %3d) %3d | %3d #### DLoss/DAcc = %6.8f/%6.8f' % (
                        da_idx, step, maxIters, epoch, maxEpoch, DLoss.data.item(), DAcc
                    ))
                optim.step()

                for td_idx in range(T_Step):
                    loss, acc = trModel.lossAndAcc(batch1)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    print('****Task Discriminative [%3d] (%3d | %3d) %3d | %3d **** Loss/Acc = %6.8f/%6.8f' % (
                        td_idx, step, maxIters, epoch, maxEpoch, loss.data.item(), acc
                    ))
                torch.cuda.empty_cache()
                if (step+1) % validEvery == 0:
                    acc_v = self.valid(trModel, validSet, validSet.labelTensor(), suffix=f"Valid_{self.suffix}")
                    if acc_v > self.best_valid_acc:
                        torch.save(trModel.state_dict(), self.model_file)
                        self.best_valid_acc = acc_v
                        self.valid(trModel, testSet, testSet.labelTensor(), suffix=f"BestTest_{self.suffix}")
                    else:
                        self.valid(trModel, testSet, testSet.labelTensor(), suffix=f"Test_{self.suffix}")


class GpDANNTrainer(DANNTrainer):
    def GpDaNNTrain(self, trModel : AdversarialModel, discriminator:nn.Module, labeledSource : CustomDataset,
                        unlabeledTarget : CustomDataset, maxEpoch):
        assert hasattr(trModel, "AdvDLossAndAcc")
        paras =  trModel.grouped_parameters(self.learning_rate) + \
                 [{"params": discriminator.parameters(), "lr": self.learning_rate}]
        base_optimizer = torch.optim.SGD
        optim = SAM(paras, base_optimizer, lr=self.learning_rate, momentum=0.9)

        for epoch in range(maxEpoch):
            maxIters, trainLoader = Generator3(labeledSource, unlabeledTarget, self.batch_size)
            for step, batch in enumerate(trainLoader()):
                loss, acc =trModel.AdvDLossAndAcc(discriminator, batch)
                loss.backward()
                optim.first_step(zero_grad=True)
                loss, acc = trModel.AdvDLossAndAcc(discriminator, batch)
                loss.backward()
                optim.second_step(zero_grad=True)
                print('####Model Update (%3d | %3d) %3d | %3d #### DLoss/DAcc = %6.8f/%6.8f' % (
                    step, maxIters, epoch, maxEpoch, loss.data.item(), acc
                ))

    def TaskTrain(self, trModel : AdversarialModel, labeledSource : CustomDataset, labeledTarget : CustomDataset,
                        validSet : CustomDataset, testSet: CustomDataset, maxEpoch, validEvery=20, test_label=None):
        print("labeled Source/labeled Target: {}/{}".format(
                len(labeledSource),
                len(labeledTarget) if labeledTarget is not None else 0,
            )
        )
        for epoch in range(maxEpoch):
            maxIters, trainLoader = Generator3(labeledSource, labeledTarget, self.batch_size)
            for step, (batch1, batch2) in enumerate(trainLoader()):
                loss, acc = trModel.lossAndAcc(batch1, temperature=self.temperature)
                loss.backward()
                torch.cuda.empty_cache()
                print('####Model Update (%3d | %3d) %3d | %3d ####, loss/acc = %6.8f/%6.8f' % (
                    step, maxIters, epoch, maxEpoch, loss.data.item(), acc
                ))
                if (step+1) % validEvery == 0:
                    acc_v = self.valid(trModel, validSet, validSet.labelTensor(), suffix=f"Valid_{self.suffix}")
                    if acc_v > self.best_valid_acc:
                        torch.save(trModel.state_dict(), self.model_file)
                        self.best_valid_acc = acc_v
                        self.valid(
                            trModel, testSet, testSet.labelTensor() if test_label is None else test_label,
                            suffix=f"BestTest_{self.suffix}"
                        )
                    else:
                        self.valid(
                            trModel, testSet, testSet.labelTensor() if test_label is None else test_label,
                            suffix=f"Test_{self.suffix}"
                        )

    def ModelTrain(self, trModel : AdversarialModel, discriminator:nn.Module,
                    labeledSource : CustomDataset, labeledTarget : CustomDataset,
                    unlabeledTarget : CustomDataset, validSet : CustomDataset,
                    testSet: CustomDataset, maxEpoch, validEvery=20, test_label=None):
        print("labeled Source/labeled Target/unlabeled Target: {}/{}/{}".format(len(labeledSource),
                                                                                len(labeledTarget) if labeledTarget is not None else 0,
                                                                                len(unlabeledTarget)))

        self.GpDaNNTrain(trModel, discriminator, labeledSource, unlabeledTarget, maxEpoch=10)
        self.TaskTrain(trModel, labeledSource, labeledTarget, validSet, testSet, maxEpoch=10, validEvery=validEvery)
        for alternate in range(20):
            self.GpDaNNTrain(trModel, discriminator, labeledSource, unlabeledTarget, maxEpoch=3)
            self.TaskTrain(trModel, labeledSource, labeledTarget, validSet, testSet, maxEpoch=3,
                                validEvery=validEvery, test_label=None)
