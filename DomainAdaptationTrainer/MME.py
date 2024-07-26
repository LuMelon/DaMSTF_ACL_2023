import sys
sys.path.append("../../")
import torch, random, os
import torch.nn as nn
import torch.nn.functional as F
from TrainingEnv import GradientReversal, BaseTrainer, CustomDataset, VirtualModel
from prefetch_generator import background

@background(max_prefetch=3)
def Generator3(labeledSource, unlabeledTarget, labeledTarget, batchSize):
    halfBS = batchSize // 2
    bs2 = halfBS if halfBS < len(labeledTarget) else len(labeledTarget)
    bs1 = batchSize - bs2
    bs3 = batchSize // 3 if batchSize < 3 * len(unlabeledTarget) else len(unlabeledTarget)
    iter1, iter2, iter3 = len(labeledSource) // bs1, \
                          len(labeledTarget) // bs2, \
                          len(unlabeledTarget) // bs3
    maxIters = max([iter1, iter2, iter3])
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
        batch1 = labeledTarget.collate_fn(items1 + items2)
        batch2 = unlabeledTarget.collate_fn(
            random.sample(items2, batchSize - bs1 - bs3) + items1 + items3
        )
        yield batch1, batch2

@background(max_prefetch=3)
def Generator2(labeledSource, unlabeledTarget, batchSize):
    bs2 = batchSize // 2 if batchSize < 2 * len(unlabeledTarget) else len(unlabeledTarget)
    iter1, iter2 = len(labeledSource) // batchSize, \
                   len(unlabeledTarget) // bs2
    maxIters = max([iter1, iter2])
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
        batch1 = labeledSource.collate_fn(items1)
        batch2 = unlabeledTarget.collate_fn(
            random.sample(items1, batchSize - bs2) + items2
        )
        yield batch1, batch2 # batch1 for CE Loss, batch2 for DA Loss

class MMETrainer(BaseTrainer):
    def __init__(self, seed, log_dir, suffix, model_file, class_num, temperature=1.0,
                 learning_rate=5e-3, batch_size=32, Lambda=0.1):
        super(MMETrainer, self).__init__()
        self.seed = seed
        if not os.path.exists(log_dir):
            os.system("mkdir %s" % log_dir)
        else:
            os.system("rm -rf %s" % log_dir)
            os.system("mkdir %s" % log_dir)
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

    def AdvEntrophy(self, trModel:VirtualModel, batch):
        assert hasattr(trModel, "AdvPredict")
        preds = trModel.AdvPredict(batch)
        epsilon = torch.ones_like(preds) * (1e-8)
        preds = (preds - epsilon).abs()
        loss = -1* self.Lambda * (preds * (preds.log())).sum(dim=1).mean()
        return loss

    def ModelTrain(self, trModel: VirtualModel, labeled_source: CustomDataset, labeled_target: CustomDataset,
                        unlabeled_target: CustomDataset, valid_target: CustomDataset, test_target:CustomDataset,
                            maxEpoch, validEvery=20, valid_label=None, test_label=None):
        self.initTrainingEnv(self.seed)
        optim = trModel.obtain_optim()
        for epoch in range(maxEpoch):
            for step, (batch1, batch2) in enumerate(Generator3(labeled_source, unlabeled_target, labeled_target, self.batch_size)):
                loss, acc = trModel.lossAndAcc(batch1, temperature=self.temperature)
                trainLoss = loss
                optim.zero_grad()
                trainLoss.backward()
                optim.step()

                HEntrophy = self.AdvEntrophy(trModel, batch2)
                optim.zero_grad()
                HEntrophy.backward()
                optim.step()
                torch.cuda.empty_cache()
                print('####Model Update (%3d | %3d) %3d | %3d ####, loss = %6.8f, entrophy = %6.8f' % (
                    step, len(unlabeled_target), epoch, maxEpoch, loss, HEntrophy.data.item()
                ))
                if (step + 1) % validEvery == 0:
                    self.valid(
                        trModel, valid_target, valid_target.labelTensor() if valid_label is None else valid_label,
                        suffix=f"{self.suffix}_valid"
                    )
                    self.valid(
                        trModel, test_target, test_target.labelTensor() if test_label is None else test_label,
                        suffix=f"{self.suffix}_test"
                    )

    def ModelTrainV2(self, trModel: VirtualModel, discriminator:nn.Module, labeled_source: CustomDataset, labeled_target: CustomDataset,
                        unlabeled_target: CustomDataset, valid_target: CustomDataset, test_target:CustomDataset,
                         maxEpoch=10, validEvery=20, E_Step=1, T_Step=5):
        self.initTrainingEnv(10086)
        optim_G = trModel.obtain_optim()
        optim_C = discriminator.obtain_optim()
        for epoch in range(maxEpoch):
            if labeled_target is None:
                trainLoader = Generator2(labeled_source, unlabeled_target, labeled_target, self.batch_size)
            for step, (batch1, batch2) in enumerate(trainLoader):
                for da_idx in range(E_Step):
                    HEntrophy = self.AdvEntrophy(trModel, batch2)
                    optim_C.zero_grad()
                    (-1.0 * self.Lambda * HEntrophy).backward()
                    optim_C.step()
                    print('####Domain Adversarial %3d [%3d] %3d | %3d #### T_Entrophy = %6.8f' % (
                        da_idx, step, epoch, maxEpoch, HEntrophy.data.item()
                    ))

                for td_idx in range(T_Step):
                    loss, acc = trModel.lossAndAcc(batch1, self.temperature)
                    optim_G.zero_grad()
                    loss.backward()
                    optim_G.step()
                    print('****Task Discriminative %3d [%3d] %3d | %3d **** Loss/Acc = %6.8f/%6.8f' % (
                        td_idx, step, epoch, maxEpoch, loss.data.item(), acc
                    ))

                if (step + 1) % validEvery == 0:
                    self.valid(trModel, valid_target, valid_target.labelTensor(), suffix=f"{self.suffix}_valid")
                    self.valid(trModel, test_target, test_target.labelTensor(), suffix=f"{self.suffix}_test")
