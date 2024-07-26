from DomainAdaptationTrainer.Utils.SelfTrainingFramework import SelfTraining
import os, fitlog, torch
from TrainingEnv import VirtualModel


class FixMatch_Trainer(SelfTraining):

    def strong_data_augmentation(self, model, training_batch):
        raise NotImplementedError("'strong_data_augmentation' is not impleted")

    def training_loss_and_acc(self, model:VirtualModel, training_batch, temperature=1.0, label_weight:torch.Tensor=None,
                                reduction='mean'):
        aug_batch = self.strong_data_augmentation(model, training_batch)
        return model.lossAndAcc(aug_batch, temperature, label_weight, reduction)