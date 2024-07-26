import sys
sys.path.append("../../")
from DomainAdaptationTrainer.Utils.WIND_Utils import InstanceReweightingV4
import torch, fitlog, random
import torch.nn as nn
from torch.utils.data import Dataset
from TrainingEnv import BaseTrainer


class WindTrainer(InstanceReweightingV4, BaseTrainer):
    def __init__(self, class_num, log_dir, suffix, weight_eta=0.1, lr4model=2e-2,
                 coeff4expandset=1.0, max_few_shot_size=20, Inner_BatchSize=5, meta_step=5):
        super(WindTrainer, self).__init__(class_num, lr4model, coeff4expandset, max_few_shot_size,
                                                      Inner_BatchSize)

        self.log_dir = log_dir
        fitlog.set_log_dir(log_dir, new_log=True)
        self.suffix = suffix
        self.weight_eta = weight_eta
        self.best_valid_acc = 0.0
        self.meta_step = meta_step

    def MetaStep(self, model:nn.Module, optim:torch.optim, batch,
                    weight:torch.Tensor, weight_mask):
        assert hasattr(self, "few_shot_data")
        assert hasattr(self, "few_shot_data_list")
        initStateDicts = model.state_dict()
        initStateDicts = {key: initStateDicts[key].clone() for key in initStateDicts}
        for step in range(self.meta_step):
            u = weight.sigmoid()
            model.zero_grad()
            loss = self.LossList(model, batch)
            sumLoss = (u * loss).sum()
            sumLoss.backward()
            optim.step()
            self.val_grad_dicts, fewLoss, fewAcc = self.meanGradOnValSet(model,
                                                                         few_shot_data=self.few_shot_data,
                                                                         few_shot_data_list=self.few_shot_data_list)
            print(f"##Perf on Meta Val Set## {step} | {self.meta_step} :  loss/acc = {fewLoss}/{fewAcc}")
            model.load_state_dict(initStateDicts)
            u_grads = self.ComputeGrads4Weights(model, batch, self.few_shot_data, self.few_shot_data_list)
            w_grads = u_grads*u*(1-u)
            weightGrads = -1 * (w_grads / (w_grads.norm(2)+1e-10))
            print("uGrads:", u_grads)
            print("wGrads:", w_grads)
            print("weightGrads:", weightGrads)
            update = self.weight_eta * weightGrads
            weight = weight - update*(weight_mask.to(update.device))
        return weight

    def OptimStep(self, model, model_optim, batch, weight):
        loss = self.LossList(model, batch)
        sumLoss = ((weight.sigmoid()) * loss).sum()
        sumLoss.backward()
        model_optim.step()

    def dataIter(self, OOD_Set, InD_Set=None, batch_size=32):
        p_idxs = list(range(len(OOD_Set)))
        p_len = len(p_idxs)
        if InD_Set is None:
            l_len = 0
            l_idxs = []
        else:
            l_idxs = list(range(len(InD_Set)))
            l_len = len(l_idxs)

        if not hasattr(self, 'collate_fn'):
            collate_fn = OOD_Set.collate_raw_batch
        else:
            collate_fn = self.collate_fn

        data_size = p_len + l_len
        idxs = random.sample(range(data_size), data_size)*2
        for start_i in range(0, data_size, batch_size):
            batch_idxs = idxs[(start_i):(start_i+batch_size)]
            items = [OOD_Set[p_idxs[idx]] if idx < p_len else \
                        InD_Set[l_idxs[idx-p_len]] for idx in batch_idxs]
            yield collate_fn(items), batch_idxs, \
                    torch.tensor([1. if idx < p_len else 0. for idx in batch_idxs])

    def Training(self, model:nn.Module, train_set:Dataset, valid_set:Dataset,
                 test_set:Dataset, indomain_set:Dataset=None, max_epoch=100, max_valid_every=100,
                 model_file="./tmp.pkl"):

        meta_optim = torch.optim.SGD([
            {'params': model.parameters(), 'lr': self.lr4model}
        ])
        model_optim = torch.optim.Adam([
            {'params': model.parameters(), 'lr': self.lr4model}
        ])
        self.few_shot_data, self.few_shot_data_list = self.FewShotDataList(valid_set)
        weights = [0.0]*len(train_set) + \
                    ([] if indomain_set is None else [10.0]*len(indomain_set))
        self.train_set_weights = torch.tensor(weights, device=self.device)
        test_label = test_set.labelTensor()
        test_label = test_label if test_label.dim() == 1 else test_label.argmax(dim=1)
        step = 0
        for epoch in range(max_epoch):
            for batch, indices, weight_mask in self.dataIter(train_set, indomain_set, self.batch_size):
                weights = self.train_set_weights[indices]
                new_weights = self.MetaStep(model, meta_optim, batch, weights, weight_mask)
                self.train_set_weights[indices] = new_weights if new_weights.dtype == torch.float32 else \
                            torch.as_tensor(new_weights, dtype=self.train_set_weights.dtype, device=new_weights.device)
                self.OptimStep(model, model_optim, batch, new_weights)
                if (step+1) % max_valid_every == 0:
                    self.valid(model, test_set, test_label, self.suffix, step)
                step += 1
        model.save_model(model_file)
