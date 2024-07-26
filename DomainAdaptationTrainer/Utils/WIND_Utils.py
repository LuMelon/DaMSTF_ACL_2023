import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.autograd import Variable

def to_var(x, requires_grad=True):
    if torch.cuda.is_available() and x.device.type == "cpu":
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def named_params(curr_module=None, memo=None, prefix=''):
    if memo is None:
        memo = set()

    if hasattr(curr_module, 'named_leaves'):  # 如果有"named_leaves", 名字返回为prefix.name, par
        for name, p in curr_module.named_leaves():
            if p is not None and p not in memo and p.requires_grad:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p
    else:
        for name, p in curr_module._parameters.items():  # 如果有参数， 返回XXX,
            if p is not None and p not in memo and p.requires_grad:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p

    for mname, module in curr_module.named_children():  # 如果没有直接的参数，则到下一级各个子模块去寻找
        submodule_prefix = prefix + ('.' if prefix else '') + mname
        for name, p in named_params(module, memo, submodule_prefix):
            yield name, p


def params(model):
    for name, param in named_params(model):
        yield param


def update_params(model:nn.Module, lr_inner, first_order=False, gradients=None, detach=False):
    tgt_device = model.device
    if gradients is not None:
        for tgt, grad in zip(named_params(model), gradients):
            name_t, param_t = tgt
            if first_order:
                grad = to_var(grad.detach().data)
            tmp = param_t - lr_inner * grad.to(tgt_device)
            set_param(model, name_t, tmp)
    else:
        for name, param in named_params(model):
            if not detach:
                grad = param.grad
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param - lr_inner * grad
                set_param(model, name, tmp)
            else:
                param = param.detach_()
                set_param(model, name, param)


def set_param(curr_mod, name, param):
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                set_param(mod, rest, param)
                break
    else:
        if hasattr(curr_mod, 'flatten_parameters'):
            curr_mod._apply(lambda x: x)
        curr_mod._parameters[name] = param

def reset_param_grad(curr_mod, name):
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                reset_param_grad(mod, rest)
                break
    else:
        if hasattr(curr_mod, 'flatten_parameters'):
            curr_mod._apply(lambda x: x)
        curr_mod._parameters[name].grad = None
        curr_mod._parameters[name].grad_fn = None


def detach_params(model):
    for name, param in named_params(model):
        set_param(model, name, param.detach())

def copy_model(model, other, same_var=False):
    for name, param in other.named_params():
        if not same_var:
            param = to_var(param.data.clone(), requires_grad=True)
        set_param(model, name, param)

def update_module(module, new_module=None, updates=None, memo=None):
    r"""
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)

    **Description**

    Updates the parameters of a module in-place, in a way that preserves differentiability.

    The parameters of the module are swapped with their update values, according to:
    \[
    p \gets p + u,
    \]
    where \(p\) is the parameter, and \(u\) is its corresponding update.


    **Arguments**

    * **module** (Module) - The module to update.
    * **updates** (list, *optional*, default=None) - A list of gradients for each parameter
        of the model. If None, will use the tensors in .update attributes.

    **Example**
    ~~~python
    error = loss(model(X), y)
    grads = torch.autograd.grad(
        error,
        model.parameters(),
        create_graph=True,
    )
    updates = [-lr * g for g in grads]
    l2l.update_module(model, updates=updates)
    ~~~
    """
    if new_module is None:
        new_module = module.clone()

    if memo is None:
        memo = {}
    if updates is not None:
        params = list(new_module.parameters())
        if not len(updates) == len(list(params)):
            msg = 'WARNING:update_module(): Parameters and updates have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(updates)) + ')'
            print(msg)
        for p, g in zip(params, updates):
            p.update = g

    # Update the params
    for param_key in module._parameters:
        p = module._parameters[param_key]
        if p is not None and hasattr(p, 'update') and p.update is not None:
            if p in memo:
                new_module._parameters[param_key] = memo[p]
            else:
                updated = p + p.update
                memo[p] = updated
                new_module._parameters[param_key] = updated

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        buff = module._buffers[buffer_key]
        if buff is not None and hasattr(buff, 'update') and buff.update is not None:
            if buff in memo:
                new_module._buffers[buffer_key] = memo[buff]
            else:
                updated = buff + buff.update
                memo[buff] = updated
                new_module._buffers[buffer_key] = updated

    # Then, recurse for each submodule
    for module_key in module._modules:
        new_module._modules[module_key] = update_module(
            module._modules[module_key],
            new_module._modules[module_key],
            updates=None,
            memo=memo,
        )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(module, 'flatten_parameters'):
        new_module._apply(lambda x: x)
    return new_module

def update_current_devices(model, new_device):
    if len(model._modules)==0:
        return
    if hasattr(model, 'device'):
        model.device = new_device
        for mod in model._modules:
            update_current_devices(model._modules[mod], new_device)
    else:
        return

def Similarity(grad_dicts_1, grad_dicts_2):
    sim_list = []
    for name in list(grad_dicts_1.keys()):
        assert name in grad_dicts_2
        sim = (grad_dicts_1[name] * grad_dicts_2[name]).sum()
        sim_list.append(sim)
    sum_sim = torch.stack(sim_list).sum()
    return sum_sim

class Data_Loss_Grad_Utils:
    def __init__(self, class_num=2, lr4model=2e-2, coeff4expandset=1.0, max_few_shot_size=20,
                 Inner_BatchSize=5):
        self.expand_idxs = None
        self.lr4model = lr4model
        self.coefff4expandset = coeff4expandset
        self.batch_size = Inner_BatchSize
        self.max_few_shot_size = max_few_shot_size
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.class_num = class_num

    def FewShotDataList(self, few_shot_set):
        # all classes share the averaged weights, so the sum of weighted loss will be equals to the mean loss
        # so we do not need to discount the loss when accumurating them
        self.f_label_weight = torch.tensor([1.0 / len(few_shot_set) for _ in range(self.class_num)],
                                           device=self.device)
        if len(few_shot_set) > self.max_few_shot_size:
            few_shot_data = None
            few_shot_data_list = [few_shot_set.collate_raw_batch(
                                            [few_shot_set[j] for j in range(i,
                                                                                 min(i+self.max_few_shot_size,
                                                                                     len(few_shot_set)))])
                                            for i in range(0, len(few_shot_set), self.max_few_shot_size)]
        else:
            few_shot_data = few_shot_set.collate_raw_batch(
                [few_shot_set[i] for i in range(len(few_shot_set))]
            )
            few_shot_data_list = None
        return few_shot_data, few_shot_data_list

    def LossList(self, model, batch):
        preds = model.predict(batch)
        # ------------------------------------------------------------#
        # to prevent extremely confident prediction, i.e., [1.0, 0.0],
        # as it can lead to a 'nan' value in log operation
        epsilon = torch.ones([len(preds), self.class_num], dtype=torch.float32, device=self.device) * 1e-8
        preds = (preds - epsilon).abs()
        # -------------------------------------------------------------#
        label = batch[-2].to(self.device) if batch[-2].dim() == 1 else batch[-2].to(self.device).argmax(dim=1)
        loss = F.nll_loss(preds.log(), label, reduce=False)
        return loss

    def GradientsSim(self, model, batch, mean_grad_dict):
        loss = self.LossList(model, batch)
        labels = batch[-2].to(self.device)
        mask_tensor = torch.eye(len(labels), device=self.device)
        grad_sim = []
        for i in range(len(labels)):
            model.zero_grad()
            retain_graph = True if i != (len(labels) - 1) else False
            (mask_tensor[i] * loss).sum().backward(retain_graph=retain_graph)
            grad_dict = {n: p.grad.clone() for n, p in model.named_parameters()}
            grad_sim.append(Similarity(grad_dict, mean_grad_dict))
        return grad_sim

    def metaLossAndAcc(self, model, batch, label_weight=None, reduction='mean'):
        preds = model.predict(batch)
        epsilon = torch.ones_like(preds) * 1e-8
        # to avoid the prediction [1.0, 0.0], which leads to the 'nan' value in log operation
        preds = (preds - epsilon).abs()
        labels = batch[-2]
        labels = labels.argmax(dim=1).to(preds.device) if labels.dim() > 1 else labels.to(preds.device)
        loss = F.nll_loss(preds.log(), labels.to(preds.device), weight=label_weight, reduction=reduction)
        acc_t = ((preds.argmax(dim=1) - labels).__eq__(0).float().sum()) / len(labels)
        acc = acc_t.data.item()
        return loss, acc

    def meanGradOnValSet(self, model, few_shot_data=None, few_shot_data_list=None):
        model.zero_grad()
        assert few_shot_data is not None or few_shot_data_list is not None
        assert hasattr(self, "f_label_weight")
        if few_shot_data_list is None:
            loss, acc = self.metaLossAndAcc(model, few_shot_data, label_weight=self.f_label_weight, reduction='sum')
            loss.backward()
            loss = loss.data.item()
        else:
            print("-------> few shot data list ------>")
            f_loss_list, f_acc_list = [], []
            for i, few_data in enumerate(few_shot_data_list):
                f_loss, f_acc = self.metaLossAndAcc(model, few_data, label_weight=self.f_label_weight, reduction='sum')
                f_loss.backward()
                f_loss_list.append(f_loss.data.item())
                f_acc_list.append(f_acc)
                torch.cuda.empty_cache()
            loss, acc = np.sum(f_loss_list), np.mean(f_acc_list)
        grad_dicts = {n: p.grad.clone() for n, p in model.named_parameters()}
        return grad_dicts, loss, acc

class InstanceReweightingV3(Data_Loss_Grad_Utils):
    def __init__(self, class_num=2, lr4model=2e-2, coeff4expandset=1.0, max_few_shot_size=20,
                 Inner_BatchSize=5):
        super(InstanceReweightingV3, self).__init__(class_num, lr4model, coeff4expandset, max_few_shot_size, Inner_BatchSize)

    def weightsLR(self, eta=0.01, grad_weights=None):
        return eta / grad_weights.abs().mean()

    def ComputeGrads4Weights(self, model, batch, few_shot_data=None,
                                            few_shot_data_list=None):
        assert few_shot_data is not None or few_shot_data_list is not None
        if not hasattr(self, "val_grad_dicts") or self.val_grad_dicts is None:
            self.val_grad_dicts, few_loss, few_acc = self.meanGradOnValSet(model, few_shot_data=few_shot_data,
                                                                          few_shot_data_list=few_shot_data_list)
        sim_list = self.GradientsSim(model, batch, self.val_grad_dicts)
        return torch.tensor(sim_list, device=self.device)

class InstanceReweightingV4(InstanceReweightingV3):
    def __init__(self, class_num=2, lr4model=2e-2, coeff4expandset=1.0, max_few_shot_size=20,
                 Inner_BatchSize=5):
        super(InstanceReweightingV4, self).__init__(class_num, lr4model, coeff4expandset, max_few_shot_size,
                                                    Inner_BatchSize)

    def ComputeGrads4Weights(self, model:nn.Module, batch, few_shot_data=None,
                                            few_shot_data_list=None):
        print("ComputeGrads4Weights")
        assert few_shot_data is not None or few_shot_data_list is not None
        if not hasattr(self, "val_grad_dicts") or self.val_grad_dicts is None:
            self.val_grad_dicts, few_loss, few_acc = self.meanGradOnValSet(model, few_shot_data=few_shot_data,
                                                                          few_shot_data_list=few_shot_data_list)
        if not hasattr(self, "epsilon"):
            self.epsilon = 1e-5

        for name, par in model.named_parameters():
            par.data = par.data + (self.epsilon) * (self.val_grad_dicts[name])
        with torch.no_grad():
            loss_1 = self.LossList(model, batch)

        for name, par in model.named_parameters():
            par.data = par.data - 2* self.epsilon * self.val_grad_dicts[name]
        with torch.no_grad():
            loss_2 = self.LossList(model, batch)

        for name, par in model.named_parameters():
            par.data = par.data + self.epsilon * self.val_grad_dicts[name]

        grad = (loss_1 - loss_2)/(2*self.epsilon)
        return grad

def WeightedAcc(y_true, y_pred, weights):
    diff = y_true - y_pred
    diff = diff.__eq__(0).long() - diff.abs()
    if all([w==0 for w in weights]):
        weights += 1
    half_high = len(weights)//2
    topK_idx = weights.argsort()[-half_high:]
    pos_cnt = (weights>0).sum()
    pos_idx = (weights>0).int().argsort()[-pos_cnt:]
    topK_acc = accuracy_score(y_true[topK_idx].cpu(), y_pred[topK_idx].cpu())
    pos_acc = accuracy_score(y_true[pos_idx].cpu(), y_pred[pos_idx].cpu())
    weights = weights / weights.sum()
    return (diff*weights).sum(), topK_acc, pos_acc