import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from TrainingEnv import VirtualModel, CustomDataset, VirtualEvaluater
from torch.utils.data import Dataset
from typing import AnyStr
from torch.autograd import Variable

def to_var(x, requires_grad=True):
    if torch.cuda.is_available() and x.device.type == "cpu":
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

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


def update_params(model: nn.Module, lr_inner, first_order=False, gradients=None, detach=False):
    tgt_device = model.device
    if gradients is not None:
        for tgt, grad in zip(named_params(model), gradients):
            name_t, param_t = tgt
            # name_s, param_s = src
            # grad = param_s.grad
            # name_s, param_s = src
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

class DARTs_Outoput:
    meta_loss:torch.Tensor=None
    meta_acc:float=None
    weight_grad:torch.Tensor=None
    def __init__(self, meta_loss=None, meta_acc=None, weight_grad=None):
        self.meta_acc = meta_acc
        self.meta_loss = meta_loss
        self.weight_grad = weight_grad

class MetaVirtualModel(VirtualModel):
    def step(self, step_size):
        """
        update the trainable parameters. For example, the trainable parameters in prompt tuning is only the
        prompt embedding vector, so the 'step' operation just needs to update the weights of the prompt embedding.
        :param step_size:
        :return:
        """
        raise NotImplementedError("'step' is not impleted")

    def meta_lossAndAcc(self, valid_batch, temperature=1.0, label_weight=None):
        raise NotImplementedError("'meta_lossAndAcc' is not impleted")

    def paras_dict(self):
        raise NotImplementedError("'paras_dict' is not impleted")

    def load_paras_dict(self, paras_dict):
        raise NotImplementedError("'load_paras_dict' is not impleted")

class MetaLearningFramework:
    ### General Training Parameters ###
    lr4model=5e-5 # learning rate for updating the model's parameters
    device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    max_batch_size=32
    grad_accum_cnt = 1
    valid_every = 100

    ### Meta-Learning Parameters ###
    num_of_meta_valid_samples = 100
    lr4weight = 0.1  # learning rate for updating the hyperparameters
    meta_step = 10  # update the hyperparameters with 'meta_step' steps
    epsilon = 1e-5

    ### Logging Parameters ###
    model_file = "./tmp.pkl"
    marker:AnyStr = "meta_learning" # mark the current trainer to output the customized results
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                print("Warning: MetaLearningFramework does not have attribute {}".format(k))

    def dataset2dataloader(self, dataset:Dataset, shuffle=False, batchsize=32):
        raise NotImplementedError("'dataset2dataloader' is not impleted")

    def gradOnMetaValSet(self, model: MetaVirtualModel, valid_data, temperature=1.0, label_weight=None):
        model.zero_grad()
        if isinstance(valid_data, tuple):
            loss, acc = model.meta_lossAndAcc(valid_data, temperature=temperature, label_weight=label_weight)
            loss.backward()
            loss = loss.data.item()
        elif isinstance(valid_data, list):
            print("-------> few shot data list ------>")
            f_loss_list, f_acc_list = [], []
            for i, few_data in enumerate(valid_data):
                loss, acc = model.meta_lossAndAcc(few_data, temperature=temperature, label_weight=label_weight)
                loss.backward()
                f_loss_list.append(loss.data.item())
                f_acc_list.append(acc)
                torch.cuda.empty_cache()
            loss, acc = np.sum(f_loss_list), np.mean(f_acc_list)
        return loss, acc

    def darts_approximation(self, model:MetaVirtualModel, valid_data, training_batch, temperature=1.0, label_weight=None):
        model.zero_grad()
        meta_loss, meta_acc = self.gradOnMetaValSet(model, valid_data,
                                                    temperature=temperature,
                                                    label_weight=label_weight)
        print(f"======> Meta Validation ===> loss/acc = {meta_loss}/{meta_acc}")
        model.step(-1*self.epsilon)
        with torch.no_grad():
            loss_1 = model.lossList(training_batch, temperature=1.0, label_weight=None)

        model.step(2*self.epsilon)
        with torch.no_grad():
            loss_2 = model.lossList(training_batch, temperature=1.0, label_weight=None)

        model.step(-1*self.epsilon)
        grad = (loss_1 - loss_2)/(2*self.epsilon)
        return DARTs_Outoput(meta_loss, meta_acc, weight_grad=grad)

    def meta_learning(self, model:MetaVirtualModel, train_set:CustomDataset, meta_valid_set:CustomDataset,
                    evaluator_dev:VirtualEvaluater, evaluator_te:VirtualEvaluater, max_epoch= 10):
        train_step = 0
        best_acc = 0.0
        optim = model.obtain_optim(self.lr4model)
        for epoch in range(max_epoch):
            with torch.no_grad():
                logits = model.dataset_logits(meta_valid_set, temperature=1.0, batch_size=20)
                vals, indexs = logits.sort(dim=1)
                probs, preds = vals[:, -1], indexs[:, -1]
            train_loader = self.dataset2dataloader(train_set)
            weights_list, idxs_list = [], []
            for batch in train_loader:
                v_batch = meta_valid_set.collate_fn(
                    [meta_valid_set[m_idx.data.item()] for m_idx in torch.multinomial(
                        probs, self.num_of_meta_valid_samples, replacement=False, generator=None
                    )]
                )
                init_state_dict = model.paras_dict()
                weights, idxs = batch[-3], batch[-4]
                weights_list.append(weights)
                idxs_list.append(idxs)
                for mstep in range(self.meta_step):
                    if mstep != 0 :
                        model.load_paras_dict(init_state_dict)
                    loss_list = model.lossList(batch, temperature=1.0, label_weight=None)
                    u = weights.sigmoid()
                    loss = (u*loss_list).sum()
                    loss.backward()
                    model.step(self.lr4model) # theta hat
                    model.zero_grad()
                    rst = self.darts_approximation(model, v_batch, batch, temperature=1.0, label_weight=None)
                    u_grads = rst.weight_grad
                    w_grads = u_grads * u * (1 - u)
                    self.print_gradient_information(w_grads)
                    normed_wGrads= -1 * (w_grads / (w_grads.norm(2) + 1e-10))
                    weights = weights - self.lr4weight*normed_wGrads
                loss_list = model.lossList(batch, temperature=1.0, label_weight=None)
                normed_weights = weights/(weights.sum())
                loss = (normed_weights * loss_list).sum()
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_step += 1
                if train_step % self.valid_every == 0 and evaluator_dev is not None:
                    acc_v = evaluator_dev(model)
                    if acc_v > best_acc:
                        best_acc = acc_v
                        torch.save(model.state_dict(), self.model_file)
            new_weights = torch.cat(weights_list)[torch.cat(idxs_list)]
            self.print_divergence(new_weights, train_set.weights)# compute the difference of two weights tensors
            train_set.weights = new_weights
        if evaluator_te is not None:
            model.load_state_dict(torch.load(self.model_file))
            evaluator_te(model)

    def print_gradient_information(self, gradient:torch.Tensor):
        g_probs = gradient.softmax(dim=-1)
        entropy = (g_probs * (g_probs.log().neg())).sum()
        print(f"weight gradient informtaion : entropy={entropy}, norm2={gradient.norm()}")

    def print_divergence(self, weights1, weights2):
        pass

