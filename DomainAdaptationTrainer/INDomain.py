from TrainingEnv import *

class VirtualDataloader:
    def __iter__(self):
        pass

    def __next__(self):
        pass


class OriginalTrainer:
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