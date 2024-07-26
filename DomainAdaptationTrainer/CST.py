import time, argparse, torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
from DomainAdaptationTrainer.CST_Utils.common.vision.transforms import ResizeImage
from DomainAdaptationTrainer.CST_Utils.common.utils.data import ForeverDataIterator
from DomainAdaptationTrainer.CST_Utils.common.utils.metric import accuracy, ConfusionMatrix
from DomainAdaptationTrainer.CST_Utils.common.utils.meter import AverageMeter, ProgressMeter
from DomainAdaptationTrainer.CST_Utils.randaugment import rand_augment_transform
from DomainAdaptationTrainer.CST_Utils.fix_utils import ImageClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rgb_mean = (0.485, 0.456, 0.406)
ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )

def entropy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H

class TsallisEntropy(nn.Module):
    def __init__(self, temperature: float, alpha: float):
        super(TsallisEntropy, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        N, C = logits.shape

        pred = F.softmax(logits / self.temperature, dim=1)
        entropy_weight = entropy(pred).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (N * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)

        sum_dim = torch.sum(pred * entropy_weight, dim=0).unsqueeze(dim=0)

        return 1 / (self.alpha - 1) * torch.sum(
            (1 / torch.mean(sum_dim) - torch.sum(pred ** self.alpha / sum_dim * entropy_weight, dim=-1)))


class TransformFixMatch(object):
    def __init__(self):
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.weak = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
        self.strong = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
            T.ToTensor(),
            normalize,
        ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return weak, strong


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model: ImageClassifier, ts: TsallisEntropy, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    rev_losses = AverageMeter('CST Loss', ':3.2f')
    fix_losses = AverageMeter('Fix Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, rev_losses, fix_losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        (x_t, x_t_u), _ = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        x_t_u = x_t_u.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_t_u, _ = model(x_t_u)

        f_s, f_t = f.chunk(2, dim=0)
        y_s, y_t = y.chunk(2, dim=0)

        # generate target pseudo-labels
        max_prob, pred_u = torch.max(F.softmax(y_t), dim=-1)
        Lu = (F.cross_entropy(y_t_u, pred_u,
                              reduction='none') * max_prob.ge(args.threshold).float().detach()).mean()

        # compute cst
        target_data_train_r = f_t
        target_data_train_r = target_data_train_r / (
            torch.norm(target_data_train_r, dim=-1).reshape(target_data_train_r.shape[0], 1))
        target_data_test_r = f_s
        target_data_test_r = target_data_test_r / (
            torch.norm(target_data_test_r, dim=-1).reshape(target_data_test_r.shape[0], 1))
        target_gram_r = torch.clamp(target_data_train_r.mm(target_data_train_r.transpose(dim0=1, dim1=0)), -0.99999999,
                                    0.99999999)
        target_kernel_r = target_gram_r
        test_gram_r = torch.clamp(target_data_test_r.mm(target_data_train_r.transpose(dim0=1, dim1=0)), -0.99999999,
                                  0.99999999)
        test_kernel_r = test_gram_r
        target_train_label_r = torch.nn.functional.one_hot(pred_u, args.num_cls) - 1 / float(args.num_cls)
        target_test_pred_r = test_kernel_r.mm(
            torch.inverse(target_kernel_r + 0.001 * torch.eye(args.batch_size).cuda())).mm(target_train_label_r)
        reverse_loss = nn.MSELoss()(target_test_pred_r,
                                    torch.nn.functional.one_hot(labels_s, args.num_cls) - 1 / float(args.num_cls))

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = ts(y_t)

        if Lu != 0:
            loss = cls_loss + transfer_loss * args.trade_off + reverse_loss * args.trade_off1 + Lu * args.trade_off3
        else:
            loss = cls_loss + transfer_loss * args.trade_off + reverse_loss * args.trade_off1

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))
        rev_losses.update(reverse_loss.item(), x_s.size(0))
        fix_losses.update(Lu.item(), x_s.size(0))

        # compute gradient and do the first SGD step
        loss.backward()
        optimizer.first_step(zero_grad=True)
        lr_scheduler.step()

        # compute gradient and do the second SGD step

        y, f = model(x)
        y_t_u, _ = model(x_t_u)

        f_s, f_t = f.chunk(2, dim=0)
        y_s, y_t = y.chunk(2, dim=0)

        # generate target pseudo-labels
        max_prob, pred_u = torch.max(F.softmax(y_t), dim=-1)
        Lu = (F.cross_entropy(y_t_u, pred_u,
                              reduction='none') * max_prob.ge(args.threshold).float().detach()).mean()

        # compute cst
        target_data_train_r = f_t
        target_data_train_r = target_data_train_r / (
            torch.norm(target_data_train_r, dim=-1).reshape(target_data_train_r.shape[0], 1))
        target_data_test_r = f_s
        target_data_test_r = target_data_test_r / (
            torch.norm(target_data_test_r, dim=-1).reshape(target_data_test_r.shape[0], 1))
        target_gram_r = torch.clamp(target_data_train_r.mm(target_data_train_r.transpose(dim0=1, dim1=0)), -0.99999999,
                                    0.99999999)
        target_kernel_r = target_gram_r
        test_gram_r = torch.clamp(target_data_test_r.mm(target_data_train_r.transpose(dim0=1, dim1=0)), -0.99999999,
                                  0.99999999)
        test_kernel_r = test_gram_r
        target_train_label_r = torch.nn.functional.one_hot(pred_u, args.num_cls) - 1 / float(args.num_cls)
        target_test_pred_r = test_kernel_r.mm(
            torch.inverse(target_kernel_r + 0.001 * torch.eye(args.batch_size).cuda())).mm(target_train_label_r)
        reverse_loss = nn.MSELoss()(target_test_pred_r,
                                    torch.nn.functional.one_hot(labels_s, args.num_cls) - 1 / float(args.num_cls))

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = ts(y_t)

        if Lu != 0:
            loss1 = cls_loss + transfer_loss * args.trade_off + reverse_loss * args.trade_off1 + Lu * args.trade_off3
        else:
            loss1 = cls_loss + transfer_loss * args.trade_off + reverse_loss * args.trade_off1

        loss1.backward()
        optimizer.second_step(zero_grad=True)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        classes = val_loader.dataset.classes
        confmat = ConfusionMatrix(len(classes))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output, _ = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        if confmat:
            print(confmat.format(classes))
    return top1.avg