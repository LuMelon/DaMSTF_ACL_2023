import sys, pickle, random, shutil, warnings
sys.path.append("..")
from DomainAdaptationTrainer.CST import *
import DomainAdaptationTrainer.CST_Utils.common.vision.datasets as datasets
import DomainAdaptationTrainer.CST_Utils.common.vision.models as models
from DomainAdaptationTrainer.CST_Utils.common.utils.logger import CompleteLogger
from DomainAdaptationTrainer.CST_Utils.common.utils.analysis import collect_feature, tsne, a_distance
from DomainAdaptationTrainer.CST_Utils.sam import SAM
import torch.backends.cudnn as cudnn
import os.path as osp


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.center_crop:
        train_transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
    else:
        train_transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])

    unlabeled_transform = TransformFixMatch()

    val_transform = T.Compose([
        ResizeImage(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    dataset = datasets.__dict__[args.data]
    train_source_dataset = dataset(root=args.root, task=args.source, download=False, transform=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_dataset = dataset(root=args.root, task=args.target, download=False, transform=unlabeled_transform)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_dataset = dataset(root=args.root, task=args.target, download=False, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    if args.data == 'DomainNet':
        test_dataset = dataset(root=args.root, task=args.target, split='test', download=True, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        test_loader = val_loader

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=True, model_path=args.model_path)
    num_classes = train_source_dataset.num_classes
    args.num_cls = num_classes
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim).to(device)

    # define optimizer and lr scheduler

    base_optimizer = SGD
    optimizer = SAM(classifier.get_parameters(), base_optimizer, lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay, adaptive=True, rho=args.rho)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    ts_loss = TsallisEntropy(temperature=args.temperature, alpha=args.alpha)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.png')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = validate(test_loader, classifier, args)
        print(acc1)
        return

    # start training
    best_acc1 = 0.
    for epoch in range(min(args.epochs, args.early)):
        print("lr:", lr_scheduler.get_last_lr()[0])
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, ts_loss, optimizer,
              lr_scheduler, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, classifier, args)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = validate(test_loader, classifier, args)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()

if __name__ == '__main__':
    with open('./cst_args.pkl', 'rb') as rb:
        args = pickle.load(rb)
    args.root = '/mnt/VMSTORE/OfficeHome/'
    args.model_path = "/mnt/VMSTORE/da-mstf_-paper_-codes/resnet50-19c8e357.pth"
    main(args)

    # Namespace(alpha=1.9, arch='resnet50', batch_size=28, bottleneck_dim=2048, center_crop=False, data='OfficeHome',
    #           early=30, epochs=30, iters_per_epoch=1000, log='logs/cst/OfficeHome_Pr2Rw', lr=0.005, lr_decay=0.75,
    #           lr_gamma=0.001, momentum=0.9, per_class_eval=False, phase='train', print_freq=100, rho=0.5,
    #           root='data/office-home', seed=None, source='Pr', target='Rw', temperature=2.5, threshold=0.97,
    #           trade_off=0.015, trade_off1=2.0, trade_off3=0.5, weight_decay=0.001, workers=2)