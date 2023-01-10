from train_eval_utils import *
from dataset import HumanSegmentationDataset, HumanMattingDataset
from torch.utils.data import DataLoader
from hvsnet import HVSNet, HVSNetV2

import os
import time
import datetime
import torch


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes

    # 用来保存训练以及验证过程中信息
    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_file = "./log/results{}.txt".format(start_time)
    model_root = "./saved/checkpoints{}".format(start_time)

    size = args.size
    if not args.matting:
        train_dataset = HumanSegmentationDataset(args.train_path, size=(size, size),
                                                 mode='train', dilate_high=args.dilate_high)
        val_dataset = HumanSegmentationDataset(args.valid_path, size=(size, size),
                                               mode='eval', dilate_high=args.dilate_high)
    else:
        train_dataset = HumanMattingDataset(args.train_path, size=(size, size),
                                            mode='train', dilate_high=args.dilate_high)
        val_dataset = HumanMattingDataset(args.valid_path, size=(size, size),
                                          mode='eval', dilate_high=args.dilate_high)

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=True)

    if args.version == 1:
        model = HVSNet(num_classes=2, mode='train', stem=args.stem,
                       type='matting' if args.matting else 'segmentation')
    else:
        model = HVSNetV2(type='matting' if args.matting else 'segmentation')
    model.to(device)

    if args.pretrained:
        weights = torch.load(args.pretrained, map_location='cpu')
        for key in list(weights.keys()):
            if key.startswith('up.'):
                del weights[key]
        model.load_state_dict(weights, strict=False)

    params_to_optimize = model.parameters()
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    start_time = time.time()
    args.epochs += args.start_epoch
    for epoch in range(args.start_epoch, args.epochs):
        if not args.matting:
            mean_loss, lr = train_iter(model, optimizer, train_loader, device, epoch, lr_scheduler,
                                       print_freq=args.print_freq, scaler=scaler, alpha=args.alpha, beta=args.beta)
        else:
            mean_loss, lr = train_iter_matting(model, optimizer, train_loader, device, epoch, lr_scheduler,
                                               print_freq=args.print_freq, scaler=scaler, theta=args.theta)

        confmat = evaluate(model, val_loader, device=device, num_classes=num_classes, matting=args.matting)
        val_info = str(confmat)
        print(val_info)
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"
            f.write(train_info + val_info + "\n\n")

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if not os.path.exists(model_root):
            os.mkdir(model_root)
        torch.save(save_file, os.path.join(model_root, "model_{}_{}_{}.pth".format(epoch, model.type, model.version)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")

    parser.add_argument("--train-path", default="./data/train/", help="train dataset root")
    parser.add_argument("--valid-path", default="./data/valid/", help="valid dataset root")
    parser.add_argument('--version', default=1, type=int, help='version')
    parser.add_argument('--matting', default=False,
                        type=lambda x: x.lower() == 'true', help='use matting or not')
    parser.add_argument("--size", default=512, type=int, help='input size')
    parser.add_argument("--stem", default='original', type=str, help='stem')
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=30, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--dilate-high', default=5, type=int,
                        help='High value of dilation')
    parser.add_argument('--alpha', default=0.9, type=float, help='alpha in loss')
    parser.add_argument('--beta', default=0.5, type=float, help='beta in loss')
    parser.add_argument('--theta', default=0.8, type=float, help='theta in loss')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--checkpoint', default='', help='resume from checkpoint')
    parser.add_argument('--pretrained', default='', help='pretrained model')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists('./saved'):
        os.mkdir('./saved')

    if not os.path.exists('./log'):
        os.mkdir('./log')

    main(args)
    pass