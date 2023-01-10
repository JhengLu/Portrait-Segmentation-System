from dataset import HumanSegmentationDataset
from tqdm import tqdm
from torch.nn import functional as F

import torch
import torch.nn as nn
import distributed_utils as utils
import numpy as np


def calculate_label_loss(output, label, trimap, beta=0.5):
    loss = F.cross_entropy(output, label)
    if beta == 1: return loss

    loss_important = F.cross_entropy(output, label, ignore_index=255)
    label_important = label.clone()
    label_important[trimap != 128] = 255
    return beta * loss + (1 - beta) * loss_important


detail_weights = np.array([0.1, 0.1, 0.8])
detail_weights = torch.from_numpy(detail_weights).to(dtype=torch.float32)


def calculate_detail_loss(output, label_edge):
    return F.cross_entropy(output, label_edge, detail_weights)


def calculate_loss(output, detail_output, label, label_edge, trimap, alpha=0.9, beta=0.5):
    if alpha == 1:
        return calculate_label_loss(output, label, trimap, beta)

    return calculate_label_loss(output, label, trimap, beta) * alpha + \
           calculate_detail_loss(detail_output, label_edge) * (1 - alpha)

def calculate_alpha_loss(output, matte, eps=1e-4):
    return torch.sqrt(torch.pow(output - matte, 2) + eps ** 2).mean()

def calculate_matting_loss(output, matte, trimap, theta=0.8):
    return F.l1_loss(output, matte) * theta + \
           F.l1_loss(output[trimap == 128], matte[trimap == 128]) * (1 - theta)
    # return calculate_alpha_loss(output, matte) * theta + \
    #        calculate_alpha_loss(output[trimap == 128], matte[trimap == 128]) * (1 - theta)


def train_iter(model, optimizer, dataloader, device, epoch,
               lr_scheduler, print_freq=10, scaler=None, alpha=0.9, beta=0.5):
    model.train()
    model.mode = 'train'

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    lr = 0

    global detail_weights
    detail_weights = detail_weights.to(device)

    for image, label, label_edge, trimap in metric_logger.log_every(dataloader, print_freq, header):

        image = image.to(device)
        label = label.to(device)
        label_edge = label_edge.to(device)
        trimap = trimap.to(device)

        output, detail_output = model(image)
        loss = calculate_loss(output, detail_output, label, label_edge, trimap, alpha, beta)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def train_iter_matting(model, optimizer, dataloader, device, epoch,
               lr_scheduler, print_freq=10, scaler=None, theta=0.8):
    model.train()
    model.mode = 'train'

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    lr = 0

    for image, matte, trimap in metric_logger.log_every(dataloader, print_freq, header):

        image = image.to(device)
        matte = matte.to(device)
        trimap = trimap.to(device)

        output = model(image)
        loss = calculate_matting_loss(output, matte, trimap, theta)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def evaluate(model, data_loader, device, num_classes, matting=False):
    model.eval()
    model.mode = 'eval'

    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)

            if matting:
                output[output < 0.5] = 0
                output[output >= 0.5] = 1
                output = output.to(dtype=torch.long)
                confmat.update(target.flatten(), output.flatten())
            else:
                confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


# https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_segmentation/fcn
def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
