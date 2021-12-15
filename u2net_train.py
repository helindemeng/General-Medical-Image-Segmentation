#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from src import SalObjDataset
from src import U2NET, U2NETP
from src import create_dir, dice_coef, get_date
import config as cfg
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_DEVICES

# 存放损失
log_dir = 'logs-' + get_date()
create_dir(log_dir)

summaryWriter = SummaryWriter(log_dir)

bce_loss = nn.BCELoss(size_average=True)


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss0, loss


def train(epochs: int, batch_size: int, load_model_path: str = None, interval: int = 10):
    """
    :param epochs: 轮次
    :param batch_size: 批次
    :param load_model_path: 如果需要在原来的参数上继续训练, 指明保存的模型地址, 例如  load_model_path='net.pth'
    :param interval: 保存模型的间隔, 如果interval=10, 的每隔10轮保存一次模型
    """
    device = torch.device('cuda')
    model_name = cfg.MODEL_NAME
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)
    create_dir(model_dir)

    # ######### dataset and dataloader
    train_dataset = SalObjDataset(cfg.TRAIN_IMAGE_DIR, cfg.TRAIN_LABEL_DIR, cfg.SIZE, cfg.COLOR_MODE, is_train=True)
    validate_dataset = SalObjDataset(cfg.VALIDATE_IMAGE_DIR, cfg.VALIDATE_LABEL_DIR, cfg.SIZE, cfg.COLOR_MODE, is_train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)

    # ######### define the net
    if cfg.COLOR_MODE == 0:
        in_ch = 1
    elif cfg.COLOR_MODE == 1:
        in_ch = 3
    else:
        raise ValueError("COLOR_MODE setup error")

    if model_name == 'u2net':
        net = U2NET(in_ch, 1).to(device)
    elif model_name == 'u2netp':
        net = U2NETP(in_ch, 1).to(device)
    else:
        raise ValueError("MODEL_NAME setup error")

    print(f"The chosen network model is '{model_name}'")

    # ######### load model
    if load_model_path is not None:
        net.load_state_dict(torch.load(load_model_path))
        print(f'load model: {load_model_path}')
    else:
        print('no model')

    net = nn.DataParallel(net)

    optimizer = optim.Adam(net.parameters())

    for epoch in range(1, epochs + 1):
        print('\nepoch:', epoch)

        train_loss = []
        test_loss = []
        test_dice_coef = []  # in test datasets

        # ######### train
        net.train()
        for i, (image, label) in enumerate(train_dataloader):
            image = image.to(device)
            label = label.to(device)

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(image)

            loss0, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, label.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            print(f"epoch:{epoch} -> {i}/{len(train_dataloader)} -> train loss: {loss.item()}")

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss0, loss, image, label

        if epoch % interval == 0:
            model_path = os.path.join(model_dir, model_name + f"-{get_date()}_{epoch}.pth")
            torch.save(net.module.state_dict(), model_path)
            print(f'save model: {model_path}')

        # ######### validate datasets
        net.eval()
        for image, label in validate_dataloader:
            image = image.to(device)
            label = label.to(device)

            d0, d1, d2, d3, d4, d5, d6 = net(image)
            loss0, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, label.unsqueeze(1))

            test_loss.append(loss.item())
            test_dice_coef.extend(list(dice_coef(d0.cpu().detach().squeeze(1).numpy(),
                                                 label.cpu().detach().numpy(), axis=(1, 2))))

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, image, label

        # ######### calculate average loss in train datasets and test datasets
        # ######### calculate average dice coefficient in test datasets
        average_train_loss = np.array(train_loss).mean()
        average_test_loss = np.array(test_loss).mean()
        average_dice_coef = np.array(test_dice_coef).mean()

        summaryWriter.add_scalars('loss', {'train': average_train_loss, 'test': average_test_loss},
                                  global_step=epoch)
        summaryWriter.add_scalar('dice coef', average_dice_coef, global_step=epoch)

        print(f"average train loss: {average_train_loss}, average test loss: {average_test_loss}, "
              f"test datasets Dice coef: {average_dice_coef}")


if __name__ == '__main__':
    train(epochs=1000, batch_size=3,
          load_model_path=None,
          interval=10)
