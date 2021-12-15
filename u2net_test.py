#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import torch
import numpy as np
import cv2
from src import tran
from src import U2NET, U2NETP
from src import create_dir, get_date
import config as cfg


def inference(load_model_path, threshold=0.5):
    """
    :param load_model_path: 模型路径
    :param threshold: 分割的阈值, 默认0.5. 可设置阈值的范围在 0-1 之间
                      由于是像素级的分割，每个像素点的值在 0-1 之间，如果值大于阈值，则认为这个像素点是分割的结果；反之不是
    """
    model_name = cfg.MODEL_NAME
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if cfg.COLOR_MODE == 0:
        in_ch = 1
    elif cfg.COLOR_MODE == 1:
        in_ch = 3
    else:
        raise ValueError("COLOR_MODE setup error")

    if model_name == 'u2net':
        net = U2NET(in_ch, 1)
    elif model_name == 'u2netp':
        net = U2NETP(in_ch, 1)
    else:
        raise ValueError("MODEL_NAME setup error")

    net.load_state_dict(torch.load(load_model_path, map_location='cpu'))
    net.to(device)
    net.eval()

    save_seg_dir = os.path.join(cfg.SAVE_RESULT, 'test-' + get_date())
    create_dir(save_seg_dir)

    for image_file in os.listdir(cfg.TEST_IMAGE_DIR):
        print('test image:', image_file)

        # ######### read image
        image_path = os.path.join(cfg.TEST_IMAGE_DIR, image_file)
        org_image = cv2.imread(image_path, flags=cfg.COLOR_MODE)

        # ######### resize the image
        image = cv2.resize(org_image, cfg.SIZE, interpolation=cv2.INTER_NEAREST)
        image = tran(image)
        image = image.unsqueeze(0)

        # ######### model predict
        image = image.to(device)
        d0, d1, d2, d3, d4, d5, d6 = net(image)

        # 分割结果
        output = d0.cpu().squeeze().detach().numpy()

        # ######### 将分割结果转为图片保存
        output[output >= threshold] = 1
        output[output < threshold] = 0
        output = cv2.resize(output, org_image.shape[:2], interpolation=cv2.INTER_NEAREST)
        output[output >= threshold] = 1
        output[output < threshold] = 0
        output = output * 255.
        output = output.astype(np.uint8)

        seg_image_name = image_file[:-len(image_file.split('.')[-1])] + 'png'  # 分割结果的名称
        cv2.imwrite(os.path.join(save_seg_dir, seg_image_name), output)

    print('test finish.')


if __name__ == '__main__':
    inference(load_model_path='saved_models/u2netp/u2netp-2021-07-26-11-55-34_24.pth',
              threshold=0.5)
