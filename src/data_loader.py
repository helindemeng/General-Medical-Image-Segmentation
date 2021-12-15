#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def tran(image):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])(image)


def rotate_img(image, label):
    random_value = random.randint(0, 4)

    if random_value == 0:
        rotate_code = cv2.ROTATE_90_CLOCKWISE
    elif random_value == 1:
        rotate_code = cv2.ROTATE_180
    elif random_value == 2:
        rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
    else:
        return image, label

    image = cv2.rotate(image, rotate_code)
    label = cv2.rotate(label, rotate_code)

    return image, label


def crop_img(image, label):
    random_value = random.randint(0, 2)
    if random_value > 0:
        return image, label

    h, w = image.shape[:2]

    left_x_range = random.randint(0, int(w / 5))
    left_y_range = random.randint(0, int(h / 5))
    right_x_range = random.randint(int(w / 5 * 4), w)
    right_y_range = random.randint(int(h / 5) * 4, h)

    # crop image and label
    crop_image = image[left_y_range:right_y_range, left_x_range:right_x_range]
    crop_label = label[left_y_range:right_y_range, left_x_range:right_x_range]

    # restore size by filling zero
    temp_image = np.zeros(image.shape, dtype=np.uint8)
    temp_label = np.zeros(label.shape, dtype=np.uint8)

    new_h, new_w = crop_image.shape[:2]
    difference_h = h - new_h
    difference_w = w - new_w

    temp_image[difference_h // 2:difference_h // 2 + new_h, difference_w // 2:difference_w // 2 + new_w] = crop_image
    temp_label[difference_h // 2:difference_h // 2 + new_h, difference_w // 2:difference_w // 2 + new_w] = crop_label

    return temp_image, temp_label


def online_enhance(image, label):
    image, label = rotate_img(image, label)
    image, label = crop_img(image, label)
    return image, label


class SalObjDataset(Dataset):
    def __init__(self, image_dir, label_dir, size, color_mode, is_train=True):
        """
        :param image_dir: 图片目录
        :param label_dir: 标签(分割图)目录
        :param size: 传入模型的图片尺寸
        :param color_mode: 色彩模式
        :param is_train: 如果为True，进行数据增强，用于训练；为False, 不进行数据增强，用于测试
        """
        self._image_dir = image_dir
        self._label_dir = label_dir
        self._size = size
        self._color_mode = color_mode
        self._is_train = is_train

        self._image_files = os.listdir(image_dir)
        self._label_files = os.listdir(label_dir)

        self._label_suffix = list(set([i.split('.')[-1] for i in self._label_files]))
        assert len(self._label_suffix) > 0, '标签的文件名称无后缀名'

    def __len__(self):
        return len(self._image_files)

    def __getitem__(self, index):
        # ######### read image
        image_file = self._image_files[index]
        image_path = os.path.join(self._image_dir, image_file)
        image = cv2.imread(image_path, flags=self._color_mode)

        # ######### read label
        label = np.zeros(image.shape[:2], dtype=np.uint8)  # image has no label, label's value is zero
        for suffix in self._label_suffix:
            label_file = image_file[:-len(image_file.split('.')[-1])] + str(suffix)
            label_path = os.path.join(self._label_dir, label_file)

            if os.path.exists(label_path):
                label = cv2.imread(label_path, flags=cv2.IMREAD_GRAYSCALE)
                break

        # ######### data online enhance
        if self._is_train:
            image, label = online_enhance(image, label)

        # ######### resize the image and label
        image = cv2.resize(image, self._size, interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, self._size, interpolation=cv2.INTER_NEAREST)

        image = tran(image)

        ret, binary = cv2.threshold(label, 10, 255, cv2.THRESH_BINARY)
        label = binary / 255.
        label = torch.tensor(label, dtype=torch.float32)

        return image, label
