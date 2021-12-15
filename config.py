#!usr/bin/env python3
# -*- coding: UTF-8 -*-

# ######### 图片和标签两个文件夹中的文件名称, 须一一对应 #########
# 训练集的图片目录
TRAIN_IMAGE_DIR = r'train/image'
# 训练集的标签(分割图)目录
TRAIN_LABEL_DIR = r'train/label'

# 验证集的图片目录
VALIDATE_IMAGE_DIR = r'validate/image'
# 验证集的标签(分割图)目录
VALIDATE_LABEL_DIR = r'validate/label'

# 测试集的图片目录
TEST_IMAGE_DIR = r'test/image'


# ######### 传入模型图片的宽和高, 通常宽高相等, 医学图像分割, 建议使用原图尺寸, 不进行缩放 #########
SIZE = (512, 512)


# ######### 传入模型图片的色彩模式 #########
# 如果图片是黑白的(灰度图), COLOR = 0
# 如果图片是彩色的, COLOR = 1
COLOR_MODE = 0


# ######### 选择网络模型 #########
# 小模型 u2netp
# 大模型 u2net
MODEL_NAME = 'u2netp'


# ######### 存放分割结果的目录 #########
SAVE_RESULT = 'result'


# ######### 选择训练的GPU型号 #########
# 假设电脑有3张显卡，且编号分别是 GPU0， GPU1， GPU2
# 如果使用三张卡训练，可设置 CUDA_DEVICES = "0, 1, 2"
CUDA_DEVICES = "0"
