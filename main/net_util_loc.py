import os
import torch
import numpy as np
from torch import nn
import torchvision
from config_loc import config_loc as cfg
import torch.utils.data
from torchvision import transforms
from PIL import Image
from data_pre_loc import json_to_numpy, generate_heatmaps

# 数据集类
class Dataset(torch.utils.data.Dataset):
    # 初始化
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.img_name_list = os.listdir(os.path.join(dataset_path, 'imgs'))

    # 根据 index 返回图像和标签
    def __getitem__(self, index):
        # 读取图像
        img_name = self.img_name_list[index]
        img_path = os.path.join(self.dataset_path,'imgs', img_name)
        img = Image.open(img_path).convert('RGB')

        # 读取标签和图像尺寸
        mask, original_h, original_w = json_to_numpy(os.path.join(self.dataset_path, 'labels', img_name.split('.')[0] + '.json'))

        # 图像预处理：转换为Tensor格式
        img_tensor = transforms.ToTensor()(img)

        # 调整图像尺寸以适应网络输入
        resize = torch.nn.Upsample(size=(cfg['input_h'], cfg['input_w']), mode='bilinear', align_corners=True)
        img_tensor = resize(img_tensor.unsqueeze(0)).squeeze(0)

        # 生成热图
        heatmaps = generate_heatmaps(mask, cfg['input_h'], cfg['input_w'], (cfg['gauss_h'], cfg['gauss_w']))
        heatmaps = torch.tensor(heatmaps, dtype=torch.float32)

        return img_tensor, heatmaps, img_name

    # 数据集的大小
    def __len__(self):
        return len(self.img_name_list)
