import os
import torch

config_loc = {
    # 网络训练部分
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'batch_size': 1,
    'epochs': 510,
    'save_epoch': 100,
    'learning_rate': 0.0001,
    'lr_scheduler': 'step1',

    # 网络输入的图像尺寸 (可以设置为动态调整)
        # 原图尺寸
    'img_h': 3840,
    'img_w': 3840,

    'input_h': 256,
    'input_w': 256,

    # 裁剪后的尺寸
    'cut_h': 3840,
    'cut_w': 3840,
    # 高斯核大小
    'gauss_h': 51,
    'gauss_w': 51,

    # 关键点个数
    'kpt_n': 14,

    # 网络评估部分
    'test_batch_size': 1,
    'test_threshold': 0.5,

    'path': "D:\\111111\\heatmap",

    # 设置路径部分
    'train_date': 'all_9_20220219',
    'train_way': 'train',
    'test_date': 'YanBen',
    'test_way': 'result',

    # 调用的模型
    'pkl_file': 'all_9_0216.pth',

    # 是否加载预训练模型
    'use_old_pkl': True,
    'old_pkl': 'min_loss.pth',

    # remember location
    'start_x': 200,
    'start_y': 200,
    'start_angle': 0,

    # max x,y
    'max_x': 300,
    'max_y': 250,
    'max_angle': 90,

    # min x,y
    'min_x': 100,
    'min_y': 100,

    # key points relative location
    'distance_12': 360,
    'distance_13': 200,
    'distance_23': 410,

    'delta': 50,
}


