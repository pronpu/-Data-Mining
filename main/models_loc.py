import torch
import torch.nn as nn
from config_loc import config_loc as cfg
from net_util_loc import *
from torchsummary import summary  # 使用 torchsummary 替代 torchsummaryX
from torchviz import make_dot
from tensorboardX import SummaryWriter


# Unet的下采样模块，两次卷积
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, channel_reduce=False):
        super(DoubleConv, self).__init__()

        # 通道减少的系数
        coefficient = 2 if channel_reduce else 1

        self.down = nn.Sequential(
            nn.Conv2d(in_channels, coefficient * out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(coefficient * out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(coefficient * out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.down(x)


# 上采样（转置卷积加残差链接）
class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 先上采样特征图
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=4, stride=2, padding=1)
        self.conv = DoubleConv(in_channels, out_channels, channel_reduce=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


# 简单的U-net模型
class U_net(nn.Module):

    def __init__(self):
        super(U_net, self).__init__()

        # 下采样
        self.double_conv1 = DoubleConv(3, 32)
        self.double_conv2 = DoubleConv(32, 64)
        self.double_conv3 = DoubleConv(64, 128)
        self.double_conv4 = DoubleConv(128, 256)
        self.double_conv5 = DoubleConv(256, 256)

        # 上采样
        self.up1 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up3 = Up(128, 32)
        self.up4 = Up(64, 16)

        # 最后一层
        self.out = nn.Conv2d(16, 14, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        # 下采样
        c1 = self.double_conv1(x)
        p1 = nn.MaxPool2d(2)(c1)
        c2 = self.double_conv2(p1)
        p2 = nn.MaxPool2d(2)(c2)
        c3 = self.double_conv3(p2)
        p3 = nn.MaxPool2d(2)(c3)
        c4 = self.double_conv4(p3)
        p4 = nn.MaxPool2d(2)(c4)
        c5 = self.double_conv5(p4)

        # 上采样
        u1 = self.up1(c5, c4)
        u2 = self.up2(u1, c3)
        u3 = self.up3(u2, c2)
        u4 = self.up4(u3, c1)

        # 输出
        out = self.out(u4)

        return out

    def summary(self):
        input_size = (3, cfg['input_h'], cfg['input_w'])  # 定义输入尺寸元组
        summary(self, input_size)  # 传递输入尺寸而不是Tensor


if __name__ == "__main__":
    # 初始化模型
    model = U_net().to(cfg['device'])

    # 打印模型结构
    print(model)

    # 生成一个随机输入张量
    x = torch.rand(cfg['batch_size'], 3, cfg['input_h'], cfg['input_w']).to(cfg['device'])

    # 通过模型进行前向传播
    y = model(x)

    # 打印输出的形状
    print("输出的形状:", y.shape)

    # 验证模型运行成功
    print("模型运行成功!")