import os
import cv2
import json
import numpy as np
from config_loc import config_loc as cfg
from scipy.ndimage import gaussian_filter
import torch
from torchvision import transforms
from PIL import Image, ImageDraw


def json_to_numpy(dataset_path):
    with open(dataset_path) as fp:
        json_data = json.load(fp)
        points = json_data['shapes']
        image_h = json_data['imageHeight']  # 读取图像高度
        image_w = json_data['imageWidth']  # 读取图像宽度

    # 初始化所有关键点为 `[-1, -1]`
    landmarks = [[-1, -1]] * cfg['kpt_n']

    for point in points:
        label = point['label']
        if label.isdigit():  # 检查label是否为数字
            label = int(label) - 1  # 将标签转换为零索引
            if 0 <= label < cfg['kpt_n']:  # 确保索引在有效范围内
                landmarks[label] = point['points'][0]

    return np.array(landmarks), image_h, image_w  # 返回所有的关键点，包括 [-1, -1]


def heatmap_to_point(heatmaps):
    """
    将热图转换为关键点坐标
    :param heatmaps: 热图的numpy数组，形状为 (N, H, W)
    :return: 关键点坐标数组，形状为 (N, 2)
    """
    points = []
    for heatmap in heatmaps:
        pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        points.append([pos[1], pos[0]])  # 将位置 (y, x) 变成坐标 (x, y)
    return np.array(points)


def generate_heatmaps(landmarks, cut_h, cut_w, sigma):
    """
    根据关键点生成热图 (heatmaps)。
    :param landmarks: 调整后的关键点坐标 (Nx2)
    :param cut_h: 裁剪后的高度
    :param cut_w: 裁剪后的宽度
    :param sigma: 高斯核大小
    :return: 热图的numpy数组，形状为 (N, cut_h, cut_w)
    """
    heatmaps = []
    for points in landmarks:
        heatmap = np.zeros((cut_h, cut_w))
        if points[0] >= 0 and points[1] >= 0:  # 对有效点生成热图
            ch = int(points[1] * cut_h / 3840)  # 将点映射到裁剪后的尺寸
            cw = int(points[0] * cut_w / 3840)
            heatmap[ch][cw] = 1

        # 生成高斯模糊的热图
        heatmap = cv2.GaussianBlur(heatmap, sigma, 0)
        am = np.amax(heatmap)
        if am > 0:
            heatmap /= am / 255
        heatmaps.append(heatmap)

    heatmaps = np.array(heatmaps)
    return heatmaps


def show_inputImg_and_keypointLabel(image, landmarks):
    """
    显示图像以及所有关键点，包括 [-1, -1] 的点
    """
    img = transforms.ToTensor()(image)
    img = img.unsqueeze(0)  # 增加一维
    img = img.squeeze(0)  # 减少一维
    img = transforms.ToPILImage()(img)
    draw = ImageDraw.Draw(img)
    for point in landmarks:
        if point[0] >= 0 and point[1] >= 0:  # 绘制所有关键点
            radius = 5  # 设置点的半径
            left_up_point = (point[0] - radius, point[1] - radius)
            right_down_point = (point[0] + radius, point[1] + radius)
            draw.ellipse([left_up_point, right_down_point], fill='yellow', outline='red')

    # 保存并展示
    img.save(os.path.join('..', 'show', 'out.jpg'))
    img.show()


if __name__ == '__main__':
    # 从JSON文件中读取关键点和图像尺寸
    landmarks, original_h, original_w = json_to_numpy('./139.json')
    print('关键点坐标', landmarks, '-------------', sep='\n')

    # 加载图像
    img = Image.open('./139.bmp').convert('RGB')

    # 显示图像和关键点
    show_inputImg_and_keypointLabel(img, landmarks)

    # 生成热图
    heatmaps = generate_heatmaps(landmarks, cfg['cut_h'], cfg['cut_w'], (cfg['gauss_h'], cfg['gauss_w']))

    # 可视化并保存热图
    for i, heatmap in enumerate(heatmaps):
        heatmap_img = Image.fromarray(heatmap.astype(np.uint8))
        heatmap_img.save(f'./heatmap_{i+1}.bmp')
        heatmap_img.show()
