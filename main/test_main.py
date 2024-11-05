import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from models_loc import U_net
from data_pre_loc import json_to_numpy, heatmap_to_point
from config_loc import config_loc as cfg
import json


def map_back_to_original_size(padded_keypoints, original_h, original_w, input_h, input_w):
    """
    将预测的关键点坐标从网络输入尺寸映射回原始图像尺寸
    """
    original_keypoints = []
    for point in padded_keypoints:
        original_x = int(point[0] * original_w / input_w)
        original_y = int(point[1] * original_h / input_h)
        original_keypoints.append([original_x, original_y])

    return np.array(original_keypoints)


def show_point_on_picture(img, landmarks, landmarks_gt):
    for point in landmarks:
        point = tuple([int(point[0]), int(point[1])])
        img = cv2.circle(img, center=point, radius=5, color=(0, 0, 255), thickness=-1)
    for point in landmarks_gt:
        point = tuple([int(point[0]), int(point[1])])
        img = cv2.circle(img, center=point, radius=5, color=(0, 255, 0), thickness=-1)
    return img


def save_to_json(predicted_points, original_points, original_json_path, output_json_path):
    """
    将预测的关键点保存到 JSON 文件中
    """
    with open(original_json_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    for i, shape in enumerate(json_data['shapes']):
        if shape['label'].isdigit() and int(shape['label']) <= len(predicted_points):
            shape['points'] = [predicted_points[i].tolist()]

    with open(output_json_path, 'w', encoding='utf-8') as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)


def evaluate(flag=False):
    img_path = os.path.join('..', 'test', 'imgs')
    label_path = os.path.join('..', 'test', 'labels')

    # 定义模型
    model = U_net()
    model.load_state_dict(torch.load("D:\\weights\\111.pth"))
    model.to(cfg['device'])
    model.eval()

    total_loss = 0
    max_keypoint_diff = 0

    # 开始预测
    for index, name in enumerate(os.listdir(img_path)):
        print('图像名称：', name, ' 图像编号：', index + 1)

        # 加载图像
        img = Image.open(os.path.join(img_path, name)).convert('RGB')

        # 从json文件加载关键点
        landmarks, original_h, original_w = json_to_numpy(os.path.join(label_path, name.split('.')[0] + '.json'))

        # 图像预处理：先将图像缩放到模型输入尺寸
        img_resized = img.resize((cfg['input_w'], cfg['input_h']))
        img_tensor = transforms.ToTensor()(img_resized).unsqueeze(0).to(cfg['device'])

        # 喂入网络进行预测
        pre = model(img_tensor)
        pre = pre.cpu().detach().numpy()

        # 解码预测的关键点
        pre_point = heatmap_to_point(pre[0])

        # 将预测的关键点从网络输入尺寸映射回原图尺寸
        original_pre_point = map_back_to_original_size(pre_point, original_h, original_w, cfg['input_h'],
                                                       cfg['input_w'])

        print('预测的关键点坐标：\n', original_pre_point)
        print('真实的关键点坐标：\n', landmarks)

        pre_label = torch.Tensor(original_pre_point.reshape(1, -1)).to(cfg['device'])
        label = torch.Tensor(landmarks.reshape(1, -1)).to(cfg['device'])

        loss_F = torch.nn.MSELoss()
        loss = loss_F(pre_label, label)

        print('+++坐标误差损失: ', loss.item())
        total_loss += loss.item()

        if loss.item() > max_keypoint_diff:
            max_keypoint_diff = loss.item()

        if flag:
            img_with_points = np.array(img)  # 直接使用原始尺寸的图像
            img_with_points = show_point_on_picture(img_with_points, original_pre_point, landmarks)

            # 存储绘制图像部分
            save_dir = os.path.join('..', 'result', cfg['test_date'], cfg['test_way'] + '_data')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            result_dir = os.path.join(save_dir, name.split('.')[0] + '_keypoint.jpg')
            cv2.imwrite(result_dir, img_with_points)

            # 存储预测结果为JSON文件
            output_json_path = os.path.join(save_dir, name.split('.')[0] + '_predicted.json')
            original_json_path = os.path.join(label_path, name.split('.')[0] + '.json')
            save_to_json(original_pre_point, landmarks, original_json_path, output_json_path)

    print('##################')
    print('# ---- Mean ---- #')
    print('##################')

    print('平均每个关键点坐标误差：', total_loss / (index + 1), ' 最大单个关键点坐标误差：', max_keypoint_diff)


if __name__ == "__main__":
    # 对一组权重进行预测
    evaluate(flag=True)
