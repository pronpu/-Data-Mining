import os
import cv2
import json
import numpy as np
from PIL import Image, ImageOps

def pad_image_and_adjust_keypoints(image, landmarks, rectangles, target_h, target_w):
    """
    将图像填充为目标大小，同时调整关键点和矩形框的坐标
    """
    original_h, original_w = image.size[1], image.size[0]
    padding_h = max(0, (target_h - original_h) // 2)
    padding_w = max(0, (target_w - original_w) // 2)

    # 调整关键点坐标
    adjusted_landmarks = []
    for point in landmarks:
        if point[0] >= 0 and point[1] >= 0:  # 跳过无效点
            new_x = point[0] + padding_w
            new_y = point[1] + padding_h
            adjusted_landmarks.append([new_x, new_y])
        else:
            adjusted_landmarks.append(point)  # 保持无效点不变

    # 调整矩形框的坐标
    adjusted_rectangles = []
    for rect in rectangles:
        new_rect = [
            [rect[0][0] + padding_w, rect[0][1] + padding_h],  # 左上角
            [rect[1][0] + padding_w, rect[1][1] + padding_h]   # 右下角
        ]
        adjusted_rectangles.append(new_rect)

    padding = (padding_w, padding_h, target_w - original_w - padding_w, target_h - original_h - padding_h)
    padded_image = ImageOps.expand(image, padding, fill=(0, 0, 0))

    return padded_image, np.array(adjusted_landmarks), np.array(adjusted_rectangles)

def json_to_numpy(dataset_path):
    with open(dataset_path) as fp:
        json_data = json.load(fp)
        points = json_data['shapes']
        image_h = json_data['imageHeight']  # 读取图像高度
        image_w = json_data['imageWidth']  # 读取图像宽度

    # 初始化所有关键点为“缺失”标记 (例如，[-1, -1])
    landmarks = []
    rectangles = []

    for point in points:
        if point['shape_type'] == 'point':  # 只处理点类型
            landmarks.append(point['points'][0])
        elif point['shape_type'] == 'rectangle':  # 处理矩形框类型
            rectangles.append(point['points'])

    return np.array(landmarks), np.array(rectangles), image_h, image_w, json_data  # 返回图像的尺寸及JSON数据

def process_images_in_folder(image_folder, json_folder, output_folder):
    """
    处理文件夹中的所有图像并更新对应的JSON文件
    """
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.bmp', '.png', '.jpg'))]

    for image_file in image_files:
        # 构建文件路径
        image_path = os.path.join(image_folder, image_file)
        json_path = os.path.join(json_folder, image_file.split('.')[0] + '.json')
        output_image_path = os.path.join(output_folder, image_file)
        output_json_path = os.path.join(output_folder, image_file.split('.')[0] + '.json')

        # 从JSON文件中读取关键点、矩形框和图像尺寸
        landmarks, rectangles, original_h, original_w, json_data = json_to_numpy(json_path)
        print(f'处理图像: {image_file}, 关键点坐标:', landmarks)

        # 加载图像并填充到目标大小，同时调整关键点和矩形框
        img = Image.open(image_path).convert('RGB')
        padded_image, adjusted_landmarks, adjusted_rectangles = pad_image_and_adjust_keypoints(
            img, landmarks, rectangles, 3840, 3840
        )

        # 保存填充后的图像
        padded_image.save(output_image_path)

        # 更新JSON文件中的坐标
        landmark_idx = 0
        rectangle_idx = 0
        for idx, shape in enumerate(json_data['shapes']):
            if shape['shape_type'] == 'point':
                shape['points'] = [list(adjusted_landmarks[landmark_idx])]
                landmark_idx += 1
            elif shape['shape_type'] == 'rectangle':
                shape['points'] = [list(adjusted_rectangles[rectangle_idx][0]), list(adjusted_rectangles[rectangle_idx][1])]
                rectangle_idx += 1

        # 更新JSON文件中的 imageHeight 和 imageWidth
        json_data['imageHeight'] = 3840
        json_data['imageWidth'] = 3840

        # 保存更新后的JSON文件
        with open(output_json_path, 'w') as outfile:
            json.dump(json_data, outfile, indent=4)

if __name__ == '__main__':
    image_folder = 'D:\\imgs'  # 图像文件夹路径
    json_folder = 'D:\\labels'    # JSON文件夹路径
    output_folder = 'D:\\post' # 输出文件夹路径

    os.makedirs(output_folder, exist_ok=True)
    process_images_in_folder(image_folder, json_folder, output_folder)
