import os
import math
import json
import cv2
import numpy as np

def calculate_anteversion(ellipse):
    """计算前倾角度 Anteversion"""
    D = ellipse['axes'][0]  # 最大直径 D (长轴)
    angle_rad = math.radians(ellipse['angle'])
    M_x = ellipse['center'][0] + (D / 5) * math.cos(angle_rad)
    M_y = ellipse['center'][1] + (D / 5) * math.sin(angle_rad)

    # 准确计算垂直距离 p
    p = abs((M_x - ellipse['center'][0]) * math.sin(angle_rad) - (M_y - ellipse['center'][1]) * math.cos(angle_rad))

    try:
        anteversion = math.degrees(math.asin(p / (0.4 * D)))
    except ValueError:
        anteversion = None

    return anteversion

def is_valid_ellipse(ellipse):
    """检查椭圆的有效性"""
    min_axis_length = 50  # 最小轴长的阈值
    axis_ratio_threshold = 10  # 最大长轴和短轴比率

    long_axis = max(ellipse['axes'])  # 使用索引1来访问axes
    short_axis = min(ellipse['axes'])  # 使用索引1来访问axes

    print(f"检查椭圆: 长轴 = {long_axis}, 短轴 = {short_axis}, 中心 = {ellipse['center']}")

    if long_axis < min_axis_length or short_axis < min_axis_length:
        print("椭圆不符合最小轴长条件。")
        return False

    if long_axis / short_axis > axis_ratio_threshold:
        print("椭圆长轴和短轴比率不合理。")
        return False

    return True

def process_image_and_json(json_file, img_file, output_dir):
    """处理每个图像和对应的 JSON 文件"""
    with open(json_file, 'r') as f:
        data = json.load(f)

    if 'ellipse_1_6' in data and is_valid_ellipse(data['ellipse_1_6']):
        anteversion_1_6 = calculate_anteversion(data['ellipse_1_6'])
        if anteversion_1_6 is not None:
            print(f"{os.path.basename(json_file)} - 1-6点前倾角: {anteversion_1_6:.2f}°")
        else:
            print(f"{os.path.basename(json_file)} - 1-6点前倾角无法计算。")
        data['anteversion_1_6'] = anteversion_1_6

    if 'ellipse_8_13' in data and is_valid_ellipse(data['ellipse_8_13']):
        anteversion_8_13 = calculate_anteversion(data['ellipse_8_13'])
        if anteversion_8_13 is not None:
            print(f"{os.path.basename(json_file)} - 8-13点前倾角: {anteversion_8_13:.2f}°")
        else:
            print(f"{os.path.basename(json_file)} - 8-13点前倾角无法计算。")
        data['anteversion_8_13'] = anteversion_8_13

    # 保存前倾角结果到JSON文件
    output_json_file = os.path.join(output_dir, os.path.basename(json_file))
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"保存处理后的图像和JSON文件至: {output_dir}")

# 主处理过程
data_dir = "D:\\DR6\\after"
output_dir = "D:\\DR6\\processed"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(data_dir):
    if filename.endswith(".json"):
        json_file = os.path.join(data_dir, filename)
        img_file = os.path.join(data_dir, filename.replace(".json", ".bmp"))
        process_image_and_json(json_file, img_file, output_dir)

print("所有文件处理完成。")
