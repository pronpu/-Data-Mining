import os
import json
import math
import numpy as np


def calculate_angle(p1, p2, p3, p4):
    """计算两条线段 (p1, p2) 和 (p3, p4) 之间的夹角"""
    vector1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    vector2 = np.array([p4[0] - p3[0], p4[1] - p3[1]])

    # 计算两个向量的夹角
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)

    dot_product = np.dot(unit_vector1, unit_vector2)

    # 确保dot_product在[-1, 1]之间，以避免出现数学错误
    dot_product = np.clip(dot_product, -1.0, 1.0)

    angle = math.degrees(math.acos(dot_product))
    return angle


def process_json_for_abduction(json_file, output_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 提取关键点
    points = {shape['label']: shape['points'][0] for shape in data['shapes'] if shape['points'][0] != [-1, -1]}

    # 确认是否存在足够的椭圆数据
    if 'ellipse_1_6' not in data or 'ellipse_8_13' not in data:
        print(f"{json_file} - 缺少必要的椭圆数据，无法计算外展角。")
        return

    # 获取坐骨最低点连线
    if '7' in points and '14' in points:
        lowest_line_start = points['7']
        lowest_line_end = points['14']
    else:
        print(f"{json_file} - 缺少坐骨最低点的连线，无法计算外展角。")
        return

    # 计算右侧的外展角
    if '1' in points and '6' in points:
        right_abduction_angle = calculate_angle(points['7'], points['14'], points['1'], points['6'])
        data['right_abduction_angle'] = right_abduction_angle
        print(f"{json_file} - 右侧外展角: {right_abduction_angle:.2f}°")
    else:
        print(f"{json_file} - 缺少右侧关键点，无法计算右侧外展角。")

    # 计算左侧的外展角
    if '8' in points and '13' in points:
        left_abduction_angle = calculate_angle(points['7'], points['14'], points['8'], points['13'])
        data['left_abduction_angle'] = left_abduction_angle
        print(f"{json_file} - 左侧外展角: {left_abduction_angle:.2f}°")
    else:
        print(f"{json_file} - 缺少左侧关键点，无法计算左侧外展角。")

    # 保存结果到JSON文件
    output_json_file = os.path.join(output_dir, os.path.basename(json_file))
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"外展角计算结果保存至: {output_json_file}")


# 主处理过程
data_dir = "D:\\DR6\\after"
output_dir = "D:\\DR6\\processed_2"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(data_dir):
    if filename.endswith(".json"):
        json_file = os.path.join(data_dir, filename)
        process_json_for_abduction(json_file, output_dir)

print("所有文件处理完成。")
