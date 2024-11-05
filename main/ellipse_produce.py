import os
import json
import cv2
import numpy as np

# 指定包含 JSON 和图像文件的目录路径
data_dir = "D:\\DR6"  

# 遍历目录中的所有文件
for filename in os.listdir(data_dir):
    if filename.endswith(".json"):
        json_file = os.path.join(data_dir, filename)
        img_file = os.path.join(data_dir, filename.replace(".json", ".bmp"))

        # 处理每个 JSON 和图像文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        points_1_6 = []
        points_8_13 = []

        # 提取1-6和8-13的点，排除无效点
        for shape in data['shapes']:
            label = shape['label']
            point = shape['points'][0]
            if label in map(str, range(1, 7)) and point != [-1, -1]:
                points_1_6.append(point)
            elif label in map(str, range(8, 14)) and point != [-1, -1]:
                points_8_13.append(point)

        img = cv2.imread(img_file)

        # 拟合并绘制1-6的椭圆
        if len(points_1_6) >= 5:
            points_1_6 = np.array(points_1_6, dtype=np.float32)
            ellipse_1_6 = cv2.fitEllipse(points_1_6)
            cv2.ellipse(img, ellipse_1_6, (255, 0, 0), 2)  # 蓝色椭圆
            print(f"1-6椭圆参数 for {filename}: {ellipse_1_6}")

            # 保存1-6椭圆的详细参数到JSON文件
            data['ellipse_1_6'] = {
                "center": [ellipse_1_6[0][0], ellipse_1_6[0][1]],
                "axes": [ellipse_1_6[1][0], ellipse_1_6[1][1]],
                "angle": ellipse_1_6[2],
                "points": points_1_6.tolist()
            }
        else:
            print(f"1-6点不足，无法拟合椭圆 for {filename}")
            data['ellipse_1_6'] = {"center": [-1, -1], "axes": [0, 0], "angle": 0}

        # 拟合并绘制8-13的椭圆
        if len(points_8_13) >= 5:
            points_8_13 = np.array(points_8_13, dtype=np.float32)
            ellipse_8_13 = cv2.fitEllipse(points_8_13)
            cv2.ellipse(img, ellipse_8_13, (0, 255, 0), 2)  # 绿色椭圆
            print(f"8-13椭圆参数 for {filename}: {ellipse_8_13}")

            # 保存8-13椭圆的详细参数到JSON文件
            data['ellipse_8_13'] = {
                "center": [ellipse_8_13[0][0], ellipse_8_13[0][1]],
                "axes": [ellipse_8_13[1][0], ellipse_8_13[1][1]],
                "angle": ellipse_8_13[2],
                "points": points_8_13.tolist()
            }
        else:
            print(f"8-13点不足，无法拟合椭圆 for {filename}")
            data['ellipse_8_13'] = {"center": [-1, -1], "axes": [0, 0], "angle": 0}

        # 保存带有椭圆的图像
        output_img_file = os.path.join("D:\\DR6\\after", filename.replace(".json", ".bmp"))
        cv2.imwrite(output_img_file, img)

        # 保存更新后的JSON文件
        updated_json_file = os.path.join('D:\\DR6\\after', filename)
        with open(updated_json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

print("所有图像处理完成。")
