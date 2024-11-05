import os
import json

# 指定包含 JSON 文件的目录路径
json_dir = "D:\\labels"  

# 定义合法的标签范围
valid_labels = {str(i) for i in range(1, 15)}

# 遍历目录中的所有文件
for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        json_path = os.path.join(json_dir, filename)

        # 加载 JSON 文件
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 查找不在 1 到 14 范围内的标签
        invalid_labels = [shape['label'] for shape in data['shapes'] if shape['label'] not in valid_labels]

        # 如果存在不合法的标签，打印文件名和这些标签
        if invalid_labels:
            print(f"文件: {filename} 包含不合法标签: {invalid_labels}")

print("检查完毕。")
