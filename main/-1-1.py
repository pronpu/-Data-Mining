import os
import json

# 指定包含 JSON 文件的目录路径
json_dir = "D:\\labels"  

# 遍历目录中的所有文件
for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        json_path = os.path.join(json_dir, filename)

        # 加载 JSON 文件
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 收集 JSON 中已有的标签
        existing_labels = {shape['label'] for shape in data['shapes']}

        # 定义需要的完整标签集（1 到 14）
        required_labels = {str(i) for i in range(1, 15)}

        # 找出缺失的标签
        missing_labels = required_labels - existing_labels

        # 将缺失的标签添加到 JSON 文件中，坐标为 [-1, -1]
        for label in missing_labels:
            data['shapes'].append({
                "label": label,
                "points": [[-1, -1]],
                "group_id": None,
                "description": None,
                "shape_type": "point",
                "flags": {},
                "mask": None
            })

        # 保存更新后的 JSON 文件
        with open(json_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

print("所有 JSON 文件已更新完毕。")
