import os
import json

# 定义label的映射
label_mapping = {
    "1": "Rlong1",
    "2": "Rass3",
    "3": "Rass1",
    "4": "Rass2",
    "5": "Rass4",
    "6": "Rlong2",
    "7": "TiR",
    "8": "Llong1",
    "9": "Lass4",
    "10": "Lass1",
    "11": "Lass2",
    "12": "Lass3",
    "13": "Llong2",
    "14": "TiL"
}


def process_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 更新shapes中的label
    new_shapes = []
    for shape in data['shapes']:
        label = shape['label']
        if label in label_mapping:
            shape['label'] = label_mapping[label]
        if label != 'sjb_rect':  # 跳过'sjb_rect'的shape
            new_shapes.append(shape)

    data['shapes'] = new_shapes

    # 保存修改后的JSON文件
    with open(filepath, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def process_all_json_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            filepath = os.path.join(folder_path, filename)
            process_json_file(filepath)
            print(f'Processed {filename}')


# 使用你要处理的文件夹路径替换这里的路径
folder_path = 'D:\\labels'
process_all_json_files_in_folder(folder_path)
