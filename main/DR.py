import os


def rename_images_in_folder(folder_path, start_number=97):
    # 获取文件夹中的所有文件名
    files = os.listdir(folder_path)
    # 对文件名进行排序，以确保按顺序重命名
    files.sort()

    current_number = start_number

    for file_name in files:
        # 获取文件的扩展名
        file_extension = os.path.splitext(file_name)[1]

        # 构造新的文件名
        new_name = f"{current_number}{file_extension}"

        # 构造完整的旧文件路径和新文件路径
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(folder_path, new_name)

        # 重命名文件
        os.rename(old_file_path, new_file_path)

        # 递增数字
        current_number += 1

    print(f"所有文件已重命名，起始数字为 {start_number}。")



folder_path = 'D:\\DR'
rename_images_in_folder(folder_path)
