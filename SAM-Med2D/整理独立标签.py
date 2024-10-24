import os
import shutil

base_dir = r'D:\508\STS2024\SAM-Med2D-main\SAM_result\23f'


def organize_files(directory):
    # 创建一个字典来存储每个组的文件
    files_dict = {}

    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.png'):  # 只处理PNG文件
            parts = filename.split('_')  # 按下划线分割文件名
            if len(parts) >= 5:  # 确保文件名符合预期的格式
                group_id = '_'.join(parts[:4])  # 文件组ID是前四部分的组合
                file_number = parts[3]  # 第四部分作为文件的编号

                # 如果字典中还没有这个组，则创建一个新的列表
                if group_id not in files_dict:
                    files_dict[group_id] = []

                # 将文件路径添加到对应组的列表中
                files_dict[group_id].append(os.path.join(directory, filename))

                # 创建新目录，如果它不存在的话
                new_dir = os.path.join(base_dir, group_id)
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)

                # 移动文件到新目录
                new_file_path = os.path.join(new_dir, filename)
                shutil.copy(os.path.join(directory, filename), new_file_path)

    print("文件整理完成。")

if __name__ == '__main__':
    # 调用函数
    directory = r'D:\508\STS2024\SAM-Med2D-main\SAM_result\23f\boxes_prompt'
    organize_files(directory)