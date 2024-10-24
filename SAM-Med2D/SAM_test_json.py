import os
import json


def generate_mapping(images_path, masks_path):
    # 初始化字典来存储映射关系
    mapping = {}

    # 获取masks文件夹中的所有文件
    mask_files = [f for f in os.listdir(masks_path) if os.path.isfile(os.path.join(masks_path, f))]

    # 遍历mask文件
    for mask_file in mask_files:
        # 获取mask文件的基础名（不含路径）
        mask_index = mask_file.split('_')[3]
        # 尝试找到匹配的image文件
        for image_file in os.listdir(images_path):
            # 如果image文件名包含在mask文件名中
            if mask_index in image_file:
                # 构建完整的键值对
                key = os.path.join(masks_path, mask_file)
                value = os.path.join(images_path, image_file)

                # 添加到映射字典中
                mapping[key] = value
        print(114514)
    return mapping


# 定义文件夹路径
images_path = r'/data/cosfs/LWX/SAM-Med2D-main/data_test_teeth/images'
masks_path = r'/data/cosfs/LWX/SAM-Med2D-main/data_test_teeth/masks'

# 生成映射
mapping = generate_mapping(images_path, masks_path)

# 将映射转换为JSON格式并打印
json_mapping = json.dumps(mapping, indent=4)
print(json_mapping)

# 可选：保存到文件
with open(r'/data/cosfs/LWX/SAM-Med2D-main/data_test_teeth/label2image_test.json', 'w') as f:
    json.dump(mapping, f, indent=4)