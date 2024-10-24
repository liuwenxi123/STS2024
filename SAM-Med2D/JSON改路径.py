import json

def update_paths_in_json(input_json_file, output_json_file):
    # 读取JSON文件
    with open(input_json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 更新路径
    new_data = {
        key.replace('D:\\508\\STS2024\\NewDataset\\LabeledTrain\\f\\MasksPNG\\', '/data/cosfs/LWX/SAM-Med2D-main/data_demo_test/masks/'):
        value.replace('D:\\508\\STS2024\\NewDataset\\LabeledTrain\\f\\ImagesPNG\\', '/data/cosfs/LWX/SAM-Med2D-main/data_demo_test/images/')
        for key, value in data.items()
    }

    # 写回新的JSON文件
    with open(output_json_file, 'w', encoding='utf-8') as file:
        json.dump(new_data, file, ensure_ascii=False, indent=4)

# 使用函数
input_json_file = r'D:\508\STS2024\NewDataset\LabeledTrain\f\label2image_test1.json'  # 输入JSON文件的路径
output_json_file = r'D:\508\STS2024\NewDataset\LabeledTrain\f\label2image_test.json'  # 输出JSON文件的路径
update_paths_in_json(input_json_file, output_json_file)