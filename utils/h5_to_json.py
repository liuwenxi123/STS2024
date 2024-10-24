import os
import numpy as np
from skimage.measure import find_contours
from skimage.morphology import dilation, erosion, opening, closing, disk
from skimage.filters import threshold_otsu
import json
import h5py
from ConnectedRegion_np import retain_largest_connected_component

npy_dir = r'D:\508\STS2024\TeethDatasetPred\real_test_c\epoch=130\predH5'
output_dir = npy_dir + '_json'
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(npy_dir):
    file_path = os.path.join(npy_dir, file)
    with h5py.File(file_path, 'r') as f:
        # 从文件中读取名为 "Mask" 的数据集
        dataset = f['Mask']
        data_array = np.array(dataset)
        mask = data_array.astype(int)

    imageHeight = mask.shape[1]
    imageWidth = mask.shape[2]

    # 定义结构元素，可以根据需要调整大小
    selem = disk(1)

    # 对每个切片进行处理
    processed_masks = []
    shapes = []
    for i in range(1, mask.shape[0]):
        slice = mask[i]

        if i >= 1 and i <= 8:
            label_idx = i + 10
        elif i >= 9 and i <= 16:
            label_idx = i + 12
        elif i >= 17 and i <= 24:
            label_idx = i + 14
        elif i >= 25 and i <= 32:
            label_idx = i + 16
        elif i >= 33 and i <= 37:
            label_idx = i + 18
        elif i >= 38 and i <= 42:
            label_idx = i + 23
        elif i >= 43 and i <= 47:
            label_idx = i + 28
        elif i >= 48 and i <= 52:
            label_idx = i + 33

        # 二值化处理
        threshold_val = threshold_otsu(slice)
        binary_slice = slice > threshold_val

        contours_slice = find_contours(binary_slice, level=0.5)
        # 合并所有轮廓的坐标
        all_points_for_label = []
        for contour in contours_slice:
            points = [(int(col), int(row)) for row, col in contour]
            all_points_for_label.extend(points)

        # 创建字典项
        if all_points_for_label:
            shape_dict = {
                "label": str(label_idx),
                "points": all_points_for_label,
                # "group_id": None,
                # "shape_type": "polygon",
                # "flags": {}
            }
            shapes.append(shape_dict)

        # 创建包含图像尺寸和轮廓信息的字典
    result_dict = {
        # "version": "1.0.0",
        # "flags": {},
        "shapes": shapes,
        "imageHeight": imageHeight,
        "imageWidth": imageWidth
    }

    # 将结果写入JSON文件
    output_json_path = os.path.join(output_dir, file.replace('.h5', '.json'))


    with open(output_json_path, 'w') as json_file:
        json.dump(result_dict, json_file, indent=4)

import os

dir = output_dir
for f in os.listdir(dir):
    temp1 = f.replace('STS24_Train_Validation', 'Validation')
    temp2 = temp1.replace('000', '00')
    temp3 = temp2.replace('pre', 'Mask')
    new_name = temp3

    old_path = os.path.join(dir, f)
    new_path = os.path.join(dir, new_name)

    os.rename(old_path, new_path)



