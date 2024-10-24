import os
from dataset import Dataset
from torch.utils.data import DataLoader
from model.Teeth53_Model import Teeth53_Model
import torch
import numpy as np
import cv2
from teeth_idx_dict import *
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
import json
from ConnectedRegion_np import filter_regions
import h5py

INPUT_DIR = 'inputs'    # 输入图像文件夹名称
OUTPUT_DIR = 'outputs2'  # 输出json文件夹名称

def hardlabel_binary(input, number_of_classes=53, threshold=None):
    """硬标签二值化"""
    np_mask_binary = np.zeros_like(input)
    max_indices = np.argmax(input, axis=0)
    if threshold is None:
        for i in range(number_of_classes):
            np_mask_binary[i] = (max_indices == i).astype(np.uint8)
    else:
        max_probs = np.max(input, axis=0)
        for i in range(number_of_classes):
            # 仅当最大概率超过阈值时，才将该位置设为1
            np_mask_binary[i] = ((max_indices == i) & (max_probs >= threshold)).astype(np.uint8)

    return np_mask_binary


def main():
    # 导入测试集图片
    images_dir = os.path.join(os.getcwd(), INPUT_DIR)
    entries = os.listdir(images_dir)
    file_paths = [os.path.join(images_dir, entry) for entry in entries]
    test_images = np.array(file_paths)
    test_labels = np.full(test_images.shape, None)

    # 创造测试集的test_dataset和test_loader
    test_dataset = Dataset(
        test_images,
        test_labels
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    #  导入检查点文件并创造模型
    checkpoint_path = './checkpoint/best-checkpoint-epoch=147-val_dice=0.87417.ckpt'
    model = Teeth53_Model.load_from_checkpoint(checkpoint_path, in_channels=1, out_classes=53)

    # 预测
    for i, batch in enumerate(test_loader):
        # 获取图像np数组
        images = batch['image'].cuda()
        # 获取图像的原始高度和宽度，为之后的尺寸还原做准备
        ori_height, ori_width = batch['origin_size'][0].item(), batch['origin_size'][1].item()
        # 模型进行预测，不进行梯度下降
        with torch.no_grad():
            logits = model(images).squeeze(0)  # (1,53,1024,1024) -> (53,1024,1024)去除batchsize维度
            logits_mask = torch.softmax(logits, dim=0)  # 对channel维度进行softmax
            pred = logits_mask.cpu().detach().numpy()   # 将模型结果从GPU上转移到CPU，并转化为np数组
        # 将预测结果转换为硬标签，即二值化
        hard_mask = hardlabel_binary(pred)
        # 将输出结果从（1024,1024）重塑回原始图像尺寸
        new_hard_mask = np.zeros((hard_mask.shape[0], ori_height, ori_width), np.uint8)
        for j in range(hard_mask.shape[0]):
            mask_temp = hard_mask[j]
            resized_mask = cv2.resize(mask_temp, (ori_width, ori_height), interpolation=cv2.INTER_NEAREST)
            new_hard_mask[j] = resized_mask
        # 恢复被裁剪部分
        pad_width = ((0, 0), (50, 50), (100, 100))
        new_hard_mask = np.pad(new_hard_mask, pad_width, mode='constant', constant_values=0)
        # 后处理
        for j in range(hard_mask.shape[0]):
            new_hard_mask[j] = filter_regions(new_hard_mask[j])

        mask_bool = new_hard_mask.astype(bool)
        h5output_dir = os.path.join(os.getcwd(), OUTPUT_DIR + '_h5')
        os.makedirs(h5output_dir, exist_ok=True)
        h5output_name = os.path.basename(test_images[i]).replace('.jpg', '.h5')
        h5output_path = os.path.join(h5output_dir, h5output_name)
        with h5py.File(h5output_path, 'w') as f:
            f.create_dataset("Mask", data=mask_bool, dtype='bool', compression="gzip", compression_opts=9)


        output_dir = os.path.join(os.getcwd(), OUTPUT_DIR+'_npdraw')
        output_name = os.path.basename(test_images[i]).replace('.jpg', 'pred_.jpg')
        os.makedirs(output_dir, exist_ok=True)
        output_npdraw_path = os.path.join(output_dir, output_name)
        from json_to_png import np_draw
        np_draw(new_hard_mask, output_npdraw_path)


        # 将模型输出结果转为json文件
        mask = new_hard_mask    # 标签np数组
        imageHeight, imageWidth = mask.shape[1], mask.shape[2]  # 获取标签的高度和宽度
        shapes = []
        for k in range(1, mask.shape[0]):   # channel=0是背景层，因此从1开始
            slice = mask[k]  # 获得第i层的标签
            label_idx = channels_dict[k]    # 通过字典找到第i层对应的牙齿序号
            # 二值化处理
            threshold_val = threshold_otsu(slice)
            binary_slice = slice > threshold_val
            # 找到所有的掩码区域
            contours_slice = find_contours(binary_slice, level=0.5)
            # 合并所有轮廓坐标
            all_points_for_label = []
            for contour in contours_slice:
                points = [(int(col), int(row)) for row, col in contour]
                all_points_for_label.extend(points)
            # 创建字典项
            if all_points_for_label:
                shape_dict = {
                    "label": str(label_idx),
                    "points": all_points_for_label,
                }
                shapes.append(shape_dict)
        # 创建包含图像尺寸和轮廓信息的字典
        result_dict = {
            "shapes": shapes,
            "imageHeight": imageHeight,
            "imageWidth": imageWidth
        }
        # 将结果写入json文件
        output_name = os.path.basename(test_images[i]).replace('jpg', 'json')
        output_dir = os.path.join(os.getcwd(), OUTPUT_DIR)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_name)
        with open(output_path, 'w') as json_file:
            json.dump(result_dict, json_file, indent=4)
        print(output_path,'已生成')

        outputpng_dir = output_dir + '_png'
        outputpng_path = os.path.join(outputpng_dir, os.path.basename(output_path).replace('json', 'png'))
        os.makedirs(outputpng_dir, exist_ok=True)
        from json_to_png import json_draw
        json_draw(output_path, outputpng_path)

if __name__ == '__main__':
    main()
