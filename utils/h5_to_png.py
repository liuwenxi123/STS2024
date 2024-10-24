import os
import h5py
import numpy as np
from PIL import Image
import imageio
from teeth_idx_dict import *
# from reserve_largest_region_np import filter_regions

def load_h5_file(file_path, save_path):
    with h5py.File(file_path, 'r') as f:
        # 从文件中读取名为 "Mask" 的数据集
        dataset = f['Mask']
        data_array = np.array(dataset)
        data_array = data_array.astype(int)
        # 将数据集转换为NumPy数组并返回
        # return data_array.shape
    # for j in range(data_array.shape[0]):
    #     data_array[j] = filter_regions(data_array[j])

    num_classes, height, width = data_array.shape[0], data_array.shape[1], data_array.shape[2]
    Masknp = np.zeros(data_array.shape, np.uint8)
    for label_index in range(1, num_classes):
        masknp = np.zeros((height, width), dtype=np.uint8)
        teeth_type = channels_dict[label_index]
        masknp[data_array[label_index]==1] = teeth_type  # 为每个标签设置不同的灰度值
        Masknp[label_index] = masknp
    foreground_mask = np.max(Masknp[1:num_classes, :, :], axis=0)
    Masknp[0, :, :] = 1 - foreground_mask
    pred_background = Masknp[0, :, :]
    pred_background = np.logical_not(pred_background).astype(pred_background.dtype)
    Masknp[0, :, :] = pred_background
    prediction = np.sum(Masknp, axis=0)
    prediction_uint8 = (prediction * 255).astype(np.uint8)

    imageio.imwrite(save_path, prediction_uint8)





if __name__ == '__main__':
    npy_dir = r'D:\508\STS2024\STS53_lwx\outputs_H5'
    save_dir = npy_dir + '_png'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for f in os.listdir(npy_dir):
        file_path = os.path.join(npy_dir, f)
        save_path = os.path.join(save_dir, f.replace('.h5', '.png'))
        loaded_array = load_h5_file(file_path, save_path)

