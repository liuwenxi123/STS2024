import os
import h5py
import numpy as np
import imageio

if __name__ == '__main__':
    labeled_h5_dir = r'D:\508\STS2024\STS53_lwx\outputs_H5'
    save_dir = labeled_h5_dir + '_teethPNG'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for filename in os.listdir(labeled_h5_dir):
        file_path = os.path.join(labeled_h5_dir, filename)
        with h5py.File(file_path, 'r') as f:
            dataset = f['Mask']
            data_array = np.array(dataset)
            data = data_array.astype(int)
        for i in range(1, data.shape[0]):
            if np.any(data[i] != 0):
                data_png = (data[i] * 255).astype(np.uint8)

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


                save_png_path = os.path.join(save_dir, filename.replace('.h5', '_{}.png'.format(label_idx)))
                imageio.imwrite(save_png_path, data_png)


# 已标记数据的JSON文件转h5的规则是：第0张切片存放背景，背景为1
# 伪标签的背景没有看到置1，需要整改