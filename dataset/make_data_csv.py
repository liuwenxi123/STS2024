import os
import pandas as pd

train = True    # 如果要生成测试集csv那就写为False

# 数据集存储的根目录
root_dir = r'/data/cosfs/LWX/STS2024/TeethDataset'
images_list = []
labels_list = []

# 获取图像及其标签的路径
for folder in os.listdir(root_dir):
    if not folder.endswith('.csv'):
        folder_path = os.path.join(root_dir, folder)
        images_folder = os.path.join(folder_path, 'Images')
        labels_folder = os.path.join(folder_path, 'Masks')

        images_list = sorted(os.listdir(images_folder))
        masksh5_lsit = sorted(os.listdir(labels_folder))

        for index, (image, label) in enumerate(zip(images_list, masksh5_lsit)):
            images_list.append(os.path.join(images_folder, image))
            labels_list.append(os.path.join(labels_folder, label))

# 创建一个DataFrame来存储以上路径
data = {'Images': images_list, 'Masks': labels_list}
df = pd.DataFrame(data)

if train:
    # 打乱数据
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # 计算每部分的大小
    total_rows = len(df)
    train_size = int(0.9 * total_rows)
    val_size = int(0.1 * total_rows)
    # 切分数据
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    # 保存文件
    train_df.to_csv(os.path.join(root_dir, 'train_file.csv'), index=False)
    val_df.to_csv(os.path.join(root_dir, 'val_file.csv'), index=False)

else:
    df.to_csv(os.path.join(root_dir, 'test_file.csv'), index=False)
