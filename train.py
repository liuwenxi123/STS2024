import os
from dataset.dataset import Dataset
import pandas as pd
from torch.utils.data import DataLoader
from model.Teeth53_Model import Teeth53_Model
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == '__main__':
    # 数据集路径
    dir = r'/data/cosfs/LWX/STS2024/TeethDataset_2'     # 存放训练集和验证集的目录
    train_path = os.path.join(dir, 'train2_file.csv')   # 训练集csv文件路径
    val_path = os.path.join(dir, 'val2_file.csv')       # 验证集csv文件路径

    # 读取CSV文件
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)

    # 获取图像和标签
    train_images = train_data.iloc[:, 0].values
    train_labels = train_data.iloc[:, 1].values
    val_images = val_data.iloc[:, 0].values
    val_labels = val_data.iloc[:, 1].values

    # 创建数据集
    train_dataset = Dataset(
        train_images,
        train_labels
    )
    val_dataset = Dataset(
        val_images,
        val_labels
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    valid_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # 训练参数
    EPOCHS = 500
    T_MAX = EPOCHS * len(train_loader)
    OUT_CLASSES = 53

    # 检查点路径
    # checkpoint_path = './checkpoints/best-checkpoint-epoch=01-valid_best_dice=0.8000.ckpt'  # 替换为你的检查点文件路径
    checkpoint_path = None

    # 加载模型并恢复训练
    if os.path.exists(checkpoint_path):
        model = Teeth53_Model.load_from_checkpoint(
            checkpoint_path,
            in_channels=1,
            out_classes=53
        )
        print(f"Loaded model from checkpoint: {checkpoint_path}")
    else:
        model = Teeth53_Model(in_channels=1, out_classes=53)
        print("No checkpoint found, starting training from scratch.")

    # 设置检查点回调
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_best_dice',  # 监控的指标
        dirpath='./checkpoints',  # 保存路径
        filename='best-checkpoint-{epoch:02d}-{valid_best_dice:.4f}',  # 文件名格式
        save_top_k=2,  # 保存最好的1个检查点
        mode='max',  # 最大化监控指标
        verbose=True  # 打印保存信息
    )

    # 创建Trainer实例
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=checkpoint_path  # 从检查点恢复训练
    )

    # 开始训练
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )