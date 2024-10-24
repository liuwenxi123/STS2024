import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch.optim import lr_scheduler
import numpy as np


class Teeth53_Model(pl.LightningModule):
    def __init__(self, in_channels=1, out_classes=53, **kwargs):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=None,
            in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=out_classes  # model output channels (number of classes in your dataset)
        )
        self.number_of_classes=53

        # 定义损失函数为多分类Dice Loss，它直接作用于模型的原始输出（logits）
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)

        # 初始化三个列表来存储每个步骤（训练、验证、测试）的结果，这些结果将在epoch结束时被聚合以计算最终指标
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.best_dice_score = 0


    def forward(self, image):
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch['image'], batch['label']

        assert image.ndim == 4  # [batch_size, 1, H, W]
        mask = mask.long()
        assert mask.ndim == 4   # [batch_size, channels, H, W]

        logits_mask = self.forward(image)
        assert (
                logits_mask.shape[1] == 53   # [batch_size, number_of_classes=53, H, W]
        )
        logits_mask = logits_mask.contiguous()

        loss = self.loss_fn(logits_mask, mask)  # torch.Size([bs])
        dice = 1 - loss.item()


        if stage == 'valid':
            prob_mask = logits_mask.softmax(dim=1)  # (bs,53,1024,1024)
            prob_hardmask = self.hardlabel_binary(prob_mask)  # (bs,53,1024,1024)
            prob_hardmask = torch.as_tensor(prob_hardmask).float().cuda()

            dice_scores = self.soft_dice_score(prob_hardmask, mask, smooth=1.0, dims=[2, 3])   # torch.Size([bs, 53])
            mean_dice_scores = dice_scores.mean(dim=1).mean(dim=0)  # torch.Size([bs])
            dice = mean_dice_scores.item()

        return {
            "loss": loss,
            "dice": dice,
            "best_dice": self.best_dice_score
        }


    def hardlabel_binary(self, input):
        """硬标签二值化"""
        np_mask = input.cpu().numpy()
        np_mask_binary = np.zeros_like(np_mask)
        max_indices = np.argmax(np_mask, axis=1)
        for i in range(self.number_of_classes):
            np_mask_binary[:,i] = (max_indices == i).astype(np.uint8)
        return np_mask_binary

    def soft_dice_score(self,
            output: torch.Tensor,
            target: torch.Tensor,
            smooth: float = 0.0,
            eps: float = 1e-7,
            dims=None,
    ) -> torch.Tensor:
        assert output.size() == target.size()
        if dims is not None:
            intersection = torch.sum(output * target, dim=dims)
            cardinality = torch.sum(output + target, dim=dims)
        else:
            intersection = torch.sum(output * target)
            cardinality = torch.sum(output + target)
        dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
        return dice_score

    def shared_epoch_end(self, outputs, stage):
        # 在epoch结束时调用此方法，将所有步骤的统计信息聚合起来
        dice = sum([x["dice"] for x in outputs])/len(outputs) # 长度是整个验证集长度
        loss = sum([x["loss"] for x in outputs])/len(outputs)

        if dice > self.best_dice_score and stage == 'valid':
            self.best_dice_score = dice
            best_dice = self.best_dice_score
            # torch.save(self.model.state_dict(), './checkpoints/best_model_{}.pth'.format(self.best_dice_score))

        # 将计算出的指标记录到日志中，并显示在进度条上
        if stage == 'train':
            metrics = {
                f"{stage}_loss": loss
            }
            self.log_dict(metrics, prog_bar=True)
        if stage == 'valid':
            metrics = {
                f"{stage}_dice": dice,
                f"{stage}_best_dice": self.best_dice_score
            }
            self.log_dict(metrics, prog_bar=True)


    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        # 配置优化器和学习率调度器。这里使用Adam优化器和余弦退火学习率调度器。返回一个字典，包含优化器和调度器的信息，其中调度器每步更新一次学习率。
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


