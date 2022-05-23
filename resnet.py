import numpy as np
import cv2 as cv
#
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
#
import torch
import timm
from torch import nn
##
class squeeze(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x.squeeze()

class bottleneck(nn.Module):
    def __init__(self, in_c, out_c, stride=2, bottleneck=4):
        super().__init__()
        self.bottleneck = bottleneck
        self.bn_c = int(out_c/bottleneck)

        self.conv = nn.Sequential(nn.Conv2d(in_c, self.bn_c, kernel_size=1, stride=1, bias=False), # this can also be implemented
                                  nn.BatchNorm2d(self.bn_c),
                                  nn.ReLU(),
                                  nn.Conv2d(self.bn_c, self.bn_c, kernel_size=1, stride=stride, bias=False),
                                  nn.BatchNorm2d(self.bn_c),
                                  nn.ReLU(),
                                  nn.Conv2d(self.bn_c, out_c, kernel_size=1, stride=1, bias=False),
                                  nn.BatchNorm2d(out_c),
                                  nn.ReLU())

        self.shortcut = nn.Sequential()
        if (stride != 1) | (in_c != out_c):
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
         x = self.conv(x) + self.shortcut(x)
         return x

class res(nn.Module):
    def __init__(self, in_c=3, out_c=10, block=bottleneck):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv1 = nn.Sequential(nn.Conv2d(in_c, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self.make_layer(block,  64, 256, 3)
        self.layer2 = self.make_layer(block, 256, 512, 4)
        self.layer3 = self.make_layer(block, 512,1024, 6)
        self.layer4 = self.make_layer(block, 1024,2048, 3)
        self.classifier   = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                          squeeze(),
                                          nn.Linear(2048,out_c))

    def make_layer(self, block, in_c, out_c, repeat):
        stacked = []
        for i in range(repeat):
            if i==0:
                stacked.append(block(in_c, out_c, stride=2))
            else:
                stacked.append(block(out_c, out_c, stride=1))
        return nn.Sequential(*stacked)

    def forward(self,x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x

#####

# class BottleNeck(nn.Module):
#     expansion = 4
#
#     def __init__(self, in_channels, out_channels, activation_function='ReLU', padding_mode='zeros', stride=1):
#         super().__init__()
#         self.padding_mode = padding_mode
#         self.activation_function = activation_function
#         self.bn_channels = int(out_channels / BottleNeck.expansion)
#         self.residual_function = nn.Sequential(
#             nn.BatchNorm3d(in_channels),
#             eval('nn.%s()' % self.activation_function),
#             nn.Conv3d(in_channels, self.bn_channels, kernel_size=1, stride=stride, bias=False),
#             nn.BatchNorm3d(self.bn_channels),
#             eval('nn.%s()' % self.activation_function),
#             nn.Conv3d(self.bn_channels, self.bn_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1),
#                       padding_mode=self.padding_mode, bias=False),
#             nn.BatchNorm3d(self.bn_channels),
#             eval('nn.%s()' % self.activation_function),
#             nn.Conv3d(self.bn_channels, out_channels, kernel_size=1, stride=1, bias=False),
#         )
#
#         self.shortcut = nn.Sequential()
#
#         if (stride != 1) | (in_channels != out_channels):
#             self.shortcut = nn.Sequential(
#                 nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#             )
#
#     def forward(self, x):
#         x = self.residual_function(x) + self.shortcut(x)
#         return x
#
#
#
#         # for i in range(len(self.num_strides)):
#         #     exec('self.conv%s_x = self._make_layer(block, self.num_out[%s], self.num_blocks[%s], self.num_strides[%s])' % (i + 2, i, i, i))  # (N, 064, 12, 8, 8)
#
#
#
#     def _make_layer(self, block, out_channels, num_block, stride):
#         strides = [stride] + [1] * (num_block - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_channels, out_channels, self.hparams.activation_function, self.hparams.padding_mode, stride))
#             self.in_channels = out_channels
#         return nn.Sequential(*layers)

###


# class net(LightningModule):
#     def __init__(self, hparams):
#         super().__init__()
#         self.save_hyperparameters(hparams)
#         #self.pretrain = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=hparams.out_c)
#         #self.pretrain = timm.create_model('tf_efficientnet_b0_ns', pretrained=True, num_classes=88, drop_rate=0.2)
#
#
#         self.result = []
#     def forward(self,x):
#         out_c = self.pretrain(x['img'])
#         return out_c
#
#     def loss_f(self, modely, targety):
#         f = nn.CrossEntropyLoss()
#         return f(modely, targety)
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
#         scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
#                                                    patience=self.hparams.lr_patience,
#                                                    min_lr=self.hparams.lr*.0001)
#         return {"optimizer": optimizer,
#                 "lr_scheduler": {"scheduler": scheduler,
#                                  "monitor": "val_loss"}}
#
#     def step(self, x):
#         label_c = x['label_c']
#         y_hat = self(x)
#         loss = self.loss_f(y_hat, label_c)
#         pred_c = y_hat.argmax(-1).detach().cpu().tolist()
#         f1_c = score_function(label_c.cpu().tolist(), pred_c)
#         return loss, f1_c
#
#     def training_step(self, batch, batch_idx):
#         loss,f1_c = self.step(batch)
#         self.log('trn_loss', loss, on_step=False, on_epoch=True)
#         self.log('trn_f1',   f1_c, on_step=False, on_epoch=True)
#         return {'loss': loss}
#
#     def validation_step(self, batch, batch_idx):
#         loss,f1_c = self.step(batch)
#         self.log('val_loss', loss, on_step=False, on_epoch=True)
#         self.log('val_f1',   f1_c, on_step=False, on_epoch=True)
#         return {'val_loss': loss,'f1':f1_c}
#
#     def validation_epoch_end(self, outputs):
#         avg_loss = torch.stack([op['val_loss'] for op in outputs]).mean()
#         avg_f1 = torch.stack([op['f1'] for op in outputs]).mean()
#
#         print("\n* EPOCH %s | loss :{%4.4f} | f1 :{%2.2f}" % (self.current_epoch, avg_loss, avg_f1))
#         return {'loss': avg_loss}
#
#     def test_step(self, batch, batch_idx):
#         y_hat = self(batch)
#         pred_c = y_hat.argmax(-1).detach().cpu().tolist()
#         self.result.extend(pred_c)