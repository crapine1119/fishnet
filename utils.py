import numpy as np
import cv2 as cv
import pickle
import random
import os
import matplotlib.pyplot as plt
from glob import glob as glob
#
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
#
import torch
import timm
from torch import nn
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import torch.optim.lr_scheduler as lr_scheduler
#
from pytorch_lightning.loggers import TestTubeLogger as Tube
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning import LightningModule
from sklearn.metrics import f1_score, accuracy_score
#
from models import *
##
def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    print('Seed : %s'%seed)

def unpickle(file):
    # http://www.cs.toronto.edu/~kriz/cifar.html
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict

def check_img(rdir):
    files = glob('%s/*data_batch*'%rdir)
    assert len(files)>0, 'No file error'
    d = unpickle(files[0])
    imgset_flat = d[b'data']
    imgset = imgset_flat.reshape(-1, 3, 32, 32)
    batch = torch.LongTensor(imgset[:64])
    plt.figure(figsize=[18,8])
    plt.imshow(make_grid(batch, padding=2).permute(1, 2, 0).numpy())

def load_cifar(rdir):
    trn_files = glob('%s/*data_batch*'%rdir)
    assert len(trn_files)>0, 'No file error'
    tst_file = '%s/test_batch'%rdir
    # train set
    trnx,trny = [],[]
    for fnm in trn_files:
        batch = unpickle(fnm)
        trnx.append(batch[b'data'])
        trny.extend(batch[b'labels'])
    trnx = np.concatenate(trnx)
    trny = np.array(trny)

    # test set
    batch = unpickle(tst_file)
    tstx = batch[b'data']
    tsty = np.array(batch[b'labels'])
    return trnx,tstx,trny,tsty

class custom(Dataset):
    def __init__(self, imgs, labels, trans, train=True):
        super().__init__()
        self.imgs =imgs
        self.labels = labels
        self.trans = trans
        self.train=train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        img = self.imgs[item].reshape(3,32,32)[::-1] # BGR to RGB
        img = np.transpose(img, (1,2,0))
        label = self.labels[item]
        mat = {}
        img = self.trans(image=img)['image']
        mat['img'] = torch.FloatTensor(img)
        if self.train:
            mat['label'] = torch.LongTensor([label])
        return mat

###
def get_callbacks(hparams):
    log_path = '%s'%(hparams.sdir)
    tube     = Tube(name=hparams.tube_name, save_dir=log_path)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          filename='{epoch:02d}-{val_loss:.4f}',
                                          save_top_k=1,
                                          mode='min')
    early_stopping        = EarlyStopping(monitor='val_loss',
                                          patience=hparams.es_patience,
                                          verbose=True,
                                          mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    print('get (1)ckp, (2)es, (3)lr_monitor callbacks with (4)tube')
    return {'callbacks':[checkpoint_callback, early_stopping, lr_monitor],
            'tube':tube}


class net(LightningModule):
    def __init__(self, hparams, **cfg):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.fish = make_fish(**cfg)
        #self.pretrain = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=hparams.out_c)
        #self.pretrain = timm.create_model('tf_efficientnet_b0_ns', pretrained=True, num_classes=88, drop_rate=0.2)
        self.result = []
    def forward(self,x):
        out = self.fish(x)
        return out

    def loss_f(self, modely, targety):
        f = nn.CrossEntropyLoss()
        return f(modely, targety)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                   patience=self.hparams.lr_patience,
                                                   min_lr=self.hparams.lr*.0001)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "monitor": "val_loss"}}

    def step(self, x):
        img, label = x['img'], x['label'].view(-1)
        y_hat = self(img)
        loss = self.loss_f(y_hat, label)
        pred_c = y_hat.argmax(-1).detach().cpu().tolist()
        f1 = self.score_function(label.cpu().tolist(), pred_c)
        return loss, f1

    def score_function(self,pred, real):
        score = f1_score(real, pred, average="macro")
        return score

    def training_step(self, batch, batch_idx):
        loss,f1 = self.step(batch)
        self.log('trn_loss', loss, on_step=False, on_epoch=True)
        self.log('trn_f1',   f1, on_step=False, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss,f1 = self.step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_f1',   f1, on_step=False, on_epoch=True)
        return {'val_loss': loss,'f1':f1}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([op['val_loss'] for op in outputs]).mean()
        avg_f1 = torch.stack([op['f1'] for op in outputs]).mean()

        print("\n* EPOCH %s | val loss :{%4.4f} | val f1 :{%2.2f}" % (self.current_epoch, avg_loss, avg_f1))
        return {'loss': avg_loss}

    def test_step(self, batch, batch_idx):
        y_hat = self(batch)
        pred_c = y_hat.argmax(-1).detach().cpu().tolist()
        self.result.extend(pred_c)