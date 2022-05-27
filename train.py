from tqdm import tqdm as tqdm
from sklearn.model_selection import train_test_split
import argparse
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from utils import *
from models import *
import warnings
warnings.filterwarnings('ignore')
##
# pretrain params
parser = argparse.ArgumentParser(description='configs')

# structure params
parser.add_argument('--model',         default='fishnet',           type=str,      help = 'One of ["fishnet", "resnet")]')
parser.add_argument('--num_c',         default='64,128,256,512',    type=str,      help = 'numbers of channels in each stage(stage 1~4)')
parser.add_argument('--Ls_tail',       default='3,4,6,2',           type=str,      help = 'numbers of tail layers in each stage(stage 1~4)')
parser.add_argument('--Ls_body',       default='1,1,1',             type=str,      help = 'numbers of body layers in each stage(stage 4~2)')
parser.add_argument('--Ls_head',       default='1,1,1,1',           type=str,      help = 'numbers of head layers in each stage(stage 1~4)')

# default
parser.add_argument('--preact',         default=True,       type=bool,     help = 'Applying preactivation')
parser.add_argument('--k',              default=2,          type=int,      help = 'k for reduction function')
parser.add_argument('--resize',         default=32,         type=int,      help = '')
parser.add_argument('--in_c',           default=3,          type=int,      help = '')
parser.add_argument('--out_c',          default=10,         type=int,      help = '')

# training params (fixed)
parser.add_argument('--max_epochs',      default=100,        type=int,      help = '')
parser.add_argument('--batch_size',      default=256,        type=int,      help = '')
parser.add_argument('--lr',             default=1e-2,       type=float,      help = '')
parser.add_argument('--wd',             default=1e-4,       type=float,      help = '') # v12 SGD2 1e-2. v14 1e-1
parser.add_argument('--lr_patience',    default=5,          type=int,      help = '')
parser.add_argument('--es_patience',    default=20,         type=int,      help = '')

# report params
parser.add_argument('--sdir',           default=r'C:\Users\82109\PycharmProjects\fishnet',  type=str,   help = '')
parser.add_argument('--tube_name',      default='log',      type=str,      help = '')

hparams = parser.parse_args()
# hparams = parser.parse_args(args=[]) # 테스트용

## check image
seed_everything()

rdir = r'D:\cv\Dataset\cifar-10-python\cifar-10-batches-py'
names = 'airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck'.split(',')
n2l = {n:e for e,n in enumerate(names)}
l2n = {e:n for e,n in enumerate(names)}

# check_img(rdir)
##
trans = A.Compose([A.RandomResizedCrop(hparams.resize,hparams.resize,scale=(0.5,1.0),ratio=(1,1),p=1.0),
                   A.HorizontalFlip(p=0.5),
                   A.FancyPCA(p=0.5),
                   A.Normalize(mean=[0.4914, 0.4822, 0.4465],
                               std=[0.2470, 0.2435, 0.2616],
                               max_pixel_value=255),
                   ToTensorV2()])

trans_test = A.Compose([A.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2470, 0.2435, 0.2616],
                                    max_pixel_value=255),
                        ToTensorV2()])

trnx,tstx,trny,tsty = load_cifar(rdir)

# get mean/std
# torch.FloatTensor(trnx/255).mean(dim=[0,-1,-2])
# torch.FloatTensor(trnx/255).std(dim=[0,-1,-2])

trnx,valx,trny,valy = train_test_split(trnx,trny,test_size=0.2)

trn_set = custom(trnx,trny,trans=trans)
val_set = custom(valx,valy,trans=trans_test)
tst_set = custom(tstx,tsty,trans=trans_test)

trn_loader = DataLoader(trn_set,batch_size=hparams.batch_size,shuffle=True, drop_last=True,num_workers=0, pin_memory=True)
val_loader = DataLoader(val_set,batch_size=hparams.batch_size,shuffle=False,drop_last=True,num_workers=0, pin_memory=True)
tst_loader = DataLoader(tst_set,batch_size=hparams.batch_size,shuffle=False,drop_last=False,num_workers=0, pin_memory=True)

##
callbacks = get_callbacks(hparams)

print('Call trainer...')
trainer = Trainer(max_epochs=hparams.max_epochs,
                  callbacks=callbacks['callbacks'],
                  gpus=2,
                  #gradient_clip_val=5.0,
                  logger=callbacks['tube'],
                  deterministic=True, accelerator='dp', accumulate_grad_batches=2)
print('Train model...')

model = net(hparams)

trainer.fit(model, trn_loader, val_loader)
print('\t\tFitting is end...')

#checkpoint_callback = callbacks['callbacks'][0]
#best_pth = checkpoint_callback.kth_best_model_path