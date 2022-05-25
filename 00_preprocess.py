from sklearn.model_selection import train_test_split
import argparse
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from utils import *
from models import *
import warnings
warnings.filterwarnings('ignore')
##
parser = argparse.ArgumentParser(description='configs')
parser.add_argument('--resize',         default=96,         type=int,      help = '')
parser.add_argument('--in_c',           default=3,          type=int,      help = '')
parser.add_argument('--out_c',          default=10,         type=int,      help = '')

parser.add_argument('--max_epochs',      default=10,         type=int,      help = '')
parser.add_argument('--batch_size',      default=128,         type=int,      help = '')

parser.add_argument('--lr',             default=1e-4,       type=float,      help = '')
parser.add_argument('--wd',             default=1e-2,       type=float,      help = '')
parser.add_argument('--lr_patience',    default=5,          type=int,      help = '')
parser.add_argument('--es_patience',    default=15,         type=int,      help = '')

parser.add_argument('--sdir',           default=r'C:\Users\82109\PycharmProjects\fishnet',  type=str,   help = '')
parser.add_argument('--tube_name',      default='log',      type=str,      help = '')

# hparams = parser.parse_args()
hparams = parser.parse_args(args=[]) # 테스트용

cfg = {'first_repeat':1,
       'k':2,
       'Ls_tail':'3,4,6,3',
       'Ls_body':'1,1,1',
       'Ls_head':'1,1,1,1'}
## check image
seed_everything()

rdir = r'D:\cv\Dataset\cifar-10-python\cifar-10-batches-py'
names = 'airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck'.split(',')
n2l = {n:e for e,n in enumerate(names)}
l2n = {e:n for e,n in enumerate(names)}

# check_img(rdir)
##
trans = A.Compose([A.Resize(hparams.resize,hparams.resize, interpolation=cv.INTER_AREA),
                   A.Normalize(mean=[0.4914, 0.4822, 0.4465],
                               std=[0.2470, 0.2435, 0.2616]),
                   ToTensorV2()])

trans_test = A.Compose([A.Resize(hparams.resize,hparams.resize, interpolation=cv.INTER_AREA),
                        A.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2470, 0.2435, 0.2616]),
                        ToTensorV2()])

trnx,tstx,trny,tsty = load_cifar(rdir)

# torch.FloatTensor((trnx/255).reshape(-1,3,32,32)).mean(dim=[0,-1,-2])
# torch.FloatTensor((trnx/255).reshape(-1,3,32,32)).std(dim=[0,-1,-2])

trnx,valx,trny,valy = train_test_split(trnx,trny,test_size=0.2)

trn_set = custom(trnx,trny,trans=trans,train=True)
val_set = custom(valx,valy,trans=trans,train=True)
tst_set = custom(tstx,tsty,trans=trans_test,train=False)

trn_loader = DataLoader(trn_set,batch_size=hparams.batch_size,shuffle=True, drop_last=True,num_workers=0, pin_memory=True)
val_loader = DataLoader(val_set,batch_size=hparams.batch_size,shuffle=False,drop_last=True,num_workers=0, pin_memory=True)
tst_loader = DataLoader(tst_set,batch_size=hparams.batch_size,shuffle=False,drop_last=False,num_workers=0, pin_memory=True)

##
callbacks = get_callbacks(hparams)

print('Call trainer...')
trainer = Trainer(max_epochs=hparams.max_epochs,
                  callbacks=callbacks['callbacks'],
                  gpus=2,
                  precision=16,
                  logger=callbacks['tube'],
                  deterministic=True, accelerator='dp', accumulate_grad_batches=2)
print('Train model...')

model = net(hparams, **cfg)
trainer.fit(model, trn_loader, val_loader)
print('\t\tFitting is end...')

checkpoint_callback = callbacks['callbacks'][0]
best_pth = checkpoint_callback.kth_best_model_path

