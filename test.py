import argparse
from torch.utils.data import DataLoader
from utils import *
from models import *
import warnings
warnings.filterwarnings('ignore')
import pickle
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

##
trans_test = A.Compose([A.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2470, 0.2435, 0.2616],
                                    max_pixel_value=255),
                        ToTensorV2()])

trnx,tstx,trny,tsty = load_cifar(rdir)
tst_set = custom(tstx,tsty,trans=trans_test)

# read log and plot the loss
best_pth = load_ckp(fig=False,n=1)
##
model = net(hparams)
get_params(model,(3,hparams.resize,hparams.resize))

tst_loader = DataLoader(tst_set,batch_size=8, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
print('get ckp and predict...')
errs = get_err(model, hparams, best_pth, tst_loader)
print(errs)
savedir = '%s/../../result.pkl'%best_pth

# save dict to pickle file
if not os.path.isfile(savedir):
    with open(savedir,'wb') as f:
        pickle.dump(errs, f)