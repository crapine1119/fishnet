from utils import *
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

##
rdir = r'D:\cv\Dataset\cifar-10-python\cifar-10-batches-py'
fnm = f'{rdir}/data_batch_1'

d = unpickle(fnm)
d.keys()

# test dset
imgset_flat = d[b'data']
imgset = imgset_flat.reshape(-1,3,32,32)

batch = torch.LongTensor(imgset[:64])
plt.imshow(make_grid(batch,padding=2).permute(1,2,0).numpy())
##