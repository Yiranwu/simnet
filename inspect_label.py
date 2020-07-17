import numpy as np
start_tid=0

from datasets import GDataset
from torch_geometric.data import DataLoader
obj=np.load('data/%05d_obj_data.npy'%start_tid)
label=np.load('data/%05d_label_data.npy'%start_tid)
cflag=np.load('data/%05d_cflag_data.npy'%start_tid)
print(obj.shape)
print(label.shape)

for i in range(100):
    # vx, dpx
    #print(obj[0,0,4] , label[0,0,2])
    pass

dataset=GDataset('/home/yiran/pc_mapping/simnet/data/00000_', nfeat=12)
dataloader=DataLoader(dataset, batch_size=1, drop_last=True)

for data in dataloader:
    print(data.x.shape)
    print(data.edge_index)
    exit()
