import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch_geometric.data import DataLoader, Data
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

#from utils import load_data, accuracy
from gen_dataset import get_data_from_string, get_obj_attrs
from models import BallEncoder, ObjectEncoder, GCN2,GCN5,GAT3,GIN, TestLinear, GINE
from datasets import GDataset, GTestDataset

class rollout_predictor():
    def __init__(self, task_id='00001:000', model_path='gin-5-60.pth', dataset_spec='00001_'):
        self.task_id=task_id
        self.model_path=model_path
        batch_size=128
        self.nfeat=nfeat=12
        self.nembed=nembed=32
        self.cuda=True
        # Model and optimizer
        net=GINE(vfeat=12, hidden=32, nclass=6)
        #net=TestLinear(cin=nfeat+nembed, cout=6)
        model=net.double()
        model.load_state_dict(torch.load(model_path))
        if self.cuda:
            model.cuda()
        model.eval()
        self.model=model
        mean_std=np.load('/home/yiran/pc_mapping/simnet/data/%smean_std.npz'%dataset_spec)
        self.mean=mean_std['mean']
        self.std=mean_std['std']

    def update_task(self, task_info, action_info):
        self.attrs = get_obj_attrs(task_info, action_info)

    def __call__(self, body_info, contact_info):
        obj, conn, manifold=get_data_from_string(body_info, contact_info, self.attrs,
                                                 self.mean, self.std)
        obj=torch.from_numpy(obj)
        if conn.shape[0] == 0:
            conn = conn.reshape([2, 0])
        if manifold.shape[0] == 0:
            manifold = manifold.reshape([0,4])
        conn=torch.from_numpy(conn)
        manifold=torch.from_numpy(manifold)
        gdata = Data(x=obj, edge_index=conn, edge_attr=manifold)
        #data,slices = self.collate([gdata])
        #print(data)
        #print(slices)
        #exit()

        #gdata.batch=torch.tensor([0]*obj.shape[0], dtype=torch.int64)
        #print(gdata.batch)
        #exit()
        if self.cuda:
            gdata = gdata.to('cuda:0')

        output = self.model(gdata)
        output += gdata.x[:,:6]
        movable_map = gdata.x[:, self.nfeat].bool()
        output = output[movable_map]#.cpu().numpy()
        output=output.detach().cpu().numpy()
        #print(output.shape)
        #print(self.std.shape)
        #exit()
        output=(output * self.std[:6]) + self.mean[:6]
        #obj_attr=get_dict_from_label(label)
        #return obj_attr
        return output, np.where(movable_map.cpu().numpy())[0]


