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
from models import BallEncoder, ObjectEncoder, GCN2,GCN5,GAT3,GIN, TestLinear
from datasets import GDataset, GTestDataset

class rollout_predictor():
    def __init__(self, task_id='00001:000', model_path='gin-5-60.pth'):
        self.task_id=task_id
        self.model_path=model_path
        batch_size=128
        self.nfeat=nfeat=12
        self.nembed=nembed=32
        self.cuda=True
        # Model and optimizer
        net=GIN(nfeat=nfeat+nembed, nclass=6).double()
        #net=TestLinear(cin=nfeat+nembed, cout=6)
        model=net.double()
        model.load_state_dict(torch.load(model_path))
        obj_encoder = ObjectEncoder(cin=nfeat,cout=nembed).double()
        if self.cuda:
            model.cuda()
            obj_encoder=obj_encoder.cuda()
        model.eval()
        self.model=model
        self.obj_encoder=obj_encoder
        mean_std=np.load('/home/yiran/pc_mapping/simnet/data/mean_std.npz')
        self.mean=mean_std['mean']
        self.std=mean_std['std']

    def update_task(self, task_info, action_info):
        self.attrs = get_obj_attrs(task_info, action_info)

    def __call__(self, body_info, contact_info):
        obj, conn=get_data_from_string(body_info, contact_info, self.attrs, self.mean, self.std)
        obj=torch.from_numpy(obj)
        if conn.shape[0] == 0:
            conn = conn.reshape([2, 0])
        conn=torch.from_numpy(conn)
        gdata = Data(x=obj, edge_index=conn)
        #data,slices = self.collate([gdata])
        #print(data)
        #print(slices)
        #exit()
        gdata.batch=torch.tensor([0], dtype=torch.int64)
        if self.cuda:
            obj = obj.to('cuda:0')
            conn = conn.to('cuda:0')
            gdata = gdata.to('cuda:0')

        obj_feat = self.obj_encoder(obj[:, :self.nfeat])
        feat_batch = torch.cat([obj_feat, obj[:, :self.nfeat]], axis=1)
        output = self.model(feat_batch, gdata)
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


