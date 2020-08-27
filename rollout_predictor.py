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
from data_utils import get_data_from_string, get_obj_attrs
from models import *
from datasets import GDataset
from gtrain_general import get_model_by_name

class rollout_predictor():
    def __init__(self, config):
        batch_size=128
        self.nfeat=nfeat=12
        self.nembed=nembed=32
        self.cuda=True
        net=get_model_by_name(config.model_name)
        #net=TestLinear(cin=nfeat+nembed, cout=6)
        model=net.double()
        model.load_state_dict(torch.load(config.model_path))
        if self.cuda:
            model.cuda()
        model.eval()
        self.model=model
        mean_std=np.load(config.data_path+'/mean_std.npz')
        self.mean=mean_std['mean']
        self.std=mean_std['std']
        self.mean_label=mean_std['mean_label']
        self.std_label=mean_std['std_label']

    def update_task(self, task_info, action_info):
        self.attrs = get_obj_attrs(task_info, action_info)

    def __call__(self, body_info, contact_info):
        obj, conn, manifold=get_data_from_string(body_info, contact_info, self.attrs,
                                                 self.mean, self.std)
        obj=torch.from_numpy(obj)
        obj_zeroed=obj.clone()
        obj_zeroed[:,1]=0.0
        obj_zeroed[:,2]=0.0
        self.mean[1]=0.0
        self.mean[2]=0.0
        self.std[1]=1.0
        self.std[2]=1.0
        if conn.shape[0] == 0:
            conn = conn.reshape([2, 0])
        if manifold.shape[0] == 0:
            manifold = manifold.reshape([0,4])
        conn=torch.from_numpy(conn)
        #print('@rollout_predictor: manifold=', manifold)
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
        movable_map = gdata.x[:, self.nfeat].bool()
        output=output[movable_map].detach().cpu().numpy()
        #print('normalized output: ', output)
        output=output*self.std_label+self.mean_label
        #print('denormed output: ', output)
        #exit()

        obj_state_np=gdata.x[movable_map][:,:6].detach().cpu().numpy()
        output+=(obj_state_np*self.std[:6])+ self.mean[:6]
        obj_attr = self.assemble_into_dict(output)
        return obj_attr, np.where(movable_map.cpu().numpy())[0]

    def assemble_into_dict(self, obj_data_np):
        obj_attr = []
        for i in range(obj_data_np.shape[0]):
            obj_data=obj_data_np[i]
            obj_attr.append({'theta': obj_data[0], 'x': obj_data[1], 'y': obj_data[2],
                             'omega': obj_data[5],'vx': obj_data[3], 'vy': obj_data[4]})
        return obj_attr

