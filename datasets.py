import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, Data
import data_utils

class GDataset(InMemoryDataset):
    def __init__(self, config, task_ids, nfeat=12, transform=None, pre_transform=None, train=True):
        self.config=config
        self.task_ids=task_ids
        self.root=config.data_path
        self.nfeat=nfeat
        self.train=train
        self.name_suffix= 'train' if train else 'test'
        super(GDataset, self).__init__(config.data_path, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        #print('@GDataset.init: x shape: ', self.data.x.shape)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.root+'/processed_%s.pt'%self.name_suffix]

    def download(self):
        pass


    def process(self):
        # Read data into huge `Data` list.

        # data.x: node feat mat [nnode x nnodefeat]
        # data.edge_index connectivity [2 x nedge] torch.long
        # data.edge_attr [nedge x nedgefeat]
        # data.y target

        obj_datas, conn_datas, manifold_datas, label_datas = [], [], [], []
        cflag_datas = []
        '''
        max_obj_size=0
        for i in range(25):
            filename = data_path+'/'+task_ids[i*100]+'.log'

            objn=get_obj_num_from_file(filename)
            print(i, objn)

            if(objn>max_obj_size):
                max_obj_size=objn
        '''
        data_list = []
        for task_id in self.task_ids:
            filename = self.config.log_dir + '/' + task_id + '.log'
            data = data_utils.get_data_from_file(filename)
            data_list=data_list+data
        data, slices = self.collate(data_list)
        #print('second collate:')
        #print('data.x: ', data.x.shape)
        #print('data.y: ', data.y.shape)

        '''
        if config.normalize:
            obj_datas = obj_datas.reshape([-1, 13])
            mean = np.mean(obj_datas[:, :7], axis=0)
            std = np.std(obj_datas[:, :7], axis=0)
            obj_datas[:, :7] = ((obj_datas[:, :7] - mean) / std)
            obj_datas = obj_datas.reshape([-1, obj_num, 13])
            for i in range(manifold_datas.shape[0]):
                if manifold_datas[i].shape[0] > 0:
                    manifold_datas[i][:, :2] = (manifold_datas[i][:, :2] - mean[1:3]) / std[1:3]
                    manifold_datas[i][:, 2:] = (manifold_datas[i][:, 2:] - mean[1:3]) / std[1:3]
        if config.corrupt:
            py_noise = np.random.normal(scale=0.01, size=(scene_num, obj_num))
            obj_datas[:, :, 2] += py_noise
            vy_noise = np.random.normal(scale=0.01, size=(scene_num, obj_num))
            obj_datas[:, :, 4] += vy_noise

        if config.normalize:
            label_datas = label_datas.reshape([-1, 6])
            mean_label = np.mean(label_datas, axis=0)
            std_label = np.std(label_datas, axis=0)
            label_datas = (label_datas - mean_label) / std_label
            label_datas = label_datas.reshape([-1, obj_num, 6])

        '''
        # data: n x 13
        if self.train:
            mean=torch.mean(data.x[:,:7], axis=0)
            std=torch.std(data.x[:,:7], axis=0)
            mean_label=torch.mean(data.y, axis=0)
            std_label=torch.std(data.y, axis=0)
        else:
            mean_std=np.load(self.root+'/mean_std.npz')
            mean=torch.from_numpy(mean_std['mean'])
            std=torch.from_numpy(mean_std['std'])
            mean_label=torch.from_numpy(mean_std['mean_label'])
            std_label=torch.from_numpy(mean_std['std_label'])
        #print('mean and std:')
        #print(torch.mean(data.x[:,:7],axis=0))
        #print(torch.std(data.x[:,:7],axis=0))
        #print(torch.mean(data.y, axis=0))
        #print(torch.std(data.y, axis=0))
        data.x[:,:7]=(data.x[:,:7]-mean)/std
        data.y=(data.y-mean_label)/std_label
        #print('data.x after normalization: ')
        #print(torch.mean(data.x[:,:7],axis=0))
        #print(torch.std(data.x[:,:7],axis=0))
        #print('data.y after normalization: ')
        #print(torch.mean(data.y, axis=0))
        #print(torch.std(data.y, axis=0))
        data.edge_attr[:, :2] = (data.edge_attr[:, :2] - mean[1:3]) / std[1:3]
        data.edge_attr[:, 2:] = (data.edge_attr[:, 2:] - mean[1:3]) / std[1:3]
        torch.save((data, slices), self.processed_paths[0])
        if self.train:
            np.savez(self.root+'/mean_std.npz', mean=mean.numpy(), std=std.numpy(),
                     mean_label=mean_label.numpy(), std_label=std_label.numpy())
        print('%s dataset generated at '%self.name_suffix,self.root)

class GStabDataset(InMemoryDataset):
    def __init__(self,root='/home/yiran/pc_mapping/simnet/dataset', transform=None, pre_transform=None):

        self.root=root
        super(GStabDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['/home/yiran/pc_mapping/simnet/dataset/3-3x30/obj_data.npy']

    @property
    def processed_file_names(self):
        return ['/home/yiran/pc_mapping/simnet/dataset/3-3x30/test_nonexist.pt']

    def download(self):
        pass

    def process(self):
        pass