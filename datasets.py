import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, Data

class GDataset(InMemoryDataset):
    def __init__(self, root, nfeat, transform=None, pre_transform=None, train=True):
        super(GDataset, self).__init__(root, transform, pre_transform)
        self.root=root
        self.nfeat=nfeat
        self.train=train
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.root+'/obj_data.npy',
                self.root+'/conn_data.npy',
                self.root+'/manifold_data.npy',
                self.root+'/label_data.npy']

    @property
    def processed_file_names(self):
        return [self.root+'/processed_data.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.

        # data.x: node feat mat [nnode x nnodefeat]
        # data.edge_index connectivity [2 x nedge] torch.long
        # data.edge_attr [nedge x nedgefeat]
        # data.y target
        obj_file, conn_file, manifold_file, label_file=self.raw_file_names
        obj_data=torch.from_numpy(np.load(obj_file))
        conn_data=np.load(conn_file, allow_pickle=True)
        manifold_data=np.load(manifold_file, allow_pickle=True)
        label_data=torch.from_numpy(np.load(label_file))
        node_list=range(0,int(obj_data.shape[0]*0.8))
        #node_list=range(0,obj_data.shape[0])
        #if self.train:
        #    node_list=range(0,40000)
        #else:
        #    node_list=range(40000, obj_data.shape[0])
        data_list = []
        for i in node_list:
            obj=obj_data[i].double()
            conn=conn_data[i]
            manifold=manifold_data[i]
            if conn.shape[0]==0:
                conn=conn.reshape([2,0])
            if manifold.shape[0]==0:
                manifold=manifold.reshape([0,4])
            gdata=Data(x=obj, edge_index=torch.from_numpy(conn),
                       y=label_data[i], edge_attr=torch.from_numpy(manifold))
            data_list.append(gdata)
        #if self.pre_filter is not None:
        #    data_list = [data for data in data_list if self.pre_filter(data)]

        #if self.pre_transform is not None:
        #    data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class GTestDataset(InMemoryDataset):
    def __init__(self, root, nfeat, transform=None, pre_transform=None, train=True):
        super(GTestDataset, self).__init__(root, transform, pre_transform)
        self.root=root
        self.nfeat=nfeat
        self.train=train
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.root+'/obj_data.npy',
                self.root+'/conn_data.npy',
                self.root+'/manifold_data.npy',
                self.root+'/label_data.npy']

    @property
    def processed_file_names(self):
        return [self.root+'/processed_data_test.pt']

    def download(self):
        pass
        # Download to `self.raw_dir`.

    def process(self):
        # Read data into huge `Data` list.

        # data.x: node feat mat [nnode x nnodefeat]
        # data.edge_index connectivity [2 x nedge] torch.long
        # data.edge_attr [nedge x nedgefeat]
        # data.y target
        obj_file, conn_file, manifold_file, label_file=self.raw_file_names
        obj_data=torch.from_numpy(np.load(obj_file))
        conn_data=np.load(conn_file, allow_pickle=True)
        manifold_data=np.load(manifold_file, allow_pickle=True)
        label_data=torch.from_numpy(np.load(label_file))
        node_list=range(int(obj_data.shape[0]*0.8),obj_data.shape[0])
        #if self.train:
        #    node_list=range(0,40000)
        #else:
        #    node_list=range(40000, obj_data.shape[0])
        data_list = []
        for i in node_list:
            obj=obj_data[i].double()
            conn=conn_data[i]
            manifold=manifold_data[i]
            if conn.shape[0]==0:
                conn=conn.reshape([2,0])
            if manifold.shape[0]==0:
                manifold=manifold.reshape([0,4])
            gdata=Data(x=obj, edge_index=torch.from_numpy(conn),
                       y=label_data[i], edge_attr=torch.from_numpy(manifold))
            data_list.append(gdata)
        #if self.pre_filter is not None:
        #    data_list = [data for data in data_list if self.pre_filter(data)]

        #if self.pre_transform is not None:
        #    data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])