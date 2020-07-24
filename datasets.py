import torch
import numpy as np
import pickle
from torch.utils.data import Dataset

class BallDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.obj_data=np.load('obj_data.npy')
        self.adj_data=np.load('adj_data.npy')
        self.label_data=np.load('label_data.npy')
        vmax=self.obj_data.reshape([-1,6]).max(axis=0, keepdims=True)
        vmin=self.obj_data.reshape([-1,6]).min(axis=0, keepdims=True)
        self.obj_data = (self.obj_data - vmin) / (vmax - vmin)
        self.label_data = self.label_data / (vmax-vmin)

        self.size=self.obj_data.shape[0]

    def __len__(self):
        return self.size
        #return 1

    def __getitem__(self, idx):
        return self.obj_data[idx], self.adj_data[idx], self.label_data[idx]


import torch
from torch_geometric.data import InMemoryDataset, Data


class GBallDataset(InMemoryDataset):
    def __init__(self, root, nfeat, transform=None, pre_transform=None, train=True):
        super(GBallDataset, self).__init__(root, transform, pre_transform)
        self.root=root
        self.nfeat=nfeat
        self.train=train
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.root+'/obj_data.npy', self.root+'/conn_data.npy', self.root+'/label_data.npy']

    @property
    def processed_file_names(self):
        return [self.root+'/data.pt']

    def download(self):
        pass
        # Download to `self.raw_dir`.

    def process(self):
        # Read data into huge `Data` list.

        # data.x: node feat mat [nnode x nnodefeat]
        # data.edge_index connectivity [2 x nedge] torch.long
        # data.edge_attr [nedge x nedgefeat]
        # data.y target
        obj_file, conn_file, label_file=self.raw_file_names
        obj_data=torch.from_numpy(np.load(obj_file))
        conn_data=np.load(conn_file, allow_pickle=True)
        #print(conn_data.shape)
        #print(conn_data[0].shape)
        #print(conn_data[0].dtype)
        #print(conn_data[0])
        #exit()
        print(obj_data.shape[0])
        label_data=torch.from_numpy(np.load(label_file))
        #if self.train:
        node_list=range(0,40000)
            #ngraph=obj_data.shape[0]
        #else:
            #node_list=range(40000, obj_data.shape[0])
        #ngraph=500
        data_list = []
        for i in node_list:
            #print(conn_data[i])
            #print(torch.from_numpy(conn_data[i]))
            obj=torch.cat([obj_data[i], torch.zeros([1,7]).double()], axis=0)
            #obj=obj_data[i]
            conn=conn_data[i]
            if conn.shape[0]==0:
                conn=conn.reshape([2,0])
            gdata=Data(x=obj, edge_index=torch.from_numpy(conn),
                       y=label_data[i])
            data_list.append(gdata)
            #print(gdata)
        #exit()
        #if self.pre_filter is not None:
        #    data_list = [data for data in data_list if self.pre_filter(data)]

        #if self.pre_transform is not None:
        #    data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class GBallTestDataset(InMemoryDataset):
    def __init__(self, root, nfeat, transform=None, pre_transform=None, train=True):
        super(GBallTestDataset, self).__init__(root, transform, pre_transform)
        self.root=root
        self.nfeat=nfeat
        self.train=train
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.root+'/obj_data.npy', self.root+'/conn_data.npy', self.root+'/label_data.npy']

    @property
    def processed_file_names(self):
        return [self.root+'/data_test.pt']

    def download(self):
        pass
        # Download to `self.raw_dir`.

    def process(self):
        # Read data into huge `Data` list.

        # data.x: node feat mat [nnode x nnodefeat]
        # data.edge_index connectivity [2 x nedge] torch.long
        # data.edge_attr [nedge x nedgefeat]
        # data.y target
        obj_file, conn_file, label_file=self.raw_file_names
        obj_data=torch.from_numpy(np.load(obj_file))
        conn_data=np.load(conn_file, allow_pickle=True)
        #print(conn_data.shape)
        #print(conn_data[0].shape)
        #print(conn_data[0].dtype)
        #print(conn_data[0])
        #exit()
        print(obj_data.shape[0])
        label_data=torch.from_numpy(np.load(label_file))
        node_list=range(40000, obj_data.shape[0])
        #ngraph=500
        data_list = []
        for i in node_list:
            #print(conn_data[i])
            #print(torch.from_numpy(conn_data[i]))
            obj=torch.cat([obj_data[i], torch.zeros([1,7]).double()], axis=0)
            #obj=obj_data[i]
            conn=conn_data[i]
            if conn.shape[0]==0:
                conn=conn.reshape([2,0])
            gdata=Data(x=obj, edge_index=torch.from_numpy(conn),
                       y=label_data[i])
            data_list.append(gdata)
            #print(gdata)

        #if self.pre_filter is not None:
        #    data_list = [data for data in data_list if self.pre_filter(data)]

        #if self.pre_transform is not None:
        #    data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class GDataset(InMemoryDataset):
    def __init__(self, root, nfeat, transform=None, pre_transform=None, train=True):
        super(GDataset, self).__init__(root, transform, pre_transform)
        self.root=root
        self.nfeat=nfeat
        self.train=train
        #print('GDataset initialized. self.train set.')
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        #print('raw_file_names() called.')
        return [self.root+'obj_data.npy',
                self.root+'conn_data.npy',
                self.root+'manifold_data.npy',
                self.root+'label_data.npy']

    @property
    def processed_file_names(self):
        #print('processed_file_names() called.')
        return [self.root+'processed_data.pt']

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
        #print(obj_data.shape)
        #exit()
        #print(conn_data.shape)
        #print(conn_data[0].shape)
        #print(conn_data[0].dtype)
        #print(conn_data[0])
        #exit()
        #print(obj_data.shape[0])
        label_data=torch.from_numpy(np.load(label_file))
        node_list=range(0,40000)
        #if self.train:
        #    node_list=range(0,40000)
        #else:
        #    node_list=range(40000, obj_data.shape[0])
        data_list = []
        for i in node_list:
            #print(conn_data[i])
            #print(torch.from_numpy(conn_data[i]))
            obj=obj_data[i].double()
            #obj=obj_data[i]
            conn=conn_data[i]
            manifold=manifold_data[i]
            if conn.shape[0]==0:
                conn=conn.reshape([2,0])
            if manifold.shape[0]==0:
                manifold=manifold.reshape([0,4])
            gdata=Data(x=obj, edge_index=torch.from_numpy(conn),
                       y=label_data[i], edge_attr=torch.from_numpy(manifold))
            data_list.append(gdata)
            #print(gdata)
        #exit()
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
        return [self.root+'obj_data.npy',
                self.root+'conn_data.npy', self.root+'label_data.npy']

    @property
    def processed_file_names(self):
        return [self.root+'processed_data_test.pt']

    def download(self):
        pass
        # Download to `self.raw_dir`.

    def process(self):
        # Read data into huge `Data` list.

        # data.x: node feat mat [nnode x nnodefeat]
        # data.edge_index connectivity [2 x nedge] torch.long
        # data.edge_attr [nedge x nedgefeat]
        # data.y target
        obj_file, conn_file, label_file=self.raw_file_names
        obj_data=torch.from_numpy(np.load(obj_file))
        conn_data=np.load(conn_file, allow_pickle=True)
        #print(obj_data.shape)
        #exit()
        #print(conn_data.shape)
        #print(conn_data[0].shape)
        #print(conn_data[0].dtype)
        #print(conn_data[0])
        #exit()
        #print(obj_data.shape[0])
        label_data=torch.from_numpy(np.load(label_file))
        node_list=range(40000, obj_data.shape[0])
        #ngraph=500
        data_list = []
        for i in node_list:
            #print(conn_data[i])
            #print(torch.from_numpy(conn_data[i]))
            obj=obj_data[i].double()
            #obj=obj_data[i]
            conn=conn_data[i]
            if conn.shape[0]==0:
                conn=conn.reshape([2,0])
            gdata=Data(x=obj, edge_index=torch.from_numpy(conn),
                       y=label_data[i])
            data_list.append(gdata)
            #print(gdata)
        #exit()
        #if self.pre_filter is not None:
        #    data_list = [data for data in data_list if self.pre_filter(data)]

        #if self.pre_transform is not None:
        #    data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
