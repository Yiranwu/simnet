import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv, GINConv, global_add_pool, GINEConv, SplineConv

class ObjectEncoder(nn.Module):
    def __init__(self, cin=12, cout=32):
        """Sparse version of GAT."""
        super(ObjectEncoder, self).__init__()
        self.l1=nn.Linear(cin, cout)
        self.l2=nn.Linear(cout,cout)
        self.l3=nn.Linear(cout,cout)

    def forward(self, x):
        #print('obj enc begin, x=', x)
        #print('x shape: ', x.shape)
        #print('x max and min: ', torch.max(x), torch.min(x))
        #print('bias:', self.l1.bias)
        #print('weight:', self.l1.weight)
        #print('weight: ', self.l1.weight.device)
        #print('data: ', x.device)
        #exit()
        x = F.relu(self.l1(x))
        #print('obj enc after l1, x=', x)
        x = F.relu(self.l2(x))
        #print('obj enc after l2, x=', x)
        x = self.l3(x)
        #print('obj enc after l3, x=', x)
        #exit()
        return x

class TestLinear(nn.Module):
    def __init__(self, cin, cout, batch_size=128):
        """Sparse version of GAT."""
        super(TestLinear, self).__init__()
        self.l1=nn.Linear(cin, 512)
        self.l2=nn.Linear(512,512)
        self.l3=nn.Linear(512,512)
        self.l4=nn.Linear(512,512)
        self.l5=nn.Linear(512,cout)

    def forward(self, x, adj):

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = self.l5(x)
        return x


class GCN2(torch.nn.Module):
    def __init__(self, nfeat=32, nclass=6):
        super(GCN2, self).__init__()
        self.conv1 = GCNConv(nfeat, 256)
        self.conv2 = GCNConv(256, nclass)

    def forward(self, x, data):
        #x, edge_index = data.x, data.edge_index
        edge_index = data.edge_index
        #print(edge_index.min(),edge_index.max())
        #exit()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        #return F.log_softmax(x, dim=1)
        return x


class GCN5(torch.nn.Module):
    def __init__(self, nfeat=32, nclass=6):
        super(GCN5, self).__init__()
        self.conv1 = GCNConv(nfeat, 256)
        self.conv2 = GCNConv(256, 1024)
        self.conv3 = GCNConv(1024, 1024)
        self.conv4 = GCNConv(1024, 256)
        self.conv5 = GCNConv(256, nclass)

    def forward(self, x, data):
        #x, edge_index = data.x, data.edge_index
        edge_index = data.edge_index
        #print(edge_index.min(),edge_index.max())
        #exit()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.conv5(x, edge_index)

        #return F.log_softmax(x, dim=1)
        return x

class GAT3(torch.nn.Module):
    def __init__(self):
        super(GAT3, self).__init__()
        self.conv1 = GATConv(32, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, 8*8, heads=8,
                             dropout=0.6)
        self.conv3 = GATConv(8 * 8 * 8, 6, heads=1, concat=False,
                             dropout=0.6)

    def forward(self, x, data):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, data.edge_index)
        return x


class GNet(torch.nn.Module):
    def __init__(self, vfeat=12, hidden=32):
        super(GNet, self).__init__()
        self.obj_encoder = ObjectEncoder(cin=vfeat, cout=hidden)
    def encode_obj(self, obj):
        return self.obj_encoder(obj)


class GIN(torch.nn.Module):
    def __init__(self, nfeat=12, nclass=6):
        super(GIN, self).__init__()

        dim = 128

        nn1 = nn.Sequential(nn.Linear(nfeat, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, nclass)

    def forward(self, x, data):

        edge_index=data.edge_index
        batch=data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x


class GINWOBN(torch.nn.Module):
    def __init__(self, nfeat=12, nclass=6):
        super(GINWOBN, self).__init__()

        dim = 128

        nn1 = nn.Sequential(nn.Linear(nfeat, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, nclass)

    def forward(self, x, data):

        edge_index=data.edge_index
        batch=data.batch
        x = F.relu(self.conv1(x, edge_index))
        #x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        #x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        #x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        #x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        #x = self.bn5(x)
        #print('after conv x: ', x.shape)
        #x = global_add_pool(x, batch)
        #print('after pool x: ', x.shape)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        #print(x.shape)
        return x
        #return F.log_softmax(x, dim=-1)

class GINE(GNet):
    def __init__(self, layers=5, vfeat=12, efeat=4, hidden=32, nclass=6):
        super(GINE, self).__init__(vfeat, hidden)
        self.layers=layers
        self.vfeat=vfeat
        self.efeat=efeat
        self.hidden=hidden
        self.nclass=nclass
        self.convs=nn.ModuleList()
        self.bns=nn.ModuleList()
        self.edge_encoders=nn.ModuleList()
        for i in range(layers):
            hfunc=nn.Sequential(nn.Linear(hidden, 2 * hidden),
                                nn.BatchNorm1d(2 * hidden),
                                nn.ReLU(),
                                nn.Linear(2 * hidden, hidden),)
            self.convs.append(GINEConv(hfunc, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden))
            self.edge_encoders.append(nn.Sequential(nn.Linear(efeat, hidden),
                                            nn.ReLU(),
                                            nn.Linear(hidden,hidden),))

        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, nclass)

    def forward(self, data):

        x=data.x
        #print('data.x= ', data.x)
        obj_feat=self.encode_obj(x[:,:self.vfeat])
        #print('obj feat= ', obj_feat)
        x=obj_feat
        #x=torch.cat([obj_feat, x[:,:self.vfeat]], axis=1)
        for i in range(self.layers):
            edge_feat=self.edge_encoders[i](data.edge_attr)
            #print('layer %d edge feat= '%i, edge_feat)
            x=self.convs[i](x, data.edge_index, edge_feat)
            #print('layer %d conv= '%i, x)
            x=F.relu(x)
            x=self.bns[i](x)
            #print('layer %d bn= '%i, x)
        #exit()
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x
        #return F.log_softmax(x, dim=-1)



class GINEWide(GNet):
    def __init__(self, layers=5, vfeat=12, efeat=4, hidden=32, nclass=6):
        super(GINEWide, self).__init__(vfeat, hidden)
        self.layers=layers
        self.vfeat=vfeat
        self.efeat=efeat
        self.hidden=hidden
        self.nclass=nclass
        self.convs=nn.ModuleList()
        self.bns=nn.ModuleList()
        self.edge_encoders=nn.ModuleList()
        for i in range(layers):
            hfunc=nn.Sequential(nn.Linear(hidden, 2 * hidden),
                                nn.BatchNorm1d(2 * hidden),
                                nn.ReLU(),
                                nn.Linear(2 * hidden, hidden),)
            self.convs.append(GINEConv(hfunc, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden))
            self.edge_encoders.append(nn.Sequential(nn.Linear(efeat, hidden),
                                            nn.ReLU(),
                                            nn.Linear(hidden,hidden),))

        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, nclass)

    def forward(self, data):

        x=data.x
        #print('data.x= ', data.x)
        obj_feat=self.encode_obj(x[:,:self.vfeat])
        #print('obj feat= ', obj_feat)
        x=obj_feat

        #x=torch.cat([obj_feat, x[:,:self.vfeat]], axis=1)
        for i in range(self.layers):
            edge_feat=self.edge_encoders[i](data.edge_attr)
            #print('layer %d edge feat= '%i, edge_feat)
            x=self.convs[i](x, data.edge_index, edge_feat)
            #print('layer %d conv= '%i, x)
            x=F.relu(x)
            x=self.bns[i](x)
            #print('layer %d bn= '%i, x)
        #exit()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc3(x)
        return x
        #return F.log_softmax(x, dim=-1)

class GINEShallow(GNet):
    def __init__(self, layers=3, vfeat=12, efeat=4, hidden=32, nclass=6):
        super(GINEWide, self).__init__(vfeat, hidden)
        self.layers=layers
        self.vfeat=vfeat
        self.efeat=efeat
        self.hidden=hidden
        self.nclass=nclass
        self.convs=nn.ModuleList()
        self.bns=nn.ModuleList()
        self.edge_encoders=nn.ModuleList()
        for i in range(layers):
            hfunc=nn.Sequential(nn.Linear(hidden, 2 * hidden),
                                nn.BatchNorm1d(2 * hidden),
                                nn.ReLU(),
                                nn.Linear(2 * hidden, hidden),)
            self.convs.append(GINEConv(hfunc, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden))
            self.edge_encoders.append(nn.Sequential(nn.Linear(efeat, hidden),
                                            nn.ReLU(),
                                            nn.Linear(hidden,hidden),))

        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, nclass)

    def forward(self, data):

        x=data.x
        #print('data.x= ', data.x)
        obj_feat=self.encode_obj(x[:,:self.vfeat])
        #print('obj feat= ', obj_feat)
        x=obj_feat
        #x=torch.cat([obj_feat, x[:,:self.vfeat]], axis=1)
        for i in range(self.layers):
            edge_feat=self.edge_encoders[i](data.edge_attr)
            #print('layer %d edge feat= '%i, edge_feat)
            x=self.convs[i](x, data.edge_index, edge_feat)
            #print('layer %d conv= '%i, x)
            x=F.relu(x)
            x=self.bns[i](x)
            #print('layer %d bn= '%i, x)
        #exit()
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x
        #return F.log_softmax(x, dim=-1)

class GINEWOBN(GNet):
    def __init__(self, layers=5, vfeat=12, efeat=4, hidden=32, nclass=6):
        super(GINEWOBN, self).__init__(vfeat, hidden)
        self.layers=layers
        self.vfeat=vfeat
        self.efeat=efeat
        self.hidden=hidden
        self.nclass=nclass
        self.convs=nn.ModuleList()
        self.bns=nn.ModuleList()
        self.edge_encoders=nn.ModuleList()
        for i in range(layers):
            hfunc=nn.Sequential(nn.Linear(hidden, 2 * hidden),
                                nn.BatchNorm1d(2 * hidden),
                                nn.ReLU(),
                                nn.Linear(2 * hidden, hidden),)
            self.convs.append(GINEConv(hfunc, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden))
            self.edge_encoders.append(nn.Sequential(nn.Linear(efeat, hidden),
                                            nn.ReLU(),
                                            nn.Linear(hidden,hidden),))

        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, nclass)

    def forward(self, data):

        x=data.x
        obj_feat=self.encode_obj(x[:,:self.vfeat])
        x=obj_feat
        #x=torch.cat([obj_feat, x[:,:self.vfeat]], axis=1)
        for i in range(self.layers):
            edge_feat=self.edge_encoders[i](data.edge_attr)
            x=self.convs[i](x, data.edge_index, edge_feat)
            x=F.relu(x)
            x=self.bns[i](x)

        x = F.relu(self.fc1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x
        #return F.log_softmax(x, dim=-1)


class GLinear(GNet):
    def __init__(self, layers=5, vfeat=12, efeat=4, hidden=32, nclass=6):
        super(GLinear, self).__init__(vfeat, hidden)
        self.layers=layers
        self.vfeat=vfeat
        self.efeat=efeat
        self.hidden=hidden
        self.nclass=nclass

        #self.fc1 = nn.Linear(hidden, hidden)
        #self.fc2 = nn.Linear(hidden, nclass)

        self.fc1 = nn.Linear(13, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, nclass)

    def forward(self, data):

        x=data.x
        #obj_feat=self.encode_obj(x[:,:self.vfeat])
        #x=obj_feat
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        x=self.fc5(x)
        return x
        #return F.log_softmax(x, dim=-1)

'''
class SplineNet(GNet):
    def __init__(self):
        super(SplineNet, self).__init__()
        self.conv1 = SplineConv(dataset.num_features, 16, dim=1, kernel_size=2)
        self.conv2 = SplineConv(16, dataset.num_classes, dim=1, kernel_size=2)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)
'''