import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer
from torch_geometric.nn import GCNConv,GATConv, GINConv, global_add_pool, GINEConv


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, batch_size):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.batch_size=batch_size

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha,
                                               concat=True, batch_size=batch_size)
                           for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha,
                                           concat=False, batch_size=batch_size)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x,adj)
        return x
        #x = F.elu(self.out_att(x, adj))
        #return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)



class BallEncoder(nn.Module):
    def __init__(self, nfeat=32):
        """Sparse version of GAT."""
        super(BallEncoder, self).__init__()
        self.l1=nn.Linear(7, nfeat)
        self.l2=nn.Linear(nfeat,nfeat)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class ObjectEncoder(nn.Module):
    def __init__(self, cin=12, cout=32):
        """Sparse version of GAT."""
        super(ObjectEncoder, self).__init__()
        self.l1=nn.Linear(cin, 32)
        self.l2=nn.Linear(32,32)
        self.l3=nn.Linear(32,cout)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
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
        #print(x.shape)

        edge_index=data.edge_index
        batch=data.batch
        #print('net init x: ', x.shape)
        #exit()
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
        #print('after conv x: ', x.shape)
        #x = global_add_pool(x, batch)
        #print('after pool x: ', x.shape)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        #print(x.shape)
        return x
        #return F.log_softmax(x, dim=-1)


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
        #print(x.shape)

        edge_index=data.edge_index
        batch=data.batch
        #print('net init x: ', x.shape)
        #exit()
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




class GINE(torch.nn.Module):
    def __init__(self, layers=5, vfeat=12, efeat=3, hidden=32, nclass=6):
        super(GINE, self).__init__()

        self.layers=layers
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

    def forward(self, x, data):
        #print(x.shape)

        for i in range(self.layers):
            edge_feat=self.edge_encoders[i](data.edge_attr)
            x=self.convs(x, data.edge_index, edge_feat)
            x=F.relu(x)
            x=self.bns[i](x)
            #possibly dropout?


        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        #print(x.shape)
        return x
        #return F.log_softmax(x, dim=-1)