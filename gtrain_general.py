from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from utils import load_data, accuracy
from models import BallEncoder, ObjectEncoder, GCN2,GCN5,GAT3,GIN, TestLinear
from datasets import GDataset, GTestDataset

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=60, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--batch_size', type=int, default=128)

args = parser.parse_args()
batch_size=args.batch_size
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

writer = SummaryWriter(log_dir='runs/train_gball')
#exit()
# Load data
#adj, features, labels, idx_train, idx_val, idx_test = load_data()
nfeat=12
nembed=32
# Model and optimizer
net=GIN(nfeat=nfeat+nembed, nclass=6)
#net=TestLinear(cin=nfeat+nembed, cout=6)
model = net.double()


obj_encoder = ObjectEncoder(cin=nfeat,cout=nembed).double()
all_param=list(model.parameters()) + list(obj_encoder.parameters())
optimizer = optim.Adam(all_param,
                       lr=args.lr,
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    obj_encoder=obj_encoder.cuda()

data_path='/home/yiran/pc_mapping/simnet/data/00001_'
os.system('rm %s/data.pt'%data_path)
os.system('rm %s/data_test.pt'%data_path)
#features, adj, labels = Variable(features), Variable(adj), Variable(labels)
dataset=GDataset(data_path, nfeat=nfeat)
dataloader=DataLoader(dataset, batch_size=batch_size, drop_last=True)
testset=GTestDataset(data_path, nfeat=nfeat)
testloader=DataLoader(testset, batch_size=batch_size, drop_last=True)
#print('num graph: ', len(dataset))
#exit()
criterion=nn.MSELoss()
criterion_sum=nn.MSELoss(reduction='sum')
#print('train len: ', len(dataloader))
#print('test len: ', len(testloader))

def train(epoch):
    t = time.time()
    model.train()
    loss_val=0
    loss_val_1, loss_val_2, loss_val_3, loss_val_4, loss_val_5, loss_val_6=0,0,0,0,0,0
    closs_val, allloss_val = 0,0
    ccnt_val, cnt_val=0,0
    for data in dataloader:
        #print (data.batch)
        #print (data.x)
        #print (data.edge_index)
        #print(data.edge_index.max())
        #print (data.y)
        if(args.cuda):
            data=data.to('cuda:0')
        optimizer.zero_grad()

        obj_feat=obj_encoder(data.x[:,:nfeat])
        #print(feat_batch.shape)
        feat_batch=torch.cat([obj_feat, data.x[:,:nfeat]], axis=1)
        #print(feat_batch.shape)
        #exit()
        #print('feat batch: ', feat_batch.shape)
        #print('bound_map: \n', bound_map)
        #print('ball_feat: \n', ball_feat)
        #print('bound_feat: \n', bound_feat)
        #print('feat_batch: \n', feat_batch)

        #feat_batch=data
        # check if axis is correct?
        #print(data)
        #print(data.num_graphs)
        #print(data.edge_index.max())

        output=model(feat_batch, data)
        #print(output.shape)
        #print(bound_map.shape)
        #exit()
        #print(output)
        #print (data.x.shape)
        #print(output.shape)
        #exit()
        movable_map=data.x[:,nfeat].long()
        output = output[~movable_map]
        label = data.y[~movable_map]
        #print(output)
        #exit()
        # output shape: batch x 3 x 6
        loss = criterion(output, label)
        loss_train_1=criterion(output[:,0], data.y[:,0]).data.item()
        loss_train_2=criterion(output[:,1], data.y[:,1]).data.item()
        loss_train_3=criterion(output[:,2], data.y[:,2]).data.item()
        loss_train_4=criterion(output[:,3], data.y[:,3]).data.item()
        loss_train_5=criterion(output[:,4], data.y[:,4]).data.item()
        loss_train_6=criterion(output[:,5], data.y[:,5]).data.item()

        #print(data.x[:,:nfeat][:,4])
        #print(label[:,4])
        #exit()
        #'''
        ccnt=torch.zeros([output.shape[0]])
        ccnt[data.edge_index[0]] += 1
        #print(ccnt)
        #print(data.edge_index)
        cflag=ccnt>0
        cnt_val+=output.shape[0]
        ccnt_val+=cflag.sum().item()

        all_loss=criterion_sum(output[:,4], data.y[:,4]).data.item()
        c_loss = criterion_sum(output[cflag][:,4], data.y[cflag][:,4]).data.item()
        closs_val+=c_loss
        allloss_val+=all_loss
        #print('closs/allloss: %f/%f=%f',c_loss,all_loss,c_loss/all_loss)
        #exit()
        #'''

        loss.backward()
        #print('--------------')
        #print(boundary_feat.data)
        #print(boundary_feat.grad.data)
        #exit()
        optimizer.step()

        #if not args.fastmode:
        #    # Evaluate validation set performance separately,
        #    # deactivates dropout during validation run.
        #    model.eval()
        #    output = model(features, adj)

        loss_val += loss.data.item()
        loss_val_1+=loss_train_1
        loss_val_2+=loss_train_2
        loss_val_3+=loss_train_3
        loss_val_4+=loss_train_4
        loss_val_5+=loss_train_5
        loss_val_6+=loss_train_6

        #print(loss_train.data.item())
        #acc_val = accuracy(output[idx_val], labels[idx_val])
    #print('Epoch: {:04d}'.format(epoch+1),
    #      'loss_train: {:.4f}'.format(loss_train.data.item()),
    #      'acc_train: {:.4f}'.format(acc_train.data.item()),
    #      'loss_val: {:.4f}'.format(loss_val.data.item()),
    #      'acc_val: {:.4f}'.format(acc_val.data.item()),
    #      'time: {:.4f}s'.format(time.time() - t))
    loss_val/=len(dataloader)
    loss_val_1/=len(dataloader)
    loss_val_2/=len(dataloader)
    loss_val_3/=len(dataloader)
    loss_val_4/=len(dataloader)
    loss_val_5/=len(dataloader)
    loss_val_6/=len(dataloader)
    print('epoch %d, loss %f, time=%f'%(epoch, loss_val,time.time()-t))
    print('loss each channel: %f %f %f %f %f %f'%(loss_val_1, loss_val_2,
                                                  loss_val_3,loss_val_4,
                                                  loss_val_5,loss_val_6))
    print('train loss: ', loss_val)
    print('closs/allloss: %f/%f=%f',closs_val,allloss_val,closs_val/allloss_val)
    print('cpoint/allpoint ', ccnt_val, cnt_val, ccnt_val/cnt_val)
    writer.add_scalar('train_loss', loss_val, global_step=epoch)
    return loss_val


def eval(epoch):
    t = time.time()
    model.train()
    loss_val=0
    loss_val_1, loss_val_2, loss_val_3, loss_val_4, loss_val_5, loss_val_6=0,0,0,0,0,0

    for data in testloader:
        #print (data.batch)
        #print (data.x)
        #print (data.edge_index)
        #print(data.edge_index.max())
        #print (data.y)
        if(args.cuda):
            data=data.to('cuda:0')
        optimizer.zero_grad()

        obj_feat=obj_encoder(data.x[:,:nfeat])
        #print(feat_batch.shape)
        feat_batch=torch.cat([obj_feat, data.x[:,:nfeat]], axis=1)
        #print(feat_batch.shape)
        #exit()
        #print('feat batch: ', feat_batch.shape)
        #print('bound_map: \n', bound_map)
        #print('ball_feat: \n', ball_feat)
        #print('bound_feat: \n', bound_feat)
        #print('feat_batch: \n', feat_batch)

        #feat_batch=data
        # check if axis is correct?
        #print(data)
        #print(data.num_graphs)
        #print(data.edge_index.max())

        output=model(feat_batch, data)
        #print(output.shape)
        #print(bound_map.shape)
        #exit()
        #print(output)
        #print (data.x.shape)
        #print(output.shape)
        #exit()
        movable_map=data.x[:,nfeat].long()
        output = output[~movable_map]
        label = data.y[~movable_map]
        #print(output)
        #exit()
        # output shape: batch x 3 x 6
        loss = criterion(output, label)
        loss_train_1=criterion(output[:,0], data.y[:,0]).data.item()
        loss_train_2=criterion(output[:,1], data.y[:,1]).data.item()
        loss_train_3=criterion(output[:,2], data.y[:,2]).data.item()
        loss_train_4=criterion(output[:,3], data.y[:,3]).data.item()
        loss_train_5=criterion(output[:,4], data.y[:,4]).data.item()
        loss_train_6=criterion(output[:,5], data.y[:,5]).data.item()

        '''
        ccnt=torch.zeros([output.shape[0]])
        ccnt[data.edge_index[0]] += 1
        print(ccnt)
        print(data.edge_index)
        cflag=ccnt>0

        all_loss=criterion_sum(output[:,4], data.y[:,4]).data.item()
        c_loss = criterion_sum(output[cflag][:,4], data.y[cflag][:,4]).data.item()
        print(all_loss, c_loss)
        exit()
        '''

        loss_val += loss.data.item()
        loss_val_1+=loss_train_1
        loss_val_2+=loss_train_2
        loss_val_3+=loss_train_3
        loss_val_4+=loss_train_4
        loss_val_5+=loss_train_5
        loss_val_6+=loss_train_6

        #print(loss_train.data.item())
        #acc_val = accuracy(output[idx_val], labels[idx_val])
    #print('Epoch: {:04d}'.format(epoch+1),
    #      'loss_train: {:.4f}'.format(loss_train.data.item()),
    #      'acc_train: {:.4f}'.format(acc_train.data.item()),
    #      'loss_val: {:.4f}'.format(loss_val.data.item()),
    #      'acc_val: {:.4f}'.format(acc_val.data.item()),
    #      'time: {:.4f}s'.format(time.time() - t))
    loss_val/=len(testloader)
    loss_val_1/=len(testloader)
    loss_val_2/=len(testloader)
    loss_val_3/=len(testloader)
    loss_val_4/=len(testloader)
    loss_val_5/=len(testloader)
    loss_val_6/=len(testloader)
    print('epoch %d, loss %f, time=%f'%(epoch, loss_val,time.time()-t))
    print('loss each channel: %f %f %f %f %f %f'%(loss_val_1, loss_val_2,
                                                  loss_val_3,loss_val_4,
                                                  loss_val_5,loss_val_6))
    writer.add_scalar('eval_loss', loss_val, global_step=epoch)
    print('test loss: ', loss_val)

    return loss_val

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))
    eval(epoch)
    #torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    #if loss_values[-1] < best:
    #    best = loss_values[-1]
    #    best_epoch = epoch
    #    bad_counter = 0
    #else:
    #    bad_counter += 1

    #if bad_counter == args.patience:
    #    break

    #files = glob.glob('*.pkl')
    #for file in files:
    #    epoch_nb = int(file.split('.')[0])
    #    if epoch_nb < best_epoch:
    #        os.remove(file)
torch.save(model.state_dict(), 'gin-5-60.pth')
files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

#print("Optimization Finished!")
#print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
#print('Loading {}th epoch'.format(best_epoch))
#model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
#compute_test()
