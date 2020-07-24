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
from models import BallEncoder, ObjectEncoder, GCN2,GCN5,GAT3,GIN, TestLinear, GINWOBN, GINE
from datasets import GDataset, GTestDataset

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--exp_name', type=str, default='gin-bn-nocor-5')
#parser.add_argument('--dataset_spec', type=str, default='00001_noise_')
parser.add_argument('--dataset_spec', type=str, default='00001_')

args = parser.parse_args()
batch_size=args.batch_size
args.cuda = not args.no_cuda and torch.cuda.is_available()
exp_name=args.exp_name
dataset_spec=args.dataset_spec

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

writer = SummaryWriter(log_dir='runs/train_gball')
#exit()
# Load data
#adj, features, labels, idx_train, idx_val, idx_test = load_data()
vfeat=12
hidden=32
# Model and optimizer
net=GINE(vfeat=12, hidden=32, nclass=6)
#net=TestLinear(cin=nfeat+nembed, cout=6)
model = net.double()

all_param=model.parameters()
optimizer = optim.Adam(all_param,
                       lr=args.lr,
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()

data_path='/home/yiran/pc_mapping/simnet/data/%s'%dataset_spec
os.system('rm %sprocessed_data.pt'%data_path)
os.system('rm %sprocessed_data_test.pt'%data_path)
#features, adj, labels = Variable(features), Variable(adj), Variable(labels)
dataset=GDataset(data_path, nfeat=vfeat, train=True)
dataloader=DataLoader(dataset, batch_size=batch_size, drop_last=True)
testset=GDataset(data_path, nfeat=vfeat, train=False)
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
        #print(data.batch)
        #print(data.batch.min())
        #exit()

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

        output=model(data)
        #print(output.shape)
        #print(bound_map.shape)
        #exit()
        #print(output)
        #print (data.x.shape)
        #print(output.shape)
        #exit()
        movable_map=data.x[:,vfeat].bool()
        masked_output = output[movable_map]
        masked_label = data.y[movable_map]
        #print(output.shape)
        #print(label.shape)
        #print(output)
        #exit()
        # output shape: batch x 3 x 6
        loss = criterion(masked_output, masked_label)
        loss_train_1=criterion(masked_output[:,0], masked_label[:,0]).data.item()
        loss_train_2=criterion(masked_output[:,1], masked_label[:,1]).data.item()
        loss_train_3=criterion(masked_output[:,2], masked_label[:,2]).data.item()
        loss_train_4=criterion(masked_output[:,3], masked_label[:,3]).data.item()
        loss_train_5=criterion(masked_output[:,4], masked_label[:,4]).data.item()
        loss_train_6=criterion(masked_output[:,5], masked_label[:,5]).data.item()

        #print(data.x[:,:nfeat][:,4])
        #print(label[:,4])
        #exit()
        #'''

        #ccnt=torch.zeros([masked_output.shape[0]])
        #ccnt[data.edge_index[0]] += 1
        #cflag=ccnt>0
        #cnt_val+=output.shape[0]
        #ccnt_val+=cflag.sum().item()

        #all_loss=criterion_sum(output[:,4], data.y[:,4]).data.item()
        #c_loss = criterion_sum(output[cflag][:,4], data.y[cflag][:,4]).data.item()
        #closs_val+=c_loss
        #allloss_val+=all_loss
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
    #print('closs/allloss: %f/%f=%f',closs_val,allloss_val,closs_val/allloss_val)
    #print('cpoint/allpoint ', ccnt_val, cnt_val, ccnt_val/cnt_val)
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
        movable_map=data.x[:,nfeat].bool()
        masked_output = output[movable_map]
        masked_label = data.y[movable_map]
        #print(output.shape)
        #print(label.shape)
        #print(output)
        #exit()
        # output shape: batch x 3 x 6
        loss = criterion(masked_output, masked_label)
        loss_train_1=criterion(masked_output[:,0], masked_label[:,0]).data.item()
        loss_train_2=criterion(masked_output[:,1], masked_label[:,1]).data.item()
        loss_train_3=criterion(masked_output[:,2], masked_label[:,2]).data.item()
        loss_train_4=criterion(masked_output[:,3], masked_label[:,3]).data.item()
        loss_train_5=criterion(masked_output[:,4], masked_label[:,4]).data.item()
        loss_train_6=criterion(masked_output[:,5], masked_label[:,5]).data.item()

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
    #eval(epoch)
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
torch.save(model.state_dict(), '%s.pth'%exp_name)
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
