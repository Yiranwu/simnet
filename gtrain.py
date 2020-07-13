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
from models import BallEncoder, GCN2,GCN5,GAT3,GIN
from datasets import GBallDataset,GBallTestDataset

os.system('rm /home/yiran/pc_mapping/simnet/data.pt')
os.system('rm /home/yiran/pc_mapping/simnet/data_test.pt')
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=60, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
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
nfeat=7
# Model and optimizer
net=GIN(nfeat=nfeat*2, nclass=6)
model = net.double()

ball_encoder = BallEncoder(nfeat=nfeat).double()
boundary_feat = nn.Parameter(torch.randn(1,nfeat).double(), requires_grad=True)
bound_map=torch.tensor([(1 if i%4==3 else 0) for i in range(batch_size*4)],
                       dtype=torch.bool)
bound_map_bc=bound_map.reshape([-1,1]).repeat(1,nfeat)
all_param=list(model.parameters()) + list(ball_encoder.parameters()) + [boundary_feat]
optimizer = optim.Adam(all_param,
                       lr=args.lr,
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    ball_encoder=ball_encoder.cuda()
    boundary_feat=boundary_feat.cuda()
    bound_map=bound_map.cuda()
    bound_map_bc=bound_map_bc.cuda()

#features, adj, labels = Variable(features), Variable(adj), Variable(labels)
dataset=GBallDataset('/home/yiran/pc_mapping/simnet', nfeat=nfeat)
dataloader=DataLoader(dataset, batch_size=batch_size, drop_last=True)
testset=GBallTestDataset('/home/yiran/pc_mapping/simnet', nfeat=nfeat)
testloader=DataLoader(testset, batch_size=batch_size, drop_last=True)
#print('num graph: ', len(dataset))
#exit()
criterion=nn.MSELoss()
#print('train len: ', len(dataloader))
#print('test len: ', len(testloader))

def train(epoch):
    t = time.time()
    model.train()
    loss_val=0
    for data in dataloader:
        #print (data.batch)
        #print (data.x)
        #print (data.edge_index)
        #print(data.edge_index.max())
        #print (data.y)
        if(args.cuda):
            data=data.to('cuda:0')
        optimizer.zero_grad()

        ball_feat=ball_encoder(data.x)
        #print (data.x.shape)
        #print(ball_feat.shape)
        n_nodes=ball_feat.shape[0]
        bound_feat=boundary_feat.repeat(n_nodes,1)

        #print(bound_map.shape)
        #print(bound_feat.shape)
        #exit()
        feat_batch=torch.where(bound_map_bc, bound_feat, ball_feat)
        #print(feat_batch.shape)
        feat_batch=torch.cat([feat_batch, data.x], axis=1)
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
        output = output[~bound_map]
        #print(output)
        #exit()
        # output shape: batch x 3 x 6
        loss = criterion(output, data.y)
        loss_train_1=criterion(output[:,0], data.y[:,0]).data.item()
        loss_train_2=criterion(output[:,1], data.y[:,1]).data.item()
        loss_train_3=criterion(output[:,2], data.y[:,2]).data.item()
        loss_train_4=criterion(output[:,3], data.y[:,3]).data.item()
        loss_train_5=criterion(output[:,4], data.y[:,4]).data.item()
        loss_train_6=criterion(output[:,5], data.y[:,5]).data.item()

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
        #print(loss_train.data.item())
        #acc_val = accuracy(output[idx_val], labels[idx_val])
    #print('Epoch: {:04d}'.format(epoch+1),
    #      'loss_train: {:.4f}'.format(loss_train.data.item()),
    #      'acc_train: {:.4f}'.format(acc_train.data.item()),
    #      'loss_val: {:.4f}'.format(loss_val.data.item()),
    #      'acc_val: {:.4f}'.format(acc_val.data.item()),
    #      'time: {:.4f}s'.format(time.time() - t))
    loss_val/=len(dataloader)
    print('epoch %d, loss %f, time=%f'%(epoch, loss_val,time.time()-t))
    print('loss each channel: %f %f %f %f %f %f'%(loss_train_1, loss_train_2,
                                                  loss_train_3,loss_train_4,
                                                  loss_train_5,loss_train_6))
    print('train loss: %f', loss_val)
    writer.add_scalar('train_loss', loss_val, global_step=epoch)
    return loss_val


def eval(epoch):
    model.eval()
    t = time.time()
    loss_val=0
    for data in testloader:
        if(args.cuda):
            data=data.to('cuda:0')

        ball_feat=ball_encoder(data.x)
        n_nodes=ball_feat.shape[0]
        bound_map=torch.tensor([(1 if i%4==3 else 0) for i in range(n_nodes)],
                               dtype=torch.bool).cuda()
        bound_map_bc=bound_map.reshape([-1,1]).repeat(1,nfeat)
        bound_feat=boundary_feat.repeat(n_nodes,1)

        #print(bound_map.shape)
        #print(bound_feat.shape)
        #exit()
        feat_batch=torch.where(bound_map_bc, bound_feat, ball_feat)
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
        #print(output)
        #print (data.x.shape)
        #print(output.shape)
        #exit()
        output = output[~bound_map]
        #print(output)
        #exit()
        # output shape: batch x 3 x 6
        angle_output=output[:,0]
        angle_label=data.y[:,0]
        loss1=criterion(angle_output,angle_label)
        loss_train_1=loss1.data.item()
        loss2 = criterion(output[:,1:], data.y[:,1:])
        loss_train_2=criterion(output[:,1], data.y[:,1]).data.item()
        loss_train_3=criterion(output[:,2], data.y[:,2]).data.item()
        loss_train_4=criterion(output[:,3], data.y[:,3]).data.item()
        loss_train_5=criterion(output[:,4], data.y[:,4]).data.item()
        loss_train_6=criterion(output[:,5], data.y[:,5]).data.item()
        loss_train=loss1+loss2
        #print(loss_train.data.item())

        #if not args.fastmode:
        #    # Evaluate validation set performance separately,
        #    # deactivates dropout during validation run.
        #    model.eval()
        #    output = model(features, adj)

        loss_val += loss_train.data.item()
        #acc_val = accuracy(output[idx_val], labels[idx_val])
    loss_val/=len(testloader)
    #print('Epoch: {:04d}'.format(epoch+1),
    #      'loss_train: {:.4f}'.format(loss_train.data.item()),
    #      'acc_train: {:.4f}'.format(acc_train.data.item()),
    #      'loss_val: {:.4f}'.format(loss_val.data.item()),
    #      'acc_val: {:.4f}'.format(acc_val.data.item()),
    #      'time: {:.4f}s'.format(time.time() - t))
    writer.add_scalar('eval_loss', loss_val, global_step=epoch)
    print('test loss: %f ', loss_val)

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
torch.save(model.state_dict(), 'gcn-2-60.pth')
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
