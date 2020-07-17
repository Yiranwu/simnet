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
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from utils import load_data, accuracy
from models import GAT, SpGAT, BallEncoder,TestLinear
from datasets import BallDataset

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=1024, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=32, help='Number of head attentions.')
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

writer = SummaryWriter(log_dir='runs/train_ball')
#exit()
# Load data
#adj, features, labels, idx_train, idx_val, idx_test = load_data()
nfeat=32
# Model and optimizer
model=TestLinear(batch_size)

ball_encoder = BallEncoder()
boundary_feat = Variable(torch.randn(1,nfeat), requires_grad=True)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    ball_encoder=ball_encoder.cuda()
    boundary_feat=boundary_feat.cuda()

#features, adj, labels = Variable(features), Variable(adj), Variable(labels)
dataset = BallDataset()
dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
criterion=nn.MSELoss()


def train(epoch):
    t = time.time()
    model.train()

    for obj_batch, adj_batch, label_batch in dataloader:
        obj_batch=obj_batch.float()
        adj_batch=adj_batch.float()
        label_batch=label_batch.float()
        if(args.cuda):
            obj_batch=obj_batch.cuda()
            adj_batch=adj_batch.cuda()
            label_batch=label_batch.cuda()
        optimizer.zero_grad()

        #ball_feat=ball_encoder(obj_batch)

        #bound_feat=boundary_feat.repeat(batch_size,1).view([batch_size,1,-1])
        #feat_batch=torch.cat([ball_feat, bound_feat], axis=1)
        # check if axis is correct?


        output = model(obj_batch, adj_batch)[:,:3,:]
        # output shape: batch x 3 x 6
        angle_output=output[:,:,0]
        sin_output=torch.sin(angle_output)
        cos_output=torch.cos(angle_output)
        angle_label=label_batch[:,:,0]
        sin_label=torch.sin(angle_label)
        cos_label=torch.cos(angle_label)
        loss1=criterion(sin_output,sin_label) + criterion(cos_output, cos_label)
        loss_train_1=loss1.data.item()
        loss2 = criterion(output[:,:,1:], label_batch[:,:,1:])
        loss_train_2=criterion(output[:,:,1], label_batch[:,:,1]).data.item()
        loss_train_3=criterion(output[:,:,2], label_batch[:,:,2]).data.item()
        loss_train_4=criterion(output[:,:,3], label_batch[:,:,3]).data.item()
        loss_train_5=criterion(output[:,:,4], label_batch[:,:,4]).data.item()
        loss_train_6=criterion(output[:,:,5], label_batch[:,:,5]).data.item()
        loss_train=loss1+loss2
        loss_train.backward()
        optimizer.step()

        #if not args.fastmode:
        #    # Evaluate validation set performance separately,
        #    # deactivates dropout during validation run.
        #    model.eval()
        #    output = model(features, adj)

        loss_val = loss_train.data.item()
        #acc_val = accuracy(output[idx_val], labels[idx_val])
    #print('Epoch: {:04d}'.format(epoch+1),
    #      'loss_train: {:.4f}'.format(loss_train.data.item()),
    #      'acc_train: {:.4f}'.format(acc_train.data.item()),
    #      'loss_val: {:.4f}'.format(loss_val.data.item()),
    #      'acc_val: {:.4f}'.format(acc_val.data.item()),
    #      'time: {:.4f}s'.format(time.time() - t))

    print('epoch %d, loss %f, time=%f'%(epoch, loss_val,time.time()-t))
    print('loss each channel: %f %f %f %f %f %f'%(loss_train_1, loss_train_2,
                                                  loss_train_3,loss_train_4,
                                                  loss_train_5,loss_train_6))
    writer.add_scalar('loss', loss_val)
    return loss_val


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data[0]),
          "accuracy= {:.4f}".format(acc_test.data[0]))

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

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

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test()