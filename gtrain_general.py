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
from torch.utils.tensorboard import SummaryWriter

#from utils import load_data, accuracy
from models import *
from datasets import GDataset, GTestDataset

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    #parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
    #parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    #parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--exp_name', type=str, default='gine-nobn-nocor-5')
    #parser.add_argument('--dataset_spec', type=str, default='00001_noise_')
    parser.add_argument('--dataset_name', type=str, default='00001_')
    parser.add_argument('--eval', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default='gine')

    args = parser.parse_args()
    return args


def train(args, epoch, model, dataloader, optimizer, vfeat, criterion, writer):
    t = time.time()
    model.train()
    loss_val=0
    loss_val_1, loss_val_2, loss_val_3, loss_val_4, loss_val_5, loss_val_6=0,0,0,0,0,0
    closs_val, allloss_val = 0,0
    ccnt_val, cnt_val=0,0
    for data in dataloader:
        if(args.cuda):
            data=data.to(args.device)
        optimizer.zero_grad()
        output=model(data)
        movable_map=data.x[:,vfeat].bool()
        masked_output = output[movable_map]
        masked_label = data.y[movable_map]
        # output shape: batch x 3 x 6
        loss = criterion(masked_output, masked_label)
        loss_train_1=criterion(masked_output[:,0], masked_label[:,0]).data.item()
        loss_train_2=criterion(masked_output[:,1], masked_label[:,1]).data.item()
        loss_train_3=criterion(masked_output[:,2], masked_label[:,2]).data.item()
        loss_train_4=criterion(masked_output[:,3], masked_label[:,3]).data.item()
        loss_train_5=criterion(masked_output[:,4], masked_label[:,4]).data.item()
        loss_train_6=criterion(masked_output[:,5], masked_label[:,5]).data.item()

        loss.backward()
        optimizer.step()

        loss_val += loss.data.item()
        loss_val_1+=loss_train_1
        loss_val_2+=loss_train_2
        loss_val_3+=loss_train_3
        loss_val_4+=loss_train_4
        loss_val_5+=loss_train_5
        loss_val_6+=loss_train_6

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


def eval(args, epoch, model, testloader, vfeat, criterion, writer):
    t = time.time()
    model.eval()
    loss_val=0
    loss_val_1, loss_val_2, loss_val_3, loss_val_4, loss_val_5, loss_val_6=0,0,0,0,0,0

    for data in testloader:
        if(args.cuda):
            data=data.to(args.device)
        output=model(data)
        movable_map=data.x[:,vfeat].bool()
        masked_output = output[movable_map]
        masked_label = data.y[movable_map]
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

def get_exp_name(args):
    return '%s_%sep%d_lr%f_h%d_%s'%(args.model_name,'ev_' if args.eval else 'noev_',
                                      args.epochs, args.lr,
                                      args.hidden, args.dataset_name)

def get_model_by_name(name):
    if name=='gine':
        return GINE(vfeat=12, hidden=32, nclass=6)
    elif name=='ginewobn':
        return GINEWOBN(vfeat=12, hidden=32, nclass=6)
    else:
        print('unrecognized model name!')
        exit()

def gtrain(model_name, dataset_name, epochs, device, name_only=False):
    args=get_config()
    args.dataset_name=dataset_name
    args.epochs=epochs
    args.device=device
    args.model_name=model_name
    batch_size = args.batch_size
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    exp_name=get_exp_name(args)
    if name_only:
        return exp_name
    root_dir=os.path.dirname(__file__)
    model_path=root_dir+'/saved_models/%s.pth'%exp_name
    #if os.path.exists(model_path):
    #    return exp_name

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    writer = SummaryWriter(log_dir='runs/train_gball')

    vfeat = 12
    hidden = 32
    net = get_model_by_name(model_name)
    #net = GINE(vfeat=12, hidden=32, nclass=6)
    # net=TestLinear(cin=nfeat+nembed, cout=6)
    model = net.double()

    all_param = model.parameters()
    optimizer = optim.Adam(all_param,
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    if args.cuda:
        model.to(device)

    data_path = '/home/yiran/pc_mapping/simnet/dataset/%s' % dataset_name

    #os.system('rm %s/processed_data.pt' % data_path)
    #os.system('rm %s/processed_data_test.pt' % data_path)
    dataset = GDataset(data_path, nfeat=vfeat, train=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    if args.eval:
        testset = GDataset(data_path, nfeat=vfeat, train=False)
        testloader = DataLoader(testset, batch_size=batch_size, drop_last=True, shuffle=True)
    criterion = nn.MSELoss()
    criterion_sum = nn.MSELoss(reduction='sum')
    # print('train len: ', len(dataloader))
    # print('test len: ', len(testloader))
    # Train model
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    for epoch in range(args.epochs):
        # args, epoch, model, dataloader, optimizer, vfeat, criterion, writer
        loss_values.append(train(args, epoch, model, dataloader, optimizer, vfeat, criterion, writer))
        if args.eval:
            eval(args, epoch, model, testloader, vfeat, criterion, writer)
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
        if(epoch%5==0):
            torch.save(model.state_dict(), root_dir + '/saved_models/%s-ep%d.pth' % (exp_name,epoch))
    dir=os.path.dirname(__file__)
    torch.save(model.state_dict(), model_path)
    print('saved to'+ model_path)
    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)
    return exp_name
