import os
import glob
import time
import random
import argparse
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


#from utils import load_data, accuracy
from models import *
from data_utils import get_dataset_name
from datasets import GDataset

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


def train(config, epoch, model, dataloader, optimizer, vfeat, criterion, writer):
    t = time.time()
    model.train()
    loss_val=0
    loss_val_1, loss_val_2, loss_val_3, loss_val_4, loss_val_5, loss_val_6=0,0,0,0,0,0
    closs_val, allloss_val = 0,0
    ccnt_val, cnt_val=0,0
    for data in dataloader:
        if(config.cuda):
            data=data.to(config.device)
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


def eval(config, epoch, model, testloader, vfeat, criterion, writer):
    t = time.time()
    model.eval()
    loss_val=0
    loss_val_1, loss_val_2, loss_val_3, loss_val_4, loss_val_5, loss_val_6=0,0,0,0,0,0

    for data in testloader:
        if(config.cuda):
            data=data.to(config.device)
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


def get_model_by_name(name):
    if name=='gine':
        return GINE(vfeat=12, hidden=32, nclass=6)
    elif name=='ginewobn':
        return GINEWOBN(vfeat=12, hidden=32, nclass=6)
    elif name=='ginewide':
        return GINEWide(vfeat=12, hidden=128, nclass=6)
    elif name=='gineshallow':
        return GINE(vfeat=12, hidden=32, nclass=6, layers=3)
    elif name=='ginedeep':
        return GINE(vfeat=12, hidden=32, nclass=6, layers=8)
    elif name=='ginewideshallow':
        return GINE(vfeat=12, hidden=128, nclass=6, layers=3)
    elif name=='ginewidedeep':
        return GINE(vfeat=12, hidden=128, nclass=6, layers=8)
    else:
        print('unrecognized model name!')
        exit()

#def gtrain(model_name, dataset_name, epochs, device, clear=False):
def gtrain(config):
    #args=get_config()
    batch_size = config.batch_size
    config.cuda = not config.no_cuda and torch.cuda.is_available()

    if config.clear_nn:
        os.system('rm -r %s'%config.model_path)
    if os.path.exists(config.model_path):
        return

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.cuda:
        torch.cuda.manual_seed(config.seed)

    writer = SummaryWriter(log_dir='runs/train_gball')

    vfeat = 12
    hidden = 32
    net = get_model_by_name(config.model_name)
    #net = GINE(vfeat=12, hidden=32, nclass=6)
    # net=TestLinear(cin=nfeat+nembed, cout=6)
    model = net.double()

    all_param = model.parameters()
    optimizer = optim.Adam(all_param,
                           lr=config.lr,
                           weight_decay=config.weight_decay)

    if config.cuda:
        model.to(config.device)

    if config.clear_graph_data:
        os.system('rm -r %s'%config.data_path)
    if not os.path.exists(config.data_path):
        os.mkdir(config.data_path)
    else:
        print('dataset detected, skipping')
    train_tasks=[]
    test_tasks=[]
    for i in range(config.end_template_id-config.start_template_id+1):
        for j in range(0,math.ceil(config.num_mods*0.8)):
            train_tasks.append(config.task_ids[i*config.num_mods+j])
        for j in range(math.ceil(config.num_mods*0.8), config.num_mods):
            test_tasks.append(config.task_ids[i*config.num_mods+j])
    print('train_tasks: ', train_tasks)
    print('test_tasks: ', test_tasks)

    dataset = GDataset(config, train_tasks, nfeat=vfeat, train=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if config.eval:
        #print('.eval=true!')
        testset = GDataset(config, test_tasks, nfeat=vfeat, train=False)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)


    criterion = nn.MSELoss()
    criterion_sum = nn.MSELoss(reduction='sum')
    # print('train len: ', len(dataloader))
    # print('test len: ', len(testloader))
    # Train model
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = config.epochs + 1
    best_epoch = 0
    for epoch in range(config.epochs):
        # args, epoch, model, dataloader, optimizer, vfeat, criterion, writer
        loss_values.append(train(config, epoch, model, dataloader, optimizer, vfeat, criterion, writer))
        if config.eval:
            eval(config, epoch, model, testloader, vfeat, criterion, writer)
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
        if(epoch%10==0):
            torch.save(model.state_dict(), config.simnet_root_dir + '/saved_models/%s-ep%d.pth' % (config.exp_name,epoch))
    torch.save(model.state_dict(), config.model_path)
    print('saved to'+ config.model_path)
    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)
