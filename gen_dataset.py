import json
import numpy as np
import pickle
import argparse
import os
from data_utils import *

def get_config():
    parser=argparse.ArgumentParser()
    #parser.add_argument("--task_id", help="input a specific task id with format xxxxx:xxx", type=str)
    parser.add_argument("--start_template_id", help="start template id", type=int, default=1)
    parser.add_argument("--end_template_id", help="end template id", type=int, default=1)
    parser.add_argument("--num_mods", help="number of mods for each template", type=int, default=100)
    parser.add_argument("--data_path", help="folder of log file", type=str, default='box2d_data')
    parser.add_argument("--shuffle", action='store_true', default=False)
    parser.add_argument('--normalize', action='store_true', default=True)
    parser.add_argument('--corrupt', action='store_true', default=False)

    config=parser.parse_args()
    return config

def get_dataset_name(config):
    return '%d-%dx%d%s'%(config.start_template_id, config.end_template_id, config.num_mods,
                          '_corr' if config.corrupt else '')

#def generate_dataset(start_tid,end_tid,num_mods, raw_dataset_name, task_ids, clear=False):
def generate_dataset(config):
    dataset_name=get_dataset_name(config)

    data_dir='/home/yiran/pc_mapping/GenBox2D/src/main/python'
    data_path=data_dir + '/' + config.data_path
    dataset_dir='/home/yiran/pc_mapping/simnet'
    dataset_path=dataset_dir + '/dataset/%s'%dataset_name

    if config.clear_npy_data:
        os.system('rm -r %s'%dataset_path)
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    else:
        print('dataset npy detected, skipping')
        return dataset_name
    normalize=config.normalize
    shuffle=config.shuffle
    corrupt=config.corrupt
    #dataset_name='%05d%s'%(start_tid, '_noise' if corrupt else '')

    obj_datas, conn_datas, manifold_datas, label_datas = [], [], [], []
    cflag_datas=[]
    '''
    max_obj_size=0
    for i in range(25):
        filename = data_path+'/'+task_ids[i*100]+'.log'

        objn=get_obj_num_from_file(filename)
        print(i, objn)

        if(objn>max_obj_size):
            max_obj_size=objn
    '''
    for task_id in config.task_ids:
        filename = data_path+'/'+task_id+'.log'
        obj_data, conn_data, manifold_data, label_data, cflag_data = get_data_from_file(filename)
        obj_data, label_data = get_padded(obj_data, label_data,11)
        # obj_data shape: timestep x numobj x stats_channel 599 x * x 13
        # label_data: timestep x numobj x label_channel     599 x * x 13
        # cflag_data: timestep x numobj                     599
        obj_datas.append(obj_data)
        #conn_datas.append(conn_data)
        conn_datas = conn_datas + conn_data
        manifold_datas=manifold_datas+manifold_data
        label_datas.append(label_data)
        cflag_datas.append(cflag_data)
    '''
    for i in range(start_tid, end_tid+1):
        for j in range(num_mods):
            filename = data_path+'/'+('%05d'%i)+':'+('%03d'%j)+'.log'
            print(filename)
            obj_data, conn_data, manifold_data, label_data, cflag_data = get_data_from_file(filename)
            obj_datas.append(obj_data)
            #conn_datas.append(conn_data)
            conn_datas = conn_datas + conn_data
            manifold_datas=manifold_datas+manifold_data
            label_datas.append(label_data)
            cflag_datas.append(cflag_data)
    '''

    obj_datas=np.concatenate(obj_datas, axis=0)
    scene_num, obj_num = obj_datas.shape[0], obj_datas.shape[1]
    conn_datas=np.array(conn_datas)
    manifold_datas=np.array(manifold_datas)
    label_datas=np.concatenate(label_datas, axis=0)
    cflag_datas=np.concatenate(cflag_datas, axis=0)

    if normalize:
        obj_datas = obj_datas.reshape([-1, 13])
        mean = np.mean(obj_datas[:,:7], axis=0)
        std = np.std(obj_datas[:,:7], axis=0)
        obj_datas[:,:7] = ((obj_datas[:,:7] - mean) / std)
        obj_datas = obj_datas.reshape([-1, obj_num, 13])
        for i in range(manifold_datas.shape[0]):
            if manifold_datas[i].shape[0]>0:
                manifold_datas[i][:,:2]=(manifold_datas[i][:,:2]-mean[1:3])/std[1:3]
                manifold_datas[i][:,2:]=(manifold_datas[i][:,2:]-mean[1:3])/std[1:3]
    if corrupt:
        py_noise=np.random.normal(scale=0.01, size=(scene_num, obj_num))
        obj_datas[:,:,2] += py_noise
        vy_noise=np.random.normal(scale=0.01, size=(scene_num, obj_num))
        obj_datas[:,:,4] += vy_noise




    label_datas = label_datas[:, :, :6]

    if normalize:
        label_datas = label_datas.reshape([-1, 6])
        mean_label=np.mean(label_datas, axis=0)
        std_label=np.std(label_datas, axis=0)
        #mean_label=mean[:6]
        #std_label=std[:6]
        #print('-----mean-----')
        #print(mean_label)
        #print('-----std-----')
        #print(std_label)
        #print('-----max-----')
        #print(np.max(label_datas,axis=0))
        #print('-----min-----')
        #print(np.min(label_datas,axis=0))
        label_datas = (label_datas-mean_label) / std_label
        label_datas = label_datas.reshape([-1, obj_num, 6])

    if shuffle:
        idx = np.arange(obj_datas.shape[0])
        np.random.shuffle(idx)
        obj_datas = obj_datas[idx]
        conn_datas = conn_datas[idx]
        manifold_datas=manifold_datas[idx]
        label_datas = label_datas[idx]
        cflag_datas= cflag_datas[idx]

    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    #print('saving at %s/*data'%data_path)
    np.save('%s/obj_data.npy'%dataset_path, obj_datas)
    np.save('%s/conn_data.npy'%dataset_path, conn_datas, allow_pickle=True)
    np.save('%s/manifold_data.npy'%dataset_path, manifold_datas, allow_pickle=True)
    np.save('%s/label_data.npy'%dataset_path, label_datas)
    np.save('%s/cflag_data.npy'%dataset_path, cflag_datas)
    np.savez('%s/mean_std.npz'%dataset_path, mean=mean, std=std,
                                             mean_label=mean_label, std_label=std_label)
    print(obj_datas.shape)
    print(conn_datas.shape)
    print(conn_datas[0].shape)
    print(label_datas.shape)
    print(cflag_datas.shape)
    print('%s/obj_data.npy  generated'%dataset_path)
    return dataset_name


if __name__ == '__main__':
    generate_dataset(1,5,100, raw_dataset_name='1-5x100')