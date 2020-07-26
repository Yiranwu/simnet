import json
import numpy as np
import pickle
import argparse
from data_utils import *

if __name__ == '__main__':
    parser=argparse.ArgumentParser()

    #parser.add_argument("--task_id", help="input a specific task id with format xxxxx:xxx", type=str)

    parser.add_argument("--start_template_id", help="start template id", type=int, default=1)
    parser.add_argument("--end_template_id", help="end template id", type=int, default=1)
    parser.add_argument("--num_mods", help="number of mods for each template", type=int, default=100)
    parser.add_argument("--data_path", help="folder of data file", type=str, default='data')
    parser.add_argument("--shuffle", action='store_true', default=False)
    parser.add_argument('--normalize', action='store_true', default=True)
    parser.add_argument('--corrupt', action='store_true', default=False)
    config=parser.parse_args()
    start_tid=config.start_template_id
    end_tid=config.end_template_id
    num_mods=config.num_mods
    data_path=config.data_path
    normalize=config.normalize
    shuffle=config.shuffle
    corrupt=config.corrupt
    dataset_spec='%05d%s'%(start_tid, '_noise' if corrupt else '')

    obj_datas, conn_datas, manifold_datas, label_datas = [], [], [], []
    cflag_datas=[]
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

    np.save('data/%s_obj_data.npy'%dataset_spec, obj_datas)
    np.save('data/%s_conn_data.npy'%dataset_spec, conn_datas, allow_pickle=True)
    np.save('data/%s_manifold_data.npy'%dataset_spec, manifold_datas, allow_pickle=True)
    np.save('data/%s_label_data.npy'%dataset_spec, label_datas)
    np.save('data/%s_cflag_data.npy'%dataset_spec, cflag_datas)
    np.savez('data/%s_mean_std.npz'%dataset_spec, mean=mean, std=std,
                                                   mean_label=mean_label, std_label=std_label)
    print(obj_datas.shape)
    print(conn_datas.shape)
    print(conn_datas[0].shape)
    print(label_datas.shape)
    print(cflag_datas.shape)
    print('data/%s_obj_data.npy  generated'%dataset_spec)
