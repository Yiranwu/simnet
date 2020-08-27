import json
import numpy as np
import pickle
import argparse
import torch

from torch_geometric.data import Data, InMemoryDataset

def get_data_from_string(body_info, contact_info, obj_attr_data, mean, std):
    data = {'body_info': body_info, 'contact_info': contact_info}
    body_array, conn_array, manifold_array = get_scene_stats(data, obj_attr_data)
    #body_array[:,1]=0
    #body_array[:,2]=0
    if conn_array.shape[0] == 0:
        conn = conn_array.reshape([2, 0])
    if manifold_array.shape[0] == 0:
        manifold = manifold_array.reshape([0, 4])
    #print('@get_data_from_string:')
    #print(manifold_array)
    #print(manifold_array.shape)
    #exit()
    body_array[:,:7] = (body_array[:,:7] - mean) / std
    if manifold_array.shape[0]>0:
        manifold_array[:,:2]=(manifold_array[:,:2]-mean[1:3])/std[1:3]
        manifold_array[:,2:]=(manifold_array[:,2:]-mean[1:3])/std[1:3]
    return body_array, conn_array, manifold_array


'''
def get_data_from_file(filename):
    datas=[json.loads(line) for line in open(filename,'r')]
    timestep=600

    obj_data, conn_data, manifold_data, label_data = [], [], [], []
    cflag_data=[]
    obj_attr_data=get_obj_attrs(datas[0], datas[1])
    body_array, conn_array, manifold_array = get_scene_stats(datas[2], obj_attr_data)
    for j in range(timestep-1):
        new_body_array, new_conn_array, new_manifold_array = get_scene_stats(datas[3+j],
                                                                             obj_attr_data)
        obj_data.append(body_array)
        conn_data.append(conn_array)
        manifold_data.append(manifold_array)
        label_data.append(new_body_array - body_array)
        cflag_data.append(0 if conn_array.shape[0]==0 else 1)
        body_array, conn_array, manifold_array = new_body_array, new_conn_array, new_manifold_array

    obj_data = np.array(obj_data)
    label_data = np.array(label_data)
    cflag_data = np.array(cflag_data)
    return obj_data, conn_data, manifold_data, label_data, cflag_data
'''

def get_data_from_file(filename):
    datas=[json.loads(line) for line in open(filename,'r')]
    timestep=100

    #obj_data, conn_data, manifold_data, label_data = [], [], [], []
    data_list = []
    cflag_data=[]
    obj_attr_data=get_obj_attrs(datas[0], datas[1])
    body_array, conn_array, manifold_array = get_scene_stats(datas[2], obj_attr_data)
    for j in range(timestep-1):
        new_body_array, new_conn_array, new_manifold_array = get_scene_stats(datas[3+j],
                                                                             obj_attr_data)
        obj = torch.from_numpy(body_array).double()
        #obj[:,1]=0
        #obj[:,2]=0
        conn = conn_array
        if conn.shape[0] == 0:
            conn = conn.reshape([2, 0])
        conn = torch.from_numpy(conn)
        label = torch.from_numpy(new_body_array - body_array).double()
        manifold = manifold_array
        if manifold.shape[0] == 0:
            manifold = manifold.reshape([0, 4])
        manifold = torch.from_numpy(manifold)
        gdata = Data(x=obj, edge_index=conn,
                     y=label[:,:6], edge_attr=manifold)
        data_list.append(gdata)

        cflag_data.append(0 if conn_array.shape[0]==0 else 1)
        body_array, conn_array, manifold_array = new_body_array, new_conn_array, new_manifold_array
    #dataset=datasets.GStabDataset()
    #data, slices = dataset.collate(data_list)
    #print('first collate: ')
    #print('data.x: ', data.x.shape)
    #print('data.y: ', data.y.shape)
    return data_list

def get_padded(obj_data,label_data, pad_num):
    obj_pad=np.zeros([599, pad_num-obj_data.shape[1], 13])
    obj_pad[11]=1

    obj_padded=np.concatenate([obj_data, obj_pad], axis=1)
    label_padded=np.concatenate([label_data, obj_pad], axis=1)
    return obj_padded, label_padded

def get_obj_num_from_file(filename):
    datas=[json.loads(line) for line in open(filename,'r')]
    timestep=600

    obj_data, conn_data, manifold_data, label_data = [], [], [], []
    cflag_data=[]
    obj_attr_data=get_obj_attrs(datas[0], datas[1])
    body_array, conn_array, manifold_array = get_scene_stats(datas[2], obj_attr_data)
    return body_array.shape[0]

def get_obj_attrs(feat_obj, action_obj):
    attrs = []
    for obj in feat_obj['featurized_objects']:
        attrs.append(obj)
    for obj in action_obj['featurized_objects']:
        attrs.append(obj)
    return attrs

def get_one_hot_attr(shape):
    if shape == 'BALL':
        return [1,0,0,0,0]
    elif shape == 'BAR':
        return [0,1,0,0,0]
    elif shape == 'JAR':
        return [0,0,1,0,0]
    elif shape == 'STANDINGSTICKS':
        return [0,0,0,1,0]
    return [0,0,0,0,1]

def get_body_array(info, attr):
    arr=np.zeros([13])
    arr[0]=info['angle']
    arr[1]=info['pos_x']
    arr[2]=info['pos_y']
    arr[3]=info['velocity_x']
    arr[4]=info['velocity_y']
    arr[5]=info['angular_velocity']


    arr[6]=attr['diameter']
    arr[7], arr[8], arr[9], arr[10], arr[11] = get_one_hot_attr(attr['shape'])
    arr[12] = attr['color']!='BLACK'
    return arr

def get_wall_array():
    arr=np.zeros(13)
    arr[7], arr[8], arr[9], arr[10], arr[11] = get_one_hot_attr('WALL')
    arr[12] = 0
    return arr

def process_body_info(body_info, attrs):
    # {"idx": 0, "pos_x": -3.0, "pos_y": -9.344138145446777, "angle": 0.0, "velocity_x": 0.0,
    # "velocity_y": -0.37500059604644775, "angular_velocity": 0.0}
    body_arrays = [get_body_array(info, attrs[id]) for id, info in enumerate(body_info)] + \
                  [get_wall_array()]
    return np.array(body_arrays)

def process_contact_info(contact_info, body_array, wall_idx):
    '''
    # contact info by adjacency matrix
    conn=np.zeros([4,4])
    for contact_pair in contact_info:
        a,b=contact_pair['body_b'], contact_pair['body_a']
        if a==-1: a=3
        if b==-1: b=3
        conn[a][b]=1
        conn[b][a]=1
    return conn
    '''
    # contact info by (x,y) pair
    #'''
    assert(len(body_array.shape)==2)
    conn = []
    manifold=[]
    for info in contact_info:
        b,a=info['body_b'], info['body_a']
        if a==-1: a=wall_idx
        if b==-1: b=wall_idx
        ax,ay=body_array[a][1], body_array[a][2]
        bx,by=body_array[b][1], body_array[b][2]
        nx,ny=info['manifold_normal']
        pts=info['points']
        npoint = info['point_count']
        # seems that len(pts)=2
        if(npoint==2):
            p1,p2=pts
            px1,py1=p1
            px2,py2=p2
            px,py=(px1+px2)/2, (py1+py2)/2
        elif(npoint==1):
            p=pts[0]
            px,py=p
        else:
            raise RuntimeError()
        conn.append([a,b])
        conn.append([b,a])
        # normal points from objA to objB
        #manifold.append([nx,ny, px, py])
        #manifold.append([nx,ny, px, py])
        #manifold.append([0.0,0.0,0.0,0.0])
        #manifold.append([0.0,0.0,0.0,0.0])
        manifold.append([nx,ny, px-ax, py-ay])
        manifold.append([-nx,-ny, px-bx, py-by])
    conn=np.array(conn, dtype=np.int64)
    manifold=np.array(manifold)
    if conn.shape[0]>0: conn = np.transpose(conn, (1,0))
    return conn, manifold
    #'''

def get_scene_stats(data, attrs):
    body_array=process_body_info(data['body_info']['bodies'], attrs)
    conn_array, manifold_array=process_contact_info(data['contact_info']['contacts'],
                                                    body_array,  len(attrs))
    return body_array, conn_array, manifold_array

