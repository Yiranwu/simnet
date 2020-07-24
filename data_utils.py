import json
import numpy as np
import pickle
import argparse

def get_data_from_string(body_info, contact_info, obj_attr_data, mean, std):
    #print('need to check @gen_dataset.get_data_from_string: split')

    data = {'body_info': body_info, 'contact_info': contact_info}
    body_array, conn_array, manifold_array = get_scene_stats(data, obj_attr_data)

    #print('@get_data_from_string: manifold normalization needed')
    body_array[:,:7] = (body_array[:,:7] - mean) / std
    if manifold_array.shape[0]>0:
        manifold_array[:,:2]=(manifold_array[:,:2]-mean[1:3])/std[1:3]
        manifold_array[:,2:]=(manifold_array[:,2:]-mean[1:3])/std[1:3]
    #body_array.reshape([1,-1,13])
    return body_array, conn_array, manifold_array

def get_data_from_file(filename):
    datas=[json.loads(line) for line in open(filename,'r')]
    timestep=600

    obj_data, conn_data, manifold_data, label_data = [], [], [], []
    cflag_data=[]
    #print(datas[0])
    #print(datas[1])
    #print(get_obj_attr(datas[0], datas[1]))
    obj_attr_data=get_obj_attrs(datas[0], datas[1])
    #print('@get_data_from_file: initial step missing')
    body_array, conn_array, manifold_array = get_scene_stats(datas[2], obj_attr_data)
    #body_array, conn_array = get_scene_stats(datas[2], obj_attr_data)
    #exit()
    #print(type(body_array))
    for j in range(timestep-1):
        #print('j')
        #print(datas[3+j]['timestamp'])
        new_body_array, new_conn_array, new_manifold_array = get_scene_stats(datas[3+j],
                                                                             obj_attr_data)
        obj_data.append(body_array)
        conn_data.append(conn_array)
        manifold_data.append(manifold_array)
        #print('body array: ', body_array)
        #print('new body array: ', new_body_array)
        #print(new_body_array-body_array)
        label_data.append(new_body_array - body_array)
        cflag_data.append(0 if conn_array.shape[0]==0 else 1)
        body_array, conn_array, manifold_array = new_body_array, new_conn_array, new_manifold_array

    obj_data = np.array(obj_data)
    #print(len(conn_data))
    #print(conn_data[0].shape)
    #conn_data = np.array(conn_data)
    label_data = np.array(label_data)
    cflag_data = np.array(cflag_data)
    return obj_data, conn_data, manifold_data, label_data, cflag_data

def get_obj_attrs(feat_obj, action_obj):
    attrs = []
    #print('feat_obj:')
    #print(feat_obj)
    #print('type feat_obj:')
    #print(type(feat_obj))
    #print('obj:')
    #print(type(feat_obj['featurized_objects']))
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

def get_init_body_array(attrs):
    body_arrays=[]
    for attr in attrs:
        arr=np.zeros([13])
        arr[0]=attr['initial_angle']
        arr[1]=attr['initial_x']
        arr[2]=attr['initial_y']
        arr[3]=0.0
        arr[4]=0.0
        arr[5]=0.0


        arr[6]=attr['diameter']
        arr[7], arr[8], arr[9], arr[10], arr[11] = get_one_hot_attr(attr['shape'])
        arr[12] = attr['color']!='BLACK'
        body_arrays.append(arr)
    # forget standard for conn array
    body_arrays += [get_wall_array()]
    return np.array(body_arrays), np.array([], dtype=np.int64)

def get_wall_array():
    arr=np.zeros(13)
    arr[7], arr[8], arr[9], arr[10], arr[11] = get_one_hot_attr('WALL')
    arr[12] = 0
    return arr

def process_body_info(body_info, attrs):
    # {"idx": 0, "pos_x": -3.0, "pos_y": -9.344138145446777, "angle": 0.0, "velocity_x": 0.0,
    # "velocity_y": -0.37500059604644775, "angular_velocity": 0.0}
    #print(len(body_info), len(attrs))
    body_arrays = [get_body_array(info, attrs[id]) for id, info in enumerate(body_info)] + \
                  [get_wall_array()]
    return np.array(body_arrays)

def process_contact_info(contact_info, wall_idx):
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
    conn = []
    manifold=[]
    for info in contact_info:
        a,b=info['body_b'], info['body_a']
        nx,ny=info['manifold_normal']
        pts=info['points']
        assert(len(pts)==2)
        #print('@data_utils.process_contact_info: need to tackle differently if #pt=1/2')
        if(len(pts)==2):
            p1,p2=pts
            px1,py1=p1
            px2,py2=p2
            px,py=(px1+px2)/2, (py1+py2)/2
        else:
            p=pts
            px,py=p
        if a==-1: a=wall_idx
        if b==-1: b=wall_idx
        conn.append([a,b])
        conn.append([b,a])
        manifold.append([nx,ny, px, py])
        manifold.append([nx,ny, px, py])
    conn=np.array(conn, dtype=np.int64)
    manifold=np.array(manifold)
    if conn.shape[0]>0: conn = np.transpose(conn, (1,0))
    #else: conn = conn.reshape([2,-1])
    return conn, manifold
    #'''

def get_scene_stats(data, attrs):
    #print(idx)
    body_array=process_body_info(data['body_info']['bodies'], attrs)
    conn_array, manifold_array=process_contact_info(data['contact_info']['contacts'], len(attrs))
    return body_array, conn_array, manifold_array

