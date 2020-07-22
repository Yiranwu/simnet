import json
import numpy as np
import pickle
import argparse

def get_data_from_string(body_info, contact_info, obj_attr_data, mean, std):
    #print('need to check @gen_dataset.get_data_from_string: split')

    data = {'body_info': body_info, 'contact_info': contact_info}
    body_array, conn_array = get_scene_stats(data, obj_attr_data)

    body_array[:,:7] = (body_array[:,:7] - mean) / std
    #body_array.reshape([1,-1,13])
    return body_array, conn_array

def get_data_from_file(filename):
    datas=[json.loads(line) for line in open(filename,'r')]
    timestep=600

    obj_data, conn_data, label_data = [], [], []
    cflag_data=[]
    #print(datas[0])
    #print(datas[1])
    #print(get_obj_attr(datas[0], datas[1]))
    obj_attr_data=get_obj_attrs(datas[0], datas[1])
    body_array, conn_array = get_scene_stats(datas[2], obj_attr_data)
    #body_array, conn_array = get_scene_stats(datas[2], obj_attr_data)
    #exit()
    #print(type(body_array))
    for j in range(timestep-1):
        #print('j')
        #print(datas[3+j]['timestamp'])
        new_body_array, new_conn_array = get_scene_stats(datas[3+j], obj_attr_data)
        obj_data.append(body_array)
        conn_data.append(conn_array)
        #print('body array: ', body_array)
        #print('new body array: ', new_body_array)
        #print(new_body_array-body_array)
        label_data.append(new_body_array - body_array)
        cflag_data.append(0 if conn_array.shape[0]==0 else 1)
        body_array, conn_array = new_body_array, new_conn_array

    obj_data = np.array(obj_data)
    #print(len(conn_data))
    #print(conn_data[0].shape)
    #conn_data = np.array(conn_data)
    label_data = np.array(label_data)
    cflag_data = np.array(cflag_data)
    return obj_data, conn_data, label_data, cflag_data

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
    for info in contact_info:
        a,b=info['body_b'], info['body_a']
        if a==-1: a=wall_idx
        if b==-1: b=wall_idx
        conn.append([a,b])
        conn.append([b,a])
    conn=np.array(conn, dtype=np.int64)
    if conn.shape[0]>0: conn = np.transpose(conn, (1,0))
    #else: conn = conn.reshape([2,-1])
    return conn
    #'''

def get_scene_stats(data, attrs):
    #print(idx)
    return process_body_info(data['body_info']['bodies'], attrs), \
        process_contact_info(data['contact_info']['contacts'], len(attrs))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()

    #parser.add_argument("--task_id", help="input a specific task id with format xxxxx:xxx", type=str)

    parser.add_argument("--start_template_id", help="start template id", type=int, default=1)
    parser.add_argument("--end_template_id", help="end template id", type=int, default=1)
    parser.add_argument("--num_mods", help="number of mods for each template", type=int, default=100)
    parser.add_argument("--data_path", help="folder of data file", type=str, default='data')
    parser.add_argument("--shuffle", action='store_true', default=True)
    parser.add_argument('--normalize', action='store_true', default=True)
    parser.add_argument('--corrupt', action='store_true', default=True)
    config=parser.parse_args()
    start_tid=config.start_template_id
    end_tid=config.end_template_id
    num_mods=config.num_mods
    data_path=config.data_path
    normalize=config.normalize
    shuffle=config.shuffle
    corrupt=config.corrupt
    dataset_spec='%05d%s'%(start_tid, '_noise' if corrupt else '')

    obj_datas, conn_datas, label_datas = [], [], []
    cflag_datas=[]
    for i in range(start_tid, end_tid+1):
        for j in range(num_mods):
            filename = data_path+'/'+('%05d'%i)+':'+('%03d'%j)+'.log'
            print(filename)
            obj_data, conn_data, label_data, cflag_data = get_data_from_file(filename)
            obj_datas.append(obj_data)
            #conn_datas.append(conn_data)
            conn_datas = conn_datas + conn_data
            label_datas.append(label_data)
            cflag_datas.append(cflag_data)

    obj_datas=np.concatenate(obj_datas, axis=0)
    scene_num, obj_num = obj_datas.shape[0], obj_datas.shape[1]
    conn_datas=np.array(conn_datas)
    label_datas=np.concatenate(label_datas, axis=0)
    cflag_datas=np.concatenate(cflag_datas, axis=0)

    if normalize:
        obj_datas = obj_datas.reshape([-1, 13])
        mean = np.mean(obj_datas[:,:7], axis=0)
        std = np.std(obj_datas[:,:7], axis=0)
        obj_datas[:,:7] = ((obj_datas[:,:7] - mean) / std)
        obj_datas = obj_datas.reshape([-1, obj_num, 13])

    if corrupt:
        py_noise=np.random.normal(scale=0.01, size=(scene_num, obj_num))
        obj_datas[:,:,2] += py_noise
        vy_noise=np.random.normal(scale=0.01, size=(scene_num, obj_num))
        obj_datas[:,:,4] += vy_noise




    label_datas = label_datas[:, :, :6]

    if normalize:
        label_datas = label_datas / std[:6]

    if shuffle:
        idx = np.arange(obj_datas.shape[0])
        np.random.shuffle(idx)
        obj_datas = obj_datas[idx]
        conn_datas = conn_datas[idx]
        label_datas = label_datas[idx]
        cflag_datas= cflag_datas[idx]

    np.save('data/%s_obj_data.npy'%dataset_spec, obj_datas)
    np.save('data/%s_conn_data.npy'%dataset_spec, conn_datas, allow_pickle=True)
    np.save('data/%s_label_data.npy'%dataset_spec, label_datas)
    np.save('data/%s_cflag_data.npy'%dataset_spec, cflag_datas)
    np.savez('data/%s_mean_std.npz'%dataset_spec, mean=mean, std=std)
    print(obj_datas.shape)
    print(conn_datas.shape)
    print(conn_datas[0].shape)
    print(label_datas.shape)
    print(cflag_datas.shape)
    print('data/%s_obj_data.npy  generated'%dataset_spec)
