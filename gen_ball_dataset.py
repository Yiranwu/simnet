import json
import numpy as np
import pickle
datas=[json.loads(line) for line in open('00000:000.log','r')]

task_num=100
timestep=600

def get_body_array(info, attr):
    arr=np.zeros([7])
    arr[0]=info['angle']
    arr[1]=info['pos_x']
    arr[2]=info['pos_y']
    arr[3]=info['velocity_x']
    arr[4]=info['velocity_y']
    arr[5]=info['angular_velocity']
    arr[6]=attr
    return arr

def process_body_info(body_info, attrs):
    # {"idx": 0, "pos_x": -3.0, "pos_y": -9.344138145446777, "angle": 0.0, "velocity_x": 0.0,
    # "velocity_y": -0.37500059604644775, "angular_velocity": 0.0}
    #print(len(body_info), len(attrs))
    body_arrays = [get_body_array(info, attrs[id]) for id, info in enumerate(body_info)]
    return np.array(body_arrays)

def process_contact_info(contact_info):
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
        if a==-1: a=3
        if b==-1: b=3
        conn.append([a,b])
        conn.append([b,a])
    conn=np.array(conn, dtype=np.int64)
    if conn.shape[0]>0: conn = np.transpose(conn, (1,0))
    return conn
    #'''

def get_scene_stats(idx, attrs):
    #print(idx)
    data_step=datas[idx]
    return process_body_info(data_step['body_info']['bodies'], attrs), \
        process_contact_info(data_step['contact_info']['contacts'])

def get_attr(idx):
    data = datas[idx]['featurized_objects']
    #print(data)
    #print(len(data))
    attrs=[]
    for obj in data:
        attrs.append(obj["diameter"])
    #print(len(attrs))
    #exit()
    return attrs

obj_data, conn_data, label_data=[], [], []

for i in range(task_num):
    idx=i*(1+timestep)+1
    obj_attr_data = get_attr(idx-1)
    body_array, conn_array=get_scene_stats(idx, obj_attr_data)
    for j in range(timestep-1):
        idx=i*(1+timestep)+2+j
        new_body_array, new_conn_array=get_scene_stats(idx, obj_attr_data)
        obj_data.append(body_array)
        conn_data.append(conn_array)
        #label_data.append(new_body_array)
        label_data.append(new_body_array-body_array)
        body_array, conn_array = new_body_array, new_conn_array

#adj_data=np.array(conn_data)
#np.save('adj_data.npy', adj_data)
#exit()


obj_data=np.array(obj_data)
conn_data=np.array(conn_data)
label_data=np.array(label_data)
idx=np.arange(obj_data.shape[0])

obj_data=obj_data.reshape([-1,7])
mean=np.mean(obj_data, axis=0)
std=np.std(obj_data, axis=0)
obj_data = ((obj_data - mean) / std).reshape([-1,3,7])
label_data = label_data / std
label_data=label_data[:,:,:6]

np.random.shuffle(idx)
obj_data=obj_data[idx]
conn_data=conn_data[idx]
label_data=label_data[idx]
np.save('obj_data.npy', obj_data)
np.save('conn_data.npy', conn_data, allow_pickle=True)
np.save('label_data.npy',label_data)
print(obj_data.shape)
print(conn_data.shape)
print(conn_data[0].shape)
print(conn_data[0].dtype)
print(label_data.shape)
