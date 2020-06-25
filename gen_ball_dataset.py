import json
import numpy as np
datas=[json.loads(line) for line in open('00000:000.log','r')]

task_num=100
timestep=600

def get_body_array(info):
    arr=np.zeros([6])
    arr[0]=info['angle']
    arr[1]=info['pos_x']
    arr[2]=info['pos_y']
    arr[3]=info['velocity_x']
    arr[4]=info['velocity_y']
    arr[5]=info['angular_velocity']
    return arr

def process_body_info(body_info):
    # {"idx": 0, "pos_x": -3.0, "pos_y": -9.344138145446777, "angle": 0.0, "velocity_x": 0.0,
    # "velocity_y": -0.37500059604644775, "angular_velocity": 0.0}
    body_arrays = [get_body_array(info) for info in body_info]
    return body_arrays

def process_contact_info(contact_info):
    adj=np.zeros([4,4])
    for contact_pair in contact_info:
        a,b=contact_pair['body_b'], contact_pair['body_a']
        if a==-1: a=3
        if b==-1: b=3
        adj[a][b]=1
        adj[b][a]=1
    return adj

def get_scene_stats(idx):
    #print(idx)
    data_step=datas[idx]
    return process_body_info(data_step['body_info']['bodies']), \
        process_contact_info(data_step['contact_info']['contacts'])

obj_data, adj_data, label_data=[], [], []

for i in range(task_num):
    idx=i*(1+timestep)+1
    body_array, adj_array=get_scene_stats(idx)
    for j in range(timestep-1):
        idx=i*(1+timestep)+2+j
        new_body_array, new_adj_array=get_scene_stats(idx)
        obj_data.append(body_array)
        adj_data.append(adj_array)
        label_data.append(new_body_array)
        body_array, adj_array = new_body_array, new_adj_array

obj_data=np.array(obj_data)
adj_data=np.array(adj_data)
label_data=np.array(label_data)
np.save('obj_data.npy', obj_data)
np.save('adj_data.npy', adj_data)
np.save('label_data.npy',label_data)
print(obj_data.shape)
print(adj_data.shape)
print(label_data.shape)
