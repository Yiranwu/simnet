import sys
sys.path.insert(0,'/home/yiran/pc_mapping/GenBox2D/src/main/python')
from rollout_simulation import simulate
from replay_compare import compare
from gtrain_general import gtrain
from gen_dataset import generate_dataset
from box2d_simulate import box2d_simulate
import os
import argparse

def get_config_from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_template_id", help="start template id", type=int, default=0)
    parser.add_argument("--end_template_id", help="end template id", type=int, default=1)
    parser.add_argument("--num_mods", help="number of mods for each template", type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='gine')
    parser.add_argument('--dataset_name', type=str, default='medium')
    parser.add_argument('--no_train', action='store_true', default=False)
    parser.add_argument('--no_datagen', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--model_name', type=str, default='gine')
    config = parser.parse_args()
    return config

def get_raw_dataset_name(sid, eid, nmods):
    return '%d-%dx%d'%(sid, eid, nmods)

def main():
    config=get_config_from_args()
    sid=config.start_template_id
    eid=config.end_template_id
    nmod=config.num_mods
    device=config.device

    raw_dataset_name=get_raw_dataset_name(sid, eid, nmod)
    task_ids=box2d_simulate(sid, eid, nmod, raw_dataset_name)
    # box2d_simulate generates *:*.log && *actions.npy in box2d path

    '''
    dataset_name = generate_dataset(sid, eid, nmod, raw_dataset_name, task_ids, name_only=config.no_datagen)
    # generate_dataset generates *_data.npy in simnet path
    config.dataset_name=dataset_name
    exp_name = gtrain(config.model_name, dataset_name, epochs=40, device=device, name_only=config.no_train)
    config.exp_name=exp_name
    print('dataset name: ',dataset_name)
    print('exp name: ', exp_name)
    '''
    dataset_name='0-11x100'
    config.exp_name='gine_ev_ep40_lr0.001000_h128_0-11x100-ep25'
    simulate(sid, eid, nmod, raw_dataset_name, exp_name=config.exp_name)
    compare(raw_dataset_name, config.exp_name)

if __name__ == '__main__':
    main()
