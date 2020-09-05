import os
import sys
sys.path.insert(0,'/home/yiran/pc_mapping/GenBox2D/src/main/python')
from get_config import get_config
config=get_config()

from rollout_simulation import simulate
from pgs_simulation import simulate as pgs_simulate
from replay_compare import compare
from gtrain_general import gtrain
from box2d_simulate import box2d_simulate


def run_box2d_simulate(config, **kwargs):
    if config is None:
        config=get_config()
    for name, val in kwargs.items():
        setattr(config, name, val)
    box2d_simulate(config)
    #config.task_ids=task_ids
    # task_ids

def run_gtrain(config, **kwargs):
    if config is None:
        config=get_config()
    for name, val in kwargs.items():
        setattr(config, name, val)
    gtrain(config)

def run_rollout(config, **kwargs):
    if config is None:
        config=get_config()
    for name, val in kwargs.items():
        setattr(config, name, val)
    config.log_path=config.exp_name
    simulate(config)

def run_pgs_rollout(config, **kwargs):
    if config is None:
        config=get_config()
    for name, val in kwargs.items():
        setattr(config, name, val)
    config.log_path=config.exp_name
    pgs_simulate(config)

def run_compare(config, **kwargs):
    if config is None:
        config=get_config()
    for name, val in kwargs.items():
        setattr(config, name, val)
    compare(config)

def run_eval(config, **kwargs):
    if config is None:
        config=get_config()
    for name, val in kwargs.items():
        setattr(config, name, val)
    run_rollout(config)
    print('rollout done')
    run_compare(config)
    print('compare done')

def run_pgs_eval(config, **kwargs):
    if config is None:
        config=get_config()
    for name, val in kwargs.items():
        setattr(config, name, val)
    run_pgs_rollout(config)
    print('rollout done')
    run_compare(config)
    print('compare done')


def main():
    #import torch
    #print(torch.cuda.device_count())
    #exit()
    #config=get_config()

    #run_eval(config, exp_name='gine_ev_ep15_lr0.001000_h128_3-3x100')
    #exit()
    #sid=config.start_template_id
    #eid=config.end_template_id
    #nmod=config.num_mods
    #device=config.device
    #clear_npy_data=config.clear_npy_data
    #clear_nn=config.clear_nn

    #raw_dataset_name=get_raw_dataset_name(sid, eid, nmod)
    #task_ids=box2d_simulate(sid, eid, nmod, raw_dataset_name)
    # box2d_simulate generates *:*.log && *actions.npy in box2d path

    #'''
    #dataset_name = generate_dataset(sid, eid, nmod, raw_dataset_name, task_ids, clear=clear_npy_data)
    # generate_dataset generates *_data.npy in simnet path
    #config.dataset_name=dataset_name
    run_box2d_simulate(config)
    #exit()
    #run_gtrain(config)
    #config.exp_name='ginewide_noev_ep200_lr0.001000_h128_0-11x100-ep10.pth'
    #config.model_path = config.simnet_root_dir + '/saved_models/' + config.exp_name
    #run_eval(config)
    run_pgs_eval(config)
    #exp_name = gtrain(config.model_name, dataset_name, epochs=40, device=device, clear=clear_nn)
    #config.exp_name=exp_name
    print('dataset name: ',config.dataset_name)
    print('exp name: ', config.exp_name)
    #'''
    #simulate(sid, eid, nmod, raw_dataset_name, exp_name=config.exp_name)
    #compare(raw_dataset_name, config.exp_name)

if __name__ == '__main__':
    main()
