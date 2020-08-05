import sys
sys.path.insert(0,'/home/yiran/pc_mapping/GenBox2D/src/main/python')
from rollout_simulation import simulate
from replay_compare import compare
from gtrain_general import gtrain
#from gen_dataset import generate_dataset
from box2d_simulate import box2d_simulate
from data_utils import get_raw_dataset_name, get_dataset_name, get_exp_name
import os
import argparse

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_template_id", help="start template id", type=int, default=0)
    parser.add_argument("--end_template_id", help="end template id", type=int, default=11)
    parser.add_argument("--num_mods", help="number of mods for each template", type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='ginewobn')
    parser.add_argument('--dataset_name', type=str, default='large0to11')
    parser.add_argument('--no_train', action='store_true', default=False)
    parser.add_argument('--clear_box2d_data', action='store_true', default=False)
    parser.add_argument('--clear_graph_data', action='store_true', default=False)
    parser.add_argument('--clear_nn', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default='gine')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--padding', action='store_true',default=False)
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--simnet_root_dir', type=str, default='/home/yiran/pc_mapping/simnet')

    # box2d_simulate args

    parser.add_argument("--task_id", help="input a specific task id with format xxxxx:xxx", type=str)
    #parser.add_argument("--start_template_id", help="start template id", type=int, default=0)
    #parser.add_argument("--end_template_id", help="end template id", type=int, default=0)
    #parser.add_argument("--num_mods", help="number of mods for each template", type=int, default=1)
    parser.add_argument("--action_tier", help="action tier <ball/two_balls>", default="ball")
    parser.add_argument("--config_path", help="name of the config file under the config directory", type=str, default="config.json")

    parser.add_argument("--no_actions", help="run tasks without actions (default is different actions)", action="store_true", default=False)
    parser.add_argument("--same_actions", help="run tasks with a unique action for all tasks (default is multdifferentiple actions)", action="store_true", default=False)
    parser.add_argument("--seed", help="random seed to selection actions", type=int, default=1)

    parser.add_argument("-i", help="enable the interactive/gui mode", action="store_true", default=False)

    parser.add_argument("--frequency", help="frequency for box2d steps", type=int, default=60)
    parser.add_argument("--total_steps", help="total sampling steps", type=int, default=600)
    parser.add_argument("--always_active", help="set objects in the scene to be active at all times", action="store_true", default=False)
    parser.add_argument("--solved_threshold", help="the minimum frames (steps) for two goal objects contacted", type=int, default=60)
    parser.add_argument('--raw_dataset_name', type=str, default='1-1x100')
    parser.add_argument('--box2d_root_dir', type=str, default='/home/yiran/pc_mapping/GenBox2D/src/main/python')

    # datagen args

    #parser.add_argument("--start_template_id", help="start template id", type=int, default=1)
    #parser.add_argument("--end_template_id", help="end template id", type=int, default=1)
    #parser.add_argument("--num_mods", help="number of mods for each template", type=int, default=100)
    #parser.add_argument("--data_path", help="folder of log file", type=str, default='box2d_data')
    parser.add_argument("--shuffle", action='store_true', default=False)
    parser.add_argument('--corrupt', action='store_true', default=False)
    #parser.add_argument('--normalize', action='store_true', default=False)
    #parser.add_argument('--padding', action='store_true',default=False)

    # training args

    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    #parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    #parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
    parser.add_argument('--batch_size', type=int, default=128)
    #parser.add_argument('--exp_name', type=str, default='gine-nobn-nocor-5')
    #parser.add_argument('--dataset_name', type=str, default='00001_')
    parser.add_argument('--eval', action='store_true', default=True)
    #parser.add_argument('--device', type=str, default='cuda:0')
    #parser.add_argument('--model_name', type=str, default='gine')

    # rollout args

    #parser.add_argument("--start_template_id", help="start template id", type=int, default=1)
    #parser.add_argument("--end_template_id", help="end template id", type=int, default=1)
    #parser.add_argument("--num_mods", help="number of mods for each template", type=int, default=100)
    #parser.add_argument("--exp_name", type=str, default='gine')

    # compare args

    parser.add_argument("--box2d_data_path", help="the path of the log file or directory", type=str,
                        default='box2d_data')
    parser.add_argument("--nn_data_path", type=str,
                        default='nn_rollout')
    parser.add_argument("--log_path", type=str, default='gine5-task1-1x100')
    #parser.add_argument('--exp_name',type=str, default='gine_bn_')
    parser.add_argument("--dir", help="the path is a directory or not", action="store_true", default=True)
    parser.add_argument("--image", help="generating images", action="store_true", default=False)
    parser.add_argument("--gif", help="generating gifs", action="store_true", default=True)

    config = parser.parse_args()
    config.raw_dataset_name=get_raw_dataset_name(config)
    config.dataset_name = get_dataset_name(config)
    config.exp_name=get_exp_name(config)
    config.log_dir=config.box2d_root_dir + '/' + config.box2d_data_path + '/' +config.raw_dataset_name
    config.data_path=config.simnet_root_dir+'/dataset/'+config.dataset_name
    config.model_path=config.simnet_root_dir+'/saved_models/'+config.exp_name
    return config


def run_box2d_simulate(config, **kwargs):
    if config is None:
        config=get_config()
    for name, val in kwargs.items():
        setattr(config, name, val)
    task_ids=box2d_simulate(config)
    config.task_ids=task_ids
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


def main():

    config=get_config()
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
    run_eval(config)
    #exp_name = gtrain(config.model_name, dataset_name, epochs=40, device=device, clear=clear_nn)
    #config.exp_name=exp_name
    print('dataset name: ',config.dataset_name)
    print('exp name: ', config.exp_name)
    #'''
    #simulate(sid, eid, nmod, raw_dataset_name, exp_name=config.exp_name)
    #compare(raw_dataset_name, config.exp_name)

if __name__ == '__main__':
    main()
