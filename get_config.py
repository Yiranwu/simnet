import argparse
from get_name import get_raw_dataset_name, get_dataset_name, get_exp_name
import os
import sys
sys.path.insert(0,'/home/yiran/pc_mapping/GenBox2D/src/main/python')
from utils.task_reader import get_task_ids
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_template_id", help="start template id", type=int, default=0)
    parser.add_argument("--end_template_id", help="end template id", type=int, default=4)
    parser.add_argument("--num_mods", help="number of mods for each template", type=int, default=3)
    #parser.add_argument('--exp_name', type=str, default='ginesuperwide')
    #parser.add_argument('--dataset_name', type=str, default='tlarge')
    parser.add_argument('--no_train', action='store_true', default=False)
    parser.add_argument('--clear_box2d_data', action='store_true', default=False)
    parser.add_argument('--clear_graph_data', action='store_true', default=False)
    parser.add_argument('--clear_nn', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    #parser.add_argument('--devices', type=int, nargs='+', default=[4,5,6,7])
    parser.add_argument('--devices', type=int, nargs='+', default=[0])
    parser.add_argument('--model_name', type=str, default='ginewider')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--padding', action='store_true',default=False)
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--simnet_root_dir', type=str, default='/home/yiran/pc_mapping/simnet')
    parser.add_argument('--corrupt', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--intermediate_save', action='store_true', default=True)

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
    #parser.add_argument('--corrupt', action='store_true', default=True)
    #parser.add_argument('--normalize', action='store_true', default=False)
    #parser.add_argument('--padding', action='store_true',default=False)

    # training args

    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    #parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    #parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
    #parser.add_argument('--exp_name', type=str, default='gine-nobn-nocor-5')
    #parser.add_argument('--dataset_name', type=str, default='00001_')
    parser.add_argument('--eval', action='store_true', default=False)
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
    config.task_ids = get_task_ids(config.start_template_id, config.end_template_id, config.num_mods)

    device_str = ','.join([str(id) for id in config.devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = device_str
    return config