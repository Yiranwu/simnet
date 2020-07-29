import sys
sys.path.insert(0,'/home/yiran/pc_mapping/GenBox2D/src/main/python')
from rollout_simulation import simulate
from replay_compare import compare
from gtrain_general import gtrain
from gen_dataset import generate_dataset
import os
import argparse

def get_config_from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_template_id", help="start template id", type=int, default=1)
    parser.add_argument("--end_template_id", help="end template id", type=int, default=1)
    parser.add_argument("--num_mods", help="number of mods for each template", type=int, default=100)
    parser.add_argument('--spec', type=str, default='gine')
    config = parser.parse_args()
    return config


def main():
    config=get_config_from_args()
    sid=config.start_template_id
    eid=config.end_template_id
    nmod=config.num_mods
    spec=config.spec
    gtrain(spec,0)
    simulate(sid, eid, nmod, spec)
    compare(spec)

if __name__ == '__main__':
    main()
