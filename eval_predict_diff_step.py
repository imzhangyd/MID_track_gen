from mid import MID
import argparse
import os
import yaml
# from pprint import pprint
from easydict import EasyDict
import numpy as np
import pdb
import shutil
import os.path as osp

def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of MID')
    parser.add_argument('--config', default='')
    parser.add_argument('--dataset', default='')
    return parser.parse_args()


def main():
    # parse arguments and load config
    args = parse_args()
    with open(args.config) as f:
       config = yaml.safe_load(f)

    for k, v in vars(args).items():
       config[k] = v
    if 'exp_name' in config.keys() and config["exp_name"] is not None:
        pass
    else:
        config["exp_name"] = args.config.split("/")[-1].split(".")[0]
    
    config["eval_mode"] = True
    config = EasyDict(config)
    agent = MID(config)

    model_dir = osp.join("./experiments", config.exp_name)
    shutil.copy(args.config, os.path.join(model_dir, os.path.basename(args.config)))
    shutil.copy('./utils/trajectron_hypers.py', os.path.join(model_dir, 'trajectron_hypers.py'))

    
    sampling = "ddpm"
    for stride in [1,2,4,5,10,20,50,100]: # steps对应[100,50,25,20,10,5,2,1]
        agent.eval(sampling, stride)


if __name__ == '__main__':
    main()
