# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:57:59 2019

@author: jbk48
"""

import argparse
import yaml
from trainer import Trainer
from translate import Translate

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="path of config file", required=True)
    parser.add_argument("--mode", help="mode (train/infer)", required=True)
    args = parser.parse_args()

    hyp_args = yaml.load(open(args.config_path))
    print('========================')
    for key,value in hyp_args.items():
        print('{} : {}'.format(key, value))
    print('========================')
    if args.mode == "train":
        ## Build model
        model = Trainer(hyp_args)
        ## Train model
        model.train(hyp_args["training_steps"])
    elif args.mode == "infer":
        ## Build model
        model = Translate(hyp_args)
        ## Infer model
        model.infer()
    else:
        raise ValueError('mode should be (train/infer)')

