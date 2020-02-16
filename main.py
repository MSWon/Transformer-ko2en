# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:57:59 2019

@author: jbk48
"""

import argparse
import yaml
from trainer import Trainer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()

    hyp_args = yaml.load(open(args.config_path))
    print('========================')
    for key,value in hyp_args.items():
        print('{} : {}'.format(key, value))
    print('========================')
    ## Build model
    model = Trainer(hyp_args)
    ## Train model
    model.train(hyp_args["training_steps"])

