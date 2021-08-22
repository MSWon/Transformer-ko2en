import argparse
import yaml

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", help="path of config file", required=True)
    parser.add_argument("--mode", "-m", help="mode (train/infer)", required=True)
    args = parser.parse_args()

    hyp_args = yaml.load(open(args.config_path))
    print('========================')
    for key,value in hyp_args.items():
        print('{} : {}'.format(key, value))
    print('========================')
    if args.mode == "train":
        from nmt.trainer import Trainer
        ## Build model
        model = Trainer(hyp_args)
        ## Train model
        model.train()
    elif args.mode == "infer":
        from nmt.nmtservice.service_transformer import ServiceTransformer
        ## Build model
        model = ServiceTransformer(hyp_args)
        ## Infer model
        model.cmd_infer()
    else:
        raise ValueError('mode should be (train/infer)')

