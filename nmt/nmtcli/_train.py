

def nmt_train(args):
    """
    :param args:
    :return:
    """
    import os
    import yaml
    from nmt.trainer import Trainer

    hyp_args = yaml.load(open(args.config_path))
    ## Build model
    model = Trainer(hyp_args)
    ## Train model
    model.train()