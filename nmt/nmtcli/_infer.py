

def nmt_infer(args):
    """
    :param args:
    :return:
    """
    import os
    import yaml
    from nmt.nmtservice.service_transformer import ServiceTransformer

    hyp_args = yaml.load(open(args.config_path))
    hyp_args["config_path"] = args.config_path
    ## Build model
    model = ServiceTransformer(hyp_args)
    ## Infer model
    model.cmd_infer()

