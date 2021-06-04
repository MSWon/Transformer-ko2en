

def nmt_decode(args):
    """
    :param args:
    :return:
    """
    import yaml
    import argparse
    import os
    from tqdm import tqdm
    from nmt.nmtservice.service_transformer import ServiceTransformer

    hyp_args = yaml.load(open(args.config_path))
    hyp_args["config_path"] = args.config_path
    ## Build model
    model = ServiceTransformer(hyp_args)
    ## Infer model
    f_in = open(args.input_file, "r", encoding="utf-8")
    lines = f_in.readlines()

    with open(f"{args.input_file}.out", "w", encoding="utf-8") as f_out:
        for sent in tqdm(lines, total=len(lines)):
            f_out.write(model.infer(sent)+"\n")
