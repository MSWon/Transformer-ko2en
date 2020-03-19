

def nmt_decode(args):
    """
    :param args:
    :return:
    """
    import yaml
    import argparse
    from ..translate import Translate
    hyp_args = yaml.load(open(args.config_path))
    print('========================')
    for key,value in hyp_args.items():
        print('{} : {}'.format(key, value))
    print('========================')
    ## Build model
    model = Translate(hyp_args)
    ## Infer model
    f_in = open(args.input_file, "r", encoding="utf-8")
    n = 1
    with open(args.save_path, "w", encoding="utf-8") as f_out:
        for sent in f_in:
            print("{} sentences decoded!".format(n))
            f_out.write(model.service_infer(sent)+"\n")
            n += 1
