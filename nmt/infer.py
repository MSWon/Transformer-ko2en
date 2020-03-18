import yaml
import argparse
from nmt.translate import Translate

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", help="path of config file", required=True)
    parser.add_argument("--input_file", "-i", help="path of input file to decode", required=True)
    parser.add_argument("--save_path", "-s", help="save path of the decoded result", required=True)
    args = parser.parse_args()

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
            print("{} sentences decoded!")
            f_out.write(model.service_infer(sent)+"\n")
            n += 1
