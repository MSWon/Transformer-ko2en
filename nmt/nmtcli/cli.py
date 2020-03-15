import argparse
from nmt import __version__
from nmt.nmtcli._download import nmt_download
from nmt.nmtcli._service import nmt_service
from nmt.nmtcli._tokenize import nmt_tokenize
from nmt.nmtcli._train import nmt_train
from nmt.nmtcli._infer import nmt_infer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", "-v", action="version", version="%(prog)s {}".format(__version__))
    parser.set_defaults(func=lambda x: parser.print_usage())
    subparsers = parser.add_subparsers()

    # nmt download
    subparser_download = subparsers.add_parser("download", help="package for downloading trained model & data")
    subparser_download.add_argument("--mode", "-m", required=True, help="download mode (data/model)")
    subparser_download.set_defaults(func=nmt_download)

    # nmt service
    subparser_service = subparsers.add_parser("service", help="package for serving on flask server")
    subparser_service.add_argument("--port", "-p", required=True, help="port number")
    subparser_service.set_defaults(func=nmt_service)

    # nmt tokenize
    subparser_tokenize = subparsers.add_parser("tokenize", help="package for tokenizing sentence")
    subparser_tokenize.add_argument("--model_path", "-m", required=True, help="BPE model path")
    subparser_tokenize.add_argument("--sentence", "-s", required=True, help="input sentence to tokenize")
    subparser_tokenize.set_defaults(func=nmt_tokenize)

    # nmt train
    subparser_train = subparsers.add_parser("train", help="package for training Transformer")
    subparser_train.add_argument("--config_path", "-c", required=True, help="config file path")
    subparser_train.set_defaults(func=nmt_train)

    # nmt infer
    subparser_infer = subparsers.add_parser("infer", help="package for inference")
    subparser_infer.add_argument("--config_path", "-c", required=True, help="config file path")
    subparser_infer.set_defaults(func=nmt_infer)

    args = parser.parse_args()

    func = args.func

    func(args)
