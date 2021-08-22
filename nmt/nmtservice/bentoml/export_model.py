import tensorflow as tf
import argparse
import yaml
import os
import sentencepiece as spm

from typing import List
from nmt.nmtservice.bentoml.model import ServiceTransformer


"""
Usage:
python nmt/nmtservice/bentoml/export_model.py \
    --config-path ${CONFIG_PATH}
"""


parser = argparse.ArgumentParser()
parser.add_argument("--config-path", "-c", required=True, help="config file path")
args = parser.parse_args()


class Tokenizer(object):
    def __init__(self, model_path: str):
        self._model_file = model_path
        self._processor = spm.SentencePieceProcessor()
        self._processor.Load(model_path)

    def tokenize(self, t: str) -> List[str]:
        return self._processor.EncodeAsPieces(t)

    def detokenize(self, t: List[str]) -> str:
        return self._processor.DecodePieces(t)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_processor"] = None
        state["_model_file"] = None
        return state, self._model_file

    def __setstate__(self, d):
        self.__dict__, self._model_file = d
        self._processor = spm.SentencePieceProcessor()
        self._processor.Load(self._model_file)

def get_vocab(vocab_path, isTF=True):
    if isTF:
        vocab_path_tensor = tf.constant(vocab_path)
        tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, vocab_path_tensor)
        vocab_dict = tf.contrib.lookup.index_table_from_file(
            vocabulary_file=vocab_path_tensor,
            num_oov_buckets=0,
            default_value=1)
        return vocab_dict
    else:
        vocab_dict = {}
        reversed_vocab_dict = {}
        with open(vocab_path, "r") as f:
            for vocab in f:
                vocab_dict[len(vocab_dict)] = vocab.strip()
                reversed_vocab_dict[vocab.strip()] = len(reversed_vocab_dict)
    return vocab_dict, reversed_vocab_dict


hyp_args = yaml.load(open(args.config_path))
hyp_args["config_path"] = args.config_path

config_abs_path = os.path.abspath(args.config_path)
model_abs_path = os.path.dirname(config_abs_path)

config_abs_path = os.path.abspath(hyp_args["config_path"])
model_abs_path = os.path.dirname(config_abs_path)
tokenizer_abs_path = os.path.join(model_abs_path, "tokenizer")
src_vocab_path = os.path.join(tokenizer_abs_path, hyp_args["src_vocab_path"])
tgt_vocab_path = os.path.join(tokenizer_abs_path, hyp_args["tgt_vocab_path"])
src_bpe_model_path = os.path.join(tokenizer_abs_path, hyp_args["src_bpe_model_path"])
tgt_bpe_model_path = os.path.join(tokenizer_abs_path, hyp_args["tgt_bpe_model_path"])

src_sp = Tokenizer(src_bpe_model_path)
tgt_sp = Tokenizer(tgt_bpe_model_path)

src_vocab_dict, src_reversed_vocab_dict = get_vocab(src_vocab_path, False)
tgt_vocab_dict, tgt_reversed_vocab_dict = get_vocab(tgt_vocab_path, False)

bento_svc = ServiceTransformer()

bento_svc.pack("model", f"{model_abs_path}/exported.model/")
bento_svc.pack("src_sp", src_sp)
bento_svc.pack("tgt_sp", tgt_sp)

bento_svc.pack("src_reversed_vocab_dict", src_reversed_vocab_dict)
bento_svc.pack("tgt_vocab_dict", tgt_vocab_dict)

saved_path = bento_svc.save()