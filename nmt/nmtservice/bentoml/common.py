import sentencepiece as spm
import tensorflow as tf
from typing import List


class Tokenizer(object):
    def __init__(self, model_path: str):
        self._model_file = model_path
        self._processor = spm.SentencePieceProcessor()
        self._processor.Load(model_path)

    def tokenize(self, t: str) -> List[str]:
        return self.tok.EncodeAsPieces(t)

    def detokenize(self, t: List[str]) -> str:
        return self.tok.DecodePieces(t)

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
