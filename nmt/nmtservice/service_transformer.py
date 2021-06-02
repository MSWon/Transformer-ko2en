import tensorflow as tf
import sentencepiece as spm
import re
import os

from tensorflow.python.saved_model import signature_constants
from nmt.nmtservice.export_transformer import ExportTransformer, MODEL_FOLDER_NAME
from nmt.nmttrain.utils.data_utils import get_vocab, idx2plainword, sent2idx, preprocess

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


class ServiceTransformer(ExportTransformer):
    """ ServiceTransformer class """
    def __init__(self, hyp_args):
        self.model_version = hyp_args["model_version"]
        tokenizer_path = os.path.join(self.model_version, "tokenizer")
        self.src_vocab_path = os.path.join(tokenizer_path, hyp_args["src_vocab_path"])
        self.tgt_vocab_path = os.path.join(tokenizer_path, hyp_args["tgt_vocab_path"])
        self.src_bpe_model_path = os.path.join(tokenizer_path, hyp_args["src_bpe_model_path"])
        self.tgt_bpe_model_path = os.path.join(tokenizer_path, hyp_args["tgt_bpe_model_path"])
        self.max_len = hyp_args["max_len"]

        self.src_sp = spm.SentencePieceProcessor()
        self.src_sp.Load(self.src_bpe_model_path)
        self.tgt_sp = spm.SentencePieceProcessor()
        self.tgt_sp.Load(self.tgt_bpe_model_path)

        self.src_vocab_dict, self.src_reversed_vocab_dict = get_vocab(self.src_vocab_path, False)
        self.tgt_vocab_dict, self.tgt_reversed_vocab_dict = get_vocab(self.tgt_vocab_path, False)

        # set Graph
        self.graph = tf.Graph()
        # initialize session
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            with self.sess.as_default() as sess:
                metagraph = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], f"{self.model_version}/{MODEL_FOLDER_NAME}")
                self._mapping = dict()
                self._mapping.update(dict(metagraph.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs))
                self._mapping.update(dict(metagraph.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs))

                self.inputs = self.sign2tensor("inputs")
                self.outputs = self.sign2tensor("outputs")
        
        # for warm up
        self.infer("warm up")

    def sign2tensor(self, sign_name):
        tensor_name = self._mapping[sign_name].name
        print(f"mapping : {sign_name} -> {tensor_name}")
        return self.graph.get_tensor_by_name(tensor_name)

    def infer(self, input_sent):
        """
        :param input_sent: string
        :return: translated sentence
        """
        input_sent, url_original = preprocess(input_sent)
        input_encoded_sent = self.src_sp.EncodeAsPieces(input_sent)
        input_idx = sent2idx(self.src_reversed_vocab_dict, input_encoded_sent, self.max_len)

        feed_dict_candidate = {
            self.inputs: input_idx
        } 
        output_idx = self.sess.run(self.outputs, feed_dict=feed_dict_candidate)
        decoded_word = idx2plainword(self.tgt_vocab_dict, output_idx, self.tgt_sp)

        if url_original:
            decoded_word = re.sub("<URL>", url_original, decoded_word)
        return decoded_word

    def cmd_infer(self):
        """
        :return: translated sentence
        """
        while True:
            input_sent = input("Input Korean sent : ")
            decoded_word = self.infer(input_sent)
            print(decoded_word)
