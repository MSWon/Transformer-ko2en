import tensorflow as tf
import sentencepiece as spm
import re
import os

from .nmttrain.model.transformer import Transformer
from .nmttrain.utils.data_utils import infer_dataset_fn, get_vocab, idx2plainword

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

class Translate(object):
    """ Translate class """
    def __init__(self, hyp_args):
        uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
        path = uppath(__file__, 1)

        self.train_src_corpus_path = os.path.join(path,hyp_args["train_src_corpus_path"])
        self.train_tgt_corpus_path = os.path.join(path,hyp_args["train_tgt_corpus_path"])
        self.test_src_corpus_path = os.path.join(path,hyp_args["test_src_corpus_path"])
        self.test_tgt_corpus_path = os.path.join(path,hyp_args["test_tgt_corpus_path"])
        self.src_vocab_path = os.path.join(path,hyp_args["src_vocab_path"])
        self.tgt_vocab_path = os.path.join(path,hyp_args["tgt_vocab_path"])
        self.src_bpe_model_path = os.path.join(path,hyp_args["src_bpe_model_path"])
        self.tgt_bpe_model_path = os.path.join(path,hyp_args["tgt_bpe_model_path"])
        self.max_len = hyp_args["max_len"]
        self.batch_size = hyp_args["batch_size"]
        self.model_path = hyp_args["model_path"]

        self.src_sp = spm.SentencePieceProcessor()
        self.src_sp.Load(self.src_bpe_model_path)
        self.tgt_sp = spm.SentencePieceProcessor()
        self.tgt_sp.Load(self.tgt_bpe_model_path)

        infer_dataset, self.input_placeholder = infer_dataset_fn(self.src_vocab_path, self.max_len, 1)

        iters = tf.data.Iterator.from_structure(infer_dataset.output_types,
                                                infer_dataset.output_shapes)
        src = iters.get_next()

        self.tgt_vocab_dict = get_vocab(self.tgt_vocab_path, False)

        # create the initialisation operations
        self.infer_init_op = iters.make_initializer(infer_dataset)

        print("Now building model")
        model = Transformer(hyp_args)
        self.decoded_idx = model.infer_fn(src["src_input_idx"])
        # build session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.tables_initializer())
        self.restore_model(self.sess)

    def restore_model(self, sess):
        """
        :param sess: tf.Session()
        :return: None
        """
        uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
        path = uppath(__file__, 1)
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(path, "model/" + self.model_path))
        print("Model loaded!")

    def prepro(self, sent):
        """
        :param sent: string
        :return: preprocessed sentence
        """
        sent = re.sub("\(.*?\)|\[.*?\]", "", sent)
        sent = re.sub("[^0-9a-zA-Z가-힣_\-@\.:&+!?'/,\s]", "", sent)
        url_regex = "(http[s]?://([a-zA-Z]|[가-힣]|[0-9]|[-_@\.&+!*/])+)|(www.([a-zA-Z]|[가-힣]|[0-9]|[-_@\.&+!*/])+)"
        is_url = re.search(url_regex, sent)
        if is_url:
            url_original = is_url.group()
        else:
            url_original = None
        sent = re.sub(url_regex, "<URL>", sent)
        return sent, url_original

    def infer(self):
        """
        :return: translated sentence
        """
        while True:
            input_sent = input("Input Korean sent : ")
            input_sent, url_original = self.prepro(input_sent)
            input_sent = [self.src_sp.EncodeAsPieces(input_sent)]

            self.sess.run(self.infer_init_op,
                          feed_dict={self.input_placeholder:input_sent})
            idx = self.sess.run(self.decoded_idx)
            decoded_word = idx2plainword(self.tgt_vocab_dict, idx,
                                       self.tgt_sp)
            if url_original:
                decoded_word = re.sub("<URL>", url_original, decoded_word)
            print(decoded_word)

    def service_infer(self, input_sent):
        """
        :param input_sent: string
        :return: translated sentence
        """
        input_sent, url_original = self.prepro(input_sent)
        input_sent = [self.src_sp.EncodeAsPieces(input_sent)]

        self.sess.run(self.infer_init_op,
                      feed_dict={self.input_placeholder:input_sent})
        idx = self.sess.run(self.decoded_idx)
        decoded_word = idx2plainword(self.tgt_vocab_dict, idx,
                                     self.tgt_sp)
        if url_original:
            decoded_word = re.sub("<URL>", url_original, decoded_word)
        return decoded_word
