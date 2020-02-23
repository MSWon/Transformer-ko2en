import tensorflow as tf
import sentencepiece as spm
import re
import os
from transformer import Transformer
from data_pipeline import infer_dataset_fn, get_vocab, idx2bpeword

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

class Translate(object):

    def __init__(self, hyp_args):
        self.train_src_corpus_path = hyp_args["train_src_corpus_path"]
        self.train_tgt_corpus_path = hyp_args["train_tgt_corpus_path"]
        self.test_src_corpus_path = hyp_args["test_src_corpus_path"]
        self.test_tgt_corpus_path = hyp_args["test_tgt_corpus_path"]
        self.src_vocab_path = hyp_args["src_vocab_path"]
        self.tgt_vocab_path = hyp_args["tgt_vocab_path"]
        self.src_bpe_model_path = hyp_args["src_bpe_model_path"]
        self.tgt_bpe_model_path = hyp_args["tgt_bpe_model_path"]
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
        self.decoded_idx = model.infer_fn(src["input_idx"])
        # build session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.tables_initializer())
        self.restore_model(self.sess)

    def restore_model(self, sess):
        saver = tf.train.Saver()
        saver.restore(sess, "./model/" + self.model_path)
        print("Model loaded!")

    def prepro(self, sent):
        sent = re.sub("\(.*?\)|\[.*?\]", "", sent)
        sent = re.sub("[^0-9a-zA-Z가-힣_\-@\.:&+!?'/,\s]", "", sent)
        sent = re.sub("(http[s]?://([a-zA-Z]|[가-힣]|[0-9]|[-_@\.&+!*/])+)|(www.([a-zA-Z]|[가-힣]|[0-9]|[-_@\.&+!*/])+)", "<URL>", sent)
        return sent

    def infer(self):
        while True:
            input_sent = input("Input Korean sent : ")
            input_sent = self.prepro(input_sent)
            input_sent = [self.src_sp.EncodeAsPieces(input_sent)]

            self.sess.run(self.infer_init_op,
                          feed_dict={self.input_placeholder:input_sent})
            idx = self.sess.run(self.decoded_idx)
            decoded_word = idx2bpeword(self.tgt_vocab_dict, idx,
                                       self.tgt_sp)
            print(decoded_word)

