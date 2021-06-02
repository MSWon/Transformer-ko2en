import tensorflow as tf
import sentencepiece as spm
import re
import os

from nmt.nmttrain.model.transformer import Transformer
from nmt.nmttrain.utils.data_utils import get_vocab, idx2plainword, preprocess


os.environ['CUDA_VISIBLE_DEVICES'] = "1"
MODEL_FOLDER_NAME = 'exported.model'

class ExportTransformer(object):
    """ ExportTransformer class """
    def __init__(self, hyp_args, input_checkpoint):
        self.max_len = hyp_args["max_len"]
        self.batch_size = hyp_args["batch_size"]
        self.checkpoint_path = input_checkpoint
        self.input_placeholder = tf.placeholder(shape=(1, self.max_len), dtype=tf.int32)

        print("Now building model")
        model = Transformer(hyp_args)
        self.decoded_idx = model.infer_fn(self.input_placeholder)
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
        saver = tf.train.Saver()
        saver.restore(sess, self.checkpoint_path)
        print("Model loaded!")
