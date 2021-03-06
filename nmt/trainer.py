import tensorflow as tf
import os
import sentencepiece as spm
from .nmttrain.model.transformer import Transformer
from .nmttrain.utils.data_utils import train_dataset_fn, test_dataset_fn, get_vocab, idx2bpeword
from .nmttrain.utils.model_utils import calc_bleu

class Trainer(object):
    """ Trainer class """
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
        self.training_steps = hyp_args["training_steps"]

        self.tgt_sp = spm.SentencePieceProcessor()
        self.tgt_sp.Load(self.tgt_bpe_model_path)

        train_dataset = train_dataset_fn(self.train_src_corpus_path,
                                         self.train_tgt_corpus_path,
                                         self.src_vocab_path, self.tgt_vocab_path,
                                         self.max_len, self.batch_size)

        test_dataset = test_dataset_fn(self.test_src_corpus_path,
                                       self.test_tgt_corpus_path,
                                       self.src_vocab_path, self.tgt_vocab_path,
                                       self.max_len, 1)

        iters = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                train_dataset.output_shapes)
        features = iters.get_next()

        self.tgt_vocab_dict = get_vocab(self.tgt_vocab_path, False)

        # create the initialisation operations
        self.train_init_op = iters.make_initializer(train_dataset)
        self.test_init_op = iters.make_initializer(test_dataset)

        print("Now building model")
        model = Transformer(hyp_args)
        global_step = tf.train.get_or_create_global_step()

        self.train_loss, self.train_opt = model.build_opt(features, hyp_args["hidden_dim"],
                                                        global_step, hyp_args["warmup_step"])

        self.decoded_idx, self.test_loss = model.test_fn(features["src_input_idx"], features["tgt_output_idx"])
        ## for tensorboard
        self.train_loss_graph = tf.placeholder(shape=None, dtype=tf.float32)
        self.test_loss_graph = tf.placeholder(shape=None, dtype=tf.float32)
        self.test_bleu_graph = tf.placeholder(shape=None, dtype=tf.float32)
        print("Done")

    def train(self):
        """
        :return: None
        """
        print("Now training")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state("./model")
        os.makedirs("./decoded")
        summary_train_loss = tf.summary.scalar("train_loss", self.train_loss_graph)
        summary_test_loss = tf.summary.scalar("test_loss", self.test_loss_graph)
        summary_test_bleu = tf.summary.scalar("test_bleu", self.test_bleu_graph)
        merged = tf.summary.merge([summary_test_loss, summary_test_bleu])

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            if(ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)):
                saver.restore(sess, "./model/" + self.model_path)
                print("Model loaded!")

            sess.run(self.train_init_op)
            sess.run(tf.tables_initializer())
            n_train_step = 0
            train_loss_, test_loss_ = 0., 0.
            best_loss = 1e8
            writer = tf.summary.FileWriter('./tensorboard/graph', sess.graph)

            for step in range(self.training_steps):
                n_train_step += 1
                batch_train_loss, _ = sess.run([self.train_loss,
                                                self.train_opt])
                train_loss_ += batch_train_loss
                train_loss = train_loss_ / n_train_step

                print("step : {} train_loss : {}".format(step+1, train_loss))

                if step % 100 == 0 and step > 0:
                    summary = sess.run(summary_train_loss,
                                       feed_dict={self.train_loss_graph:train_loss})
                    writer.add_summary(summary, step)
                
                if step % 2000 == 0 and step > 0:
                    print("Now for test data")
                    sess.run(self.test_init_op)
                    n_test_step = 0
                    with open("./decoded/test_output_{}.en".format(step), "w") as f:
                        try:
                            while True:
                                n_test_step += 1
                                batch_test_loss, idx = sess.run([self.test_loss,
                                                                 self.decoded_idx])
                                test_loss_ += batch_test_loss
                                decoded_word = idx2bpeword(self.tgt_vocab_dict, idx)
                                f.write(decoded_word + "\n")

                        except tf.errors.OutOfRangeError:
                            pass

                    test_bleu = calc_bleu(self.test_tgt_corpus_path, "./decoded/test_output_{}.en".format(step))
                    test_loss = test_loss_ / n_test_step
                    summary = sess.run(merged, feed_dict={self.test_loss_graph: test_loss,
                                                          self.test_bleu_graph: test_bleu})
                    writer.add_summary(summary, step)

                    if test_loss < best_loss or step % 2000 == 0:
                        save_path = saver.save(sess, "./model/" + self.model_path)
                        best_loss = test_loss

                    sess.run(self.train_init_op)


