import tensorflow as tf
from transformer import Transformer
from data_pipeline import train_dataset_fn, test_dataset_fn, get_vocab, idx2bpeword


class Trainer(object):

    def __init__(self, hyp_args):
        self.train_src_corpus_path = hyp_args["train_src_corpus_path"]
        self.train_tgt_corpus_path = hyp_args["train_tgt_corpus_path"]
        self.test_src_corpus_path = hyp_args["test_src_corpus_path"]
        self.test_tgt_corpus_path = hyp_args["test_tgt_corpus_path"]
        self.src_vocab_path = hyp_args["src_vocab_path"]
        self.tgt_vocab_path = hyp_args["tgt_vocab_path"]
        self.max_len = hyp_args["max_len"]
        self.batch_size = hyp_args["batch_size"]
        self.model_path = hyp_args["model_path"]

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
        src, tgt = iters.get_next()

        self.tgt_vocab_dict = get_vocab(self.tgt_vocab_path, False)

        # create the initialisation operations
        self.train_init_op = iters.make_initializer(train_dataset)
        self.test_init_op = iters.make_initializer(test_dataset)

        print("Now building model")
        model = Transformer(hyp_args)
        global_step = tf.train.get_or_create_global_step()

        self.train_loss, self.train_opt = model.build_opt(src, tgt, hyp_args["hidden_dim"],
                                                        global_step, hyp_args["warmup_step"])

        self.decoded_idx, self.test_loss = model.test_fn(src["input_idx"], tgt["output_idx"])
        print("Done")

    def train(self, training_steps):
        print("Now training")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state("./model" + self.model_path)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            if(ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)):
                saver.restore(sess, "./model" + self.model_path)
                print("Model loaded!")

            sess.run(self.train_init_op)
            sess.run(tf.tables_initializer())
            n_train_step = 0
            train_loss_, test_loss_ = 0., 0.
            best_loss = 1e8

            for step in range(training_steps):
                n_train_step += 1
                batch_train_loss, _ = sess.run([self.train_loss,
                                                self.train_opt])
                train_loss_ += batch_train_loss
                train_loss = train_loss_ / n_train_step

                print("step : {} train_loss : {}".format(step+1, train_loss))

                if step % 100 == 0 and step > 0:
                    print("Now for test data")
                    sess.run(self.test_init_op)
                    n_test_step = 0
                    with open("test_output_{}.txt".format(step), "w") as f:
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

                    test_loss = test_loss_ / n_test_step
                    if test_loss < best_loss:
                        save_path = saver.save(sess, "./model" + self.model_path)
                        best_loss = test_loss

                    sess.run(self.train_init_op)


