import tensorflow as tf
import model_utils
from encoder import Encoder
from decoder import Decoder
from gpu_utils import average_gradients

class Transformer(object):

    def __init__(self, hyp_args):
        self.num_layers = hyp_args['num_layers']
        self.num_heads = hyp_args['num_heads']
        self.hidden_dim = hyp_args['hidden_dim']
        self.linear_key_dim = hyp_args['linear_key_dim']
        self.linear_value_dim = hyp_args['linear_value_dim']
        self.ffn_dim = hyp_args['ffn_dim']
        self.dropout = hyp_args['dropout']
        self.vocab_size = hyp_args['vocab_size']
        self.bos_idx = hyp_args['bos_idx']
        self.eos_idx = hyp_args['eos_idx']
        self.max_len = hyp_args['max_len']
        self.n_gpus = hyp_args['n_gpus']

    def build_embed(self, inputs, isTrain):
        '''
        inputs : (batch_size, max_len)
        '''
        batch_size = tf.shape(inputs)[0]
        max_seq_length = tf.shape(inputs)[1]
        # Positional Encoding
        with tf.variable_scope("Positional-encoding", reuse=tf.AUTO_REUSE):
            position_emb = model_utils.get_position_encoding(max_seq_length, self.hidden_dim)
        # Word Embedding
        with tf.variable_scope("Embeddings", reuse=tf.AUTO_REUSE):
            self.embedding_weights = tf.get_variable('Weights', [self.vocab_size, self.hidden_dim],
                                                     dtype=tf.float32,
                                                     initializer=tf.random_normal_initializer(
                                                         0., self.hidden_size ** -0.5))
            mask = tf.to_float(tf.not_equal(inputs, 0))
            word_emb = tf.nn.embedding_lookup(self.embedding_weights, inputs)  ## batch_size, length, dim
            word_emb *= tf.expand_dims(mask, -1)  ## zeros out masked positions
            word_emb *= self.hidden_dim ** 0.5  ## Scale embedding by the sqrt of the hidden size
        ## Add Word emb & Positional emb
        encoded_inputs = tf.add(word_emb, position_emb)
        if isTrain:
            return tf.nn.dropout(encoded_inputs, 1.0 - self.dropout)
        else:
            return encoded_inputs

    def build_encoder(self, enc_input_idx, isTrain):
        ## enc_input_idx : (batch_size, enc_len)
        with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE):
            padding_bias = model_utils.get_padding_bias(enc_input_idx)
            padding = model_utils.get_padding(enc_input_idx)
            encoder_emb_inp = self.build_embed(enc_input_idx, isTrain)
            encoder = Encoder(num_layers=self.num_layers,
                              num_heads=self.num_heads,
                              linear_key_dim=self.linear_key_dim,
                              linear_value_dim=self.linear_value_dim,
                              model_dim=self.hidden_dim,
                              ffn_dim=self.ffn_dim,
                              dropout=self.dropout,
                              isTrain=isTrain)
            return encoder.build(encoder_emb_inp, padding_bias, padding=padding)

    def build_decoder(self, enc_input_idx, encoder_outputs, dec_input_idx, isTrain):
        '''
        enc_input_idx : (batch_size, enc_len)
        dec_input_idx : (batch_size, dec_len)
        '''
        with tf.variable_scope("Decoder", reuse=tf.AUTO_REUSE):
            dec_len = tf.shape(dec_input_idx)[1]
            dec_bias = model_utils.get_decoder_self_attention_bias(dec_len)
            enc_dec_bias = model_utils.get_padding_bias(enc_input_idx)
            decoder_emb_inp = self.build_embed(dec_input_idx, isTrain)
            decoder = Decoder(num_layers=self.num_layers,
                              num_heads=self.num_heads,
                              linear_key_dim=self.linear_key_dim,
                              linear_value_dim=self.linear_value_dim,
                              model_dim=self.hidden_dim,
                              ffn_dim=self.ffn_dim,
                              dropout=self.dropout,
                              isTrain=isTrain)
            return decoder.build(decoder_emb_inp, encoder_outputs, dec_bias, enc_dec_bias)

    def build_logits(self, decoder_outputs):
        with tf.variable_scope("Output_layer", reuse=tf.AUTO_REUSE):
            dec_len = tf.shape(decoder_outputs)[1]
            decoder_outputs = tf.reshape(decoder_outputs, [-1, self.hidden_dim])
            dec_shared_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Decoder/Embeddings/Weights')[0]
            logits = tf.matmul(decoder_outputs, dec_shared_weights, transpose_b=True)
            logits = tf.reshape(logits, [-1, dec_len, self.vocab_size])
        return logits

    def build_loss(self, dec_output_idx, dec_len, logits):
        self.masks = tf.sequence_mask(lengths=dec_len, maxlen=self.max_len, dtype=tf.float32)
        smoothed_label = self.label_smoothing(tf.one_hot(dec_output_idx, depth=self.vocab_size))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=smoothed_label, logits=logits)
        return tf.reduce_sum(cross_entropy * self.masks) / (tf.reduce_sum(self.masks) + 1e-10)

    def label_smoothing(self, inputs, epsilon=0.1):
        V = tf.shape(inputs)[-1] # number of channels
        return ((1 - epsilon) * inputs) + (epsilon / V)

    def noam_scheme(self, d_model, global_step, warmup_steps=4000):
        '''
        init_lr: initial learning rate. scalar.
        global_step: scalar.
        warmup_steps: scalar. During warmup_steps, learning rate increases
                      until it reaches init_lr.
        '''
        step = tf.cast(global_step + 1, dtype=tf.float32)
        return d_model ** (-0.5) * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

    def build_opt(self, src, tgt, d_model, global_step, warmup_steps=4000):
        # define optimizer
        learning_rate = self.noam_scheme(d_model, global_step, warmup_steps)
        opt = tf.contrib.opt.LazyAdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9)

        ''' Multi-GPU '''
        train_loss = tf.get_variable('total_loss', [],
                                     initializer=tf.constant_initializer(0.0), trainable=False)

        tower_grads = []
        total_batch = tf.shape(src['input_idx'])[0]
        batch_per_gpu = total_batch // self.n_gpus

        with tf.variable_scope(tf.get_variable_scope()):
            for k in range(self.n_gpus):
                with tf.device("/gpu:{}".format(k)):
                    print("Building model tower_{}".format(k + 1))
                    print("Could take few minutes")
                    # calculate the loss for one model replica
                    start = tf.to_int32(batch_per_gpu * k)
                    end = tf.to_int32(batch_per_gpu * (k + 1)) if k<self.n_gpus-1 else total_batch
                    enc_input_idx = src['input_idx'][start:end]
                    dec_input_idx = tgt['input_idx'][start:end]
                    dec_output_idx = tgt['output_idx'][start:end]
                    dec_idx_len = tgt['len'][start:end]
                    train_logits = self.train_fn(enc_input_idx, dec_input_idx)
                    loss = self.build_loss(dec_output_idx, dec_idx_len, train_logits)
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()
                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)
                    train_loss += loss / self.n_gpus

        grads = average_gradients(tower_grads)
        train_opt = opt.apply_gradients(grads, global_step=global_step)
        return train_loss, train_opt

    def train_fn(self, enc_input_idx, dec_input_idx):
        '''
        enc_input_idx : (batch_size, enc_len)
        dec_input_idx : (batch_size, dec_len)
        '''
        ## Encoder
        encoder_outputs = self.build_encoder(enc_input_idx, isTrain=True)
        ## Decoder
        decoder_outputs = self.build_decoder(enc_input_idx, encoder_outputs, dec_input_idx, isTrain=True)
        ## Logits
        logits = self.build_logits(decoder_outputs)
        return logits

    def test_fn(self, enc_input_idx, dec_output_idx):
        '''
        enc_input_idx : (batch_size, enc_len)
        dec_input_idx : (batch_size, dec_len)
        '''
        batch_size = tf.shape(enc_input_idx)[0]
        ## Initial values for while loop
        init_timestep = tf.constant(0, dtype=tf.int32)
        init_input = tf.fill([batch_size, 1], self.bos_idx)
        init_output = tf.constant([], dtype=tf.int32)
        init_output_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        init_loss_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        ## Encoder
        encoder_outputs = self.build_encoder(enc_input_idx, isTrain=False)
        ## Greedy Decoder
        def cond(timestep, input, output, output_array, loss_array):
            ''' Ends 'while-loop' when it returns False '''
            return tf.logical_and(tf.less(timestep, self.max_len), tf.not_equal(output, self.eos_id))
        def body(timestep, input, output, output_array, loss_array):
            ''' Main function of the while loop '''
            decoder_outputs = self.build_decoder(enc_input_idx, encoder_outputs, input, isTrain=False)
            decoder_logits = self.build_logits(decoder_outputs)[:,timestep,:] ## (N, vocab_size)
            next_output = tf.argmax(decoder_logits, axis=1) ## (N, )
            next_output_array = output_array.write(next_output)
            next_input = tf.concat([input, next_output[:,None]], axis=1) ## (N, timestep+1)
            smoothed_label = self.label_smoothing(tf.one_hot(dec_output_idx[:,timestep], depth=self.vocab_size)) ## (N, vocab_size)
            batch_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=smoothed_label,
                                                                                   logits=decoder_logits))
            next_loss_array = loss_array.write(batch_loss)
            next_timestep = timestep + 1
            return next_timestep, next_input, next_output, next_output_array, next_loss_array

        _, _, _, decoded_array, loss_array = tf.while_loop(cond,
                                                           body,
                                                           [init_timestep,
                                                            init_input,
                                                            init_output,
                                                            init_output_array,
                                                            init_loss_array])
        decoded_idx = tf.squeeze(decoded_array.stack()) ## (N, )
        total_loss = tf.reduce_mean(loss_array.stack()) ## ()
        return decoded_idx, total_loss

