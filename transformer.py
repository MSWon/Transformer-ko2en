import tensorflow as tf
import model_utils
from encoder import Encoder
from decoder import Decoder


class Transformer(object):

    def __init__(self):
        None

    def build_embed(self, inputs, reuse=False):
        batch_size = tf.shape(inputs)[0]
        max_seq_length = tf.shape(inputs)[1]
        # Positional Encoding
        with tf.variable_scope("Positional-encoding", reuse=reuse):
            positional_encoded = model_utils.get_position_encoding(max_seq_length, self.hidden_dim)
            position_inputs = tf.tile(tf.range(0, max_seq_length), [batch_size])
            position_inputs = tf.reshape(position_inputs, [batch_size, max_seq_length])
            position_emb = tf.nn.embedding_lookup(positional_encoded, position_inputs)
        # Word Embedding
        with tf.variable_scope("Word-embeddings", reuse=reuse, initializer="TODO"):
            self.shared_weights = tf.get_variable('shared_weights', [self.vocab, self.hidden_dim], dtype=tf.float32)
            mask = tf.to_float(tf.not_equal(inputs, 0))
            word_emb = tf.nn.embedding_lookup(self.shared_weights, inputs)  ## batch_size, length, dim
            word_emb *= tf.expand_dims(mask, -1)  ## zeros out masked positions
            word_emb *= self.hidden_dim ** 0.5  ## Scale embedding by the sqrt of the hidden size
        ## Add
        encoded_inputs = tf.add(word_emb, position_emb)
        return tf.nn.dropout(encoded_inputs, 1.0 - self.dropout)

    def build_encoder(self, x, encoder_emb_inp, attention_bias, reuse=False):
        ## x: (batch_size, enc_len)
        padding_bias = attention_bias
        with tf.variable_scope("Encoder", reuse=reuse, initializer="TODO"):
            encoder = Encoder(num_layers=self.num_layers,
                              num_heads=self.num_heads,
                              linear_key_dim=self.linear_key_dim,
                              linear_value_dim=self.linear_value_dim,
                              model_dim=self.hidden_dim,
                              ffn_dim=self.ffn_dim)
            padding = model_utils.get_padding(x)
            return encoder.build(encoder_emb_inp, padding_bias, padding=padding)

    def build_decoder(self, decoder_emb_inp, encoder_outputs, dec_bias, attention_bias, reuse=False):
        enc_dec_bias = attention_bias
        with tf.variable_scope("Decoder", reuse=reuse, initializer="TODO"):
            decoder = Decoder(num_layers=self.num_layers,
                              num_heads=self.num_heads,
                              linear_key_dim=self.linear_key_dim,
                              linear_value_dim=self.linear_value_dim,
                              model_dim=self.hidden_dim,
                              ffn_dim=self.ffn_dim)
            return decoder.build(decoder_emb_inp, encoder_outputs, dec_bias, enc_dec_bias)

    def build_output(self, decoder_outputs, reuse=False):
        with tf.variable_scope("Output", reuse=reuse):
            t_shape = decoder_outputs.get_shape().as_list()  ## batch_size, seq_length, dim
            seq_length = t_shape[1]
            decoder_outputs = tf.reshape(decoder_outputs, [-1, self.hidden_dim])
            logits = tf.matmul(decoder_outputs, self.shared_weights, transpose_b=True)
            logits = tf.reshape(logits, [-1, seq_length, self.vocab])
        return logits

    def decoder_train(self, x, y):
        ## x: (batch_size, enc_len) , y: (batch_size, dec_len)
        dec_bias = model_utils.get_decoder_self_attention_bias(self.max_dec_len)
        attention_bias = model_utils.get_padding_bias(x)
        # Encoder
        encoder_emb_inp = self.build_embed(x, encoder=True, reuse=False)
        encoder_outputs = self.build_encoder(x, encoder_emb_inp, attention_bias, reuse=False)
        # Decoder
        batch_size = tf.shape(x)[0]
        start_tokens = tf.fill([batch_size, 1], self.bos_idx)  # 2: <s> ID
        target_slice_last_1 = tf.slice(y, [0, 0], [batch_size, self.max_dec_len - 1])
        decoder_inputs = tf.concat([start_tokens, target_slice_last_1], axis=1)  ## shift to right
        decoder_emb_inp = self.build_embed(decoder_inputs, encoder=False, reuse=True)
        decoder_outputs = self.build_decoder(decoder_emb_inp, encoder_outputs, dec_bias, attention_bias, reuse=False)
        train_prob = self.build_output(decoder_outputs, reuse=False)
        return train_prob