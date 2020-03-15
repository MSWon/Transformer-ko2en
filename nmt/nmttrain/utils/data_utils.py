import tensorflow as tf

_buffer_size = 2000000
_bucket_size = 10
_thread_num = 16

def get_vocab(vocab_path, isTF=True):
    if isTF:
        vocab_path_tensor = tf.constant(vocab_path)
        tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, vocab_path_tensor)
        vocab_dict = tf.contrib.lookup.index_table_from_file(
            vocabulary_file=vocab_path_tensor,
            num_oov_buckets=0,
            default_value=1)
    else:
        vocab_dict = {}
        with open(vocab_path, "r") as f:
            for vocab in f:
                vocab_dict[len(vocab_dict)] = vocab.strip()
    return vocab_dict

def idx2bpeword(vocab_dict, idx):
    word_list = list(map(lambda x: vocab_dict[x], idx))
    return " ".join(word_list)

def idx2plainword(vocab_dict, idx, sp):
    word_list = list(map(lambda x: vocab_dict[x], idx))
    return sp.DecodePieces(word_list)

def train_dataset_fn(src_corpus_path, tgt_corpus_path,
                     src_vocab_path, tgt_vocab_path, max_len, batch_size):

  with tf.device("/cpu:0"):
      tf_vocab_src = get_vocab(src_vocab_path)
      tf_vocab_tgt = get_vocab(tgt_vocab_path)
      dataset_src = tf.data.TextLineDataset(src_corpus_path)
      dataset_tgt = tf.data.TextLineDataset(tgt_corpus_path)

      dataset = tf.data.Dataset.zip((dataset_src, dataset_tgt))
      dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(_buffer_size))

      dataset = dataset.map(lambda src,tgt:(
                                tf.concat([tf.string_split([src]).values, ["</s>"]], axis=0),
                                tf.concat([["<s>"], tf.string_split([tgt]).values, ["</s>"]], axis=0)
                                ),
                                num_parallel_calls=_thread_num
                            )

      dataset = dataset.map(lambda src,tgt:(
                                tf_vocab_src.lookup(src),
                                tf_vocab_tgt.lookup(tgt)
                                ),
                                num_parallel_calls=_thread_num
                            )

      dataset = dataset.map(lambda src, tgt:(
                                tf.to_int32(src),
                                tf.to_int32(tgt)
                                ),
                                num_parallel_calls=_thread_num
                            )

      dataset = dataset.filter(lambda src, tgt: tf.logical_and(tf.less_equal(tf.shape(src)[0], max_len),
                                                               tf.less_equal(tf.shape(tgt)[0], max_len)))

      dataset = dataset.map(lambda src,tgt:{
                                "src_input_idx": src,
                                "src_len": tf.shape(src)[0],
                                "tgt_input_idx": tgt[:-1],
                                "tgt_output_idx": tgt[1:],
                                "tgt_len": tf.shape(tgt[:-1])[0],
                               },
                            num_parallel_calls=_thread_num
                          )

      # Use bucket batch
      bucket_boundaries = [i for i in range(10, max_len + 1, int(max_len / _bucket_size))]
      bucket_batch_sizes = [max(1, batch_size // length) for length in bucket_boundaries + [max_len]]

      dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
                                     lambda x: tf.maximum(x["src_len"], x["tgt_len"]),
                                     bucket_boundaries,
                                     bucket_batch_sizes
                                 )
                             )

      # Prefetch the next element to improve speed of input pipeline.
      dataset = dataset.prefetch(3)
  return dataset


def test_dataset_fn(src_corpus_path, tgt_corpus_path,
                    src_vocab_path, tgt_vocab_path, max_len, batch_size):
    with tf.device("/cpu:0"):
        tf_vocab_src = get_vocab(src_vocab_path)
        tf_vocab_tgt = get_vocab(tgt_vocab_path)
        dataset_src = tf.data.TextLineDataset(src_corpus_path)
        dataset_tgt = tf.data.TextLineDataset(tgt_corpus_path)

        dataset = tf.data.Dataset.zip((dataset_src, dataset_tgt))

        dataset = dataset.map(lambda src, tgt: (
                                    tf.concat([tf.string_split([src]).values, ["</s>"]], axis=0),
                                    tf.concat([["<s>"], tf.string_split([tgt]).values, ["</s>"]], axis=0)
                                ),
                                num_parallel_calls=_thread_num
                              )

        dataset = dataset.map(lambda src, tgt: (
                                    tf_vocab_src.lookup(src),
                                    tf_vocab_tgt.lookup(tgt)
                                 ),
                                num_parallel_calls=_thread_num
                              )

        dataset = dataset.map(lambda src, tgt:(
                                tf.to_int32(src),
                                tf.to_int32(tgt)
                                ),
                                num_parallel_calls=_thread_num
                             )

        dataset = dataset.filter(lambda src, tgt: tf.logical_and(tf.less_equal(tf.shape(src)[0], max_len),
                                                                 tf.less_equal(tf.shape(tgt)[0], max_len)))

        dataset = dataset.map(lambda src, tgt: {
                                "src_input_idx": src,
                                "src_len": tf.shape(src)[0],
                                "tgt_input_idx": tgt[:-1],
                                "tgt_output_idx": tgt[1:],
                                "tgt_len": tf.shape(tgt[:-1])[0],
                                },
                                num_parallel_calls=_thread_num
                              )
        dataset = dataset.padded_batch(
            batch_size,
            {
                "src_input_idx": [tf.Dimension(None)],
                "tgt_input_idx": [tf.Dimension(None)],
                "tgt_output_idx": [tf.Dimension(None)],
                "src_len": [],
                "tgt_len": []
            },
            {
                "src_input_idx": 0,
                "tgt_input_idx": 0,
                "tgt_output_idx": 0,
                "src_len": 0,
                "tgt_len": 0
            }
        )
    return dataset

def infer_dataset_fn(src_vocab_path, max_len, batch_size):
    tf_vocab_src = get_vocab(src_vocab_path)
    input_src = tf.placeholder(shape=(1, None), dtype=tf.string)
    dataset_src = tf.data.Dataset.from_tensor_slices(input_src)
    dataset_src = dataset_src.map(lambda src: tf.concat([src, ["</s>"]], axis=0))
    dataset_src = dataset_src.map(lambda src: tf_vocab_src.lookup(src))
    dataset_src = dataset_src.map(lambda src: tf.to_int32(src))
    dataset_src = dataset_src.map(lambda src: {"src_input_idx": src,
                                               "src_len": tf.shape(src)[0]})

    dataset_src = dataset_src.padded_batch(
            batch_size,
            {
            "src_input_idx": [tf.Dimension(None)],
            "src_len": []
            },
            {
            "src_input_idx": 0,
            "src_len": 0
            }
        )

    return dataset_src, input_src

