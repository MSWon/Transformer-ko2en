import tensorflow as tf

_buffer_size = 100000
_bucket_size = 10

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
    word_list = map(lambda x: vocab_dict[x], idx)
    return " ".join(word_list)

def train_dataset_fn(src_corpus_path, tgt_corpus_path,
                     src_vocab_path, tgt_vocab_path, max_len, batch_size):

  tf_vocab_src = get_vocab(src_vocab_path)
  dataset_src = tf.data.TextLineDataset(src_corpus_path)
  dataset_src = dataset_src.map(lambda token: tf.string_split([token]).values)
  dataset_src = dataset_src.map(lambda token: {"input_idx": tf_vocab_src.lookup(token),
                                               "len": tf.shape(token)[0]})

  tf_vocab_tgt = get_vocab(tgt_vocab_path)
  dataset_tgt = tf.data.TextLineDataset(tgt_corpus_path)
  dataset_tgt = dataset_tgt.map(lambda token: tf.concat([["<s>"], tf.string_split([token]).values, ["</s>"]], axis=0))
  dataset_tgt = dataset_tgt.map(lambda token: {"input_idx": tf_vocab_tgt.lookup(token[:-1]),
                                               "output_idx": tf_vocab_tgt.lookup(token[1:]),
                                               "len": tf.shape(token[:-1])[0]})

  dataset = tf.data.Dataset.zip((dataset_src, dataset_tgt))
  dataset = dataset.filter(lambda src, tgt: tf.logical_and(tf.less_equal(src["len"], max_len),
                                                           tf.less_equal(tgt["len"], max_len)))

  dataset = dataset.shuffle(_buffer_size)
  # Use bucket batch
  bucket_boundaries = [i for i in range(10, max_len + 1, int(max_len / _bucket_size))]
  bucket_batch_sizes = [max(1, batch_size // length) for length in bucket_boundaries + [max_len]]

  dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
                         lambda src, tgt: (tf.maximum(src["len"], tgt["len"])),
                         bucket_boundaries,
                         bucket_batch_sizes
                         ))

  dataset = dataset.repeat()
  # Prefetch the next element to improve speed of input pipeline.
  dataset = dataset.prefetch(buffer_size=batch_size)
  return dataset


def test_dataset_fn(src_corpus_path, tgt_corpus_path,
                    src_vocab_path, tgt_vocab_path, max_len, batch_size):

    tf_vocab_src = get_vocab(src_vocab_path)
    dataset_src = tf.data.TextLineDataset(src_corpus_path)
    dataset_src = dataset_src.map(lambda token: tf.string_split([token]).values)
    dataset_src = dataset_src.map(lambda token: {"input_idx": tf_vocab_src.lookup(token),
                                                 "len": tf.shape(token)[0]})

    tf_vocab_tgt = get_vocab(tgt_vocab_path)
    dataset_tgt = tf.data.TextLineDataset(tgt_corpus_path)
    dataset_tgt = dataset_tgt.map(lambda token: tf.concat([["<s>"], tf.string_split([token]).values, ["</s>"]], axis=0))
    dataset_tgt = dataset_tgt.map(lambda token: {"input_idx": tf_vocab_tgt.lookup(token[:-1]),
                                                 "output_idx": tf_vocab_tgt.lookup(token[1:]),
                                                 "len": tf.shape(token[:-1])[0]})

    dataset = tf.data.Dataset.zip((dataset_src, dataset_tgt))
    dataset = dataset.filter(lambda src, tgt: tf.logical_and(tf.less_equal(src["len"], max_len),
                                                             tf.less_equal(tgt["len"], max_len)))

    dataset = dataset.padded_batch(batch_size, padded_shapes=({"input_idx": [max_len], "len": []},
                                                              {"input_idx": [max_len],
                                                               "output_idx": [max_len], "len": []}))

    # Prefetch the next element to improve speed of input pipeline.
    dataset = dataset.prefetch(buffer_size=batch_size)
    return dataset

def infer_dataset_fn(src_vocab_path, tgt_vocab_path, max_len, batch_size):

    tf_vocab_src = get_vocab(src_vocab_path)

    input_src = tf.placeholder(shape=(1, None), dtype=tf.string)
    dataset_src = tf.data.Dataset.from_tensor_slices(input_src)
    dataset_src = dataset_src.map(lambda token: {"input_idx": tf_vocab_src.lookup(token),
                                                 "len": tf.shape(token)[0]})

    dataset_src = dataset_src.padded_batch(1, padded_shapes=({"input_idx": [max_len], "len": []}))

    tf_vocab_tgt = get_vocab(tgt_vocab_path)

    return dataset_src

