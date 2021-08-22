import bentoml
import tensorflow as tf
import sentencepiece as spm
import re
import os

from bentoml.frameworks.tensorflow import TensorflowSavedModelArtifact
from bentoml.service.artifacts.common import PickleArtifact
from bentoml.adapters import JsonInput

MAX_LEN = 50
tf.compat.v1.enable_eager_execution() # required


@bentoml.env(
    infer_pip_packages=True,
    auto_pip_dependencies=True
)
@bentoml.artifacts(
    [
        TensorflowSavedModelArtifact('model'),
        PickleArtifact('src_sp'),
        PickleArtifact('tgt_sp'),
        PickleArtifact('src_reversed_vocab_dict'),
        PickleArtifact('tgt_vocab_dict')
    ]
)
class ServiceTransformer(bentoml.BentoService):
    def idx2plainword(self, vocab_dict, idx, sp):
        word_list = list(map(lambda x: vocab_dict[x], idx))
        return sp.detokenize(word_list)

    def sent2idx(self, reversed_vocab_dict, sent_list, max_len):
        word_list = sent_list + ["</s>"]
        unk_idx = len(reversed_vocab_dict) - 1
        idx_list = list(map(lambda x: reversed_vocab_dict[x] if x in reversed_vocab_dict else unk_idx, word_list))
        padded_idx_list = idx_list + [0] * (max_len - len(idx_list))
        return padded_idx_list

    def preprocess(self, sent):
        """
        :param sent: string
        :return: translated sentence, urls
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
        input_encoded_sent = self.artifacts.src_sp.tokenize(sent)
        input_idx = self.sent2idx(self.artifacts.src_reversed_vocab_dict, input_encoded_sent, MAX_LEN)
        return input_idx, url_original
    
    def postprocess(self, output_idx, url_original):
        decoded_word = self.idx2plainword(self.artifacts.tgt_vocab_dict, output_idx, self.artifacts.tgt_sp)
        if url_original:
            decoded_word = re.sub("<URL>", url_original, decoded_word)
        return decoded_word

    @bentoml.api(input=JsonInput(), batch=False)
    def predict(self, text):
        loaded_func = self.artifacts.model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        input_idx, input_url = self.preprocess(text)
        tensor_idx = tf.constant([input_idx])
        func_result = loaded_func(inputs=tensor_idx)
        output_idx = func_result['outputs']
        output_idx = output_idx.numpy().tolist()
        result = self.postprocess(output_idx, input_url)
        return result
