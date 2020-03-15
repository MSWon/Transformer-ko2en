import sentencepiece as spm
import re
import os

class Tokenizer(object):
    """ Tokenizer class """
    def __init__(self):
        self.sp = spm.SentencePieceProcessor()

    def load(self, bpe_model_path):
        """
        :param bpe_model_path: BPE model path
        :return: None
        """
        self.sp.Load(bpe_model_path)

    def train(self, corpus_path, model_name, vocab_size=32000):
        """
        :param corpus_path: corpus path to train BPE
        :param model_name: output model prefix name
        :param vocab_size: size of the vocab (default 32k)
        :return:
        """
        train_sp = '--input={} --pad_id=0 --unk_id=1 \
                    --bos_id=2 --eos_id=3 \
                    --model_prefix={} \
                    --user_defined_symbols=<URL> \
                    --vocab_size={} \
                    --model_type=bpe'.format(corpus_path, model_name, vocab_size)

        print("training BPE model")
        for config in train_sp.replace("\t","").split("--"):
            print(config)

        spm.SentencePieceTrainer.Train(train_sp)

        uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
        model_path = os.path.join(uppath(corpus_path, 1), model_name)

        f = open("{}.vocab".format(model_path), "r")
        with open("{}.vocab2".format(model_path), "w") as f1:
            for sent in f:
                f1.write(sent.split()[0] + "\n")

        os.system("rm {}.vocab".format(model_path))
        os.system("mv {}.vocab2 {}.vocab".format(model_path, model_path))

    def preprocess(self, sent):
        """
        :param sent: input sentence
        :return: preprocessed sentence
        """
        sent = re.sub("\(.*?\)|\[.*?\]", "", sent)
        sent = re.sub("[^0-9a-zA-Z가-힣_\-@\.:&+!?'/,\s]", "", sent)
        return sent

    def url_replace(self, sent):
        """
        :param sent: input sentence
        :return: url replaced sentence
        """
        url_regex = "(http[s]?://([a-zA-Z]|[가-힣]|[0-9]|[-_@\.&+!*/])+)|(www.([a-zA-Z]|[가-힣]|[0-9]|[-_@\.&+!*/])+)"
        sent = re.sub(url_regex, "<URL>", sent)
        return sent

    def tokenize(self, sent):
        """
        :param sent: input sentence
        :return: BPE tokenized list
        """
        return self.sp.EncodeAsPieces(sent)

