from nmt.nmttokenize.tokenizer import Tokenizer

def nmt_tokenize(args):
    """
    :param args:
    :return:
    """
    tok = Tokenizer()
    tok.load(args.model_path)
    print(tok.tokenize(args.sentence))



