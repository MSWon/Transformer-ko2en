
def nmt_tokenize(args):
    """
    :param args:
    :return:
    """
    from ..nmttokenize.tokenizer import Tokenizer
    tok = Tokenizer()
    tok.load(args.model_path)
    print(tok.tokenize(args.sentence))



