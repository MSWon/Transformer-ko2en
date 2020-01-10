# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:57:59 2019

@author: jbk48
"""
import sentencepiece as spm
import pickle

class Data(object):

    def __init__(self, path, max_enc_len=128, max_dec_len=128):

        self.path = path

        self.max_enc_len , self.max_dec_len = max_enc_len, max_dec_len
        
        self.pad_token, self.pad_idx = "<pad>", 0
        self.unk_token, self.unk_idx = "<unk>", 1
        self.bos_token, self.bos_idx = "<s>", 2
        self.eos_token, self.eos_idx = "</s>", 3
        
        self.get_SentPiece()
        self.w2idx, self.idx2w = self.read_vocab()
        self.vocab = len(self.w2idx)

    def get_SentPiece(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.path + "/" + "bpe.model")
        
    def read_vocab(self):
        vocab = [line.split()[0] for line in open(self.path + "/" + "bpe.vocab", 'r', encoding='utf-8').read().splitlines()]
        w2idx = {token: idx for idx, token in enumerate(vocab)}
        idx2w = {idx: token for idx, token in enumerate(vocab)}
        return w2idx, idx2w

    def read_file(self, name):
        print("preparing {} file".format(name))

        with open(self.path + "/" + name + ".ko", "rb") as f:
            ##enc = f.readlines()
            enc = pickle.load(f)
        
        with open(self.path + "/" + name + ".en", "rb") as f:
            ##dec = f.readlines()
            dec = pickle.load(f)
        
        enc_idx, enc_len = [], []
        dec_idx, dec_len = [], []
        
        for sent1, sent2 in zip(enc,dec):
            if(len(self.sp.EncodeAsPieces(sent1)) > self.max_enc_len-1):
                continue
            if(len(self.sp.EncodeAsPieces(sent2)) > self.max_dec_len-1):
                continue
            if(sent1 != "\n" and sent2 != "\n"):
                sent1 = self.sp.EncodeAsPieces(sent1) + ["</s>"]
                enc_len.append(len(sent1))
                enc_idx.append(self.sent2idx(sent1, self.w2idx, self.max_enc_len))
            
                sent2 = self.sp.EncodeAsPieces(sent2) + ["</s>"]
                dec_len.append(len(sent2))
                dec_idx.append(self.sent2idx(sent2, self.w2idx, self.max_dec_len))
        
        return enc_idx, dec_idx, enc_len, dec_len

    def sent2idx(self, sent, w2idx, max_len):
        idx = []
        for word in sent:
            if(word in w2idx):
                idx.append(w2idx[word])
            else:
                idx.append(self.unk_idx)
        
        return idx + [self.pad_idx]*(max_len-len(idx)) ## PAD for max length

