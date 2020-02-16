# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 18:13:34 2019

@author: jbk48
"""

import pandas as pd
import sentencepiece as spm
import re
import os
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.utils import shuffle

tokenizer = ToktokTokenizer()

filename_list = [dir for dir in sorted(os.listdir()) if ".xlsx" in dir]

def prepro(sent):
    sent = re.sub("\(.*?\)|\[.*?\]", "", sent)
    sent = re.sub("[^0-9a-zA-Z가-힣_\-@\.:&+!?'/,\s]", "", sent)
    sent = re.sub("(http[s]?://([a-zA-Z]|[가-힣]|[0-9]|[-_@\.&+!*/])+)|(www.([a-zA-Z]|[가-힣]|[0-9]|[-_@\.&+!*/])+)", "<URL>", sent)
    return sent

ko_corpus, en_corpus = [], []
idx = 1
for filename in filename_list:
    print("file num {} in progress".format(idx))
    df = pd.read_excel(filename)
    ko_corpus += [prepro(sent) for sent in df['원문']]
    en_corpus += [prepro(sent) for sent in df['번역문']]
    idx += 1

print("Done")
print("Now shuffling data")

df = pd.DataFrame({'ko':ko_corpus,
                   'en':en_corpus})

df = shuffle(df)

ko_train = list(df['ko'])
en_train = list(df['en'])

with open('./train.ko','w') as f:
    for sent in ko_train:
        f.write(sent + "\n")

with open('./train.en','w') as f:
    for sent in en_train:
        f.write(sent + "\n")

print("Now training sentencepiece model")

train_ko =  '--input=train.ko --pad_id=0 --unk_id=1 \
             --bos_id=2 --eos_id=3\
             --model_prefix=bpe.ko \
             --user_defined_symbols=<URL> \
             --vocab_size={} \
             --model_type=bpe'.format(32000)
             
spm.SentencePieceTrainer.Train(train_ko)

f = open("bpe.ko.vocab","r")
with open("bpe.ko.vocab2", "w") as f1:
	for sent in f:
		f1.write(sent.split()[0] + "\n")


train_en = '--input=train.en --pad_id=0 --unk_id=1 \
             --bos_id=2 --eos_id=3\
             --model_prefix=bpe.en \
             --user_defined_symbols=<URL> \
             --vocab_size={} \
             --model_type=bpe'.format(32000)

spm.SentencePieceTrainer.Train(train_en)


f = open("bpe.en.vocab","r")
with open("bpe.en.vocab2", "w") as f1:
    for sent in f:
        f1.write(sent.split()[0] + "\n")

print("Converting to bpe tokenized version")

sp = spm.SentencePieceProcessor()
sp.Load("bpe.en.model")

f = open("train.en", "r")

with open("train.en.bpe","w") as f1:
    for sent in f:
        f1.write(" ".join(sp.EncodeAsPieces(sent)) + "\n")


sp = spm.SentencePieceProcessor()
sp.Load("bpe.ko.model")

f = open("train.ko", "r")

with open("train.ko.bpe","w") as f1:
    for sent in f:
        f1.write(" ".join(sp.EncodeAsPieces(sent)) + "\n")



'''
sp = spm.SentencePieceProcessor()
sp.Load("bpe.model")

encoded = sp.EncodeAsPieces(ko_sent)
decoded = sp.DecodePieces(encoded)

vocab_fpath = "bpe.vocab"
def load_vocab(vocab_fpath):
    vocab = [line.split()[0] for line in open(vocab_fpath, 'r', encoding='utf-8').read().splitlines()]
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}
    return token2idx, idx2token

token2idx, idx2token = load_vocab(vocab_fpath)
''' 