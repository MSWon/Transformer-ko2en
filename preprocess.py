# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 18:13:34 2019

@author: jbk48
"""

import pandas as pd
import sentencepiece as spm
import re
import os
import pickle
from nltk import tokenize
from sklearn.utils import shuffle
from translate.storage.tmx import tmxfile


'''
df1 = pd.read_excel("1.구어체.xlsx", sheetname = "Sheet1")

df2_1 = pd.read_excel("2.대화체.xlsx", sheetname = "Sheet1")
df2_2 = pd.read_excel("2.대화체.xlsx", sheetname = "Sheet3")

df3_1 = pd.read_excel("3.문어체-뉴스.xlsx", sheetname = "번역")
df3_2 = pd.read_excel("3.문어체-뉴스.xlsx", sheetname = "MTPE")

df4 = pd.read_excel("4.문어체-한국문화.xlsx")
df5 = pd.read_excel("5.문어체-조례.xlsx")
df6 = pd.read_excel("6.문어체-지자체웹사이트.xlsx")


ko_corpus = list(df1['ko'])+list(df2_1['한국어'])+list(df2_2['한국어'])+list(df3_1['한국어'])\
+ list(df3_2['한국어'])+list(df4['원문'])+list(df5['원문'])+list(df6['원문'])

ko_corpus = [" ".join(tokenize.word_tokenize(sent)) for sent in ko_corpus]

en_corpus = list(df1['en'])+list(df2_1['영어'])+list(df2_2['영어'])+list(df3_1['영어'])\
+list(df3_2['영어'])+list(df4['PE'])+list(df5['PE'])+list(df6['PE'])

en_corpus = [" ".join(tokenize.word_tokenize(str(sent))) for sent in en_corpus]

'''

with open("en-ko_1.tmx", 'rb') as fin:
    tmx_file = tmxfile(fin, 'en', 'ko')

ko_corpus = []
en_corpus = []
    
for node in tmx_file.unit_iter():
    ko_corpus.append(" ".join(tokenize.word_tokenize(node.gettarget())))
    en_corpus.append(" ".join(tokenize.word_tokenize(node.getsource())))

'''

with open("korean-english-park.train.ko", 'r', encoding = "utf-8") as f:
    for sent in f:
        ko_corpus.append(" ".join(tokenize.word_tokenize(sent)))

with open("korean-english-park.train.en", 'r', encoding = "utf-8") as f:
    for sent in f:
        en_corpus.append(" ".join(tokenize.word_tokenize(sent)))

'''

df = pd.DataFrame({'ko':ko_corpus, 
                   'en':en_corpus})

df = shuffle(df)


ko_train = list(df['ko'])[2000:]
en_train = list(df['en'])[2000:]

ko_test = list(df['ko'])[:2000]
en_test = list(df['en'])[:2000]



if(not os.path.exists("./ko2en")):
    os.mkdir("./ko2en")

with open('./ko2en/train.ko' ,'wb') as f:
    pickle.dump(ko_train, f)

with open('./ko2en/train.en' ,'wb') as f:
    pickle.dump(en_train, f)

with open('./ko2en/train' , 'w', encoding = 'utf-8') as f:    
    for sent in en_train+ko_train:
        f.write(sent+"\n")
   
train = '--input=ko2en/train --pad_id=0 --unk_id=1 \
             --bos_id=2 --eos_id=3\
             --model_prefix=bpe \
             --vocab_size={} \
             --model_type=bpe'.format(30000)
             
spm.SentencePieceTrainer.Train(train)

ko_test, en_test = [], []

with open("korean-english-park.test.ko", 'r', encoding = "utf-8") as f:
    for sent in f:
        ko_test.append(" ".join(tokenize.word_tokenize(sent)))

with open("korean-english-park.test.en", 'r', encoding = "utf-8") as f:
    for sent in f:
        en_test.append(" ".join(tokenize.word_tokenize(sent)))

with open('./ko2en/test.ko' ,'wb') as f:
    pickle.dump(ko_test, f)

with open('./ko2en/test.en' ,'wb') as f:
    pickle.dump(en_test, f)


ko_dev, en_dev = [], []

with open("korean-english-park.dev.ko", 'r', encoding = "utf-8") as f:
    for sent in f:
        ko_dev.append(" ".join(tokenize.word_tokenize(sent)))

with open("korean-english-park.dev.en", 'r', encoding = "utf-8") as f:
    for sent in f:
        en_dev.append(" ".join(tokenize.word_tokenize(sent)))

with open('./ko2en/dev.ko' ,'wb') as f:
    pickle.dump(ko_dev, f)

with open('./ko2en/dev.en' ,'wb') as f:
    pickle.dump(en_dev, f)

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