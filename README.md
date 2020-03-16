# Transformer-ko2en
Translation using Transformer Model for ko2en dataset
```python
>>> from nmt.nmttokenize.tokenizer import Tokenizer
>>> tok = Tokenizer()
```
```python
>>> tok.train("./train.en", "bpe.en", 32000)
```
```
training BPE model

input=./train.en 
pad_id=0 
unk_id=1                     
bos_id=2 
eos_id=3                     
model_prefix=bpe.en                     
user_defined_symbols=<URL>                     
vocab_size=32000                     
model_type=bpe
```

```python
>>> tok.load("bpe.en.model")
>>> tok.tokenize("I use sentencepiece model for tokenizing words")
```
```
['▁I', '▁use', '▁sentence', 'piece', '▁model', '▁for', '▁token', 'izing', '▁words']
```

