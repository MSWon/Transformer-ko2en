# Transformer-ko2en
**Transformer** 모델을 이용한 한->영 번역모델 패키지입니다

## 설치 방법

- git clone 및 setup.py 파일을 통해 설치합니다
```
$ git clone https://github.com/MSWon/Transformer-ko2en.git
```
```
$ cd Transformer-ko2en
$ python setup.py install
```

- nmt download 명령어를 통해 학습 데이터 및 학습된 모델을 다운받습니다
```
$ nmt download -m data
$ nmt download -m model
```

## Python 패키지 사용방법

### nmt tokenizer
```python
>>> from nmt.nmttokenize.tokenizer import Tokenizer
>>> tok = Tokenizer()
```
- BPE tokenizer를 학습합니다
```python
>>> tok.train("./train.en", "bpe.en", 32000)
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
- tokenizer를 사용해 문장을 토큰나이징합니다
```python
>>> tok.load("bpe.en.model")
>>> tok.tokenize("I use sentencepiece model for tokenizing words")
['▁I', '▁use', '▁sentence', 'piece', '▁model', '▁for', '▁token', 'izing', '▁words']
```

### nmt train
- config 파일을 준비합니다
```python
>>> import yaml
>>> hyp_args = yaml.load(open("./train_config.yaml"))
```
- Trainer 모듈로부터 모델을 생성합니다
```python
>>> from nmt.trainer import Trainer
>>> model = Trainer(hyp_args)
Now building model
Building model tower_1
Could take few minutes
Now building model
Building model tower_2
Could take few minutes
Now building model
Building model tower_3
Could take few minutes
...
```
- 모델을 학습합니다
```python
>>> model.train(hyp_args["training_steps"])
Now training
```

### nmt infer
- Translate 모듈로부터 모델을 생성합니다
```python
>>> from nmt.translate import Translate
>>> model = Translate(hyp_args)
```
- 문장을 직접 모델에 입력하여 번역문을 확인합니다
```python
>>> model.service_infer("나는 학교에 간다.")
'I go to school.'
```
```python
>>> model.service_infer("인공신경망의 발달로 인해 높은 품질의 번역이 가능해졌습니다.")
'The development of the artificial neural network has enabled high-quality translation.'
```
