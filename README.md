# Transformer-ko2en
**Transformer** 모델을 이용한 **한->영 번역모델** 패키지입니다

## 설치 방법

- git clone 및 **setup.py** 파일을 통해 설치합니다
```
$ git clone https://github.com/MSWon/Transformer-ko2en.git
```
```
$ cd Transformer-ko2en
$ python setup.py install
```

- **nmt download** 명령어를 통해 학습 데이터 및 학습된 모델을 다운받습니다
- 학습 데이터, 학습 모델 및 config 파일은 /opt/anaconda3/envs/${user이름}/lib/python3.6/site-packages/nmt-1.0.0-py3.6.egg/nmt에 저장됩니다
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

## CLI 커맨드 사용방법

### nmt tokenizer
- BPE 모델을 준비합니다
```
$ nmt tokenize --model_path ./bpe.en.model --sentence "I use sentencepiece model for tokenizing words"
['▁I', '▁use', '▁sentence', 'piece', '▁model', '▁for', '▁token', 'izing', '▁words']
```

### nmt train
- config 파일을 준비합니다
```
$ nmt train --config_path ./train_config.yaml
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

### nmt infer

```
$ nmt infer --config_path ./train_config.yaml
Now building model
Model loaded!
Input Korean sent : 인공신경망의 발달로 인해 높은 품질의 번역이 가능해졌습니다.
The development of the artificial neural network has enabled high-quality translation.
```

### nmt service for website
- nmt service 모드를 website로 입력합니다
- 원하는 port 번호를 입력합니다
- 브라우저를 키고 ${ip주소}:${port번호}로 접속합니다
```
$ nmt service -m website -p 6006
Now building model
Model loaded!
 * Serving Flask app "app" (lazy loading)
 * Environment: production
 * Debug mode: on
 * Running on http://0.0.0.0:6006/ (Press CTRL+C to quit)
 * Restarting with stat
```

### nmt service for rest-api

- nmt service 모드를 restapi로 입력합니다
- 원하는 port 번호를 입력합니다

```
$ nmt service -m restapi -p 6006
Now building model
Model loaded!
 * Serving Flask app "app" (lazy loading)
 * Environment: production
 * Debug mode: on
 * Running on http://0.0.0.0:6006/ (Press CTRL+C to quit)
 * Restarting with stat
```

- python에서 아래와 같이 요청을 합니다

```python
>> import json
>> import urllib

>> encText = urllib.parse.quote("요즘 어린이들은 유튜브로 뽀로로를 즐겨봅니다")
>> url = "http://아이피 주소:6006/nmt?source=ko&target=en&text={}".format(encText)

>> request = urllib.request.Request(url)
>> response = urllib.request.urlopen(request)
>> response_body = response.read()
>> json_body = json.loads(response_body.decode('utf-8'))

>> print(json_body)
{'srcLangType': 'ko',
 'tgtLangType': 'en',
 'translatedText': 'Nowadays children enjoy Pororo with Youtube these days.'}
```


## 성능

- BLEU 점수
- multi-bleu.perl에 의해 측정

| tst2016.en    | tst2017.en    |
|:-------------:|:-------------:|
| 10.89 | 8.69 |



## 데모

![alt text](https://github.com/MSWon/Transformer-ko2en/blob/develop/img/ko2en_demo.gif)
