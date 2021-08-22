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
Now downloading file 'data.tar.gz'
6311it [00:09, 676.18it/s]

Now unpacking file 'data.tar.gz'
```

```
$ tree ./data
data
├── bpe.en.model
├── bpe.en.vocab
├── bpe.ko.model
├── bpe.ko.vocab
├── train.en.bpe
├── train.ko.bpe
├── tst2016.en
├── tst2016.en.bpe
├── tst2016.ko
├── tst2016.ko.bpe
├── tst2017.en
├── tst2017.en.bpe
├── tst2017.ko
└── tst2017.ko.bpe
```

```
$ nmt download -m model
Now downloading file 'koen.2021.0704.tar.gz'
10594it [00:06, 1568.49it/s]

Now unpacking file 'koen.2021.0704.tar.gz'
```

```
$ tree ./koen.2021.0704
koen.2021.0704/
├── exported.model
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── service_config.yaml
└── tokenizer
    ├── bpe.model.en
    ├── bpe.model.ko
    ├── vocab.en
    └── vocab.ko
```

## config 파일

### train_config

- `examples/configs/train_config.yaml`

```yaml
## Data config
train_src_corpus_path: ./data/train.ko.bpe
train_tgt_corpus_path: ./data/train.en.bpe
test_src_corpus_path: ./data/tst2016.ko.bpe
test_tgt_corpus_path: ./data/tst2016.en.bpe
src_vocab_path: ./data/bpe.ko.vocab
tgt_vocab_path: ./data/bpe.en.vocab
src_bpe_model_path: ./data/bpe.ko.model
tgt_bpe_model_path: ./data/bpe.en.model
## Training config
training_steps: 100000
warmup_step: 4000
max_len: 50
batch_size: 40000
bos_idx: 2
eos_idx: 3
n_gpus: 8
model_path: transformer_ko2en.ckpt
## Model config 
num_layers: 6
num_heads: 8
hidden_dim: 512
linear_key_dim: 512
linear_value_dim: 512
ffn_dim: 2048
dropout: 0.1
vocab_size: 32000 
shared_dec_inout_emb: False
```

### service_config

- `examples/configs/service_config.yaml`

```yaml
model_version: koen.2021.0704
src_vocab_path: vocab.ko
tgt_vocab_path: vocab.en
src_bpe_model_path: bpe.model.ko
tgt_bpe_model_path: bpe.model.en
max_len: 50
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
>>> model.train()
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
$ nmt infer -c ./koen.2021.0704/service_config.yaml
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
$ nmt service -m -c ./koen.2021.0704/service_config.yaml website -p 6006
Now loading 'koen.2021.0704' model
====================================================================================================
model_version : koen.2021.0704
src_vocab_path : vocab.ko
tgt_vocab_path : vocab.en
src_bpe_model_path : bpe.model.ko
tgt_bpe_model_path : bpe.model.en
max_len : 50
config_path : ./koen.2021.0704/service_config.yaml
====================================================================================================

mapping : inputs -> Placeholder:0
mapping : outputs -> Squeeze:0
 * Serving Flask app "app_restapi" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Running on all addresses.
   WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://0.0.0.0:6006/ (Press CTRL+C to quit)
 * Restarting with stat
```

### nmt service for rest-api & serving

- nmt service 모드를 `api`로 입력합니다
- 원하는 port 번호를 입력합니다

```
$ nmt service -m api -c ./koen.2021.0704/service_config.yaml -p 6006
Now loading 'koen.2021.0704' model
====================================================================================================
model_version : koen.2021.0704
src_vocab_path : vocab.ko
tgt_vocab_path : vocab.en
src_bpe_model_path : bpe.model.ko
tgt_bpe_model_path : bpe.model.en
max_len : 50
config_path : ./koen.2021.0704/service_config.yaml
====================================================================================================

mapping : inputs -> Placeholder:0
mapping : outputs -> Squeeze:0
 * Serving Flask app "app_restapi" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Running on all addresses.
   WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://0.0.0.0:6006/ (Press CTRL+C to quit)
 * Restarting with stat
```

- nmt service 모드를 `serving`로 입력합니다
- 원하는 port 번호를 입력합니다

```
$ nmt service -m serving -c ./koen.2021.0704/service_config.yaml -p 6006
[2021-08-22 19:57:56 +0900] [11649] [INFO] Starting gunicorn 20.1.0
[2021-08-22 19:57:56 +0900] [11649] [INFO] Listening at: http://0.0.0.0:6006 (11649)
[2021-08-22 19:57:56 +0900] [11649] [INFO] Using worker: sync
[2021-08-22 19:57:56 +0900] [11674] [INFO] Booting worker with pid: 11674
WARNING:tensorflow:From /Users/user/Desktop/minsub/python_code/Transformer-ko2en/nmt/nmttrain/utils/model_utils.py:122: The name tf.layers.Layer is deprecated. Please use tf.compat.v1.layers.Layer instead.

/Users/user/Desktop/minsub/python_code/Transformer-ko2en/nmt/serving/api.py:44: YAMLLoadWarning: calling yaml.load() without Loader=... is deprecated, as the default Loader is unsafe. Please read https://msg.pyyaml.org/load for full details.
  hyp_args = yaml.load(open(config_path))
Now loading 'koen.2021.0704' model
====================================================================================================
model_version : koen.2021.0704
src_vocab_path : vocab.ko
tgt_vocab_path : vocab.en
src_bpe_model_path : bpe.model.ko
tgt_bpe_model_path : bpe.model.en
max_len : 50
config_path : ./koen.2021.0704/service_config.yaml
====================================================================================================

mapping : inputs -> Placeholder:0
mapping : outputs -> Squeeze:0
[2021-08-22 22:07:30 +0900] [11649] [CRITICAL] WORKER TIMEOUT (pid:11674)
[2021-08-22 22:07:30 +0900] [11674] [INFO] Worker exiting (pid: 11674)
[2021-08-22 22:07:31 +0900] [11649] [WARNING] Worker with pid 11674 was terminated due to signal 9
[2021-08-22 22:07:31 +0900] [41950] [INFO] Booting worker with pid: 41950
[2021-08-22 22:07:33 +0900] [11649] [INFO] Handling signal: winch
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
