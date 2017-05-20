# QUASI-RECURRENT NEURAL NETWORKS

- code for the [paper](https://arxiv.org/abs/1611.01576v1)

## Requirements

- Chainer 2.0+
- Python 2 or 3

## language Modeling

### Dataset

[Penn Tree Bank](https://github.com/wojzaremba/lstm/tree/master/data)

### Usage

```
cd rnn
```

```
python train.py -train data/train.txt -dev data/dev.txt -lr 0.1 -dense -zoneout -dropout -b 64 -wd 2e-5
```

```
python error.py -dev data/dev.txt -test data/test.txt 
```

```
python generate.py -n 10
```

## Neural Machine Translation

### Dataset

[京都フリー翻訳タスク (KFTT)](http://www.phontron.com/kftt/index-ja.html#dataonly)

### Preprocessing

We use [SentencePiece](https://github.com/google/sentencepiece) to tokenize text.

```
cat dev.en train.en test.en > english.txt
cat dev.ja train.ja test.ja > japanese.txt
spm_train  --input english.txt --model_prefix english --vocab_size 16000
spm_train  --input japanese.txt --model_prefix japanese --vocab_size 16000
```

```
spm_encode --model english.model --output_format=piece < train.en > train.en.txt
spm_encode --model english.model --output_format=piece < dev.en > dev.en.txt
spm_encode --model english.model --output_format=piece < test.en > test.en.txt
spm_encode --model japanese.model --output_format=piece < train.ja > train.ja.txt
spm_encode --model japanese.model --output_format=piece < dev.ja > dev.ja.txt
spm_encode --model japanese.model --output_format=piece < test.ja > test.ja.txt
```

### Usage

```
cd seq2seq
```

```
python train.py --source-train data/train.ja.txt --target-train data/train.en.txt --source-dev data/dev.ja.txt --target-dev data/dev.en.txt --source-test data/test.ja.txt --target-test data/test.en.txt --batchsize 64 -zoneout -dense -attention -lr 0.1
```

```
python error.py --source-test data/test.ja.txt --target-test data/test.en.txt -beam 8 -alpha 0.6
```

```
python translate.py --source-test data/test.ja.txt  -beam 8 -alpha 0.6
```