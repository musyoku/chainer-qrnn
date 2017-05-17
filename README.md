in progress ...

todo:
- [x] f-pooling
- [x] fo-pooling
- [x] ifo-pooling
- [x] zoneout
- [x] encoder-decoder
- [ ] seq2seq
- [ ] seq2seq with attention
- [ ] experiments

[京都フリー翻訳タスク (KFTT)](http://www.phontron.com/kftt/index-ja.html#dataonly)

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
spm_encode --model english.model --output_format=piece < train.ja > train.ja.txt
spm_encode --model english.model --output_format=piece < dev.ja > dev.ja.txt
spm_encode --model english.model --output_format=piece < test.ja > test.ja.txt
```