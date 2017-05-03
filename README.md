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

```
spm_train  --input target.txt --model_prefix english --vocab_size 16000
spm_train  --input source.txt --model_prefix japanese --vocab_size 16000
```

```
spm_encode --model english.model --output_format=piece < target.txt > english.txt
spm_encode --model japanese.model --output_format=piece < source.txt > japanese.txt
```