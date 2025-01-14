import sentencepiece as spm

spm.SentencePieceTrainer.train(input='data/train.txt', model_prefix='bpe_model', vocab_size=10000)