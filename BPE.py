import collections
import re
from sentiment_data import read_sentiment_examples, read_word_embeddings

def build_vocab_with_bpe(training_data_path, num_merges=5000):
    examples = read_sentiment_examples(training_data_path)
    vocab = collections.defaultdict(int)

    # 构建初始词汇表，逐个字符表示
    for ex in examples:
        for word in ex.words:
            vocab[' '.join(list(word)) + ' </w>'] += 1  # 每个字符之间用空格分隔，并在结尾添加</w>

    # 进行BPE合并
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)

        # 限制 vocab 的大小为当前的 merge 次数，保留最频繁的词汇
        if len(vocab) > num_merges:
            sorted_vocab = sorted(vocab.items(), key=lambda item: item[1], reverse=True)
            vocab = dict(sorted_vocab[:num_merges])

    # vocab["UNK"] = 1
    # 将词汇表保存到文件中
    with open('subword_vocab.txt', 'w') as f:
        for word, freq in vocab.items():
            f.write(f'{word}\t{freq}\n')
    print("finished generating subword_vocab.txt")

    return vocab



def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

build_vocab_with_bpe('data/train.txt', num_merges=20000)