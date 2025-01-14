import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, WordEmbeddings, read_word_embeddings
from torch.utils.data import Dataset

class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, word_indexer):
        self.examples = read_sentiment_examples(infile)

        self.sentences = [ex.words for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]

        self.word_indexer = word_indexer
        self.embeddings = [
            [self.word_indexer.index_of(word) if self.word_indexer.index_of(word) != -1 else self.word_indexer.index_of(
                "UNK") for word in sentence]
            for sentence in self.sentences
        ]

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]




class SentimentDatasetSubwordDAN(Dataset):

    def __init__(self, infile, subword_vocab_file):
        self.examples = read_sentiment_examples(infile)
        self.sentences = [ex.words for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]

        # 加载子词词典（按出现顺序生成索引）
        self.subword_vocab = self.load_subword_vocab(subword_vocab_file)

        # 对每个句子生成嵌入
        self.embeddings = [
            [index for word in sentence for index in self.get_subword_indices(word)]
            for sentence in self.sentences
        ]

        self.labels = torch.tensor(self.labels, dtype=torch.long)


    def subword_vocab_length(self, subword_vocab_file):
        subword_vocab = self.load_subword_vocab(subword_vocab_file)
        return len(subword_vocab)

    def load_subword_vocab(self, subword_vocab_file):
        subword_vocab = {}
        with open(subword_vocab_file, 'r') as f:
            for idx, line in enumerate(f):
                subword = line.strip().split('\t')[0]  # 只取子词部分
                subword_vocab[subword] = idx  # 使用行号作为索引
        return subword_vocab


    def get_subword_indices(self, word):
        subword_indices = []
        i = 0
        while i < len(word):
            longest_subword = None
            longest_subword_index = -1

            for j in range(i + 1, len(word) + 1):
                subword_candidate = word[i:j] + ('</w>' if j == len(word) else '')
                if subword_candidate in self.subword_vocab:
                    longest_subword = subword_candidate
                    longest_subword_index = self.subword_vocab[subword_candidate]

            if longest_subword:
                subword_indices.append(longest_subword_index)
                i += len(longest_subword.replace('</w>', ''))
            else:
                subword_indices.append(0)  # 默认返回索引0
                i += 1

        return subword_indices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]





class SentimentDataset_bpe_sentencepiece(Dataset):
    def __init__(self, infile, bpe_processor, train=True):
        # Read sentiment examples
        self.examples = read_sentiment_examples(infile)
        self.sentences = [ex.words for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]
        self.bpe_processor = bpe_processor
        self.train = train

        self.bpe_sentences = [' '.join(sentence) for sentence in self.sentences]

        # Convert labels to tensor
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        indexed_sentence = torch.tensor(self.bpe_processor.encode(self.bpe_sentences[idx], out_type=int), dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return indexed_sentence, label


class DAN(nn.Module):
    def __init__(self, word_embeddings, vocab_size, embedding_dim, hidden_dim: int=50, output_dim: int=2):
        super(DAN, self).__init__()
        self.embedding_dim = embedding_dim
        if word_embeddings is not None:
            self.embedding = word_embeddings.get_initialized_embedding_layer(frozen=False)
        else:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_dim, padding_idx=0)

        self.fn1 = nn.Linear(self.embedding_dim, hidden_dim)
        self.fn3 = nn.Linear(hidden_dim, hidden_dim)
        self.fn2 = nn.Linear(hidden_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, word_indices):
        embedded = self.embedding(word_indices)
        averaged = torch.mean(embedded, dim=1)

        hidden_output = F.relu(self.fn1(averaged))
        hidden_output = F.relu(self.fn3(hidden_output))
        output = self.fn2(hidden_output)
        return self.log_softmax(output)


