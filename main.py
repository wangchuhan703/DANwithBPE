# models.py

import torch
from nltk import infile
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, read_word_embeddings
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import SentimentDatasetDAN, DAN, SentimentDatasetSubwordDAN, SentimentDataset_bpe_sentencepiece

from torch.nn.utils.rnn import pad_sequence

import sentencepiece as spm

# from BPE import build_vocab_with_bpe

def collate_fn(batch):
    embeddings, labels = zip(*batch)
    embeddings_padded = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in embeddings],
                                     batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return embeddings_padded, labels


# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        # X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        # X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    # loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')

    # Parse the command-line arguments
    args = parser.parse_args()


    # Check if the model type is "BOW"
    if args.model == "BOW":

        # Load dataset
        start_time = time.time()

        train_data = SentimentDatasetBOW("data/train.txt")
        dev_data = SentimentDatasetBOW("data/dev.txt")
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")

        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model == "DAN":
        #TODO:  Train and evaluate your DAN
        # print("DAN model not implemented yet")
        glove_embeddings = "data/glove.6B.50d-relativized.txt"
        word_embeddings = read_word_embeddings(glove_embeddings)
        word_indexes = word_embeddings.word_indexer
        vocab_size = len(word_indexes)
        embedding_dim = word_embeddings.get_embedding_length()
        # print(f'Vocab size: {vocab_size}')

        # Load dataset
        start_time = time.time()

        train_data = SentimentDatasetDAN("data/train.txt", word_indexes)
        dev_data = SentimentDatasetDAN("data/dev.txt", word_indexes)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False, collate_fn=collate_fn)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")

        # Train and evaluate DAN
        start_time = time.time()


        # 1a
        dan_train_accuracy, dan_test_accuracy = experiment(
            DAN(word_embeddings, vocab_size, embedding_dim, hidden_dim=100, output_dim=2), train_loader, test_loader)
        # 1b
        # dan_train_accuracy, dan_test_accuracy = experiment(
        #     DAN(None, vocab_size, embedding_dim, hidden_dim=100, output_dim=2), train_loader, test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for DAN Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'dan_train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_test_accuracy, label='DAN')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for DAN Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dan_dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()


    elif args.model == "SUBWORDDAN":

        embedding_dim = 50

        train_data = SentimentDatasetSubwordDAN("data/train.txt", "subword_vocab.txt")
        vocab_size = train_data.subword_vocab_length(subword_vocab_file="subword_vocab.txt")
        print(vocab_size)
        dev_data = SentimentDatasetSubwordDAN("data/dev.txt", "subword_vocab.txt")
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False, collate_fn=collate_fn)

        dan_train_accuracy, dan_test_accuracy = experiment(
            DAN(None, vocab_size, embedding_dim, hidden_dim=100, output_dim=2), train_loader, test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for DAN Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'bpe_train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_test_accuracy, label='DAN')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for DAN Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'bpe_dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()



    # elif args.model == "bpe_sentencepiece":
    #     start_time = time.time()
    #     bpe_processor = spm.SentencePieceProcessor(model_file='bpe_model.model')
    #     # Load dataset
    #
    #     train_data = SentimentDataset_bpe_sentencepiece("data/train.txt", bpe_processor=bpe_processor)
    #     dev_data = SentimentDataset_bpe_sentencepiece("data/dev.txt", bpe_processor=bpe_processor)
    #     train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
    #     test_loader = DataLoader(dev_data, batch_size=16, shuffle=False, collate_fn=collate_fn)
    #
    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
    #     print(f"Data loaded in : {elapsed_time} seconds")
    #
    #
    #     vocab_size = bpe_processor.get_piece_size()
    #     embed_size = 50
    #
    #     dan_train_accuracy, dan_test_accuracy = experiment(
    #         DAN(None, vocab_size, embed_size, hidden_dim=100, output_dim=2), train_loader, test_loader)
    #
    #     # Plot the training accuracy
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(dan_train_accuracy, label='3 layers')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Training Accuracy')
    #     plt.title('Training Accuracy for DAN Networks')
    #     plt.legend()
    #     plt.grid()
    #
    #     # Save the training accuracy figure
    #     training_accuracy_file = 'dan_bpe_train_accuracy.png'
    #     plt.savefig(training_accuracy_file)
    #     print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")
    #
    #     # Plot the testing accuracy
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(dan_test_accuracy, label='DAN')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Dev Accuracy')
    #     plt.title('Dev Accuracy for DAN Networks')
    #     plt.legend()
    #     plt.grid()
    #
    #     # Save the testing accuracy figure
    #     testing_accuracy_file = 'dan_bpe_dev_accuracy.png'
    #     plt.savefig(testing_accuracy_file)
    #     print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

if __name__ == "__main__":
    main()
