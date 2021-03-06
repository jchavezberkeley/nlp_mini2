# models.py

import numpy as np
import random
from sentiment_data import *
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1

class SentFFNN(nn.Module):
    def __init__(self, input, hid, output):
        super(SentFFNN, self).__init__()
        self.V = nn.Linear(input, hid)
        #self.g = nn.Tanh()
        self.g = nn.ReLU()
        self.W = nn.Linear(hid, output)
        self.log_softmax = nn.LogSoftmax(dim=0)
        
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x):
        return self.log_softmax(self.W(self.g(self.V(x))))



class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, model, word_vectors: WordEmbeddings, isLSTM=False, embeddings=None):
        self.model = model
        self.word_vectors = word_vectors
        self.isLSTM = isLSTM
        self.embeddings = embeddings

    def predict(self, ex_words: List[str]) -> int:
        ex_len = len(ex_words)
        feat_vec_size = self.word_vectors.get_embedding_length()
        
        if self.isLSTM:
            word_indices = np.zeros(ex_len, dtype=int)
            for i in range(ex_len):
                index_from_indexer = self.word_vectors.word_indexer.index_of(ex_words[i])
                if index_from_indexer == -1:
                    index_to_append = 1
                else:
                    index_to_append = index_from_indexer
                word_indices[i] = index_to_append

            word_indices = torch.from_numpy(word_indices).type(torch.LongTensor)
            embed_input = self.embeddings(word_indices)
            log_probs = self.model.forward(embed_input)
            prediction = torch.argmax(log_probs)
            
        else:
            avg_embed = np.zeros(feat_vec_size)
            for i in range(ex_len):
                avg_embed += self.word_vectors.get_embedding(ex_words[i])
            avg_embed /= ex_len

            avg_embed_input = torch.from_numpy(avg_embed).float()

            log_probs = self.model.forward(avg_embed_input)
            prediction = torch.argmax(log_probs)
        
        return prediction



def pad_to_length(np_arr, length):
    """
    Forces np_arr to length by either truncation (if longer) or zero-padding (if shorter)
    :param np_arr:
    :param length: Length to pad to
    :return: a new numpy array with the data from np_arr padded to be of length length. If length is less than the
    length of the base array, truncates instead.
    """
    result = np.zeros(length)
    result[-np_arr.shape[0]:-1] = np_arr
    return result


class SentLSTM(nn.Module):
    def __init__(self, inp, hid, out, seq_max_len):
        super(SentLSTM, self).__init__()
        self.num_layers = 1
        self.input_size = inp
        self.hidden_size = hid
        self.output_size = out
        self.seq_max_len = seq_max_len
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        input_x = torch.unsqueeze(x, 0)

        h_0 = torch.zeros(self.num_layers, input_x.size(0), self.hidden_size) #hidden state
        c_0 = torch.zeros(self.num_layers, input_x.size(0), self.hidden_size) #internal state

        lstm_out, (hn, cn) = self.lstm(input_x, (h_0, c_0))
        hidden = hn.squeeze()

        out_space = self.linear(hidden)
        tag_scores = F.log_softmax(out_space, dim=0)
        return tag_scores


def train_ffnn(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_vectors: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Train a feedforward neural network on the given training examples, using dev_exs for development, and returns
    a trained NeuralSentimentClassifier.
    :param args: Command-line args so you can access them here
    :param train_exs:
    :param dev_exs:
    :param word_vectors:
    :return: the trained NeuralSentimentClassifier model -- this model should wrap your PyTorch module and implement
    prediction on new examples
    """
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60 # This will come in handy for padding I think

    lr_rate = 0.0001 # This will need to be adjusted
    num_epochs = 15
    hidden_size = 200
    #hidden_size = args.hidden_size ## This will need to be adjusted
    batch_size = args.batch_size

    num_train_exs = len(train_exs)
    num_labels = 2

    # Setting up input side and hidden size in SentFFNN
    feat_vector_size = word_vectors.get_embedding_length()
    ffnn = SentFFNN(feat_vector_size, hidden_size, num_labels)

    # Setting up optimizer
    optimizer = optim.Adam(ffnn.parameters(), lr=lr_rate)

    for epoch in range(num_epochs):
        ex_indices = [i for i in range(num_train_exs)]
        random.shuffle(ex_indices)

        total_loss = 0.0
        for idx in ex_indices:

            sent_ex = train_exs[idx]
            sent_words = sent_ex.words
            sent_label = sent_ex.label
            sent_ex_len = len(sent_words)

            # Averaging word embeddings across all words in the sentence
            avg_embed = np.zeros(feat_vector_size)
            for j in range(sent_ex_len):
                avg_embed += word_vectors.get_embedding(sent_words[j])
            avg_embed /= sent_ex_len

            avg_embed_input = torch.from_numpy(avg_embed).float()

            y_onehot = torch.zeros(num_labels)
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(sent_label,dtype=np.int64)), 1)

            ffnn.zero_grad()
            log_probs = ffnn.forward(avg_embed_input)
            loss = torch.neg(log_probs).dot(y_onehot)

            total_loss += loss
            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
    
    return NeuralSentimentClassifier(ffnn, word_vectors)


# Analogous to train_ffnn, but trains your fancier model.
def train_fancy(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_vectors: WordEmbeddings) -> NeuralSentimentClassifier:
    lr_rate = args.lr
    num_epochs = args.num_epochs
    hidden_size = args.hidden_size
    
    batch_size = args.batch_size

    num_train_exs = len(train_exs)
    num_labels = 2
    seq_max_len = 60

    # Setting up input side and hidden size in SentLSTM
    feat_vector_size = word_vectors.get_embedding_length()
    lstm_model = SentLSTM(feat_vector_size, hidden_size, num_labels, seq_max_len)

    # Setting up optimizer
    optimizer = optim.Adam(lstm_model.parameters(), lr=lr_rate)

    embedding_weights = torch.FloatTensor(word_vectors.vectors)
    embeddings = nn.Embedding.from_pretrained(embedding_weights)
    embeddings.weight.requires_grad = False

    for epoch in range(num_epochs):
        ex_indices = [i for i in range(num_train_exs)]
        random.shuffle(ex_indices)

        total_loss = 0.0
        for idx in ex_indices:
            sent_ex = train_exs[idx]
            sent_words = sent_ex.words
            sent_label = sent_ex.label
            sent_ex_len = len(sent_words)

            word_indices = np.zeros(sent_ex_len, dtype=int)
            for i in range(sent_ex_len):
                index_from_indexer = word_vectors.word_indexer.index_of(sent_words[i])
                if index_from_indexer == -1:
                    index_to_append = 1
                else:
                    index_to_append = index_from_indexer
                word_indices[i] = index_to_append

            word_indices = torch.from_numpy(word_indices).type(torch.LongTensor)
            embed_input = embeddings(word_indices)

            y_onehot = torch.zeros(num_labels)
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(sent_label,dtype=np.int64)), 1)

            lstm_model.zero_grad()
            log_probs = lstm_model.forward(embed_input)
            loss = torch.neg(log_probs).dot(y_onehot)

            total_loss += loss
            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))

    return NeuralSentimentClassifier(lstm_model, word_vectors, True, embeddings)
