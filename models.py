# models.py

import numpy as np
import random
from sentiment_data import *


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


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self):
        raise NotImplementedError


def pad_to_length(np_arr, length):
    """
    Forces np_arr to length by either truncation (if longer) or zero-padding (if shorter)
    :param np_arr:
    :param length: Length to pad to
    :return: a new numpy array with the data from np_arr padded to be of length length. If length is less than the
    length of the base array, truncates instead.
    """
    result = np.zeros(length)
    result[0:np_arr.shape[0]] = np_arr
    return result


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
    seq_max_len = 60
    raise Exception("Not implemented")


# Analogous to train_ffnn, but trains your fancier model.
def train_fancy(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_vectors: WordEmbeddings) -> NeuralSentimentClassifier:
    raise Exception("Not implemented")
