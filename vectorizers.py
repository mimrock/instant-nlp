import logging
import math

import numpy as np
from tokenizers import WhiteSpaceTokenizer
from preprocessors import EmptyPreprocessor

class LabelVectorizer():
    def __init__(self):
        pass


    def update_alphabet(self, target):
        return

    def vectorize_target(self, target):
        return target

    def vector_length(self):
        return 1


class NGramVectorizer():

    def __init__(self, size=2, min_occurence = 0,
                 tokenizer=WhiteSpaceTokenizer(),
                 target_vectorizer=LabelVectorizer(),
                 preprocessor=EmptyPreprocessor()):

        self.alphabet_map = {}
        self.alphabet = []
        self.target_vectorizer = target_vectorizer
        self.size = size
        self.data = []
        self.min_occurence = min_occurence
        self.preprocessor = preprocessor

        self.tokenizer = tokenizer


    def prepare(self, data):
        for sample in data.samples():
            tokens = self.tokenizer.tokenize(self.preprocessor.preprocess(sample.text))
            self.update_alphabet(tokens)
            # necessary for e.g. One-hot encoding
            self.target_vectorizer.update_alphabet(tokens)


        logging.debug("Alphabet size before purging: %s", len(self.alphabet))

        self.purge_alphabet()

        # logging.debug("Creating arrays for vectors. X shape: %s", (len(self.data.samples()), len(self.alphabet)))


    def vectorize(self, sample):
        tokens = self.tokenizer.tokenize(self.preprocessor.preprocess(sample.text))
        ngrams = self.split_to_ngrams(tokens)
        v = self.tf_idf(ngrams)
        return v

    def vectorize_target(self, sample):
        target = self.target_vectorizer.vectorize_target(sample.label)
        return target

    def tf_idf(self, ngrams):
        # @todo better tf_idf models
        # @todo more efficient algorithm
        # @todo use a separate scorer
        v = np.zeros(len(self.alphabet),)
        for i in range(0, len(self.alphabet)):
            if self.alphabet[i] in ngrams:
                v[i] += 1.0

        for i in range(len(v)):
            v[i] = math.log(1 + v[i])

        return v


    def split_to_ngrams(self, tokens):
        ngrams = []
        for i in range(len(tokens) - self.size):
            start = i
            end = i + self.size
            ngrams.append(tuple(tokens[start:end]))

        return ngrams


    def update_alphabet(self, text):
        tokens = self.split_to_ngrams(text)
        for ngram in tokens:
            if ngram not in self.alphabet_map:
                self.alphabet_map[ngram] = 1
            else:
                self.alphabet_map[ngram] += 1
        self.alphabet = list(self.alphabet_map.keys())


    def purge_alphabet(self):
        new_map = {}
        for ngram in self.alphabet_map:
            if self.alphabet_map[ngram] >= self.min_occurence:
                new_map[ngram] = self.alphabet_map[ngram]

        self.alphabet_map = new_map
        self.alphabet = list(self.alphabet_map.keys())

    def vector_len(self):
        return len(self.alphabet)