nlptoolkit
==========

This code will be a part of a mini-framework for rapid prototyping of nlp classification tasks.

Today, it serves as a baseline model for the Large Movie Review Dataset available at http://ai.stanford.edu/~amaas/data/sentiment/

There are multiple models (unigram, bigram, trigram, char-level trigram) implemented, and there is an ensemble class that aggregates separate models to improve prediction quality, but it's only the unigram model that yields the expected results (0.83-0.85 depending on the configuration and the number of samples)