import numpy as np
import math
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVR

import csv

from metrics import ClassificationMetrics
from metrics import RegressionMetrics

from vectorizers import NGramVectorizer
from vectorizers import LabelVectorizer

from tokenizers import CharTokenizer

from preprocessors import regexPurgePreprocessor

from validators import KfoldCrossValidator



class Sample():

    def __init__(self, text, label):
        self.text = text
        self.label = label

        self.context = {}

        # self.X = None
        # self.y = None

    def add_context(self, key, value):
        if key in self.context:
            self.context[key].append(value)
        else:
            self.context[key] = [value]

class DataWrapper:

    def __init__(self, text_field, label_field):
        self._samples = [] # Do not write this manually after initialized.
        self.text_field = text_field
        self.label_field = label_field


    def add_doc(self, document):
        # @todo Maybe move this to the constructor since calling this method after something started reading the samples can mess with everything.
        self._samples.append(Sample(document[self.text_field], row[self.label_field]))


    def samples(self):
        return self._samples


class Recipe:

    def __init__(self, vectorizer, algorithm, name):
        self.vectorizer = vectorizer
        self.algorithm= algorithm
        self.name = name


class Ensemble:

    def __init__(self, recipes, data, master_classifier):
        self.data = data
        self.recipes = recipes # It is important that the loop order of self.recipes must be fixed during the processing. Same goes for the samples in the datamodel.
        self.master_classifier = master_classifier

    def do(self):
        for recipe in self.recipes:
            recipe.vectorizer.prepare(self.data) # <- What if different vectorizers need differently parametrized prepatations? (args indenependent from the data can be set in __init__)

            X = np.zeros((len(self.data.samples()), recipe.vectorizer.vector_len()))
            y = np.zeros((len(self.data.samples()), recipe.vectorizer.target_vectorizer.vector_length())) # y_true
            logging.debug("X shape: %s y shape: %s", X.shape, y.shape)

            for i, s in enumerate(self.data.samples()):
                # add vectors(X and y) to the data samples

                X[i] = recipe.vectorizer.vectorize(s)
                y[i] = recipe.vectorizer.vectorize_target(s)

                s.add_context("X_vectors", X[i])
                # s.add_context("target_vectors", y[i])

            logging.debug("Vectors ready.")
            validator = KfoldCrossValidator(algorithm=recipe.algorithm)
            y_true, y_preds = validator.validate(4, X, y, shuffle=False) # Do not shuffle.

            logging.debug("Validating member: %s", recipe.name)
            m = RegressionMetrics()
            m.display_result(y_true, y_preds)

            for i,s in enumerate(self.data.samples()):
                s.add_context("predictions", y_preds[i])


        target_vectorizer = LabelVectorizer()
        X = np.zeros((len(self.data.samples()), len(self.recipes)))
        y_true = np.zeros((len(self.data.samples()), target_vectorizer.vector_length()))
        for i, s in enumerate(self.data.samples()): # <- Example for datamodel, generator
            # @todo Build X, where the features are the predictions. y is just y_true stacked
            # use the simulator and validator to validate the result
            X[i] = np.array(s.context["predictions"])
            y_true[i] = target_vectorizer.vectorize_target(s.label)

        validator = KfoldCrossValidator(algorithm=self.master_classifier)
        _, y_preds = validator.validate(4, X, y, shuffle=False)  # Do not shuffle.

        logging.debug("Evaluating ensemble model.")
        m = ClassificationMetrics()
        m.display_result(y_true, y_preds)



if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    logging.info("Reading reviews")

    '''reviews = []
    # with open('results-20171009-235316.csv', mode='r') as infile:
    with open('train.tsv', mode='r') as infile:
        reader = csv.DictReader(infile, dialect='excel-tab')
        for row in reader:
            # print(rows)
            sample = Sample(row["Phrase"], row["Sentiment"])
            reviews.append(sample)

    logging.info("Vectorization")


    d = DataHandler(reviews)
    logging.debug("Number of labels %d, labels: %s", len(d.text_map.keys()), d.text_map.keys())
    X, y = d.get_vectors()

    logging.info("Startin up ML: shape: %s, y shape: %s", X.shape, y.shape)

    n = NeuralRegressor(X.shape[1], y.shape[1])

    # clf = svm.SVC()
    clf = RandomForestClassifier()



    v = ClassificationValidator(clf, X, y.reshape(y.shape[0],))
    # v = Validator(linear_model.LinearRegression(), X, y)
    # v = Validator(SVR(C=1.0, epsilon=0.2), X, y)

    v.evaluate()'''

    '''d = DataWrapper("Phrase", "Sentiment") # read rotten-tomato data from kaggle
    #reviews = []
    i = 0
    with open('train.tsv', mode='r') as infile:
        reader = csv.DictReader(infile, dialect='excel-tab')
        for row in reader:
            # print(rows)
            #reviews.append(row)
            d.add_doc(row)
            i += 1
            if i >= 100000:
                break


    #@todo this should be automatic
    d._samples = shuffle(d._samples)'''

    d = DataWrapper(1, 0)
    with open('imdb-data/imdb_clas/train.csv') as f:
        r = csv.reader(f)
        for row in r:
            d.add_doc(row)
            if len(d._samples) == 5000:
                break


# @todo better tfidf
# @todo single experiment instead of ensembles.
# @todo merge back info for the master classifier (length(chars, tokens), pca-reduced vectors, confidence scores)
# @todo finish fastai test, try in an ensemble model with the tradional methods

    recipes = [
        Recipe(
            vectorizer=NGramVectorizer(size=3, min_occurence=100,
                                       tokenizer=CharTokenizer(),
                                       target_vectorizer=LabelVectorizer(),
                                       preprocessor=regexPurgePreprocessor(regex="[\W]")),
            algorithm=LinearSVR(), name="charTrigram"),
        Recipe(
            vectorizer=NGramVectorizer(size=1, min_occurence=100,
                                       target_vectorizer=LabelVectorizer(),
                                       preprocessor=regexPurgePreprocessor(regex="[\W]")),
            algorithm=LinearSVR(), name="uniigram"),
        Recipe(
            vectorizer=NGramVectorizer(size=2, min_occurence=100,
                                       target_vectorizer=LabelVectorizer(),
                                       preprocessor=regexPurgePreprocessor(regex="[\W]")),
            algorithm=LinearSVR(), name="bigram"),
        Recipe(
            vectorizer=NGramVectorizer(size=3, min_occurence=75,
                                       target_vectorizer=LabelVectorizer(),
                                       preprocessor=regexPurgePreprocessor(regex="[\W]")),
            algorithm=LinearSVR(), name="trigram")
    ]
    logging.info("Vectorization")
    experiment = Ensemble(recipes, d, RandomForestClassifier())
    experiment.do()

