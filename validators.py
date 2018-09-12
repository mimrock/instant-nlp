import numpy as np
import logging
from sklearn.base import clone

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def str_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


class KfoldCrossValidator():

    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.random_state = 0


    def fixed_length_validate(self, p_size):
        # @todo validator fails if the number of samples is not dividable by p_size
        # to be deleted
        X_validate = self.X[-p_size:]
        X = self.X[:-p_size]
        y_validate = self.y[-p_size:]
        y = self.y[:-p_size]

        # clf = linear_model.LinearRegression()
        self.algorithm.fit(X, y)

        y_predict = self.algorithm.predict(X_validate)
        return y_validate, y_predict

    def validate(self, k_size, X, y_true, shuffle=False):
        if shuffle:
            X, y_true = shuffle(X, y_true, random_state=self.random_state)
        logging.debug("X shape: %s, y shape: %s", X.shape, y_true.shape)
        X_chunks = np.split(X,k_size)
        y_chunks = np.split(y_true,k_size)
        logging.debug("X chunks l: %s, Xc[0]: %s y chunks l: %s yc[0]: %s", len(X_chunks), len(X_chunks[0]), len(y_chunks), len(y_chunks[0]))

        y_validate = np.array([])
        y_predict = np.array([])
        # @todo Remove concatenations since they copy a lot of data.
        for i,y_chunk in enumerate(y_chunks):
            X_train = np.array([])
            y_train = np.array([])
            for j,X_chunk in enumerate(X_chunks):
                if i != j:
                    if len(X_train) == 0:
                        X_train = X_chunks[j]
                        y_train = y_chunks[j]
                    else:
                        X_train = np.concatenate((X_train,X_chunks[j]))
                        y_train = np.concatenate((y_train, y_chunks[j]))

            logging.debug("Fitting the %d. chunk", i)
            algorithm_instance = clone(self.algorithm)
            algorithm_instance.fit(X_train, y_train)
            logging.debug("Predicting the %d. chunk", i)
            y_p = algorithm_instance.predict(X_chunks[i])
            print("k_fold y_p:", type(y_p))
            logging.debug("y_predict_shape: %s, y_p shape: %s", y_predict.shape, y_p.shape)
            if len(y_predict) == 0:
                y_predict = y_p
                y_validate = y_chunks[i]
            else:
                y_predict = np.concatenate((y_predict, y_p))
                y_validate = np.concatenate((y_validate, y_chunks[i]))

        # y_true is required for validation when the result is shuffled
        return y_true, y_predict