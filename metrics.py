import logging
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class ClassificationMetrics:

    def __init__(self):
        pass


    def evaluate(self):

        # y_validate, y_predict = self.fixed_length_validate(500)
        y_validate, y_predict = self.simulator.k_fold_cross_validate(4)
        return self.display_result(y_validate, y_predict)

    def display_result(self, y_true, y_predict):
        labels = set()
        y_p = y_predict.reshape(y_predict.shape[0],)
        for label in y_p:
            # logging.debug("label: %s", label)
            # labels.add(str(label))
            pass

        labels = list(labels)
        logging.debug("labels=%s",labels)
        logging.debug("y_true shape: %s dtype: %s y_predict shape: %s dtype: %s", y_true.shape, y_true.dtype, y_predict.shape, y_predict.dtype)

        m = confusion_matrix(y_true, y_predict)

        print("labels=", labels)
        print(m)
        print("accuracy=%s", accuracy_score(y_true, y_predict))


class RegressionMetrics():

    def __init__(self):
        pass


    def mse(self, true, predict):
        return (true - predict) ** 2


    def mae(self, true, predict):
        return abs(true - predict)

    def greater(self, a, limit):
        n = 0
        for i in range(len(a)):
            if a[i] > limit:
                n += 1

        return n / len(a)

    def display_result(self, y_true, y_predict):
        y_t = y_true.reshape(y_predict.shape[0], )
        y_p = y_predict.reshape(y_predict.shape[0], )
        mse = []
        mae = []
        for i,pred in enumerate(y_p):
            mse.append(self.mse(y_t[i], y_p[i]))
            mae.append(self.mae(y_t[i], y_p[i]))

        #@todo percentiles
        logging.info("average mse: %s median mse: %s percentile(60): %s percentile(75): %s mse bigger than 0.49: %s",
                     np.mean(mse), np.median(mse), np.percentile(mse, 60), np.percentile(mse, 75), self.greater(mse, 0.49))
        logging.info("average mae: %s median mae: %s percentile(60): %s percentile(75): %s mae bigger than 0.49: %s",
                     np.mean(mae), np.median(mae), np.percentile(mae, 60), np.percentile(mae, 75), self.greater(mae, 0.49))


class OneHotRankingMetrics():

    def process_result(self, y_validate, y_predict):
        tops = {}
        for i in range(0, len(y_predict)):
            author = np.nonzero(y_validate[i])[0][0]
            better = 0
            for j, guess in enumerate(y_predict[i]):
                if guess >= y_predict[i][author] and j != author:
                    better += 1
            if better not in tops:
                tops[better] = 1
            else:
                tops[better] += 1

        logging.info("There was %d samples.", len(y_predict))

        for better in tops:  # / len(y_predict)
            logging.info("In %s of the cases %s wrong guess scored higher than the right.", tops[better], better)

        best_10 = 0
        best_half = 0
        h = len(tops) // 2
        for better in tops:
            if better < 10:
                best_10 += tops[better]
            if better < h:
                best_half += tops[better]

        logging.info("\nBingo: %d - %f \nTop10: %d %f\ntop%d: %d %f",
                     tops[0], tops[0]/len(y_predict), best_10, best_10 / len(y_predict), h, best_half, best_half / len(y_predict))