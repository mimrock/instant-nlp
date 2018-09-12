from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.initializers import RandomNormal

import time
import math

class NeuralRegressor():


    def __init__(self, input_dim, output_dim):
        self.verbose = 2
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.model = self.get_model()


    def get_model(self):
        start_time = time.time()

        model = Sequential()

        i = RandomNormal(mean=0.0, stddev=math.sqrt(1/self.input_dim), seed=None)

        model.add(Dense(self.input_dim, input_dim=self.input_dim, activation='selu', init=i))
        # model.add(Dropout(0.4))
        # model.add(Dense(self.input_dim, activation='selu', init=i))
        # model.add(Dropout(0.5))
        # model.add(Dense(400, activation='selu', init=i))
        # model.add(Dense(1044, activation='selu', init='normal'))
        model.add(Dense(self.output_dim, activation='softmax', init=i))

        '''model.add(Dense(100, input_dim=self.input_dim, activation='relu', init='normal'))
        model.add(Dense(200, activation='relu', init='normal'))
        model.add(Dropout(0.4))
        model.add(Dense(500, activation='relu', init='normal'))
        model.add(Dense(self.output_dim, activation='linear', init='normal'))'''


        rms = RMSprop()
        model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
        print('Model compield in {0} seconds'.format(time.time() - start_time))
        return model


    def fit(self, X, y):

        print("rebuilding the model")
        self.model = self.get_model()

        self.model.fit(X, y, epochs=12, batch_size=200, verbose=self.verbose)

    def predict(self, X, y=None):
        pred = self.model.predict(X, batch_size=200, verbose=self.verbose)
        # print(pred)
        # p = numpy.array(pred)
        # print("p shape: "p.shape)

        if y is not None:
            p_loss = self.model.evaluate(X, y, batch_size=200, verbose=self.verbose)
            logging.info("Prediction loss: %s", p_loss)

        return pred