import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

# keras imports
from keras import backend as K
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

import shutil
from abc import ABCMeta, abstract_attribute, abstractproperty

class DataModel(metaclass=ABCMeta):
    """ Each of the developed architectures should have a model property
    """
    @abstractproperty
    def model(self):
        pass

class ModelUse(object):
    """ The class to be inherited by the models. It contains the train methods. This class is
    particularly developed for the region of Brabant.
    """
    def __init__(self, model, X_train, X_val, X_test, y_train, y_val, y_test):
        self.model = model
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.predHVB = pd.DataFrame()
        self.predWB = pd.DataFrame()
        self.predBZO = pd.DataFrame()

    def train_model(self, epochs, times, nbest, Perf):
        ''' The train method.
        times: int
            Number of times to train and test the model
        epochs: int
            Number of epochs of the training process.
        nbest: int
            Number of the best models to average their predictions
        Perf: int
            Reduce heuristic methods interval.
        '''

        def reset_weights(model):
            ''' Every time we start the training from scratch we have to reset
            the weights of the model.
            '''
            session = K.get_session()
            for layer in model.layers:
                if hasattr(layer, 'kernel_initializer'):
                    layer.kernel.initializer.run(session=session)

        for time in range(times):
            reset_weights(self.model)

            self.model.fit(self.X_train, self.y_train, epochs=epochs,
                           validation_data=(self.X_val, self.y_val))
            y_predict = np.round(self.model.predict(self.X_test))

            # store the predictions for each region
            self.predHVB[time] = y_predict[:, 0]
            self.predWB[time] = y_predict[:, 1]
            self.predBZO[time] = y_predict[:, 2]


class DeepNN_1(DataModel):
    """ A DeepNN architecture used and tested for our analysis."""
    def __init__(self, numfeatures, numtargets):
        """Here we add the arcitecture of the model
        """

        model = Sequential()
        # First layer
        model.add(Dense(numfeatures - 6, input_dim=numfeatures))
        model.add(Activation('linear'))
        model.add(Dropout(0.1))

        # Second layer
        model.add(Dense(numfeatures - 6))
        model.add(Activation('linear'))
        model.add(Dropout(0.2))

        # Third layer
        model.add(Dense(numfeatures - 8))
        model.add(Activation('linear'))
        model.add(Dropout(0.12))

        # Final layer
        model.add(Dense(numtargets))

        model.compile(loss=mean_squared_error,  optimizer='adam', metrics=['mse', 'accuracy'])

        self.model = model

class DeepNN_2(DataModel):
    """ The main DeepNN used for our analysis."""
    def __init__(self, numfeatures, numtargets):
        """Here we add the arcitecture of the model """

        model = Sequential()
        # First layer
        model.add(Dense(numfeatures - 6, input_dim=numfeatures))
        model.add(Activation('linear'))
        model.add(Dropout(0.15))

        # Second layer
        model.add(Dense(numfeatures - 6))
        model.add(Activation('linear'))

        # Final layer
        model.add(Dense(numtargets))

        model.compile(loss=mean_squared_error,  optimizer='adam', metrics=['mse', 'accuracy'])

        self.model = model
