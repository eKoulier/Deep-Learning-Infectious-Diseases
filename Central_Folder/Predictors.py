import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import backend as K
import numpy as np
import os
import sys
sys.path.append("../Data/Map")

# First we read the Files to be use
cwd = os.getcwd()
os.chdir(r'../Data/Map')
post_to_mun = pd.read_csv('Post_to_Mun.txt', sep=';')
mun_to_GGD = pd.read_csv('Mun_to_GGD.csv', sep=';')
mun_to_GGD['Municipality'] = mun_to_GGD['Municipality'].replace('Nuenen', 'Nuenen, ' +
                                                                'Gerwen en Nederwetten')
os.chdir(cwd)


class PrepareTimeDf(object):
    """ The class that prepares the data to be used by the predictors.
    """
    def __init__(self, df):
        """ Initialize the object.
        """
        assert type(df) == pd.core.frame.DataFrame,\
            'What you have provided is not a Pandas DataFrame'

        self.df = df

    def create_lag(self, brabant=True, nlags=4):
        """ Preprocesses the dataset in order to create time lags for 'HVB', 'WB', 'BZO' and 'Trends'.
        df: Pandas DataFrame
        brabat: Bool
            If True, then only the columns named 'BZO', 'WB', 'HVB' will be kept
        lag: int
            This is the time lag to be created for each column
        """
        assert type(nlags) == int

        if 'Date' in self.df.columns:
            del self.df['Date']

        if brabant:
            df = self.df[['HVB', 'BZO', 'WB', 'Trends']]
        else:
            df = self.df

        for column in df.columns.tolist():
            for lag in range(1, nlags+1):
                df[column+'-'+str(lag)] = df[column].shift(lag)
            df[column+'+1'] = df[column].shift(-1)

        # We don't need Trends +1, no reason to forecast it. Remove also the last lag of trends
        del df['Trends+1']
        del df['Trends-'+str(nlags)]

        # Due to the shift that creates nans, we delete the first nlags rows and the last row.
        df = df[nlags:].reset_index(drop=True)
        df = df[:-1]

        self.df = df

    def split_data(self, train=0.33, val=0.1):
        """Splits the data to create a train, a validation, and a test set.
        train: float
            Percentage of the train set.
        val: float
            Percentage of the validation set.
        """
        assert train < 1 and train > 0,\
            "Train set must be between 0 and 1"
        assert val < 1 and val > 0,\
            "Validation set must be between 0 and 1"

        # Create a list of the target columns
        target_cols = [col for col in self.df.columns if '+1' in col]
        # Isolate the target columns
        y = self.df[target_cols]
        # Isolate the feature columns
        features = [col for col in self.df.columns if col not in target_cols]
        X = self.df[features]

        # Split the data
        X_T, X_test, y_T, y_test = train_test_split(X, y, test_size=train, shuffle=False)
        # Make the validation dataset
        X_train, X_val, y_train, y_val = train_test_split(X_T, y_T, test_size=val, shuffle=True)

        return X_train, X_test, X_val, y_train, y_test, y_val


class ModelUse(object):
    """ The class to be inherited by the models. It contains the train methods.
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
            self


class DeepNN(object):
    """ The main DeepNN used for our analysis."""
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
