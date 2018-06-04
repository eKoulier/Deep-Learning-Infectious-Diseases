import shutil
from keras import backend as K
from abc import ABCMeta, abstract_attribute


import shutil
from keras import backend as K
from abc import ABCMeta, abstractproperty


class DataModel(metaclass=ABCMeta):
    @abstractproperty
    def model(self):
        pass

    def train_model(self, epochs, times, NBestPerf, Perf, val=True):
        '''
        epochs: number of epochs for the training part
        times : Number of times to train and test the model.
        NBestPerf : Number of top sucessful trainings.
        val :
        '''
        assert times >= NBestPerf,\
            'The number of the best models should be higher than the number of' + \
            + 'total models.'

        def reset_weights(model):
            ''' This is a way to reset weights using the Keras Framework
            '''
            session = K.get_session()
            for layer in model.layers:
                if hasattr(layer, 'kernel_initializer'):
                    layer.kernel.initializer.run(session=session)

        def fit():
            if val == True:
                self.model.fit(self.X_train, self.y_train, epochs=epochs, validation_data=(self.X_val, self.y_val))
            else:
                self.model.fit(self.X_train, self.y_train, epochs=epochs)


        # Here starts the training
        for time in range(times):
            reset_weights(self.model)

            fit()





            reset_weights(self.model)
