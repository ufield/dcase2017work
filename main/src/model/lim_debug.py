from __future__ import print_function, absolute_import
import sys
import os

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
import numpy
import textwrap
import platform
sys.path.append('/store/projects/ml/mathG/DCASE2017/baseline/DCASE2017-baseline-system/')

from dcase_framework.application_core import BinarySoundEventAppCore
from dcase_framework.parameters import ParameterContainer
from dcase_framework.utils import *
from dcase_framework.learners import EventDetectorKerasSequential


from keras.layers import MaxPooling1D, Conv2D, Conv1D, MaxPooling2D, Input, RepeatVector
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Activation, Dropout, Flatten, Reshape, Permute
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import L1L2
from keras.layers.wrappers import TimeDistributed
from keras.legacy.layers import MaxoutDense

class LimModel():

    def __init__(self):
        pass

    def create_model(self):
        input_shape = 40

        self.nb_input_freq = input_shape
        # self.nb_input_time = self.learner_params['input_data']['subdivs']
        # self.nb_chunk_time = int(self.params['win_length_seconds'] / self.params['hop_length_seconds'])
        self.nb_input_time = 1501
        self.nb_chunk_time = 50

        nb_classes = 1
        channels = 1
        inputs = Input(shape=(self.nb_input_time, self.nb_chunk_time, self.nb_input_freq, nb_classes))

        # 1D-Conv Nets Layer =====================================
        filters = 64
        kernel_size = (1, 20)
        conv_stride = (1, 1)
        conv_act = 'relu'
        conv_bord_mode = 'valid'
        conv_init = 'he_normal'

        model = TimeDistributed(Conv2D(filters=filters, kernel_size=kernel_size,
                                       padding=conv_bord_mode, kernel_initializer=conv_init,
                                       data_format='channels_last'),
                                input_shape=(self.nb_input_time, self.nb_chunk_time, self.nb_input_freq, nb_classes))(inputs)
        model = Activation(conv_act)(model)
        model = Dropout(0.2)(model)

        # RNN-LSTM Layer ===================================
        nb_rnn_hidden_units = 40
        rnn_hid_act = 'tanh'
        # model = TimeDistributed(LSTM(units=nb_rnn_hidden_units,
        #                              input_shape=(self.nb_chunk_time, self.nb_input_freq),
        #                              return_sequences=True, activation=rnn_hid_act, stateful=False, implementation=2),
        #                         input_shape=(self.nb_input_time, self.nb_chunk_time, self.nb_input_freq))(inputs)
        # model = TimeDistributed(LSTM(units=nb_rnn_hidden_units,
        #                              input_shape=(self.nb_chunk_time, self.nb_input_freq),
        #                              return_sequences=True, activation=rnn_hid_act, stateful=False, implementation=2),
        #                         input_shape=(self.nb_input_time, self.nb_chunk_time, self.nb_input_freq))(model)

        # FC Layer =====================
        # model = TimeDistributed(Dense(units=nb_classes, kernel_initializer='he_normal', activation='relu'))(model)
        # model = TimeDistributed(Dense(units=nb_classes, kernel_initializer=fnn_init, activation=output_act))(model)

        # Sliding-Average  =====================


        final_model = Model(inputs=inputs, outputs=model)
        self.model = final_model
        self.model.summary()
        self.logger.debug(self.params)

if __name__ == "__main__":
    lm = LimModel()
    lm.create_model()
