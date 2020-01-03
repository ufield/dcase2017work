import sys
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

class CakirModel(EventDetectorKerasSequential):
    def __init__(self, *args, **kwargs):
        super(CakirModel, self).__init__(*args, **kwargs)
        self.method = 'mysample' # TODO

    def _set_global(self, input_shape):
        self.nb_channels = 1
        self.nb_input_freq = input_shape
        self.nb_input_time = self.learner_params['input_data']['subdivs']

    def _input_layer(self):
        channels = 1

        return Input(shape=(channels, self.nb_input_time, self.nb_input_freq))

    def _add_conv_layers(self, inputs):
        learner_params = self.learner_params

        nb_conv_filters = learner_params['conv_params']['nb_conv_filters']
        nb_conv_layers  = len(nb_conv_filters)
        nb_conv_time = learner_params['conv_params']['nb_conv_time']
        nb_conv_freq = learner_params['conv_params']['nb_conv_freq']
        conv_act = learner_params['conv_params']['conv_act']
        conv_bord_mode = learner_params['conv_params']['conv_border_mode']
        conv_init = learner_params['conv_params']['conv_init']
        conv_stride = tuple(learner_params['conv_params']['conv_stride'])

        w_reg = None
        b_reg = None
        b_constr = None
        w_constr = None

        if len(nb_conv_time) == 1:
            nb_conv_time = [nb_conv_time[0]] * nb_conv_layers
        assert len(nb_conv_time) == nb_conv_layers

        if len(nb_conv_freq) == 1:
            nb_conv_freq = [nb_conv_freq[0]] * nb_conv_layers
        assert len(nb_conv_freq) == nb_conv_layers

        batchnorm_flag = learner_params['conv_params']['batchnorm_flag']
        batchnorm_axis = learner_params['conv_params']['batchnorm_axis']

        dropout_flag = learner_params['general']['dropout_flag']
        dropout_rate = learner_params['general']['dropout_rate']

        if len(dropout_rate) == 1:
            dropout_rate = [dropout_rate[0]] * nb_conv_layers
        assert len(dropout_rate) == nb_conv_layers

        nb_conv_pool_freq = learner_params['conv_params']['nb_conv_pool_freq']
        pool_stride_freq = learner_params['conv_params']['pool_stride_freq']

        if len(pool_stride_freq) == 0:
            pool_stride_freq = nb_conv_pool_freq
        elif len(pool_stride_freq) == 1:
            pool_stride_freq = [pool_stride_freq[0]] * nb_conv_layers


        # if len(nb_conv_pool_time) == 1:
        #     nb_conv_pool_time = [nb_conv_pool_time[0]] * nb_conv_layers
        # assert len(nb_conv_pool_time) == nb_conv_layers
        model = None
        for i in range(nb_conv_layers):

            if i == 0:
                model = Conv2D(filters=nb_conv_filters[i], kernel_size=(nb_conv_time[i], nb_conv_freq[i]),
                           strides=conv_stride,
                           padding=conv_bord_mode, data_format="channels_first",
                           kernel_initializer=conv_init,
                           kernel_regularizer=w_reg, bias_regularizer=b_reg, kernel_constraint=w_constr,
                           bias_constraint=b_constr)(inputs)
            else:
                model = Conv2D(filters=nb_conv_filters[i], kernel_size=(nb_conv_time[i], nb_conv_freq[i]),
                           strides=conv_stride,
                           padding=conv_bord_mode, data_format="channels_first",
                           kernel_initializer=conv_init,
                           kernel_regularizer=w_reg, bias_regularizer=b_reg, kernel_constraint=w_constr,
                           bias_constraint=b_constr)(model)

            if batchnorm_flag:
                model = BatchNormalization(axis=batchnorm_axis)(model) # TODO: batchnorm_axis?
            if 'LeakyReLU' in conv_act:
                model = LeakyReLU()(model)
            else:
                model = Activation(conv_act)(model)
            if dropout_flag:
                model = Dropout(dropout_rate[i])(model)

            # 周波数方向のpooling
            model = MaxPooling2D(pool_size=(1, nb_conv_pool_freq[i]),
                                 strides=(1, pool_stride_freq[i]), padding='valid', data_format='channels_first')(model)
            self.nb_channels = nb_conv_filters[i]
            if conv_bord_mode == 'valid':  # 'valid' convolution decreases the CNN output length by nb_conv_x -1.
                self.nb_input_freq = int(
                    (self.nb_input_freq - nb_conv_freq[i] + conv_stride[1]) / nb_conv_pool_freq[i] / conv_stride[1])
            elif conv_bord_mode == 'same':
                self.nb_input_freq = int(self.nb_input_freq / nb_conv_pool_freq[i] / conv_stride[1])


        return model

    def _reshape(self, model):
        self.nb_input_freq = self.nb_channels * self.nb_input_freq
        model = Permute((2, 1, 3))(model)
        model = Reshape((self.nb_input_time, self.nb_input_freq))(model)
        return model

    def _add_rnn_layers(self, model):
        learner_params = self.learner_params
        nb_rnn_hidden_units = learner_params['rnn_params']['nb_rnn_hidden_units']
        nb_rnn_layers = len(nb_rnn_hidden_units)

        if nb_rnn_layers > 0:  # no need to read RNN parameters if there is no RNN layer (nb_rnn_layers = 0).
            rnn_dropout_U = learner_params['rnn_params']['rnn_dropout_U']
            rnn_dropout_W = learner_params['rnn_params']['rnn_dropout_W']
            rnn_hid_act = learner_params['rnn_params']['rnn_hid_act']
            rnn_projection_downsampling = learner_params['rnn_params']['rnn_projection_downsampling']
            rnn_type = learner_params['rnn_params']['rnn_type']
            if rnn_type == 'GRU':
                rnn_type = GRU
            elif rnn_type == 'LSTM':
                rnn_type = LSTM
            else:
                raise ('unknown RNN type!')
            statefulness_flag = learner_params['rnn_params']['statefulness_flag']


        for i in range(nb_rnn_layers):
            if i == 0:
                inp_shape = (self.nb_input_time, self.nb_input_freq)
            else:
                # if rnn_projection_downsampling:
                #     inp_shape = (nb_input_time, nb_rnn_proj_hidden_units[i - 1])
                # else:
                inp_shape = (self.nb_input_time, nb_rnn_hidden_units[i - 1])

            # if i == 0 and conv_layers == 0:
            #     model = Reshape((nb_input_time, nb_input_freq))(inputs)

            model = rnn_type(units=nb_rnn_hidden_units[i], input_shape=inp_shape,
                             activation=rnn_hid_act, return_sequences=True, stateful=False,
                             dropout=rnn_dropout_W, recurrent_dropout=rnn_dropout_U, implementation=2)(model)

        return model

    def _add_fnn_layers(self, model):
        learner_params = self.learner_params
        fnn_type = learner_params['fnn_params']['fnn_type']
        if fnn_type == 'Dense':
            fnn_type = Dense
        elif fnn_type == 'MaxoutDense':
            fnn_type = MaxoutDense
        else:
            raise ('unknown FNN type!')
        # maxout_pool_size = learner_params['fnn_params']['maxout_pool_size']
        # if fnn_type == Dense:
        #     maxout_pool_size = None
        fnn_init = learner_params['fnn_params']['fnn_init']
        output_act = learner_params['general']['output_act']
        nb_classes = learner_params['output_data']['nb_classes']

        model = TimeDistributed(Dense(units=nb_classes, kernel_initializer=fnn_init, activation=output_act))(model)
        # final_model = Model(inputs=inputs, outputs=model)
        return model

    def create_model(self, input_shape):
        self._set_global(input_shape)

        inputs = self._input_layer()
        model = self._add_conv_layers(inputs)
        model = self._reshape(model)
        model = self._add_rnn_layers(model)
        model = self._add_fnn_layers(model)
        final_model = Model(inputs=inputs, outputs=model)

        self.model = final_model
        self.logger.debug(self.params)
        self.model.summary()
        self.compile_model()

    def compile_model(self):
        from keras.optimizers import SGD, RMSprop, Adagrad, Adam, Adadelta
        optimizer = self.learner_params['general']['optimizer']
        loss_func = self.learner_params['general']['loss']
        l_rate = self.learner_params['general']['learning_rate']
        if optimizer == 'sgd':
            optimizer = SGD(lr=l_rate)
        elif optimizer == 'rmsprop':
            optimizer = RMSprop(lr=l_rate)
        elif optimizer == 'adagrad':
            optimizer = Adagrad(lr=l_rate)
        elif optimizer == 'adam':
            optimizer = Adam(lr=l_rate)
        elif optimizer == 'adadelta':
            optimizer = Adadelta(lr=l_rate)
        else:
            pass
        self.model.compile(optimizer=optimizer, loss=loss_func, metrics=self.learner_params.get_path('model.metrics'))
