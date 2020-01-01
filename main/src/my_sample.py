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


__version_info__ = ('0', '0', '1')
__version__ = '.'.join(__version_info__)


class MySampleDeepLearningSeaquential(EventDetectorKerasSequential):

    def __init__(self, *args, **kwargs):
        super(MySampleDeepLearningSeaquential, self).__init__(*args, **kwargs)
        self.method = 'mysample'

    def create_model(self, input_shape):
        from keras.layers import MaxPooling1D, Conv2D, Conv1D, MaxPooling2D, Input, RepeatVector
        from keras.layers.advanced_activations import LeakyReLU
        from keras.layers.core import Dense, Activation, Dropout, Flatten, Reshape, Permute
        from keras.layers.recurrent import LSTM, GRU
        from keras.layers.normalization import BatchNormalization
        from keras.models import Model
        from keras.regularizers import L1L2
        from keras.layers.wrappers import TimeDistributed
        from keras.legacy.layers import MaxoutDense

        learner_params = self.learner_params

        nb_channels = 1
        nb_input_time = 1501
        nb_input_freq = 40

        nb_fnn_hidden_units = [30]
        nb_fnn_layers = len(nb_fnn_hidden_units)

        filters = 10
        kernel_size = (5, 5) # nb_conv_time = 5, nb_input_freq=5
        conv_stride = (1, 1)
        padding = 'same'
        conv_init = 'he_normal'
        # conv_filters = [96, 96]
        conv_filters = learner_params['conv_params']['nb_conv_filters']
        conv_layers  = len(conv_filters)
        conv_pool_freq = learner_params['conv_params']['nb_conv_pool_freq']
        conv_bord_mode = learner_params['conv_params']['conv_border_mode']
        conv_act = learner_params['conv_params']['conv_act']
        batchnorm_axis = learner_params['conv_params']['batchnorm_axis']

        rnn_hidden_units = learner_params['rnn_params']['nb_rnn_hidden_units']
        rnn_layers = len(rnn_hidden_units)

        w_reg = None
        b_reg = None
        b_constr = None
        w_constr = None

        batchnorm_flag = True
        dropout_flag = True

        nb_classes = 1

        fnn_type = 'Dense'
        fnn_init = 'he_normal'
        if fnn_type == 'Dense':
            fnn_type = Dense

        output_act = 'sigmoid'

        print('input shape ===================')
        print(input_shape)

        model = None
        inputs = Input(shape=(nb_channels, nb_input_time, nb_input_freq))

        for i in range(len(conv_filters)):

            if i == 0:
                model = Conv2D(filters=conv_filters[i], kernel_size=kernel_size,
                           strides=conv_stride,
                           padding=padding, data_format="channels_first",
                           kernel_initializer=conv_init,
                           kernel_regularizer=w_reg, bias_regularizer=b_reg, kernel_constraint=w_constr,
                           bias_constraint=b_constr)(inputs)
            else:
                model = Conv2D(filters=conv_filters[i], kernel_size=kernel_size,
                           strides=conv_stride,
                           padding=padding, data_format="channels_first",
                           kernel_initializer=conv_init,
                           kernel_regularizer=w_reg, bias_regularizer=b_reg, kernel_constraint=w_constr,
                           bias_constraint=b_constr)(model)

            if batchnorm_flag:
                model = BatchNormalization(axis=1)(model) # TODO: batchnorm_axis?
                # model = BatchNormalization(axis=batchnorm_axis)(model) # TODO: batchnorm_axis?
            if 'LeakyReLU' in conv_act:
                model = LeakyReLU()(model)
            else:
                model = Activation(conv_act)(model)
            if dropout_flag:
                model = Dropout(0.5)(model)

            # 周波数方向のpooling
            # nb_conv_pool_freq = 40
            model = MaxPooling2D(pool_size=(1, conv_pool_freq[i]),
                                 strides=(1, conv_pool_freq[i]), padding='valid', data_format='channels_first')(model)

        # FFN 投入用に
        model = Permute((2, 1, 3))(model)
        nb_input_freq = 96

        # final_model = Model(inputs=inputs, outputs=model)
        # final_model.summary()
        # import pdb
        # pdb.set_trace()

        # condense the last two axes (number of channels * outputs per channel).
        model = Reshape((nb_input_time, nb_input_freq))(model)

        if rnn_layers > 0:  # no need to read RNN parameters if there is no RNN layer (nb_rnn_layers = 0).
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


        for i in range(rnn_layers):
            if i == 0:
                inp_shape = (nb_input_time, nb_input_freq)
            else:
                # if rnn_projection_downsampling:
                #     inp_shape = (nb_input_time, nb_rnn_proj_hidden_units[i - 1])
                # else:
                inp_shape = (nb_input_time, rnn_hidden_units[i - 1])

            if i == 0 and conv_layers == 0:
                model = Reshape((nb_input_time, nb_input_freq))(inputs)

            model = rnn_type(units=rnn_hidden_units[i], input_shape=inp_shape,
                             activation=rnn_hid_act, return_sequences=True, stateful=False,
                             dropout=rnn_dropout_W, recurrent_dropout=rnn_dropout_U, implementation=2)(model)

        # =============================================
        fnn_hid_act = learner_params['fnn_params']['fnn_hid_act']
        fnn_batchnorm_flag = False
        # for i in range(nb_fnn_layers):  # FNN layers loop.
        #     model = TimeDistributed(fnn_type(units=nb_fnn_hidden_units[i], kernel_initializer=fnn_init))(model)
        # if fnn_batchnorm_flag:
        #     model = BatchNormalization(axis=-1)(model)
        # model = Activation(fnn_hid_act)(model)

        # final_model = Model(inputs=inputs, outputs=[model_short, model_long])
        model = TimeDistributed(Dense(units=nb_classes, kernel_initializer=fnn_init, activation=output_act))(model)
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


class CustomAppCore(BinarySoundEventAppCore):
    def __init__(self, *args, **kwargs):
        kwargs['Learners'] = {
            'mysample': MySampleDeepLearningSeaquential,
        }
        super(CustomAppCore, self).__init__(*args, **kwargs)


def main(argv):
    numpy.random.seed(123456)  # let's make randomization predictable

    parser = argparse.ArgumentParser(
        prefix_chars='-+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            DCASE 2017
            Task 2: Rare Sound Event Detection
            A custom learner where you can create FNNs, CNNs, RNNs and CRNNs of any size and depth
            Code used in the paper:
            Convolutional Recurrent Neural Networks for Rare Sound Event Detection (Emre Cakir, Tuomas Virtanen)
            http://www.cs.tut.fi/sgn/arg/dcase2017/documents/challenge_technical_reports/DCASE2017_Cakir_104.pdf

            ---------------------------------------------
                Tampere University of Technology / Audio Research Group
                Author:  Emre Cakir ( emre.cakir@tut.fi )


        '''))


    # Setup argument handling
    parser.add_argument('-m', '--mode',
                        choices=('dev', 'challenge'),
                        default=None,
                        help="Selector for system mode",
                        required=False,
                        dest='mode',
                        type=str)

    parser.add_argument('-p', '--parameters',
                        help='parameter file override',
                        dest='parameter_override',
                        required=False,
                        metavar='FILE',
                        type=argument_file_exists)

    parser.add_argument('-s', '--parameter_set',
                        help='Parameter set id',
                        dest='parameter_set',
                        required=False,
                        type=str)

    parser.add_argument("-n", "--node",
                        help="Node mode",
                        dest="node_mode",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_sets",
                        help="List of available parameter sets",
                        dest="show_set_list",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_datasets",
                        help="List of available datasets",
                        dest="show_dataset_list",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_parameters",
                        help="Show parameters",
                        dest="show_parameters",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_eval",
                        help="Show evaluated setups",
                        dest="show_eval",
                        action='store_true',
                        required=False)

    parser.add_argument("-o", "--overwrite",
                        help="Overwrite mode",
                        dest="overwrite",
                        action='store_true',
                        required=False)


    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)

    # Parse arguments
    args = parser.parse_args()

    # Load default parameters from a file
    default_parameters_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../parameters',
                                               os.path.splitext(os.path.basename(__file__))[0]+'.defaults.yaml')
    # default_parameters_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
    #                                            os.path.splitext(os.path.basename(__file__))[0] + '.defaults.yaml')
    print('default_parameters_filename')
    print(default_parameters_filename)
    if args.parameter_set:
        parameters_sets = args.parameter_set.split(',')
    else:
        parameters_sets = [None]

    print('hoge')
    print(parameters_sets)
    for parameter_set in parameters_sets:
        # Initialize ParameterContainer
        params = ParameterContainer(project_base=os.path.dirname(os.path.realpath(__file__)),
                                    path_structure={
                                        'feature_extractor': [
                                            'dataset',
                                            'feature_extractor.parameters.*'
                                        ],
                                        'feature_normalizer': [
                                            'dataset',
                                            'feature_extractor.parameters.*'
                                        ],
                                        'learner': [
                                            'dataset',
                                            'feature_extractor',
                                            'feature_normalizer',
                                            'feature_aggregator',
                                            'learner'
                                        ],
                                        'recognizer': [
                                            'dataset',
                                            'feature_extractor',
                                            'feature_normalizer',
                                            'feature_aggregator',
                                            'learner',
                                            'recognizer'
                                        ],
                                    }
                                    )

        # Load default parameters from a file
        params.load(filename=default_parameters_filename)

        if args.parameter_override:
            # Override parameters from a file
            params.override(override=args.parameter_override)

        if parameter_set:
            # Override active_set
            params['active_set'] = parameter_set

        # Process parameters
        params.process()

        # Force overwrite
        if args.overwrite:
            params['general']['overwrite'] = True

        # Override dataset mode from arguments
        if args.mode == 'dev':
            # Set dataset to development
            params['dataset']['method'] = 'development'

            # Process dataset again, move correct parameters from dataset_parameters
            params.process_method_parameters(section='dataset')

        elif args.mode == 'challenge':
            # Set dataset to training set for challenge
            params['dataset']['method'] = 'challenge_train'
            params['general']['challenge_submission_mode'] = True
            # Process dataset again, move correct parameters from dataset_parameters
            params.process_method_parameters(section='dataset')

        if args.node_mode:
            params['general']['log_system_progress'] = True
            params['general']['print_system_progress'] = False

        # Force ascii progress bar under Windows console
        if platform.system() == 'Windows':
            params['general']['use_ascii_progress_bar'] = True

        # Setup logging
        setup_logging(parameter_container=params['logging'])

        app = CustomAppCore(name='DCASE 2017::Rare Sound Event Detection / Custom MySample Deep Learning',
                             params=params,
                             system_desc=params.get('description'),
                             system_parameter_set_id=params.get('active_set'),
                             setup_label='Development setup',
                             log_system_progress=params.get_path('general.log_system_progress'),
                             show_progress_in_console=params.get_path('general.print_system_progress'),
                             use_ascii_progress_bar=params.get_path('general.use_ascii_progress_bar')
                             )

        # Show parameter set list and exit
        if args.show_set_list:
            params_ = ParameterContainer(
                project_base=os.path.dirname(os.path.realpath(__file__))
            ).load(filename=default_parameters_filename)

            if args.parameter_override:
                # Override parameters from a file
                params_.override(override=args.parameter_override)
            if 'sets' in params_:
                app.show_parameter_set_list(set_list=params_['sets'])

            return

        # Show parameter set list and exit
        if args.show_set_list:
            params_ = ParameterContainer(
                project_base=os.path.dirname(os.path.realpath(__file__))
            ).load(filename=default_parameters_filename)

            if args.parameter_override:
                # Override parameters from a file
                params_.override(override=args.parameter_override)
            if 'sets' in params_:
                app.show_parameter_set_list(set_list=params_['sets'])

            return

        # Show dataset list and exit
        if args.show_dataset_list:
            app.show_dataset_list()
            return

        # Show system parameters
        if params.get_path('general.log_system_parameters') or args.show_parameters:
            app.show_parameters()

        # Show evaluated systems
        if args.show_eval:
            app.show_eval()
            return

        # Initialize application
        # ==================================================
        if params['flow']['initialize']:
            app.initialize()

        # Extract features for all audio files in the dataset
        # ==================================================
        if params['flow']['extract_features']:
            app.feature_extraction()

        # Prepare feature normalizers
        # ==================================================
        if params['flow']['feature_normalizer']:
            app.feature_normalization()

        # System training
        # ==================================================
        if params['flow']['train_system']:
            # app.system_training()
            app.system_training(overwrite=True)

        # System evaluation
        if not args.mode or args.mode == 'dev':

            # System testing
            # ==================================================
            if params['flow']['test_system']:
                app.system_testing()

            # System evaluation
            # ==================================================
            if params['flow']['evaluate_system']:
                app.system_evaluation()

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
