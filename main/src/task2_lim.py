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

__version_info__ = ('1', '0', '0')
__version__ = '.'.join(__version_info__)

class LimModel(EventDetectorKerasSequential):
    def __init__(self, *args, **kwargs):
        super(LimModel, self).__init__(*args, **kwargs)
        self.method = 'mysample'

    def create_model(self, input_shape):
        self.nb_input_freq = input_shape
        self.nb_input_time = self.learner_params['input_data']['subdivs']
        self.nb_chunk_time = int(self.params['win_length_seconds']/self.params['hop_length_seconds'])

        nb_classes = 1
        channels = 1
        inputs = Input(shape=(self.nb_input_time, self.nb_chunk_time, self.nb_input_freq))


        # 1D-Conv Nets Layer =====================================
        filters = 64
        kernel_size = (32, 1)
        conv_stride = (1, 1)
        conv_act = 'relu'
        conv_bord_mode = 'valid'
        conv_init = 'he_normal'


        model = TimeDistributed(Conv2D(filters=filters, kernel_size=kernel_size, strides=conv_stride,
                                       padding=conv_bord_mode, kernel_initializer=conv_init),
                                input_shape=(self.nb_input_time, self.nb_chunk_time, self.nb_input_freq))(inputs)
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

        import pdb
        pdb.set_trace()


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
            'mysample': LimModel,
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
                                               os.path.splitext(os.path.basename(__file__))[0]+'.yaml')
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

        app = CustomAppCore(name='DCASE 2017::Rare Sound Event Detection / Lim Model Deep Learning',
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
            app.system_training()
            # app.system_training(overwrite=True)

        # System evaluation
        if not args.mode or args.mode == 'dev':

            # System testing
            # ==================================================
            if params['flow']['test_system']:
                # app.system_testing(overwrite=True)
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