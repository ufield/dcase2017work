import os
import sys

env = os.environ
sys.path.append(env['DCASE_BASE'])
sys.path.append(env['MYWORK_BASE'])

from dcase_framework.application_core import BinarySoundEventAppCore
from dcase_framework.parameters import ParameterContainer
from dcase_framework.utils import *
from dcase_framework.features import FeatureContainer, FeatureRepository, FeatureExtractor, FeatureNormalizer, \
    FeatureStacker, FeatureAggregator, FeatureMasker

from keras.models import load_model, Model


class CustomAppCore(BinarySoundEventAppCore):
    def __init__(self, *args, **kwargs):
        from main.src.model.cakir import CakirModel
        kwargs['Learners'] = {
            'mysample': CakirModel,
        }
        super(CustomAppCore, self).__init__(*args, **kwargs)

class CakirAnalyzer():

    def __init__(self, model_name, project_name, set_id):
        self.params = self._set_params(project_name, set_id)
        self.app = CustomAppCore(
            name='DCASE 2017::Detection of rare sound events / Cakir 2017',
            params=self.params,
            system_desc=self.params.get('description'),
            system_parameter_set_id=self.params.get('active_set'),
            setup_label='Development setup',
            log_system_progress=self.params.get_path('general.log_system_progress'),
            show_progress_in_console=self.params.get_path('general.print_system_progress'),
            use_ascii_progress_bar=self.params.get_path('general.use_ascii_progress_bar')
        )
        self.model_pickle_filename = model_name + '.cpickle'
        self.model_hdf5_filename = model_name + '.model.hdf5'
        self.model_container = self.app._get_learner(method=self.app.params.get_path('learner.method')).load(filename=self.model_pickle_filename)

    def _set_params(self, project_name, set_id):
        project_base = env['MYWORK_BASE'] + '/main'
        params = ParameterContainer(
            project_base=project_base,
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
                    'feature_stacker',
                    'feature_normalizer',
                    'feature_aggregator',
                    'learner'
                ],
                'recognizer': [
                    'dataset',
                    'feature_extractor',
                    'feature_stacker',
                    'feature_normalizer',
                    'feature_aggregator',
                    'learner',
                    'recognizer'
                ],
            }
        )

        params.load(filename=project_base + '/parameters/' + project_name + '.defaults.yaml')
        params['active_set'] = set_id  # ex) 'crnn'
        params.process() # 実行用のパラメータ構造に変える？

        return params

    def process_feature_data(self, feature_filename):
        feature_list = {}
        feature_list['mel'] = FeatureContainer().load(filename=feature_filename)
        feature_data = self.model_container.feature_stacker.process(
            feature_data=feature_list
        )

        # Normalize features
        if self.model_container.feature_normalizer:
            feature_data = self.model_container.feature_normalizer.normalize(feature_data)

        # Aggregate features
        if self.model_container.feature_aggregator:
            feature_data = self.model_container.feature_aggregator.process(feature_data)
        return feature_data


    def predict(self, feature_filename):
        feature_data = self.process_feature_data(feature_filename)
        frame_probabilities = self.model_container.predict(
            feature_data=feature_data,
        )
        return frame_probabilities

    def mid_output(self, feature_filename, layer_name, input_shape=(1,1,1501,40)):
        '''
        :param feature_filename:
        :param layer_name: # ex.) gpu_1
        :param shape:
        :return:
        '''
        model = self.model_container.model
        middle_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        feature_data = self.process_feature_data(feature_filename)
        data = feature_data['feat'][0].reshape(input_shape)
        return middle_model.predict(data)

    def model_summary(self):
        self.model_container.model.summary()
