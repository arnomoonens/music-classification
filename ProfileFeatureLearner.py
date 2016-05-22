import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
import logging
from sklearn.preprocessing import LabelEncoder

from Learner import Learner
from profile import get_profile


import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne.base import TrainSplit

logging.getLogger("theano").setLevel(logging.ERROR)

class ProfileFeatureLearner(Learner):
    """Naive bayes and linear regression learner using n-gram counts as features"""

    __learners = []
    __dictionary = None
    __v = None

    def __init__(self, N, ngram_type, **kwargs):
        super(ProfileFeatureLearner, self).__init__(**kwargs)
        self.N = N
        self.ngram_type = ngram_type

    def __make_neural_net(self, nr_topics, regression, nr_outputs):
        """Make a neural network for classification or regression"""
        return NeuralNet(
            layers=[('input', layers.InputLayer),
                    ('hidden', layers.DenseLayer),
                    ('output', layers.DenseLayer),
                    ],
            # layer parameters:
            input_shape=(None, nr_topics),
            hidden_num_units=1000,  # number of units in 'hidden' layer
            output_nonlinearity=None if regression else lasagne.nonlinearities.softmax,
            output_num_units=nr_outputs,

            # optimization method:
            update=nesterov_momentum,
            update_learning_rate=0.001,
            update_momentum=0.9,
            regression=regression,
            train_split=TrainSplit(eval_size=0.2),

            max_epochs=50,
            verbose=0,
        )

    def learn(self, input_data_file):
        """Train learners using profiles of songs"""
        df = pd.read_csv(input_data_file, sep=';', index_col=0, names=self.column_names)
        logging.info('Making profiles for songs')
        df['profile'] = df.apply(lambda row: get_profile(pd.read_csv("unigram/" + str(row.name) + ".csv", index_col=0), self.N, self.ngram_type), axis=1)  # Make profile for each song in input
        self.v = DictVectorizer(sparse=False)
        self.v.fit(df['profile'])
        training_input = self.v.transform(df['profile'])
        logging.info('Made profiles, now making and fitting learners')
        for output_name in self.output_names:
            training_output = np.array(df[output_name])
            if output_name in ['Year', 'Tempo']:
                regressor = self.__make_neural_net(training_input.shape[1], True, 1)  # regression=True, 1 output
                regressor.fit(np.float32(training_input), np.float32(training_output))
                self.__learners.append({'output name': output_name, 'learner': regressor, 'type': 'regressor'})
            else:
                label_encoder = LabelEncoder()
                labels = label_encoder.fit_transform(training_output)
                classifier = self.__make_neural_net(training_input.shape[1], False, len(np.unique(training_output)))  # regression=False
                classifier.fit(np.float32(training_input), np.int32(labels))
                self.__learners.append({'ouput name': output_name, 'label encoder': label_encoder, 'learner': classifier, 'type': 'classifier'})
        logging.info('Made and fit learners')
        return

    def classify(self, song_df):
        """Classify a specific song"""
        test_input = np.float32(self.v.transform(get_profile(song_df, self.N, self.ngram_type)))
        return {learner['output name']: learner['label encoder'].inverse_transform(learner['learner'].predict(test_input)) if learner['type'] == 'classifier' else learner['learner'].predict(test_input) for learner in self.__learners}

    def test(self, test_data_file):
        """Classify all songs in a test set"""
        df = pd.read_csv(test_data_file, sep=';', index_col=0, names=self.column_names)
        df['profile'] = df.apply(lambda row: get_profile(pd.read_csv("unigram/" + str(row.name) + ".csv", index_col=0), self.N, self.ngram_type), axis=1)
        test_input = np.float32(self.v.transform(df['profile']))
        output_arrays = [learner['label encoder'].inverse_transform(learner['learner'].predict(test_input)) if learner['type'] == 'classifier' else learner['learner'].predict(test_input).flatten() for learner in self.__learners]
        dicts = [dict(zip(self.output_names, row)) for row in zip(*output_arrays)]
        output_df = pd.DataFrame(columns=self.output_names)
        output_df = output_df.append(dicts, ignore_index=True)
        return output_df
