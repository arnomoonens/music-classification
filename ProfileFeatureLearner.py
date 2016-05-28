import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import logging
from sklearn.preprocessing import LabelEncoder

from Learner import Learner
from profile import get_profile, get_union_profile


# Classifiers and regressors
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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

    def __make_neural_net(self, nr_topics, regression, nr_outputs, hidden_num_units=100, max_epochs=100):
        """Make a neural network for classification or regression"""
        return NeuralNet(
            layers=[('input', layers.InputLayer),
                    ('hidden', layers.DenseLayer),
                    ('output', layers.DenseLayer),
                    ],
            # layer parameters:
            input_shape=(None, nr_topics),
            hidden_num_units=hidden_num_units,  # number of units in 'hidden' layer
            output_nonlinearity=None if regression else lasagne.nonlinearities.softmax,
            output_num_units=nr_outputs,

            # optimization method:
            update=nesterov_momentum,
            update_learning_rate=0.001,
            update_momentum=0.9,
            regression=regression,
            train_split=TrainSplit(eval_size=0.2),

            max_epochs=max_epochs,
            verbose=0,
        )

    def get_classifier(self, algorithm_name, output_name, nr_inputs, nr_outputs, algorithm_args):
        """Return a function to make a classifier of a certain type"""
        possible_classifiers = {
            'svm': lambda: svm.SVC(**{'decision_function_shape': 'ovo', **algorithm_args}),
            'naive bayes': lambda: GaussianNB(**algorithm_args),
            'random forest': lambda: RandomForestClassifier(**{'n_jobs': -1, 'warm_start': True, **algorithm_args}),
            'neural network': lambda: self.__make_neural_net(nr_inputs, False, nr_outputs, **algorithm_args)
        }
        return possible_classifiers[algorithm_name]()

    def get_regressor(self, algorithm_name, output_name, nr_inputs, nr_outputs, algorithm_args):
        """Return a function to make a regressor of a certain type"""
        possible_regressors = {
            'svm': lambda: svm.SVR(**algorithm_args),
            'linear regression': lambda: LinearRegression(**algorithm_args),
            'random forest': lambda: RandomForestRegressor(**{'n_jobs': -1, 'warm_start': True, **algorithm_args}),
            'neural network': lambda: self.__make_neural_net(nr_inputs, True, nr_outputs, **algorithm_args)
        }
        return possible_regressors[algorithm_name]()

    def __init__(self, N, ngram_type, classifier='svm', classifier_args={}, regressor='svm', regressor_args={}, **kwargs):
        super(ProfileFeatureLearner, self).__init__(**kwargs)
        self.N = N
        self.ngram_type = ngram_type
        self.classifier_type = classifier
        self.classifier_args = classifier_args
        self.regressor_type = regressor
        self.regressor_args = regressor_args

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
                regressor = self.get_regressor(self.regressor_type, output_name, training_input.shape[1], 1, self.regressor_args)
                regressor.fit(np.float32(training_input), np.float32(training_output))
                self.__learners.append({'output name': output_name, 'learner': regressor, 'type': 'regressor'})
            else:
                label_encoder = LabelEncoder()
                labels = label_encoder.fit_transform(training_output)
                classifier = self.get_classifier(self.classifier_type, output_name, training_input.shape[1], len(np.unique(training_output)), self.classifier_args)
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
