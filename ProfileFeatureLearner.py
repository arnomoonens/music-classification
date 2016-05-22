import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import logging

from Learner import Learner
from profile import get_profile

# Classifiers and regressors
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


class ProfileFeatureLearner(Learner):
    """Naive bayes and linear regression learner using n-gram counts as features"""

    learners = {}

    def get_classifier(self, name, algorithm_args):
        possible_classifiers = {
            'svm': lambda: svm.SVC(decision_function_shape='ovo', **algorithm_args),
            'naive_bayes': lambda: GaussianNB(**algorithm_args),
            'random forest': lambda: RandomForestClassifier(**algorithm_args)
        }
        return possible_classifiers[name]

    def get_regressor(self, name, algorithm_args):
        possible_regressors = {
            'svm': lambda: svm.SVR(**algorithm_args),
            'linear_regression': lambda: LinearRegression(**algorithm_args),
            'random forest': lambda: RandomForestRegressor(**algorithm_args)
        }
        return possible_regressors[name]

    def __init__(self, N, ngram_type, classifier='svm', classifier_args={}, regressor='svm', regressor_args={}, **kwargs):
        super(ProfileFeatureLearner, self).__init__(**kwargs)
        self.N = N
        self.ngram_type = ngram_type
        self.classifier = self.get_classifier(classifier, classifier_args)
        self.regressor = self.get_regressor(regressor, regressor_args)

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
            if output_name in ['Year', 'Tempo']:
                self.learners[output_name] = self.regressor()
            else:
                self.learners[output_name] = self.classifier()
            self.learners[output_name].fit(training_input, df[output_name])
        logging.info('Made and fit learners')

    def classify(self, song_df):
        """Classify a specific song"""
        test_input = self.v.transform(get_profile(song_df, self.N, self.ngram_type))
        return {output_name: clf.predict(test_input)[0] for output_name, clf in self.learners.items()}

    def test(self, test_data_file):
        """Classify all songs in a test set"""
        df = pd.read_csv(test_data_file, sep=';', index_col=0, names=self.column_names)
        df['profile'] = df.apply(lambda row: get_profile(pd.read_csv("unigram/" + str(row.name) + ".csv", index_col=0), self.N, self.ngram_type), axis=1)
        test_input = self.v.transform(df['profile'])
        output_arrays = [self.learners[output_name].predict(test_input) for output_name in self.output_names]
        dicts = [dict(zip(self.output_names, row)) for row in zip(*output_arrays)]
        output_df = pd.DataFrame(columns=self.output_names)
        output_df = output_df.append(dicts, ignore_index=True)
        return output_df
