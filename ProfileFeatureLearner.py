import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import logging

from Learner import Learner
from profile import get_profile


class ProfileFeatureLearner(Learner):
    """Naive bayes and linear regression learner using n-gram counts as features"""

    classifiers = {}

    def __init__(self, N, ngram_type, **kwargs):
        super(ProfileFeatureLearner, self).__init__(**kwargs)
        self.N = N
        self.ngram_type = ngram_type

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
                self.classifiers[output_name] = RandomForestRegressor(n_jobs=-1, n_estimators=20)
            else:
                self.classifiers[output_name] = RandomForestClassifier(n_jobs=-1, n_estimators=60)
            self.classifiers[output_name].fit(training_input, df[output_name])
        logging.info('Made and fit learners')

    def classify(self, song_df):
        """Classify a specific song"""
        test_input = self.v.transform(get_profile(song_df, self.N, self.ngram_type))
        return {output_name: clf.predict(test_input)[0] for output_name, clf in self.classifiers.items()}

    def test(self, test_data_file):
        """Classify all songs in a test set"""
        df = pd.read_csv(test_data_file, sep=';', index_col=0, names=self.column_names)
        df['profile'] = df.apply(lambda row: get_profile(pd.read_csv("unigram/" + str(row.name) + ".csv", index_col=0), self.N, self.ngram_type), axis=1)
        test_input = self.v.transform(df['profile'])
        output_arrays = [self.classifiers[output_name].predict(test_input) for output_name in self.output_names]
        dicts = [dict(zip(self.output_names, row)) for row in zip(*output_arrays)]
        output_df = pd.DataFrame(columns=self.output_names)
        output_df = output_df.append(dicts, ignore_index=True)
        return output_df
