from Learner import Learner
import pandas as pd
import numpy as np
from profile import get_profile, similarity, manhatten, dice
import logging


class ProfileLearner(Learner):
    """Classify using profiles"""
    def __init__(self, N, profile_size, similarity=None, **kwargs):
        super(ProfileLearner, self).__init__(**kwargs)
        self.N = N
        self.profile_size = profile_size
        self.similarity = similarity

    def learn(self, input_data_file):
        """Make a classifier based on ngram profiles"""
        df = pd.read_csv(input_data_file, sep=';', index_col=0, names=self.column_names)
        logging.info('Making profiles for songs')
        df['profile'] = df.apply(lambda row: get_profile(pd.read_csv("unigram/" + str(row.name) + ".csv", index_col=0), 3, '--both'), axis=1)  # Make profile for each song in input
        logging.info('Profiles for songs made')
        logging.info('Making profiles for types')
        type_profiles = pd.DataFrame(columns=['type', 'name', 'profile'])
        for output_column in self.output_names:  # For each output type (e.g. Performer, Year,…)…
            grouped = df.groupby(output_column)  # …group the rows by that output type…
            for name, group in grouped:
                profile = group.loc[:, 'profile'].sum()  # …and build a profile for each instance of a type (e.g. profile of a specific performer) by summing the profiles of the songs with that specific output type
                if self.profile_size > 0:
                    profile = profile.most_common(self.profile_size)
                type_profiles = type_profiles.append({'type': output_column, 'name': name, 'profile': profile}, ignore_index=True)
        logging.info('Profiles for types made')
        self.types_grouped = type_profiles.groupby('type')  # Group of all performers, group of all years,…
        return

    def classify(self, song_df):
        """
        Classifier checks for each value's profile of output_column which is
        most similar to the song's profile (by an argmax in each group of types_grouped)
        """
        if self.similarity == '-m':
            return {t: group.loc[np.argmin(group.apply(lambda row: manhatten(row['profile'], get_profile(song_df, 3, '--both')), axis=1))]['name'] for t, group in self.types_grouped}
        elif self.similarity == '-d':
            return {t: group.loc[np.argmax(group.apply(lambda row: dice(row['profile'], get_profile(song_df, 3, '--both')), axis=1))]['name'] for t, group in self.types_grouped}
        else:
            return {t: group.loc[np.argmax(group.apply(lambda row: similarity(row['profile'], get_profile(song_df, 3, '--both')), axis=1))]['name'] for t, group in self.types_grouped}
