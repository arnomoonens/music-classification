from Learner import Learner
from collections import Counter
import pandas as pd
import numpy as np
from ngram import generate_ngram
import logging


class ProfileLearner(Learner):
    """Classify using profiles"""
    def __init__(self, N, profile_size, **kwargs):
        super(ProfileLearner, self).__init__(**kwargs)
        self.N = N
        self.profile_size = profile_size

    # Generate a profile using song data
    def __get_profile(self, song_df):
        ngram = generate_ngram(song_df, self.N, '--length')
        c = Counter()
        for x in ngram:
            c[tuple([float(nr) for nr in x])] += 1
        return c

    # Calculate the similarity between the profile of specific
    # output_columns value (e.g. specific composer) and the profile of a song
    def __similarity(self, type_profile, song_profile):
        new_type_profile = dict()
        new_song_profile = dict()
        for k in list(type_profile.keys()) + list(set(song_profile.keys()) - set(type_profile.keys())):
            new_type_profile[k] = type_profile[k] if k in type_profile else 0
            new_song_profile[k] = song_profile[k] if k in song_profile else 0
        # Similarity formula from original paper
        return sum([4 - ((2 * (new_type_profile[k] - new_song_profile[k])) / (new_type_profile[k] + new_song_profile[k])) ** 2 for k in new_type_profile.keys()])

    # Make a classifier based on ngram profiles
    def learn(self, input_data_file):
        df = pd.read_csv(input_data_file, sep=';', index_col=0, names=self.column_names)
        logging.info('Making profiles for songs')
        df['profile'] = df.apply(lambda row: self.__get_profile(pd.read_csv("unigram/" + str(row.name) + ".csv", index_col=0)), axis=1)  # Make profile for each song in input
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

    # classifier checks for each value's profile of output_column which is most similar to the song's profile (by an argmax in each group of types_grouped)
    def classify(self, song_df):
        return {t: group.loc[np.argmax(group.apply(lambda row: self.__similarity(row['profile'], self.__get_profile(song_df)), axis=1))]['name'] for t, group in self.types_grouped}
