from Learner import Learner
from collections import Counter
import pandas as pd
import numpy as np
import math
from ngram import generate_ngram
import logging

class LexRankLearner(Learner):
    """Classify using profiles"""
    """Profiles are reduced using LexRank algorithm"""
    def __init__(self, N, profile_size, **kwargs):
        super(LexRankLearner, self).__init__(**kwargs)
        self.N = N
        self.profile_size = profile_size

    # Generate a profile using song data
    def __get_profile(self, song_df, N):
        ngram = generate_ngram(song_df, N, '--both')
        c = Counter()
        for x in ngram:
            c[tuple([float(nr) for nr in x])] += 1
        return c

    # Calculate the similarity between the profile of specific output_columns value (e.g. specific composer) and the profile of a song
    def __similarity(self, type_profile, song_profile):
        new_type_profile = dict()
        new_song_profile = dict()
        for k in list(type_profile.keys()) + list(set(song_profile.keys()) - set(type_profile.keys())):
            new_type_profile[k] = type_profile[k] if k in type_profile else 0
            new_song_profile[k] = song_profile[k] if k in song_profile else 0
        return sum([4 - ((2 * (new_type_profile[k] - new_song_profile[k])) / (new_type_profile[k] + new_song_profile[k])) ** 2 for k in new_type_profile.keys()])  # Similarity formula from original paper

    # Input: Array S of n sentences (i.e. the Document)
    # Input: cosine threshold t
    def __lexrank(self, doc, t):
        n = len(doc)
        flat_doc = list(doc.items())
        cosine_matrix = np.zeros((n, n))
        degree = np.zeros(n)
        for i in range(n):
            for j in range(n):
                sim = self.__cosine_distance(flat_doc[i][0], flat_doc[j][0])
                if sim > t:
                    cosine_matrix[i,j] = 1
                    degree[i] += 1
        for i in range(n):
            for j in range(n):
                if degree[i] == 0:
                    degree[i] = 1 #At least similar to itself
                cosine_matrix[i,j] = cosine_matrix[i,j] / degree[i]
        return self.__power_method(cosine_matrix, n, 0.15)

    def __cosine_distance(self, sen1, sen2):
        return np.dot(sen1, sen2)  / (math.sqrt(np.dot(sen1, sen1) * np.dot(sen2, sen2)))

    def __power_method(self, M, n, epsilon):
        p = [1.0 / n] * n
        while True:
            new_p = [0] * n
            new_p = np.dot(np.transpose(M), p)
            delta = np.linalg.norm(new_p - p)
            p = new_p
            if delta < epsilon:
                break
        return p

    # Make a classifier based on ngram profiles
    def learn(self, input_data_file):
        df = pd.read_csv(input_data_file, sep=';', index_col=0, names=self.column_names)
        logging.info('Making profiles for songs')
        df['profile'] = df.apply(lambda row: self.__get_profile(pd.read_csv("unigram/" + str(row.name) + ".csv", index_col=0), self.N), axis=1)  # Make profile for each song in input
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
        logging.info('Applying LexRank to profiles')
        for idx, row in type_profiles.iterrows():
            importance = self.__lexrank(row['profile'], 0.01)
            mask = np.where(importance > np.mean(importance), True, False)
            flat_profile = list(row['profile'].keys())
            # I was hoping to do this using a one-liner with np.where, but it doesn't work
            reduced_profile = []
            for i,x in enumerate(mask):
                if x: reduced_profile.append(flat_profile[i])
            row['profile'] = Counter(reduced_profile)
        logging.info('LexRank applied to profiles')
        self.types_grouped = type_profiles.groupby('type')  # Group of all performers, group of all years,…
        return

    # classifier checks for each value's profile of output_column which is most similar to the song's profile (by an argmax in each group of types_grouped)
    def classify(self, song_df):
        return {t: group.loc[np.argmax(group.apply(lambda row: self.__similarity(row['profile'], self.__get_profile(song_df, self.N)), axis=1))]['name'] for t, group in self.types_grouped}
