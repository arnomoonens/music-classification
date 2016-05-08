#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import random
import itertools
import logging
from collections import Counter
from ngram import generate_ngram

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

column_names = ['Id', 'Performer', 'Title', 'Inst.', 'Style', 'Year', 'Tempo', 'Number of Notes']
output_columns = ['Performer', 'Inst.', 'Style', 'Year', 'Tempo']

# Random with probability
def learn_random(input_data_file):
    df = pd.read_csv(input_data_file, sep=';', index_col=0, names=column_names)
    frequencies = {x: Counter(df[x]) for x in output_columns}
    return lambda x: {k: next(itertools.islice(frequencies[k].elements(), random.randrange(sum(frequencies[k].values())), None)) for k in frequencies}

# Generate a profile using song data
def get_profile(song_df, N):
    ngram = generate_ngram(song_df, N, '--both')
    c = Counter()
    for x in ngram:
        c[tuple([float(nr) for nr in x])] += 1
    return c

# Calculate the similarity between the profile of specific output_columns value (e.g. specific composer) and the profile of a song
def similarity(type_profile, song_profile):
    new_type_profile = dict()
    new_song_profile = dict()
    for k in list(type_profile.keys()) + list(set(song_profile.keys()) - set(type_profile.keys())):
        new_type_profile[k] = type_profile[k] if k in type_profile else 0
        new_song_profile[k] = song_profile[k] if k in song_profile else 0
    return sum([4 - ((2 * (new_type_profile[k] - new_song_profile[k])) / (new_type_profile[k] + new_song_profile[k])) ** 2 for k in new_type_profile.keys()])  # Similarity formula from original paper

# Make a classifier based on ngram profiles
def learn_with_profiles(input_data_file, N, profile_size):
    df = pd.read_csv(input_data_file, sep=';', index_col=0, names=column_names)
    logging.info('Making profiles for songs')
    df['profile'] = df.apply(lambda row: get_profile(pd.read_csv("unigram/" + str(row.name) + ".csv", index_col=0), N), axis=1)  # Make profile for each song in input
    logging.info('Profiles for songs made')
    logging.info('Making profiles for types')
    type_profiles = pd.DataFrame(columns=['type', 'name', 'profile'])
    for output_column in output_columns:  # For each output type (e.g. Performer, Year,…)…
        grouped = df.groupby(output_column)  # …group the rows by that output type…
        for name, group in grouped:
            profile = group.loc[:, 'profile'].sum()  # …and build a profile for each instance of a type (e.g. profile of a specific performer) by summing the profiles of the songs with that specific output type
            flat_profile = list(profile.items())
            flat_profile = flat_profile[:profile_size] # limit profile to profile_size
            profile = Counter(dict(flat_profile))
            type_profiles = type_profiles.append({'type': output_column, 'name': name, 'profile': profile}, ignore_index=True)
    logging.info('Profiles for types made')
    types_grouped = type_profiles.groupby('type')  # Group of all performers, group of all years,…
    # classifier checks for each value's profile of output_column which is most similar to the song's profile (by an argmax in each group of types_grouped)
    return lambda song_df: {t: group.loc[np.argmax(group.apply(lambda row: similarity(row['profile'], get_profile(song_df, N)), axis=1))]['name'] for t, group in types_grouped}


# Apply a classifier on test data and output the results
def test_classifier(classifier, test_data_file):
    df = pd.read_csv(test_data_file, sep=';', index_col=0, names=column_names)
    output_df = pd.DataFrame(columns=output_columns)
    n_songs = len(df)
    ctr = 0
    for song in df.iterrows():
        output_df = output_df.append(classifier(pd.read_csv("unigram/" + str(song[0]) + ".csv", index_col=0)), ignore_index=True)
        ctr += 1
        print("Classified " + str(ctr) + "/" + str(n_songs), end='\r')
    return output_df

# This code will predict the Performer, Instrument, Style, Year and Tempo
# based on the probability of their occurence. This is a temporary mockup
# while we work on the classification using N-grams.
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Please provide parameters: [1] training input csv file, [2] test input csv file, [3] output file.')
    else:
        training_data_file = sys.argv[1]
        test_data_file = sys.argv[2]
        output_file = sys.argv[3]
        classifier = learn_with_profiles(training_data_file, 3, 1000)
        logging.info('Classifier made, now testing it.')
        results = test_classifier(classifier, test_data_file)
        logging.info('Test results gathered, writing them to file')
        results.to_csv(output_file, header=False, sep=';', index=False)
        logging.info('Test results wrote to file')
