#!/usr/bin/env python3
import sys
import pandas as pd
import random
import itertools
from collections import Counter
from ngram import generate_ngram

column_names = ['Id', 'Performer', 'Title', 'Inst.', 'Style', 'Year', 'Tempo', 'Number of Notes']
output_columns = ['Performer', 'Inst.', 'Style', 'Year', 'Tempo']

# Random with probability
def learn_random(input_data_file):
    df = pd.read_csv(input_data_file, sep=';', index_col=0, names=column_names)
    frequencies = {x: Counter(df[x]) for x in output_columns}
    return lambda x: {k: next(itertools.islice(frequencies[k].elements(), random.randrange(sum(frequencies[k].values())), None)) for k in frequencies}

# Generate a profile using song data
def get_profile(song_df):
    ngram = generate_ngram(song_df, 3, '--both')
    c = Counter()
    for x in ngram:
        c[tuple([float(nr) for nr in x])] += 1
    return c

# Calculate the similarity between the profile of specific output_columns value (e.g. specific composer) and the profile of a song
def similarity(type_profile, song_profile):
    return sum([4 - ((2 * (type_profile[i] - song_profile[i])) / (type_profile[i] + song_profile[i])) ** 2 for i in range(len(type_profile))])  # How to make this work with the result from get_profile?

# Make a classifier based on ngram profiles
def learn_with_profiles(input_data_file):
    df = pd.read_csv(input_data_file, sep=';', index_col=0, names=column_names)
    df['profile'] = df[1:].apply(lambda row: get_profile(pd.read_csv("unigram/" + row.name + ".csv", index_col=0)), axis=1)
    # Make profiles of every output_column (except tempo?)
    # classifier checks for each value's profile of output_column which is most similar to the song's profile
    return df  # should be changed to classifier once it's ready


# Apply a classifier on test data and output the results
def test_classifier(classifier, test_data_file):
    df = pd.read_csv(test_data_file, sep=';', index_col=0, names=column_names)
    output_df = pd.DataFrame(columns=output_columns)
    for song in df.iterrows():
        output_df = output_df.append(classifier(song), ignore_index=True)
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
        classifier = learn_random(training_data_file)
        results = test_classifier(classifier, test_data_file)
        results.to_csv(output_file, header=False, sep=';', index=False)
