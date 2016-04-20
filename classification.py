#!/usr/bin/env python3
import sys
import pandas as pd
import random
import itertools
from collections import Counter

column_names = ['Id', 'Performer', 'Title', 'Inst.', 'Style', 'Year', 'Tempo', 'Number of Notes']
output_columns = ['Performer', 'Inst.', 'Style', 'Year', 'Tempo']

# Random with probability
def learn(input_data_file):
    df = pd.read_csv(input_data_file, sep=';', index_col=0, names=column_names)
    frequencies = {x: Counter(df[x]) for x in output_columns}
    return lambda x: {k: next(itertools.islice(frequencies[k].elements(), random.randrange(sum(frequencies[k].values())), None)) for k in frequencies}

#Output order: Performer, Instrument, Style, Year, Tempo
def test_classifier(classifier, test_data_file):
    df = pd.read_csv(test_data_file, sep=';', index_col=0, names=column_names)
    output_df = pd.DataFrame(columns=output_columns)
    for song in df.iterrows():
        output_df=output_df.append(classifier(song), ignore_index=True)
    return output_df

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Please provide a training input csv file, test input csv file and output file.')
    else:
        training_data_file = sys.argv[1]
        test_data_file = sys.argv[2]
        output_file = sys.argv[3]
        classifier = learn(training_data_file)
        results = test_classifier(classifier, test_data_file)
        results.to_csv(output_file, header=False, sep=';', index=False)