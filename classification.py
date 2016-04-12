#!/usr/bin/env python
import sys
import pandas
import random
import itertools
from collections import Counter

# Random with probability
def learn(ngram_folder, descriptions):
    df = pandas.read_csv(descriptions, sep=';', index_col=0)
    frequencies = {x: Counter(df[x]) for x in ['Performer', 'Inst.', 'Style', 'Year']}
    return lambda x: {k: next(itertools.islice(frequencies[k].elements(), random.randrange(sum(frequencies[k].values())), None)) for k in frequencies}



if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Please provide a folder with the n-grams, descriptions file and number of song to classify")
    else:
        classifier = learn(sys.argv[1], sys.argv[2])
        print(classifier(sys.argv[3]))