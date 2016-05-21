#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from os.path import isfile
from nltk.util import ngrams


def generate_ngram(df, n, content='--both'):
    nlength = list(ngrams(df.loc[:, 'note length'], n))
    pitch = list(ngrams(df.loc[:, 'pitch'], n))
    if content == '--both':
        return [tuple(np.array(x).flatten()) for x in list(zip(nlength, pitch))]
    elif content == '--length':
        return nlength
    elif content == '--pitch':
        return pitch


# This file will take a unigram, a size for N and a type
# (length, pitch, both) and transform this into an n-gram
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("""
              Please provide arguments:
              [1] unigram source file,
              [2] size for N
              [3] the n-gram content (--length, --pitch, --both)
              """)
    else:
        file = sys.argv[1]
        if isfile(file):
            unigram = pd.read_csv(file, index_col=0)
            print(generate_ngram(unigram, int(sys.argv[2]), sys.argv[3]))
