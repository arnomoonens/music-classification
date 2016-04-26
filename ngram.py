#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from os.path import isfile

def generate_ngram(df, n, content='--both'):
    if content == '--both':
        ngram = np.zeros((len(df) - n + 1, n * 2))
        for i in range(len(df) - n + 1):
            nlength = np.array(df['note length'].iloc[i:i + n])
            pitch = np.array(df['pitch'].iloc[i:i + n])
            gram = np.append(nlength, pitch)
            ngram[i] = gram
    elif content == '--length':
        ngram = np.zeros((len(df) - n + 1, n))
        for i in range(len(df) - n + 1):
            nlength = np.array(df['note length'].iloc[i:i + n])
            ngram[i] = nlength
    elif content == "--pitch":
        ngram = np.zeros((len(df) - n + 1, n))
        for i in range(len(df) - n + 1):
            pitch = np.array(df['pitch'].iloc[i:i + n])
            ngram[i] = pitch
    return ngram

# This file will take a unigram, a size for N and a type
# (length, pitch, both) and transform this into an n-gram
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Please provide arguments: [1] unigram source file, [2] size for N [3] the n-gram content (--length, --pitch, --both)")
    else:
        file = sys.argv[1]
        if isfile(file):
            unigram = pd.read_csv(file, index_col=0)
            print(generate_ngram(unigram, int(sys.argv[2]), sys.argv[3]))
