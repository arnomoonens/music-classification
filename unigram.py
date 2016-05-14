#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

def round_down(num, divisor):
    return num - (num % divisor)

def generate_unigram(path):
    rows_to_skip = 0
    with open(path, "r") as f:
        while "Note_on_c" not in f.readline():
            rows_to_skip += 1

    # error_bad_lines=False to skip unnecessary and unparseable information.
    # The last 2 lines however, are also unnecessary, but the parameter to skip those
    # (skipfooter) can't be used with the C engine, while error_bad_lines can't be used
    # with the python engine. As such, we skip the last 2 lines manually by reducing N.
    df = pd.read_csv(path,
                     skiprows=rows_to_skip,
                     error_bad_lines=False,
                     # skipfooter=2,
                     # engine='python',
                     header=None,
                     usecols=[1, 4],
                     names=["time", "pitch"])
    N = len(df.index) - 2  # Don't use the last 2 lines, as discussed above

    # First caculate the note length
    # To calculate the note length: subtract time of uneven rows by time of even rows
    df = pd.DataFrame({'note length': df[1:N:2]['time'].values - df[:N:2]['time'].values,
                       'pitch': df[1:N:2]['pitch']})

    df['pitch'] = round_down(df.shift(-1)['pitch'] - df['pitch'], 0.2)
    df.set_value(N - 1, 'pitch', 0)

    df['note length'] = round_down(np.log2(df.shift(-1)['note length'] / df['note length']), 0.2)
    df.set_value(N - 1, 'note length', 0)
    df = df.round(1)
    return df.reset_index(drop=True)

# This file will generate unigrams from the song-csv files and store
# them in corresponding csv files
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Please provide arguments: [1] source folder, [2] output folder.")
    else:
        for f in listdir(sys.argv[1]):
            filepath = join(sys.argv[1], f)
            if isfile(filepath):
                print("Generating file", f)
                generate_unigram(filepath).to_csv(join(sys.argv[2], f))
