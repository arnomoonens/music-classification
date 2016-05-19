from collections import Counter
from ngram import generate_ngram

def get_profile(song_df, N, ngram_type):
    """Generate a profile using song data"""
    ngram = generate_ngram(song_df, N, ngram_type)
    c = Counter()
    for x in ngram:
        c[tuple([float(nr) for nr in x])] += 1
    return c

def similarity(type_profile, song_profile):
    """
    Calculate the similarity between the profile of specific
    output_columns value (e.g. specific composer) and the profile of a song
    """
    new_type_profile = dict()
    new_song_profile = dict()
    for k in list(type_profile.keys()) + list(set(song_profile.keys()) - set(type_profile.keys())):
        new_type_profile[k] = type_profile[k] if k in type_profile else 0
        new_song_profile[k] = song_profile[k] if k in song_profile else 0
    # Similarity formula from original paper
    return sum([4 - ((2 * (new_type_profile[k] - new_song_profile[k])) / (new_type_profile[k] + new_song_profile[k])) ** 2 for k in new_type_profile.keys()])
