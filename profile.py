from nltk import FreqDist
from ngram import generate_ngram


def get_profile(song_df, N, ngram_type):
    """Generate a profile using song data"""
    ngram = generate_ngram(song_df, N, ngram_type)
    return FreqDist(ngram)


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


def manhatten(type_profile, song_profile):
    """
    Calculate the Manhatten distance between the profile of specific
    output_colums value (e.g. specific composer) and the profile of a
    song
    """
    # Sort profiles by frequency
    type_profile = type_profile.most_common()
    song_profile = song_profile.most_common()
    flat_type_profile = [ngram for (ngram, freq) in type_profile]
    flat_song_profile = [ngram for (ngram, freq) in song_profile]
    manhatten = 0
    for i in range(len(flat_song_profile)):
        ngram = flat_song_profile[i]
        if ngram in flat_type_profile:
            manhatten += abs(flat_type_profile.index(ngram) - i)
        else:
            manhatten += abs(len(flat_type_profile) - i)
    return manhatten  # Minimization!


def dice(type_profile, song_profile):
    """
    Calculate the Dice similarity measure between the profile of
    specific output_columns value (e.g. specific composer) and
    the profile of a song
    """
    flat_type_profile = list(type_profile.keys())
    flat_song_profile = list(song_profile.keys())
    type_profile_len = len(type_profile)
    song_profile_len = len(song_profile)
    overlap = set(flat_type_profile).intersection(set(flat_song_profile))
    return 2 * len(overlap) / (type_profile_len + song_profile_len)  # Maximization!
