import pandas as pd


class Learner(object):
    """basic Learner"""

    def __init__(self, column_names, output_names):
        super(Learner, self).__init__()
        self.column_names = column_names
        self.output_names = output_names

    def learn(self, input_data_file):
        pass

    def classify(self, song_df):
        pass

    # Apply a classifier on test data and output the results
    def test(self, test_data_file):
        df = pd.read_csv(test_data_file, sep=';', index_col=0, names=self.column_names)
        output_df = pd.DataFrame(columns=self.output_names)
        n_songs = len(df)
        ctr = 0
        for song in df.iterrows():
            output_df = output_df.append(self.classify(pd.read_csv("unigram/" + str(song[0]) + ".csv", index_col=0)), ignore_index=True)
            ctr += 1
            print("Classified " + str(ctr) + "/" + str(n_songs), end='\r')
        return output_df
