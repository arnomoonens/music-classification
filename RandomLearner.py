from Learner import Learner
from collections import Counter
import random
import itertools
import pandas as pd

class RandomLearner(Learner):
    """Classify random using a probability"""

    def learn(self, input_data_file):
        df = pd.read_csv(input_data_file, sep=';', index_col=0, names=self.column_names)
        self.frequencies = {x: Counter(df[x]) for x in self.output_names}
        return

    def classify(self, x):
        return {k: next(itertools.islice(self.frequencies[k].elements(), random.randrange(sum(self.frequencies[k].values())), None)) for k in self.frequencies}
