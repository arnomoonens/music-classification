from Learner import Learner
from gensim import corpora, models
from ngram import generate_ngram
import pandas as pd
import numpy as np
import logging
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm

logging.getLogger("gensim").setLevel(logging.WARNING)


class LDALearner(Learner):
    classifiers = {}

    def __init__(self, N, ngram_type, **kwargs):
        super(LDALearner, self).__init__(**kwargs)
        self.N = N
        self.ngram_type = ngram_type

    def learn(self, input_data_file):
        df = pd.read_csv(input_data_file,
                         sep=';',
                         index_col=0,
                         names=self.column_names)
        logging.info('Making ngrams')
        songs = [[str(x) for x in generate_ngram(pd.read_csv("unigram/" + str(i) + ".csv", index_col=0), self.N, self.ngram_type)] for i in df.index]
        logging.info('Made ngrams, now generating LDA model')
        self.dictionary = corpora.Dictionary(songs)
        corpus = [self.dictionary.doc2bow(song) for song in songs]
        self.ldamodel = models.ldamodel.LdaModel(corpus, num_topics=50, id2word=self.dictionary, passes=20)
        logging.info('LDA model generated')
        doc_topics = [dict(self.ldamodel.get_document_topics(corpus_song)) for corpus_song in corpus]  # probability of topics for each song
        self.v = DictVectorizer(sparse=False)
        training_input = self.v.fit_transform(doc_topics)  # "unsparse" the {topic_id: probability} dictionary
        logging.info('Learning classifiers')
        for output_name in self.output_names:
            if output_name in ['Tempo', 'Year']:
                self.classifiers[output_name] = svm.SVR()
            else:
                self.classifiers[output_name] = svm.SVC(decision_function_shape='ovo')
            self.classifiers[output_name].fit(training_input, np.array(df[output_name]))
        logging.info('Classifiers learned')
        return

    def classify(self, song_df):
        """Classify a specific song"""
        ngram = [str(x) for x in generate_ngram(song_df, self.N, self.ngram_type)]
        bow = self.dictionary.doc2bow(ngram)
        topics = self.ldamodel.get_document_topics(bow)
        return {output_name: clf.predict([self.v.transform(topics)])[0] for output_name, clf in self.classifiers.items()}

    def test(self, test_data_file):
        """Classify all songs in a test set"""
        df = pd.read_csv(test_data_file, sep=';', index_col=0, names=self.column_names)
        songs = [[str(x) for x in generate_ngram(pd.read_csv("unigram/" + str(i) + ".csv", index_col=0), self.N, self.ngram_type)] for i in df.index]
        corpus = [self.dictionary.doc2bow(song) for song in songs]
        doc_topics = [dict(self.ldamodel.get_document_topics(corpus_song)) for corpus_song in corpus]
        test_input = self.v.transform(doc_topics)
        output_arrays = [self.classifiers[output_name].predict(test_input) for output_name in self.output_names]
        dicts = [dict(zip(self.output_names, row)) for row in zip(*output_arrays)]
        output_df = pd.DataFrame(columns=self.output_names)
        output_df = output_df.append(dicts, ignore_index=True)
        return output_df
