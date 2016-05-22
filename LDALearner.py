from Learner import Learner
from gensim import corpora, models
from ngram import generate_ngram
import pandas as pd
import numpy as np
import logging
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
# from sklearn import svm

# Classifiers and regressors
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne.base import TrainSplit

logging.getLogger("gensim").setLevel(logging.WARNING)

class LDALearner(Learner):
    """Learner that uses Latent Dirichlet Allocation"""

    __learners = []
    __dictionary = None
    __v = None

    def get_classifier(self, name, algorithm_args):
        """Return a function to make a classifier of a certain type"""
        possible_classifiers = {
            'svm': lambda: svm.SVC(**{'decision_function_shape': 'ovo', **algorithm_args}),
            'naive bayes': lambda: GaussianNB(**algorithm_args),
            'random forest': lambda: RandomForestClassifier(**{'n_jobs': -1, 'warm_start': True, **algorithm_args})
        }
        return possible_classifiers[name]

    def get_regressor(self, name, algorithm_args):
        """Return a function to make a regressor of a certain type"""
        possible_regressors = {
            'svm': lambda: svm.SVR(**algorithm_args),
            'linear regression': lambda: LinearRegression(**algorithm_args),
            'random forest': lambda: RandomForestRegressor(**{'n_jobs': -1, 'warm_start': True, **algorithm_args})
        }
        return possible_regressors[name]

    def __init__(self, N, ngram_type, classifier='svm', classifier_args={}, regressor='svm', regressor_args={}, **kwargs):
        super(LDALearner, self).__init__(**kwargs)
        self.N = N
        self.ngram_type = ngram_type
        self.classifier = self.get_classifier(classifier, classifier_args)
        self.regressor = self.get_regressor(regressor, regressor_args)

    def __make_neural_net(self, nr_topics, regression, nr_outputs):
        return NeuralNet(
            layers=[('input', layers.InputLayer),
                    ('hidden', layers.DenseLayer),
                    ('output', layers.DenseLayer),
                    ],
            # layer parameters:
            input_shape=(None, nr_topics),
            hidden_num_units=20,  # number of units in 'hidden' layer
            output_nonlinearity=None if regression else lasagne.nonlinearities.softmax,
            output_num_units=nr_outputs,

            # optimization method:
            update=nesterov_momentum,
            update_learning_rate=0.001,
            update_momentum=0.9,
            regression=regression,
            train_split=TrainSplit(eval_size=0.2),

            max_epochs=1000,
            verbose=0,
        )

    def learn(self, input_data_file):
        df = pd.read_csv(input_data_file,
                         sep=';',
                         index_col=0,
                         names=self.column_names)
        logging.info('Making ngrams')
        songs = [[str(x) for x in generate_ngram(pd.read_csv("unigram/" + str(i) + ".csv", index_col=0), self.N, self.ngram_type)] for i in df.index]
        logging.info('Made ngrams, now generating LDA model')
        self.__dictionary = corpora.Dictionary(songs)
        corpus = [self.__dictionary.doc2bow(song) for song in songs]
        self.__ldamodel = models.ldamodel.LdaModel(corpus, num_topics=50, id2word=self.__dictionary, passes=20)
        logging.info('LDA model generated')
        doc_topics = [dict(self.__ldamodel.get_document_topics(corpus_song)) for corpus_song in corpus]  # probability of topics for each song
        self.__v = DictVectorizer(sparse=False)
        training_input = self.__v.fit_transform(doc_topics)  # "unsparse" the {topic_id: probability} dictionary
        logging.info('Learning classifiers')
        for output_name in self.output_names:
            training_output = np.array(df[output_name])
            if output_name in ['Tempo', 'Year']:
                # regressor = self.__make_neural_net(training_input.shape[1], True, 1)  # regression=True, 1 output
                regressor = self.regressor()
                regressor.fit(np.float32(training_input), np.float32(training_output))
                self.__learners.append({'output name': output_name, 'learner': regressor, 'type': 'regressor'})
            else:
                label_encoder = LabelEncoder()
                labels = label_encoder.fit_transform(training_output)
                # classifier = self.__make_neural_net(training_input.shape[1], False, len(np.unique(training_output)))  # regression=False
                classifier = self.classifier()
                classifier.fit(np.float32(training_input), np.int32(labels))
                self.__learners.append({'ouput name': output_name, 'label encoder': label_encoder, 'learner': classifier, 'type': 'classifier'})
        logging.info('learners learned')
        return

    def classify(self, song_df):
        """Classify a specific song"""
        ngram = [str(x) for x in generate_ngram(song_df, self.N, self.ngram_type)]
        bow = self.__dictionary.doc2bow(ngram)
        topics = self.__ldamodel.get_document_topics(bow)
        test_input = np.float32([self.__v.transform(topics)])
        return {learner['output name']: learner['label encoder'].inverse_transform(learner['learner'].predict(test_input)) if learner['type'] == 'classifier' else learner['learner'].predict(test_input) for learner in self.__learners}

    def test(self, test_data_file):
        """Classify all songs in a test set"""
        df = pd.read_csv(test_data_file, sep=';', index_col=0, names=self.column_names)
        songs = [[str(x) for x in generate_ngram(pd.read_csv("unigram/" + str(i) + ".csv", index_col=0), self.N, self.ngram_type)] for i in df.index]
        corpus = [self.__dictionary.doc2bow(song) for song in songs]
        doc_topics = [dict(self.__ldamodel.get_document_topics(corpus_song)) for corpus_song in corpus]
        test_input = np.float32(self.__v.transform(doc_topics))
        output_arrays = [learner['label encoder'].inverse_transform(learner['learner'].predict(test_input)) if learner['type'] == 'classifier' else learner['learner'].predict(test_input).flatten() for learner in self.__learners]
        dicts = [dict(zip(self.output_names, row)) for row in zip(*output_arrays)]
        output_df = pd.DataFrame(columns=self.output_names)
        output_df = output_df.append(dicts, ignore_index=True)
        return output_df
