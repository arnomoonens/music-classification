#!/usr/bin/env python3
import sys
import logging
# from RandomLearner import RandomLearner
# from ProfileLearner import ProfileLearner
# from LexRankLearner import LexRankLearner
# from LDALearner import LDALearner
from ProfileFeatureLearner import ProfileFeatureLearner

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

column_names = ['Id', 'Performer', 'Title', 'Inst.',
                'Style', 'Year', 'Tempo', 'Number of Notes']
output_names = ['Performer', 'Inst.', 'Style', 'Year', 'Tempo']

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("""
              Please provide parameters: [1] training input csv file,
                                         [2] test input csv file,
                                         [3] output file.
              """)
    else:
        training_data_file = sys.argv[1]
        test_data_file = sys.argv[2]
        output_file = sys.argv[3]
        # learner = LexRankLearner(2, -1, '--both', column_names=column_names, output_names=output_columns)

        # learner = ProfileLearner(1, -1, '--both', similarity=None, column_names=column_names, output_names=output_columns)

        # learner = LDALearner(
        #     2,
        #     '--pitch',
        #     classifier='random forest', classifier_args={'n_estimators': 60},
        #     regressor='random forest', regressor_args={'n_estimators': 20},
        #     column_names=column_names,
        #     output_names=output_names)

        learner = ProfileFeatureLearner(
            2,
            '--pitch',
            # classifier='random forest', classifier_args={'n_estimators': 60},
            # regressor='random forest', regressor_args={'n_estimators': 20},
            classifier='svm', classifier_args={'C': 100},
            regressor='svm', regressor_args={'C': 100},
            # classifier='neural network', classifier_args={'hidden_num_units': 200, 'max_epochs': 1000},
            # regressor='neural network', regressor_args={'hidden_num_units': 200, 'max_epochs': 1000},
            column_names=column_names,
            output_names=output_names)
        
        learner.learn(training_data_file)
        logging.info('Classifier made, now testing it.')
        results = learner.test(test_data_file)
        logging.info('Test results gathered, writing them to file')
        results.to_csv(output_file, header=False, sep=';', index=False)
        logging.info('Test results wrote to file')
