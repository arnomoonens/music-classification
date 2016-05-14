#!/usr/bin/env python3
import sys
import logging
# from RandomLearner import RandomLearner
# from ProfileLearner import ProfileLearner
# from LexRankLearner import LexRankLearner
from LDALearner import LDALearner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

column_names = ['Id', 'Performer', 'Title', 'Inst.', 'Style', 'Year', 'Tempo', 'Number of Notes']
output_columns = ['Performer', 'Inst.', 'Style', 'Year', 'Tempo']

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Please provide parameters: [1] training input csv file, [2] test input csv file, [3] output file.')
    else:
        training_data_file = sys.argv[1]
        test_data_file = sys.argv[2]
        output_file = sys.argv[3]
        learner = LDALearner(3, '--both',
                             column_names=column_names,
                             output_names=output_columns)
        learner.learn(training_data_file)
        logging.info('Classifier made, now testing it.')
        results = learner.test(test_data_file)
        logging.info('Test results gathered, writing them to file')
        results.to_csv(output_file, header=False, sep=';', index=False)
        logging.info('Test results wrote to file')
