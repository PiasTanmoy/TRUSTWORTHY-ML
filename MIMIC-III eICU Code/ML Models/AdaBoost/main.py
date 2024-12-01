from __future__ import absolute_import
from __future__ import print_function

from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models import common_utils
from mimic3models.metrics import print_metrics_binary
from mimic3models.in_hospital_mortality.utils import save_results
#from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer as Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

import os
import numpy as np
import argparse
import json
import pandas as pd

def read_and_extract_features(reader, period, features):
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    # ret = common_utils.read_chunk(reader, 100)
    X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    return (X, ret['y'], ret['name'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=float, default=1.0, help='inverse of L1 / L2 regularization')
    parser.add_argument('--l1', dest='l2', action='store_false')
    parser.add_argument('--l2', dest='l2', action='store_true')
    parser.set_defaults(l2=True)
    parser.add_argument('--period', type=str, default='all', help='specifies which period extract features from',
                        choices=['first4days', 'first8days', 'last12hours', 'first25percent', 'first50percent', 'all'])
    parser.add_argument('--features', type=str, default='all', help='specifies what features to extract',
                        choices=['all', 'len', 'all_but_len'])
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                        default=os.path.join(os.path.dirname(__file__), '../../../data/task-data/in-hospital-mortality/'))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')
    args = parser.parse_args()
    print(args)

    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                             listfile=os.path.join(args.data, 'train_listfile.csv'),
                                             period_length=48.0)

    # train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test-2/Temperature set - Seed1 - SD 0.7'),
    #                                          listfile=os.path.join(args.data, 'test-2/Temperature set - Seed1 - SD 0.7/name_list.csv'),
    #                                          period_length=48.0)


    val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                           listfile=os.path.join(args.data, 'val_listfile.csv'),
                                           period_length=48.0)

    # val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
    #                                        listfile=os.path.join(args.data, 'test_listfile.csv'),
    #                                        period_length=48.0)

    # test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
    #                                         listfile=os.path.join(args.data, 'test_listfile.csv'),
    #                                         period_length=48.0)

    # test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test-2/Oxygen set - Seed5 - SD 11'),
    #                                         listfile=os.path.join(args.data, 'test-2/Oxygen set - Seed5 - SD 11/name_list.csv'),
    #                                         period_length=48.0)

    #mimic3models/in_hospital_mortality/pred-DBP/Original_model
    #data/task-data/in-hospital-mortality/test-2/LR-high-cases/64580
    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                             listfile=os.path.join(args.data, 'test_listfile.csv'),
                                             period_length=48.0)
    

    print('Reading data and extracting features ...')
    (train_X, train_y, train_names) = read_and_extract_features(train_reader, args.period, args.features)
    (val_X, val_y, val_names) = read_and_extract_features(val_reader, args.period, args.features)
    (test_X, test_y, test_names) = read_and_extract_features(test_reader, args.period, args.features)
    print('  train data shape = {}'.format(train_X.shape))
    print('  validation data shape = {}'.format(val_X.shape))
    print('  test data shape = {}'.format(test_X.shape))

    print('Imputing missing values ...')
    #imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0, verbose=0, copy=True)
    imputer = Imputer(missing_values=np.nan, strategy='mean', verbose=0, copy=True)
    imputer.fit(train_X)
    train_X = np.array(imputer.transform(train_X), dtype=np.float32)
    val_X = np.array(imputer.transform(val_X), dtype=np.float32)
    test_X = np.array(imputer.transform(test_X), dtype=np.float32)

    print('Normalizing the data to have zero mean and unit variance ...')
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)


    penalty = ('l2' if args.l2 else 'l1')
    file_name = '{}.{}.{}.C{}'.format(args.period, args.features, penalty, args.C)

    # Define the parameter grid (example)
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01, 0.1, 1],
    }

    print(param_grid)





    #logreg = LogisticRegression(penalty=penalty, C=args.C, random_state=42)
    model = AdaBoostClassifier(random_state=42)

    # Create a grid search object
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='roc_auc')

    # Fit the grid search to the data
    grid_search.fit(train_X, train_y)

    # Get the best model parameters based on AUROC
    best_params = grid_search.best_params_

    print("Best Parameters:", best_params)

    # Evaluate the model on the test set
    logreg = grid_search.best_estimator_

    #logreg.fit(train_X, train_y)

    result_dir = os.path.join(args.output_dir, 'results')
    common_utils.create_directory(result_dir)

    train_prediction = logreg.predict_proba(train_X)[:, 1]

    with open(os.path.join(result_dir, 'train_{}.json'.format(file_name)), 'w') as res_file:
        ret = print_metrics_binary(train_y, logreg.predict_proba(train_X))
        ret = {k : float(v) for k, v in ret.items()}
        json.dump(ret, res_file)

    save_results(train_names, train_prediction, train_y,
                 os.path.join(args.output_dir, 'predictions', 'train_{}.csv'.format(file_name)))


    val_prediction = logreg.predict_proba(val_X)[:, 1]

    with open(os.path.join(result_dir, 'val_{}.json'.format(file_name)), 'w') as res_file:
        ret = print_metrics_binary(val_y, logreg.predict_proba(val_X))
        ret = {k: float(v) for k, v in ret.items()}
        json.dump(ret, res_file)
    
    save_results(val_names, val_prediction, val_y,
                 os.path.join(args.output_dir, 'predictions', 'val_{}.csv'.format(file_name)))

                 

    test_prediction = logreg.predict_proba(test_X)[:, 1]

    with open(os.path.join(result_dir, 'test_{}.json'.format(file_name)), 'w') as res_file:
        ret = print_metrics_binary(test_y, test_prediction)
        ret = {k: float(v) for k, v in ret.items()}
        json.dump(ret, res_file)

    save_results(test_names, test_prediction, test_y,
                 os.path.join(args.output_dir, 'predictions', 'test_{}.csv'.format(file_name)))
    


    # Assuming grid_search is your GridSearchCV object
    results = pd.DataFrame(grid_search.cv_results_)
    print(results)
    file_path = os.path.join(args.output_dir, "grid_search_training_log.csv")
    results.to_csv(file_path, index=False)


if __name__ == '__main__':
    main()
