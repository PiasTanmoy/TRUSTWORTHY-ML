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
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression as IR
import os
import numpy as np
import argparse
import json
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

import xgboost as xgb
from sklearn.isotonic import IsotonicRegression as IR
import pandas as pd
import numpy as np
import heapq
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score

def Select_Threshold_calibrated(df):
    full_threshold_list = []
    for threshold in np.arange(0.01, 1.0, 0.01):
        #df.drop(columns = ['y_pred'])
        df['y_pred'] = df['score y calibrated'].apply(lambda x: 1 if x >= threshold else 0)

        # survival => 1, death => 0 where 1 minority in this case
        y_pred = df["y_pred"].values
        y_true = df["true y"].values
        f1_C1 = f1_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        
        full_threshold_list.append([threshold, f1_C1, balanced_accuracy]) # use minority class 1 (death) for LCS and ICU MR
        
    df_varying_threshold = pd.DataFrame(full_threshold_list, columns = ['threshold', 'f1_score', 'balanced_accuracy'])
    
    # select three highest F1 score and the the highest balanced accuracy
    f1_scores = df_varying_threshold["f1_score"].values
    thresholds = df_varying_threshold["threshold"].values
    bal_acc_values = list(df_varying_threshold["balanced_accuracy"].values)
    
    #print(heapq.nlargest(3, f1_scores))
    list_index = heapq.nlargest(3, range(len(f1_scores)), key=f1_scores.__getitem__)
    opt_threshold = thresholds[bal_acc_values.index(max(bal_acc_values[list_index[0]], bal_acc_values[list_index[1]], bal_acc_values[list_index[2]]))]
    
    
    return opt_threshold, df_varying_threshold



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

    # data/task-data/in-hospital-mortality/eICU_dataset/all_samples
    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'eICU_dataset/all_samples'),
                                             listfile=os.path.join(args.data, 'eICU_dataset/eICU_list_file_train_0.csv'),
                                             period_length=48.0)

    val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'eICU_dataset/all_samples'),
                                           listfile=os.path.join(args.data, 'eICU_dataset/eICU_list_file_val_0.csv'),
                                           period_length=48.0)

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'eICU_dataset/all_samples'),
                                             listfile=os.path.join(args.data, 'eICU_dataset/eICU_list_file_test_0.csv'),
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


    best_params = {'var_smoothing': 1e-07}
    print(best_params)

    file_name = ""
    for k in best_params:
        file_name += (k + "_" +str(best_params[k]) + "_")

    best_model = GaussianNB(**best_params)
    best_model.fit(train_X, train_y)

    result_dir = os.path.join(args.output_dir, 'results')
    common_utils.create_directory(result_dir)

    train_prediction = best_model.predict_proba(train_X)[:, 1]
    val_prediction = best_model.predict_proba(val_X)[:, 1]
    test_prediction = best_model.predict_proba(test_X)[:, 1]

    
    ir = IR(out_of_bounds='clip')
    ir.fit( val_prediction, val_y )

    val_pred_calibrated = ir.transform( val_prediction )
    valid_preds_df = pd.DataFrame()
    valid_preds_df['true y'] = val_y
    valid_preds_df['score y'] = val_prediction
    valid_preds_df['score y calibrated'] = val_pred_calibrated
    valid_preds_df.to_csv(os.path.join(args.output_dir, 'val_preds.csv'), index = False)


    train_prediction_calibrated = ir.transform( train_prediction )
    train_preds_df = pd.DataFrame()
    train_preds_df['true y'] = train_y
    train_preds_df['score y'] = train_prediction
    train_preds_df['score y calibrated'] = train_prediction_calibrated
    train_preds_df.to_csv(os.path.join(args.output_dir, 'train_preds.csv'), index = False)


    test_prediction_calibrated = ir.transform( test_prediction )
    test_preds_df = pd.DataFrame()
    test_preds_df['true y'] = test_y
    test_preds_df['score y'] = test_prediction
    test_preds_df['score y calibrated'] = test_prediction_calibrated
    test_preds_df.to_csv(os.path.join(args.output_dir, 'test_preds.csv'), index = False)


    opt_threshold, df_varying_threshold = Select_Threshold_calibrated(valid_preds_df)
    print("Optimal threshold: ", opt_threshold)
    df_varying_threshold.to_csv(os.path.join(args.output_dir, 'varying_threshold.csv'), index = False)

    

    with open(os.path.join(result_dir, 'train_{}.json'.format(file_name)), 'w') as res_file:
        ret = print_metrics_binary(train_y, train_prediction)
        ret = {k : float(v) for k, v in ret.items()}
        json.dump(ret, res_file)

    save_results(train_names, train_prediction, train_y,
                 os.path.join(args.output_dir, 'predictions', 'train_{}.csv'.format(file_name)))


    with open(os.path.join(result_dir, 'val_{}.json'.format(file_name)), 'w') as res_file:
        ret = print_metrics_binary(val_y, val_prediction)
        ret = {k: float(v) for k, v in ret.items()}
        json.dump(ret, res_file)
    
    save_results(val_names, val_prediction, val_y,
                 os.path.join(args.output_dir, 'predictions', 'val_{}.csv'.format(file_name)))

                 
    with open(os.path.join(result_dir, 'test_{}.json'.format(file_name)), 'w') as res_file:
        ret = print_metrics_binary(test_y, test_prediction)
        ret = {k: float(v) for k, v in ret.items()}
        json.dump(ret, res_file)

    save_results(test_names, test_prediction, test_y,
                 os.path.join(args.output_dir, 'predictions', 'test_{}.csv'.format(file_name)))
    



    for i in range(1, 6):
        p1 = "test-2/Respiratory rate set - Seed"+str(i)+" - SD 20"
        p2 = p1 + "/name_list.csv"
        file_name = p1.split("/")[1].split(" ")[0] +str(i)+ ".csv"

        test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, p1),
                                                listfile=os.path.join(args.data, p2),
                                                period_length=48.0)

        (test_X, test_y, test_names) = read_and_extract_features(test_reader, args.period, args.features)
        print('Custom test data shape = {}'.format(test_X.shape))

        test_X = np.array(imputer.transform(test_X), dtype=np.float32)
        test_X = scaler.transform(test_X)
        test_prediction = best_model.predict_proba(test_X)[:, 1]

        test_prediction_calibrated = ir.transform( test_prediction )
        test_preds_df = pd.DataFrame()
        test_preds_df['stay'] = test_names
        test_preds_df['true y'] = test_y
        test_preds_df['score y'] = test_prediction
        test_preds_df['score y calibrated'] = test_prediction_calibrated

        
        test_preds_df.to_csv(os.path.join(args.output_dir, file_name), index = False)
        print("Completed", file_name)



    #Temperature set - Seed1 - SD 0.7
    for i in range(1, 6):
        p1 = "test-2/Temperature set - Seed"+str(i)+" - SD 0.7"
        p2 = p1 + "/name_list.csv"
        file_name = p1.split("/")[1].split(" ")[0] +str(i)+ ".csv"

        test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, p1),
                                                listfile=os.path.join(args.data, p2),
                                                period_length=48.0)

        (test_X, test_y, test_names) = read_and_extract_features(test_reader, args.period, args.features)
        print('Custom test data shape = {}'.format(test_X.shape))

        test_X = np.array(imputer.transform(test_X), dtype=np.float32)
        test_X = scaler.transform(test_X)
        test_prediction = best_model.predict_proba(test_X)[:, 1]

        test_prediction_calibrated = ir.transform( test_prediction )
        test_preds_df = pd.DataFrame()
        test_preds_df['stay'] = test_names
        test_preds_df['true y'] = test_y
        test_preds_df['score y'] = test_prediction
        test_preds_df['score y calibrated'] = test_prediction_calibrated

        test_preds_df.to_csv(os.path.join(args.output_dir, file_name), index = False)
        print("Completed", file_name)



    #Systolic BP set - Seed1 - SD 15
    for i in range(1, 6):
        p1 = "test-2/Systolic BP set - Seed"+str(i)+" - SD 15"
        p2 = p1 + "/name_list.csv"
        file_name = p1.split("/")[1].split(" ")[0] +str(i)+ ".csv"

        test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, p1),
                                                listfile=os.path.join(args.data, p2),
                                                period_length=48.0)

        (test_X, test_y, test_names) = read_and_extract_features(test_reader, args.period, args.features)
        print('Custom test data shape = {}'.format(test_X.shape))

        test_X = np.array(imputer.transform(test_X), dtype=np.float32)
        test_X = scaler.transform(test_X)
        test_prediction = best_model.predict_proba(test_X)[:, 1]

        test_prediction_calibrated = ir.transform( test_prediction )
        test_preds_df = pd.DataFrame()
        test_preds_df['stay'] = test_names
        test_preds_df['true y'] = test_y
        test_preds_df['score y'] = test_prediction
        test_preds_df['score y calibrated'] = test_prediction_calibrated

        test_preds_df.to_csv(os.path.join(args.output_dir, file_name), index = False)
        print("Completed", file_name)




if __name__ == '__main__':
    main()
