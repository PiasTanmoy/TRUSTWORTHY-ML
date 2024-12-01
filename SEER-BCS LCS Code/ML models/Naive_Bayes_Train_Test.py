import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from Performance_metrics import measure_performance
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression as IR
import pandas as pd
import numpy as np
import heapq
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

def Select_Threshold_calibrated(df):
    full_threshold_list = []
    for threshold in np.arange(0.01, 1.0, 0.01):
        #df.drop(columns = ['y_pred'])
        df['y_pred'] = df['score y calibrated'].apply(lambda x: 1 if x >= threshold else 0)

        # survival => 1, death => 0
        y_pred = df["y_pred"].values
        y_true = df["true y"].values
        f1_C1 = f1_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        #print(f1_C1, balanced_accuracy)

        # survival => 0, death => 1
        y_pred_flip = (1- np.array(y_pred))
        y_true_flip = (1 - np.array(y_true))
        f1_C0 = f1_score(y_true_flip, y_pred_flip)
        balanced_accuracy = balanced_accuracy_score(y_true_flip, y_pred_flip) #threshold agnostic
        #print(f1_C0, balanced_accuracy)

        
        full_threshold_list.append([threshold, f1_C0, balanced_accuracy])
        
    df_varying_threshold = pd.DataFrame(full_threshold_list, columns = ['threshold', 'f1_score', 'balanced_accuracy'])
    
    # select three highest F1 score and the the highest balanced accuracy
    f1_scores = df_varying_threshold["f1_score"].values
    thresholds = df_varying_threshold["threshold"].values
    bal_acc_values = list(df_varying_threshold["balanced_accuracy"].values)
    
    #print(heapq.nlargest(3, f1_scores))
    list_index = heapq.nlargest(3, range(len(f1_scores)), key=f1_scores.__getitem__)
    opt_threshold = thresholds[bal_acc_values.index(max(bal_acc_values[list_index[0]], bal_acc_values[list_index[1]], bal_acc_values[list_index[2]]))]
    
    
    return opt_threshold, df_varying_threshold


# ... (load and preprocess data as before)
import os
os.chdir('/home/tanmoysarkar/Trustworthiness/SEER')

X_train= np.load("seer_data/processed_data_TS_2/X_train_normalized.npy")
y_train = np.load("seer_data/processed_data_TS_2/y_train_normalized.npy")

X_valid = np.load("seer_data/processed_data_TS_2/X_valid_normalized.npy")
y_valid = np.load("seer_data/processed_data_TS_2/y_valid_normalized.npy")

X_test = np.load("seer_data/processed_data_TS_2/X_test_normalized.npy")
y_test = np.load("seer_data/processed_data_TS_2/y_test_normalized.npy")


test_dirs = ['seer_data/Test2/CS_Tumor_Size/CS_Turmor_Size_valid_norm_range_0_42_seed_', #0_x.csv,
     'seer_data/Test2/Num_lymph/num_lymph_valid_norm_range_0_10_seed_',
     'seer_data/Test2/Pos_lymph/pos_lymph_valid_norm_range_0_25_seed_']

save_directory_path = "seer_data/Test_results/Naive_Bayes/"
if not os.path.exists(save_directory_path):
    os.makedirs(save_directory_path)


print("data loaded !")


best_params = {'var_smoothing': 1e-09}
best_model = GaussianNB(**best_params)

print(best_params)
# Create a grid search object

# Fit the grid search to the data
best_model.fit(X_train, y_train)

y_valid_pred_proba = best_model.predict_proba(X_valid)[:, 1]

ir = IR(out_of_bounds='clip')
ir.fit( y_valid_pred_proba, y_valid )

y_valid_pred_proba_calibrated = ir.transform( y_valid_pred_proba )

valid_preds_df = pd.DataFrame()
valid_preds_df['score y calibrated'] = y_valid_pred_proba_calibrated
valid_preds_df['true y'] = y_valid

opt_threshold, df_varying_threshold = Select_Threshold_calibrated(valid_preds_df)

print("Optimal threshold: ", opt_threshold)
df_varying_threshold.to_csv(save_directory_path + 'varying_threshold.csv', index = False)

for test_path in test_dirs: 
    df = pd.DataFrame()

    for i in range(3):
        path = test_path + str(i) + '_x.npy'
        test_name = path.split("/")[2] + " " + str(i)

        print(path, test_name)

        X_test = np.load(path)
        y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]

        y_test_pred_proba_calibrated = ir.transform( y_test_pred_proba ) 

        df[test_name] = y_test_pred_proba
        df[test_name + " calibrated"] = y_test_pred_proba_calibrated

        file_path = os.path.join(save_directory_path, path.split("/")[2]+ ".csv")

    df.to_csv(file_path, index = False)



