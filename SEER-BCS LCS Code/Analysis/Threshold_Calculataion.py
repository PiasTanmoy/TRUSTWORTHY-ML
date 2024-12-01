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