import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from Performance_metrics import measure_performance

# ... (load and preprocess data as before)
import os
os.chdir('/home/tanmoysarkar/Trustworthiness/SEER')

X_train= np.load("seer_data/processed_data_TS_2/X_train_normalized.npy")
y_train = np.load("seer_data/processed_data_TS_2/y_train_normalized.npy")

X_valid = np.load("seer_data/processed_data_TS_2/X_valid_normalized.npy")
y_valid = np.load("seer_data/processed_data_TS_2/y_valid_normalized.npy")

X_test = np.load("seer_data/processed_data_TS_2/X_test_normalized.npy")
y_test = np.load("seer_data/processed_data_TS_2/y_test_normalized.npy")

print("data loaded !")

adaboost = AdaBoostClassifier(random_state=42)

# Define the parameter grid (example)
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.1, 1],
}

print(param_grid)
# Create a grid search object
grid_search = GridSearchCV(estimator=adaboost, param_grid=param_grid, cv=5, n_jobs=-1, scoring='roc_auc')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)


directory_path = "seer_data/AdaBoost/AdaBoost_grid_search_training"
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

# Get the best model parameters based on AUROC
best_params = grid_search.best_params_

print("Best Parameters:", best_params)

# Evaluate the model on the test set
best_model = grid_search.best_estimator_

y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
df = measure_performance(y_test, y_test_pred_proba)
file_path = os.path.join(directory_path, "best_model_test_performance.csv")
df.to_csv(file_path)

y_valid_pred_proba = best_model.predict_proba(X_valid)[:, 1]
df = measure_performance(y_valid, y_valid_pred_proba)
file_path = os.path.join(directory_path, "best_model_valid_performance.csv")
df.to_csv(file_path)

y_train_pred_proba = best_model.predict_proba(X_train)[:, 1]
df = measure_performance(y_train, y_train_pred_proba)
file_path = os.path.join(directory_path, "best_model_train_performance.csv")
df.to_csv(file_path)

print("best model performance saved !")

# Assuming grid_search is your GridSearchCV object
results = pd.DataFrame(grid_search.cv_results_)
print(results)
file_path = os.path.join(directory_path, "grid_search_training_log.csv")
results.to_csv(file_path, index=False)

'''
thf following code is to print performance for each combination 
of params; however, due to long run-time, it can be reported later.
'''

# # Iterate over the grid search results
# for idx in range(len(grid_search.cv_results_['params'])):
#     params = grid_search.cv_results_['params'][idx]
    
#     # Reinitialize the AdaBoost classifier with the specific parameters
#     estimator = AdaBoostClassifier(random_state=42, **params)

#     # Fit the estimator on the training data
#     estimator.fit(X_train, y_train)
    
#     # Predict probabilities on the validation set
#     y_pred_proba = estimator.predict_proba(X_valid)[:, 1]
    
#     # Calculate AUROC and AUPRC
#     auc_roc = roc_auc_score(y_valid, y_pred_proba)
#     auc_prc = average_precision_score(y_valid, y_pred_proba)
    
#     print(f"Valid: {idx + 1} Params: {params}, AUROC: {auc_roc:.3f}, AUPRC: {auc_prc:.3f}")
    
#     # Optional: Measure additional performance metrics
#     df = measure_performance(y_valid, y_pred_proba)  # Assuming this function is defined
    
#     # Save performance metrics to CSV
#     file_name = f"params_set_{idx + 1}_valid_performance.csv"
#     file_path = os.path.join(directory_path, file_name)
#     df.to_csv(file_path, index=False)


#     # Predict probabilities on the validation set
#     y_pred_proba = estimator.predict_proba(X_test)[:, 1]
    
#     # Calculate AUROC and AUPRC
#     auc_roc = roc_auc_score(y_test, y_pred_proba)
#     auc_prc = average_precision_score(y_test, y_pred_proba)
    
#     print(f"Test: {idx + 1} Params: {params}, AUROC: {auc_roc:.3f}, AUPRC: {auc_prc:.3f}")
    
#     # Optional: Measure additional performance metrics
#     df = measure_performance(y_test, y_pred_proba)  # Assuming this function is defined
    
#     # Save performance metrics to CSV
#     file_name = f"params_set_{idx + 1}_test_performance.csv"
#     file_path = os.path.join(directory_path, file_name)
#     df.to_csv(file_path, index=False)


#     # Predict probabilities on the validation set
#     y_pred_proba = estimator.predict_proba(X_train)[:, 1]
    
#     # Calculate AUROC and AUPRC
#     auc_roc = roc_auc_score(y_train, y_pred_proba)
#     auc_prc = average_precision_score(y_train, y_pred_proba)
    
#     print(f"Train: {idx + 1} Params: {params}, AUROC: {auc_roc:.3f}, AUPRC: {auc_prc:.3f}")
    
#     # Optional: Measure additional performance metrics
#     df = measure_performance(y_train, y_pred_proba)  # Assuming this function is defined
    
#     # Save performance metrics to CSV
#     file_name = f"params_set_{idx + 1}_train_performance.csv"
#     file_path = os.path.join(directory_path, file_name)
#     df.to_csv(file_path, index=False)



