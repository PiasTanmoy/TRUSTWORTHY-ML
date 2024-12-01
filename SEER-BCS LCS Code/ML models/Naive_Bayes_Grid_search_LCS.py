import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

import numpy as np
from Performance_metrics import measure_performance
import xgboost as xgb


# ... (load and preprocess data as before)
import os
os.chdir('/home/tanmoysarkar/Trustworthiness/SEER/seer_data/LCS')

X_train= np.load("processed_data_TS_2/X_train_normalized.npy")
y_train = np.load("processed_data_TS_2/y_train_normalized.npy")

X_valid = np.load("processed_data_TS_2/X_valid_normalized.npy")
y_valid = np.load("processed_data_TS_2/y_valid_normalized.npy")

X_test = np.load("processed_data_TS_2/X_test_normalized.npy")
y_test = np.load("processed_data_TS_2/y_test_normalized.npy")

directory_path = "Naive_Bayes/Grid_search_training"

if not os.path.exists(directory_path):
    os.makedirs(directory_path)

print("data loaded !")


# Define the parameter grid (example)
# Define parameter grids
param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
}


print(param_grid)
# Create a grid search object
# grid_search = GridSearchCV(estimator=adaboost, param_grid=param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='roc_auc', verbose=2)

# Initialize GaussianNB model
gnb = GaussianNB()

# Create GridSearchCV object
grid_search = GridSearchCV(
    estimator=gnb,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    n_jobs=-20,  # Use all available cores
    scoring='roc_auc',  # Evaluate using AUROC
    verbose=2  # Print progress messages
)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and AUROC score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("=====================================================================")
print("Best Parameters:", best_params)
print("=====================================================================")

# Evaluate the best model on the test set
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
file_path = os.path.join(directory_path, "best_model_training_log.csv")
results.to_csv(file_path, index=False)


d = {
    "y train": y_train,
    'y train pred': y_train_pred_proba
}

file_path = os.path.join(directory_path, "best_model_train_preds.csv")
pd.DataFrame(d).to_csv(file_path, index = False)


d = {
    "y test": y_test,
    'y test pred': y_test_pred_proba
}

file_path = os.path.join(directory_path, "best_model_test_preds.csv")
pd.DataFrame(d).to_csv(file_path, index = False)



d = {
    "y valid": y_valid,
    'y valid pred': y_valid_pred_proba
}

file_path = os.path.join(directory_path, "best_model_valid_preds.csv")
pd.DataFrame(d).to_csv(file_path, index = False)


