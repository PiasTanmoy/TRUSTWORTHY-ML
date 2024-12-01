import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from Performance_metrics import measure_performance

# ... (load and preprocess data as before)
import os
os.chdir('/home/tanmoysarkar/Trustworthiness/SEER')

X_train= np.load("seer_data/LCS/processed_data_TS_2/X_train_normalized.npy")
y_train = np.load("seer_data/LCS/processed_data_TS_2/y_train_normalized.npy")

X_valid = np.load("seer_data/LCS/processed_data_TS_2/X_valid_normalized.npy")
y_valid = np.load("seer_data/LCS/processed_data_TS_2/y_valid_normalized.npy")

X_test = np.load("seer_data/LCS/processed_data_TS_2/X_test_normalized.npy")
y_test = np.load("seer_data/LCS/processed_data_TS_2/y_test_normalized.npy")

print("data loaded !")

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


# Create a Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Create a grid search object
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='roc_auc', verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

directory_path = "seer_data/LCS/RF/RF_grid_search"
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

# Get the best model parameters based on AUROC
best_params = grid_search.best_params_


print("=====================================================================")
print("Best Parameters:", best_params)
print("=====================================================================")

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


d = {
    "y train": y_train,
    'y train pred': y_train_pred_proba
}

file_path = os.path.join(directory_path, "grid_search_train_preds.csv")
pd.DataFrame(d).to_csv(file_path, index = False)



d = {
    "y test": y_test,
    'y test pred': y_test_pred_proba
}

file_path = os.path.join(directory_path, "grid_search_test_preds.csv")
pd.DataFrame(d).to_csv(file_path, index = False)



d = {
    "y valid": y_valid,
    'y valid pred': y_valid_pred_proba
}

file_path = os.path.join(directory_path, "grid_search_valid_preds.csv")
pd.DataFrame(d).to_csv(file_path, index = False)