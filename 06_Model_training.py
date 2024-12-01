
# **QSAR Model
"""

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

"""# Read in data"""

dataset = pd.read_csv('Merged_data_06_bioactivity_data_3class_pIC50_pubchem_fp.csv')
dataset

# if the dataset contains NaN values than apply this
dataset = dataset.dropna()

# Define the target variable Y
Y = dataset.iloc[:, -1]

# after deleting the NaN values, and save it to csv files

dataset.to_csv('Merged_data_06_bioactivity_data_3class_pIC50_pubchem_fp_nan.csv')

# Remove low variance features
def remove_low_variance(input_data, threshold=0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]

X = dataset.drop(['pIC50'], axis=1)
X = remove_low_variance(X, threshold=0.1)

X

# Save the descriptor list
X.to_csv('descriptor_list.csv', index=False)

# Data split (80/20 ratio)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2021)

!nvidia-smi

# Ensembel Model (HGBRegressor & NuSVR)
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score, mean_squared_log_error
from sklearn.ensemble import HistGradientBoostingRegressor as HGBRegressor
from sklearn.svm import NuSVR
from sklearn.ensemble import VotingRegressor
import pickle

# Assuming 'X_train', 'X_test', 'Y_train', 'Y_test' are defined

# Parameter grid to search for HGBRegressor
param_dist_hgb = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_iter': [50, 100, 150, 200, 250],
    'max_depth': [3, 4, 5, 6, 7],
    'min_samples_leaf': [1, 3, 5, 7, 10],
    'max_leaf_nodes': [10, 20, 30, 40],
    'l2_regularization': [0, 0.001, 0.01, 0.1, 1],
}

# Parameter grid to search for NuSVR
param_dist_nusvr = {
    'C': [0.1, 1, 10],
    'nu': [0.1, 0.3, 0.5],
    'kernel': ['linear', 'rbf'],
}

# Create the HGBRegressor model
hgb_model = HGBRegressor()

# Create the RandomizedSearchCV object for HGBRegressor
hgb_random_search = RandomizedSearchCV(hgb_model, param_dist_hgb, n_iter=100, cv=5, scoring='r2', n_jobs=-1)

# Fit the RandomizedSearchCV object on the training data
print("Training HGBRegressor started...")
hgb_random_search.fit(X_train, Y_train)
print("Training HGBRegressor completed.")

# Get the best HGBRegressor model
best_hgb_model = hgb_random_search.best_estimator_

# Create the NuSVR model
nusvr_model = NuSVR()

# Create the RandomizedSearchCV object for NuSVR
nusvr_random_search = RandomizedSearchCV(nusvr_model, param_dist_nusvr, n_iter=100, cv=5, scoring='r2', n_jobs=-1)

# Fit the RandomizedSearchCV object on the training data
print("Training NuSVR started...")
nusvr_random_search.fit(X_train, Y_train)
print("Training NuSVR completed.")

# Get the best NuSVR model
best_nusvr_model = nusvr_random_search.best_estimator_

# Create the ensemble model
ensemble_model = VotingRegressor([('hgb', best_hgb_model), ('nusvr', best_nusvr_model)])

# Fit the ensemble model on the training data
print("Training Ensemble started...")
ensemble_model.fit(X_train, Y_train)
print("Training Ensemble completed.")

# Save the ensemble model using pickle
pickle.dump(ensemble_model, open('Combine_IL1_2_ensembel.pkl', 'wb'))

# Make predictions on the training and test data using the ensemble model
train_predictions = ensemble_model.predict(X_train)
test_predictions = ensemble_model.predict(X_test)

# Calculate Pearson correlation and Spearman's rank correlation
pearson_corr_train, _ = pearsonr(Y_train, train_predictions)
pearson_corr_test, _ = pearsonr(Y_test, test_predictions)

spearman_corr_train, _ = spearmanr(Y_train, train_predictions)
spearman_corr_test, _ = spearmanr(Y_test, test_predictions)

# Calculate performance metrics for the training set
metrics_train = {
    "R-squared": r2_score(Y_train, train_predictions),
    "Mean Squared Error": mean_squared_error(Y_train, train_predictions),
    "Root Mean Squared Error": np.sqrt(mean_squared_error(Y_train, train_predictions)),
    "Mean Absolute Error": mean_absolute_error(Y_train, train_predictions),
    "Explained Variance Score": explained_variance_score(Y_train, train_predictions),
    "Pearson Correlation": pearson_corr_train,
    "Spearman's Rank Correlation": spearman_corr_train,
    "Mean Absolute Percentage Error": np.mean(np.abs((Y_train - train_predictions) / Y_train)) * 100,
    "Adjusted R-squared": 1 - (1 - r2_score(Y_train, train_predictions)) * (len(Y_train) - 1) / (len(Y_train) - X_train.shape[1] - 1),
    "Mean Squared Logarithmic Error": mean_squared_log_error(Y_train, train_predictions),
    "Median Absolute Error": np.median(np.abs(Y_train - train_predictions)),
}

# Calculate performance metrics for the test set
metrics_test = {
    "R-squared": r2_score(Y_test, test_predictions),
    "Mean Squared Error": mean_squared_error(Y_test, test_predictions),
    "Root Mean Squared Error": np.sqrt(mean_squared_error(Y_test, test_predictions)),
    "Mean Absolute Error": mean_absolute_error(Y_test, test_predictions),
    "Explained Variance Score": explained_variance_score(Y_test, test_predictions),
    "Pearson Correlation": pearson_corr_test,
    "Spearman's Rank Correlation": spearman_corr_test,
    "Mean Absolute Percentage Error": np.mean(np.abs((Y_test - test_predictions) / Y_test)) * 100,
    "Adjusted R-squared": 1 - (1 - r2_score(Y_test, test_predictions)) * (len(Y_test) - 1) / (len(Y_test) - X_test.shape[1] - 1),
    "Mean Squared Logarithmic Error": mean_squared_log_error(Y_test, test_predictions),
    "Median Absolute Error": np.median(np.abs(Y_test - test_predictions)),
}

# Convert metrics to DataFrame and save to CSV file
metrics_df = pd.DataFrame({'Training Set': metrics_train, 'Test Set': metrics_test})
metrics_df.to_csv('performance_metrics.csv', index=False)

# Print the performance metrics
print("\nTraining Set Metrics:")
print(metrics_train)

print("\nTest Set Metrics:")
print(metrics_test)

"""# Save Model as Pickle Object"""

import pickle

pickle.dump(ensemble, open('Combine_IL1_2.pkl', 'wb'))

pip install --upgrade scikit-learn

import sklearn

print(sklearn.__version__)
