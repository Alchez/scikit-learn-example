import numpy as np  # More efficient numerical computation
import pandas as pd  # Manage dataframes

from sklearn import preprocessing  # Scale, transform, and wrangle data
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib  # Persist model for future use
from sklearn.metrics import mean_squared_error, r2_score  # Evaluate model performance
from sklearn.model_selection import GridSearchCV  # Perform cross-validation
from sklearn.model_selection import train_test_split  # Separate working datasets
from sklearn.pipeline import make_pipeline  # Perform cross-validation

# -------------------------------------------------------------------------

# IMPORTANT CONCEPTS

# -------------------------------------------------------------------------

# STANDARDIZATION:

# Standardization is the process of subtracting
# the means from each feature and then dividing
# by the feature standard deviations.

# Many algorithms assume that all features are
# centered around zero and have approximately
# the same variance.

# -------------------------------------------------------------------------

# CROSS-VALIDATION (WITH DATA PREPROCESSING):

# Cross-validation is a process for reliably
# estimating the performance of a method for
# building a model by training and evaluating
# your model multiple times using the same method.

# Practically, that "method" is simply a set
# of hyperparameters in this context.

# Steps for CV:

# 1. Split data into 'k' equal parts, or "folds".
# 2. Preprocess 'k-1' training folds.
# 3. Train the model on those 'k-1' folds.
# 4. Preprocess 'hold-out' fold using transformations from (2).
# 5. Evaluate it on the remaining hold-out fold.
# 6. Perform (2) - (5) 'k' times, with a different hold-out fold.
# 7. Aggregate the performance across all 'k' folds.
#
# The result is the model's performance metric.

# -------------------------------------------------------------------------

# HYPERPARAMETERS:

# There are two types of parameters: model parameters
# and hyperparameters. Model parameters can be
# learned directly from the data (i.e. regression
# coefficients), while hyperparameters cannot.

# Hyperparameters express "higher-level" structural
# information about the model, and they are
# typically set before training the model.

# -------------------------------------------------------------------------

dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')

# Separate training and target features
y = data.quality
X = data.drop('quality', axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=123,
                                                    stratify=y)

# Standardize the dataset for easier processing
scaler = preprocessing.StandardScaler().fit(X_train)

# Set up a cross-validation pipeline -
# substitute for creating the 'scaler' object
pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=100))

hyperparameters = {
    'randomforestregressor__max_features': ['sqrt'],
    'randomforestregressor__max_depth': [None]
}

# Cross-validation with the pipeline using 10 'folds'
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

# Fit and tune model - 'clf.best_params_' will
# contain the suggested hyperparameter config
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the performance of the selected model
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

# Save model to a .pkl file - to load the model
# later, use `joblib.load(_file_name_)`.
joblib.dump(clf, 'rf_regressor.pkl')
