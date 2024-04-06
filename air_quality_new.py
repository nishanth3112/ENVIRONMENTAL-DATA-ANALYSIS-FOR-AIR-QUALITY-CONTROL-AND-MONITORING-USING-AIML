# -*- coding: utf-8 -*-
"""air quality new.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DFI8Ta9po84Hhp0r4mz41KbELk6AMfco
"""

!pip install matplotlib-venn

!pip install matplotlib

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as snc
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
#from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.applications import resnet
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from tensorflow.python.keras.layers.core import Dense, Activation

air_quality = pd.read_csv("/content/AirQualityUCI.csv", sep=";", decimal=",")

air_quality.dropna(axis=0, how= 'all', inplace=True)
air_quality.dropna(axis=1, inplace=True)

air_quality.replace(to_replace= -200, value= np.NaN, inplace= True)
air_quality

# Convert the entire DataFrame to numeric, excluding non-convertible values
air_quality = air_quality.apply(pd.to_numeric, errors='coerce')

# Fill NaN values with the mean
air_quality.fillna(air_quality.mean(), inplace=True)

air_quality.loc[:,'Date']=air_quality['Date']
air_quality.head()

import pandas as pd
from datetime import datetime

# Assuming air_quality is your DataFrame and 'Time' is the column causing the error
# Convert 'Time' column to datetime objects using pd.to_datetime
air_quality['Time'] = pd.to_datetime(air_quality['Time'], errors='coerce')

# Display data types after conversion
print(air_quality.dtypes)

air_quality['Date']=air_quality['Date'].astype(float)
air_quality.dtypes
air_quality

air_quality.loc[:,'Time']=air_quality['Time']

import pandas as pd
from datetime import datetime

# Assuming air_quality is your DataFrame and 'Time' is the column causing the error
# Convert 'Time' column to datetime objects using pd.to_datetime
air_quality['Time'] = pd.to_datetime(air_quality['Time'], errors='coerce')

# Display data types after conversion
print(air_quality.dtypes)

air_quality.tail()

air_quality2=air_quality.corr('pearson')
air_quality2

abs(air_quality2['T']).sort_values(ascending=False)

from sklearn.preprocessing import MinMaxScaler

# num = air_quality.keys()
# scaler = MinMaxScaler()
# scaler.fit(air_quality[num])
# air_quality[num] = scaler.transform(air_quality[num])

features=air_quality
target=air_quality['T']

features=features.drop('Date',axis=1)
features=features.drop('Time',axis=1)
features=features.drop('T',axis=1)
features=features.drop('CO(GT)',axis=1)
features=features.drop('PT08.S5(O3)',axis=1)
features=features.drop('NMHC(GT)',axis=1)
features=features.drop('PT08.S1(CO)',axis=1)
features.tail()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target)
y_test.tail()

"""**FEATURE SCALING**"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""**XG BOOST - EXTREME GRADIENT BOOST**"""

!pip install xgboost

from xgboost import XGBRegressor
from sklearn.metrics import r2_score
# Initialize the XGBoost model with appropriate hyperparameters
xgb_regressor = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Fit the XGBoost model on the training data
xgb_regressor.fit(X_train, y_train)

# Predict the target variable on the test set
y_pred_xgb = xgb_regressor.predict(X_test)

# Calculate the R-squared score
accuracy_xgb = r2_score(y_test, y_pred_xgb)

print("Accuracy of the XGBoost algorithm: {:.2f}".format(accuracy_xgb))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize lists to store metrics
xgb_mae = []
xgb_mse = []
xgb_r2 = []

# Train and evaluate XGBoost model over different iterations
for i in range(10, 101, 10):
    xgb_regressor.set_params(n_estimators=i)
    xgb_regressor.fit(X_train, y_train)
    y_pred_xgb = xgb_regressor.predict(X_test)

    # Calculate evaluation metrics
    xgb_mae.append(mean_absolute_error(y_test, y_pred_xgb))
    xgb_mse.append(mean_squared_error(y_test, y_pred_xgb))
    xgb_r2.append(r2_score(y_test, y_pred_xgb))

# Plotting for XGBoost
plt.figure(figsize=(10, 6))
plt.plot(range(10, 101, 10), xgb_mae, label='Mean Absolute Error', marker='o')
plt.plot(range(10, 101, 10), xgb_mse, label='Mean Squared Error', marker='o')
plt.plot(range(10, 101, 10), xgb_r2, label='R-squared', marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Scores')
plt.title('XGBoost Metrics over Iterations')
plt.legend()
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Compute regression metrics
mae = mean_absolute_error(y_test, y_pred_xgb)
mse = mean_squared_error(y_test, y_pred_xgb)
r2 = r2_score(y_test, y_pred_xgb)

# Print regression metrics
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize lists to store metrics
xgb_mae = []
xgb_mse = []
xgb_r2 = []

# Train and evaluate XGBoost model over different iterations
for i in range(10, 101, 10):
    xgb_regressor.set_params(n_estimators=i)
    xgb_regressor.fit(X_train, y_train)
    y_pred_xgb = xgb_regressor.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred_xgb)
    mse = mean_squared_error(y_test, y_pred_xgb)
    r2 = r2_score(y_test, y_pred_xgb)

    xgb_mae.append(mae)
    xgb_mse.append(mse)
    xgb_r2.append(r2)

# Print the metrics
for i in range(len(xgb_mae)):
    print(f"Iterations: {(i+1)*10}")
    print(f"Mean Absolute Error: {xgb_mae[i]}")
    print(f"Mean Squared Error: {xgb_mse[i]}")
    print(f"R-squared: {xgb_r2[i]}")
    print()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

# Function to plot learning curve for XGBoost
def plot_learning_curve_xgboost(estimator, X, y, cv=5):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
    plt.plot(train_sizes, test_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15)
    plt.title('XGBoost Learning Curve')
    plt.xlabel('Number of training examples')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

# Function to plot confusion matrix for XGBoost
def plot_confusion_matrix_xgboost(estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_mat, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title('XGBoost Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks([0, 1], ['Class 0', 'Class 1'])
    plt.yticks([0, 1], ['Class 0', 'Class 1'])
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(j, i, format(conf_mat[i, j], 'd'), horizontalalignment="center", color="white" if conf_mat[i, j] > conf_mat.max() / 2 else "black")
    plt.show()

# Generate synthetic dataset for example
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost classifier
xgb_classifier = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Fit XGBoost classifier on training data
xgb_classifier.fit(X_train, y_train)

# Plot learning curve for XGBoost
plot_learning_curve_xgboost(xgb_classifier, X_train, y_train)

# Plot confusion matrix for XGBoost
plot_confusion_matrix_xgboost(xgb_classifier, X_test, y_test)

"""**CATBOOST**"""

!pip install catboost



from catboost import CatBoostRegressor

# Initialize the CatBoost model with appropriate hyperparameters
catboost_regressor = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=3, random_state=42)

# Fit the CatBoost model on the training data
catboost_regressor.fit(X_train, y_train, verbose=0)

# Predict the target variable on the test set
y_pred_catboost = catboost_regressor.predict(X_test)

# Calculate the R-squared score
accuracy_catboost = r2_score(y_test, y_pred_catboost)

print("Accuracy of the CatBoost algorithm: {:.2f}".format(accuracy_catboost))

!pip install catboost

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize the CatBoost model with appropriate hyperparameters
catboost_regressor = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=3, random_state=42)

# Fit the CatBoost model on the training data
catboost_regressor.fit(X_train, y_train, verbose=0)

# Predict the target variable on the test set
y_pred_catboost = catboost_regressor.predict(X_test)

# Calculate mean absolute error (MAE)
mae_catboost = mean_absolute_error(y_test, y_pred_catboost)

# Calculate mean squared error (MSE)
mse_catboost = mean_squared_error(y_test, y_pred_catboost)

# Calculate R-squared score
r2_catboost = r2_score(y_test, y_pred_catboost)

print("Mean Absolute Error (MAE) of the CatBoost algorithm: {:.2f}".format(mae_catboost))
print("Mean Squared Error (MSE) of the CatBoost algorithm: {:.2f}".format(mse_catboost))
print("R-squared score of the CatBoost algorithm: {:.2f}".format(r2_catboost))

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Initialize lists to store metrics
catboost_mae = []
catboost_mse = []
catboost_r2 = []

# Train and evaluate CatBoost model over different iterations
for i in range(10, 101, 10):
    # Initialize the CatBoost model with appropriate hyperparameters
    catboost_regressor = CatBoostRegressor(iterations=i, learning_rate=0.1, depth=3, random_state=42, verbose=0)

    # Fit the CatBoost model on the training data
    catboost_regressor.fit(X_train, y_train)

    # Predict the target variable on the test set
    y_pred_catboost = catboost_regressor.predict(X_test)

    # Calculate evaluation metrics
    catboost_mae.append(mean_absolute_error(y_test, y_pred_catboost))
    catboost_mse.append(mean_squared_error(y_test, y_pred_catboost))
    catboost_r2.append(r2_score(y_test, y_pred_catboost))

# Plotting for CatBoost
plt.figure(figsize=(10, 6))
plt.plot(range(10, 101, 10), catboost_mae, label='Mean Absolute Error', marker='o')
plt.plot(range(10, 101, 10), catboost_mse, label='Mean Squared Error', marker='o')
plt.plot(range(10, 101, 10), catboost_r2, label='R-squared', marker='o')
plt.xlabel('Number of Iterations')
plt.ylabel('Scores')
plt.title('CatBoost Metrics over Iterations')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from catboost import CatBoostClassifier

# Function to plot learning curve for CatBoost
def plot_learning_curve_catboost(estimator, X, y, cv=5):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
    plt.plot(train_sizes, test_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15)
    plt.title('CatBoost Learning Curve')
    plt.xlabel('Number of training examples')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

# Function to plot confusion matrix for CatBoost
def plot_confusion_matrix_catboost(estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_mat, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title('CatBoost Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks([0, 1], ['Class 0', 'Class 1'])
    plt.yticks([0, 1], ['Class 0', 'Class 1'])
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(j, i, format(conf_mat[i, j], 'd'), horizontalalignment="center", color="white" if conf_mat[i, j] > conf_mat.max() / 2 else "black")
    plt.show()

# Generate synthetic dataset for example
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CatBoost classifier
catboost_classifier = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=3, random_state=42, verbose=0)

# Fit CatBoost classifier on training data
catboost_classifier.fit(X_train, y_train)

# Plot learning curve for CatBoost
plot_learning_curve_catboost(catboost_classifier, X_train, y_train)

# Plot confusion matrix for CatBoost
plot_confusion_matrix_catboost(catboost_classifier, X_test, y_test)

"""**RANDOM FOREST**"""

from sklearn.ensemble import RandomForestRegressor

regressor=RandomForestRegressor()
regressor.fit(X_train,y_train)

print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_train, y_train)))

from sklearn.model_selection import cross_val_score
score=cross_val_score(regressor,X_train,y_train,cv=5)

score.mean()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Predictions on the test set
y_pred = regressor.predict(X_test)

# Compute evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Plotting the graph for mean_absolute_error, mean_squared_error, and r2_score
plt.figure(figsize=(10, 6))

# Mean Absolute Error
plt.subplot(1, 3, 1)
plt.bar(['MAE'], [mae], color='blue')
plt.title('Mean Absolute Error')
plt.ylabel('Error')

# Mean Squared Error
plt.subplot(1, 3, 2)
plt.bar(['MSE'], [mse], color='green')
plt.title('Mean Squared Error')
plt.ylabel('Error')

# R-squared
plt.subplot(1, 3, 3)
plt.bar(['R^2'], [r2], color='red')
plt.title('R-squared')
plt.ylabel('Score')

plt.tight_layout()
plt.show()

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Initialize and train the RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)

# Predict on the training set
y_pred_train = regressor.predict(X_train)

# Compute the R-squared score on the training set
r2_train = r2_score(y_train, y_pred_train)
print("Coefficient of determination R^2 on the training set:", r2_train)

# Predict on the test set
y_pred_test = regressor.predict(X_test)

# Compute the R-squared score on the test set
r2_test = r2_score(y_test, y_pred_test)
print("Coefficient of determination R^2 on the test set:", r2_test)

import matplotlib.pyplot as plt

# Plotting actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')  # Plotting the diagonal line
plt.title('Actual vs. Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Function to plot learning curve for Random Forest
def plot_learning_curve_random_forest(estimator, X, y, cv=5):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
    plt.plot(train_sizes, test_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15)
    plt.title('Random Forest Learning Curve')
    plt.xlabel('Number of training examples')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

# Function to plot confusion matrix for Random Forest
def plot_confusion_matrix_random_forest(estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_mat, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title('Random Forest Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks([0, 1], ['Class 0', 'Class 1'])
    plt.yticks([0, 1], ['Class 0', 'Class 1'])
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(j, i, format(conf_mat[i, j], 'd'), horizontalalignment="center", color="white" if conf_mat[i, j] > conf_mat.max() / 2 else "black")
    plt.show()

# Generate synthetic dataset for example
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit Random Forest classifier on training data
random_forest_classifier.fit(X_train, y_train)

# Plot learning curve for Random Forest
plot_learning_curve_random_forest(random_forest_classifier, X_train, y_train)

# Plot confusion matrix for Random Forest
plot_confusion_matrix_random_forest(random_forest_classifier, X_test, y_test)

"""**NAIVE BAYES**"""

# Importing necessary libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Generating synthetic dataset for example
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing Naive Bayes classifier (GaussianNB)
nb_classifier = GaussianNB()

# Training the classifier
nb_classifier.fit(X_train, y_train)

# Making predictions on the test set
y_pred = nb_classifier.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import learning_curve

# Generating synthetic dataset for example
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing Naive Bayes classifier (GaussianNB)
nb_classifier = GaussianNB()

# Training the classifier
nb_classifier.fit(X_train, y_train)

# Making predictions on the test set
y_pred = nb_classifier.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plotting learning curve
train_sizes, train_scores, test_scores = learning_curve(nb_classifier, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
plt.plot(train_sizes, test_mean, label='Cross-validation score')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15)
plt.title('Learning Curve')
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.show()

# Plotting confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(conf_mat, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks([0, 1], ['Class 0', 'Class 1'])
plt.yticks([0, 1], ['Class 0', 'Class 1'])
for i in range(conf_mat.shape[0]):
    for j in range(conf_mat.shape[1]):
        plt.text(j, i, format(conf_mat[i, j], 'd'), horizontalalignment="center", color="white" if conf_mat[i, j] > conf_mat.max() / 2 else "black")
plt.show()