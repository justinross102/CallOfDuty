import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer

## K Neighbors Regression - Justin
## Random Forest - Tyler
## Boosting - Bryce
## NN - Cameron
## SVM - Justin

cod = pd.read_csv("cod.csv")
cod = cod.drop('name', axis=1)
print(cod.head())

print(cod.describe())

# Calculate Accuracy by dividing hits by shots, handling division by zero
mask = cod['shots'] != 0  # Create a mask for non-zero shots
cod.loc[mask, 'Accuracy'] = cod['hits'] / cod['shots']
cod.loc[~mask, 'Accuracy'] = pd.NA  # Set Accuracy to pd.NA for zero shots

# Calculate Headshot Ratio by dividing headshots by kills, handling division by zero
mask = cod['kills'] != 0  # Create a mask for non-zero shots
cod.loc[mask, 'Headshot Ratio'] = cod['headshots'] / cod['kills']
cod.loc[~mask, 'Headshot Ratio'] = pd.NA  # Set Accuracy to pd.NA for zero shots

print(cod.columns)

X = cod.drop('wins', axis=1)
y = cod['wins']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=102)

pipe = Pipeline([
  ('impute', SimpleImputer(strategy = 'median')),
  ('poly', PolynomialFeatures(degree = 1, include_bias = False)),
  ('standard', StandardScaler()),
  ('model', KNeighborsRegressor(n_neighbors=5, weights='uniform'))
])

# fit pipeline to training data
pipe.fit(X_train, y_train)

# training MSE
train_preds = pipe.predict(X_train)
mse_train = mean_squared_error(y_train, train_preds)
print(f"Train MSE: {mse_train} \n")

# test MSE
test_preds = pipe.predict(X_test)
mse_test = mean_squared_error(y_test, test_preds)
print(f"Test MSE: {mse_test} \n")
print(f"Test RMSE: {np.sqrt(mse_test)} \n")

print(f"The variance of y_test is {np.var(y_test)} \n")
print(f"The sd of y_test is {np.std(y_test)} \n")

# pipe = Pipeline([
#   ('impute', SimpleImputer(strategy = 'mean')),
#   ('poly', PolynomialFeatures(degree = 2, include_bias = False)),
#   ('standard', StandardScaler()),
#   ('model', KNeighborsRegressor())
# ])

# # find optimal hyperparameters using GridSearchCV
# params = {
#   'impute__strategy':('mean','median'), 
#   'poly__degree':(1,2,3),
#   'model__n_neighbors': list(range(5, 101, 5)),
#   'model__weights': ['uniform', 'distance']
# }

# gs = GridSearchCV(pipe, param_grid = params, scoring = 'neg_mean_squared_error', cv = 10)
# gs.fit(X_train, y_train)

# # best hyperparameter combinations
# best_params = gs.best_params_
# print(f'Best parameters: {best_params}')
# print()

# # best MSE
# best_mse = -gs.best_score_ 
# print(f'Best MSE: {best_mse}')
# print()

# best_model = gs.best_estimator_
# preds = best_model.predict(X_test)

# # test MSE
# test_mse = mean_squared_error(y_test, preds)
# print(f'Test MSE: {test_mse}')
# print(f'Test RMSE: {np.sqrt(test_mse)}')
# print()