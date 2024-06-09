import pandas as pd
from pygam import LogisticGAM, s, f, l, te
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import GridSearchCV
from data_preprocessing import X_train, y_train, X_test, y_test, X_dev, y_dev

def evaluate(model, x, y):
    y_pred = model.predict(x)
    return np.sqrt(mean_squared_error(y, y_pred))

'''Step 1: Training and evaluating the first iteration of GAM to tune the hyperparameters'''
# Use linear term for continuous variables (spline terms can also be used if non-linear relationships are expected)
# Use factor terms for categorical variables
# Use te() for interactions between variables
# LogiticGAM assumes binomial distribution with logit link function
base_gam = LogisticGAM(terms= l() + s() + f()).fit(X_train, y_train)
base_RMSE = evaluate(base_gam, X_dev, y_dev)

'''Step 2: Tune hyperparameters with RMSE as evaluation'''
# Uitzoeken: wat kunnen/willen we allemaal tunen?
param_grid = {
    'lam': (),
    'n_splines': []      
}

# Find best parameters and model
grid_search = GridSearchCV(estimator=base_gam, param_grid=param_grid, scoring=mean_squared_error) #nog aanvullen evt
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_regr = LogisticGAM(**best_params)

best_regr.fit(X_train, y_train)

# Predict again on development set and get RMSE
tuned_RMSE = evaluate(best_regr, X_dev, y_dev)
print('First RMSE: {base_RMSE}, Tuned RMSE: {tuned_RMSE}')

# Take best model and predict on test values and report RMSE
# TODO

'''Step 3: PFI'''
