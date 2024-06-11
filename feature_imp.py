import pandas as pd
from pygam import LogisticGAM, s, f, l, te
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from data_preprocessing import X_train, y_train, X_dev, y_dev, X_test, y_test

def evaluate(model, x, y):
    y_pred = model.predict(x)
    return accuracy_score(y, y_pred)

'''Step 1: Training and evaluating the first iteration of GAM to tune the hyperparameters'''
def GAM(X_train, y_train, X_dev, y_dev, X_test, y_test):
    # Use linear term for continuous variables (spline terms can also be used if non-linear relationships are expected)
    # Use factor terms for categorical variables
    # Use te() for interactions between variables
    # LogiticGAM assumes binomial distribution with logit link function
    #terms= l() + s() + f()
    base_gam = LogisticGAM()
    base_gam_trained = LogisticGAM().fit(X_train, y_train)
    base_acc = evaluate(base_gam_trained, X_dev, y_dev)

    # Uitzoeken: wat kunnen/willen we allemaal tunen? nu random waardes hier
    param_grid = {
        'lam': np.logspace(-3, 3, 7),
        'n_splines': [30, 40, 50]      
    }

    # Find best parameters and model
    grid_search = GridSearchCV(estimator=base_gam, param_grid=param_grid, scoring='accuracy') #nog aanvullen evt
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_regr = LogisticGAM(lam=best_params['lam'], n_splines=best_params['n_splines'])

    best_regr.fit(X_train, y_train)

    # Predict again on development set and get RMSE
    tuned_acc = evaluate(best_regr, X_dev, y_dev)
    print(f'First Accuracy GAM: {base_acc}, Tuned Accuracy GAM: {tuned_acc}')
    return best_regr

def logreg(X_train, y_train, X_dev, y_dev, X_test, y_test):
    base_log = LogisticRegression()
    base_log_trained = LogisticRegression().fit(X_train, y_train)
    base_acc = evaluate(base_log_trained, X_dev, y_dev)

    # Uitzoeken: wat kunnen/willen we allemaal tunen? nu random waardes hier
    param_grid = {
            
    }

    # Find best parameters and model
    grid_search = GridSearchCV(estimator=base_log, param_grid=param_grid, scoring='accuracy') #nog aanvullen evt
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_regr = LogisticRegression(**best_params)

    best_regr.fit(X_train, y_train)

    # Predict again on development set and get RMSE
    tuned_acc = evaluate(best_regr, X_dev, y_dev)
    print(f'First Accuracy LogReg: {base_acc}, Tuned Accuracy LogReg: {tuned_acc}')

    return best_regr
# Take best model and predict on test values and report RMSE
# TODO

'''Step 3: PFI'''
