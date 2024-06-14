import pandas as pd
from pygam import LogisticGAM, s, f, l, te
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import shap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data_preprocessing import X_train, X_p_train, y_train,X_dev ,X_p_dev, y_dev, X_test,X_p_test, y_test

def evaluate(model, x, y):
    y_pred = model.predict(x)
    return accuracy_score(y, y_pred)

'''Step 1: Training and evaluating the first iteration of GAM to tune the hyperparameters'''
def GAM(X_train, y_train, X_dev, y_dev):
    # Use linear term for continuous variables (spline terms can also be used if non-linear relationships are expected)
    # Use factor terms for categorical variables
    # Use te() for interactions between variables
    # LogiticGAM assumes binomial distribution with logit link function

    base_gam = LogisticGAM(    
    l(0) +       # AGEP (Age) - linear term for continuous variable
    f(1) +       # COW (Class of Worker) - factor term for categorical variable
    f(2) +       # SCHL (Educational Attainment) - factor term for categorical variable
    f(3) +       # MAR (Marital Status) - factor term for categorical variable
    f(4) +       # OCCP (Occupation) - factor term for categorical variable
    f(5) +       # POBP (Place of Birth) - factor term for categorical variable
    f(6) +       # RELP (Relationship) - factor term for categorical variable
    s(7) +       # WKHP (Working Hours per Week) - spline term for continuous variable
    f(8) +       # SEX (Sex) - factor term for categorical variable
    f(9)         # RAC1P (Race) - factor term for categorical variable))
    )

    base_gam_trained = base_gam.fit(X_train, y_train)
    base_acc = evaluate(base_gam_trained, X_dev, y_dev)

    # Uitzoeken: wat kunnen/willen we allemaal tunen? nu random waardes hier
    lams = np.random.rand(10, 10) # random points on [0, 1], with shape (100, 3)
    lams = lams * 6 - 3 # shift values to -3, 3
    lams = 10 ** lams # transforms values to 1e-3, 1e3

    tuned_gam = base_gam.gridsearch(X_train, y_train, lam=lams)

    return tuned_gam

# def logreg(X_train, y_train, X_dev, y_dev):

    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=10000))  # Increase max_iter
])
    base_log_trained = pipeline.fit(X_train, y_train)
    logreg_model = base_log_trained.named_steps['logreg']
    
    return logreg_model

'''Step 3: SHAP'''
def shap_explainer(model, X_train, X_test):
    # we use a subset to save computation time
    explainer = shap.KernelExplainer(model.predict, X_train[:100])

    # Calculate SHAP values for the test set
    shap_values = explainer.shap_values(X_test)

    # Visualize the SHAP values for a single prediction
    shap.initjs()
    shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test.iloc[0])

    # Summary plot for feature importance
    shap.summary_plot(shap_values[1], X_test)

    return shap_values

# shap_explainer(GAM_trained, X_train, X_test)

"""    # Find best parameters and model
    grid_search = GridSearchCV(estimator=base_gam, param_grid=param_grid, scoring='accuracy', cv=3) #nog aanvullen evt
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_regr = LogisticGAM(**best_params)

    best_regr_trained = best_regr.fit(X_train, y_train)

    # Predict again on development set and get RMSE
    tuned_acc = evaluate(best_regr, X_dev, y_dev)
    print(f'First Accuracy GAM: {base_acc}, Tuned Accuracy GAM: {tuned_acc}')"""

if __name__ == '__main__':
    # Results to compare
    GAM_trained = GAM(X_train, y_train, X_dev, y_dev)
    # LOG_trained = logreg(X_train, y_train, X_dev, y_dev)

    GAM_summary = GAM_trained.summary()
    # LOG_summary = LOG_trained.coef_

    GAM_tested = evaluate(GAM_trained, X_test, y_test)
    # LOG_tested = evaluate(LOG_trained, X_test, y_test)