import pandas as pd
import numpy as np
import shap


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
    # LOG_trained = logreg(X_train, y_train, X_dev, y_dev)

    # LOG_summary = LOG_trained.coef_

    pass
    # LOG_tested = evaluate(LOG_trained, X_test, y_test)