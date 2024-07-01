import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import shap
from scipy.stats import pearsonr

def feature_correlation(X_train_with_p, X_dev_with_p, X_test_with_p):
    """Classic method for finding proxies: check how much the features correlate with each other."""
    X_all = pd.DataFrame(np.concatenate((X_train_with_p, X_dev_with_p, X_test_with_p), axis=0))
    correlation = X_all.corr()

    print(X_all[:10])

    feature_tags = ["Age", "Class", "Education", "Married?", "Occupation",
                    "Birth place", "Relationship", "Work hours", "Race", "Sex (xp)"]
    feature_names = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "RAC1P", "SEX"]

    fig, ax = plt.subplots()
    mappable = ax.matshow(correlation)
    # ax.set_xticklabels(feature_names)
    # ax.set_yticklabels(feature_names)

    # Add a color bar (legend) with vertical orientation
    cb = plt.colorbar(mappable)
    cb.ax.tick_params(labelsize=14)

    plt.xticks(range(10), feature_names, rotation=45)
    plt.yticks(range(10), feature_names, rotation=45)
    
    plt.show()

    return correlation

def shap_explainer(model, X_train, X_test):
    # we use a subset to save computation time
    explainer = shap.KernelExplainer(model.predict, X_train[:100])

    # Calculate SHAP values for the test set
    shap_values = explainer.shap_values(X_test)

    # Visualize the SHAP values for a single prediction
    shap.initjs()
    #shap.force_plot(explainer.expected_value, shap_values[1], X_test.iloc[0])

    # Summary plot for feature importance
    shap.summary_plot(shap_values, X_test)

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
    # LOG_tested = evaluate(LOG_trained, X_test, y_test)

    from data_preprocessing import preprocess

    (X_train_without_p, xp_train, y_train,
     X_dev_without_p, xp_dev, y_dev,
     X_test_without_p, xp_test, y_test,
     feature_names) = preprocess()

    # Generate X sets with protected attribute
    xp_train = xp_train.reshape(-1, 1)
    X_train_with_p = np.concatenate((X_train_without_p, xp_train), axis=1)
    xp_dev = xp_dev.reshape(-1, 1)
    X_dev_with_p = np.concatenate((X_dev_without_p, xp_dev), axis=1)
    xp_test = xp_test.reshape(-1, 1)
    X_test_with_p = np.concatenate((X_test_without_p, xp_test), axis=1)

    feature_correlation(X_train_with_p, X_dev_with_p, X_test_with_p)