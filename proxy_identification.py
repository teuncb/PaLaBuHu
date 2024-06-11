import numpy as np
import pandas as pd
#from sklearn import

from data_preprocessing import preprocess
from feature_imp import GAM, logreg

def model(X_train, y_train, X_dev, y_dev, X_test, y_test, model_type="logreg"):
    """Return a model object for determining y based on X"""
    match model_type:
        case 'GAM':
            trained_model = GAM(X_train, y_train, X_dev, y_dev, X_test, y_test)
        case 'logreg':
            trained_model = logreg(X_train, y_train, X_dev, y_dev, X_test, y_test)
        case _:
            raise Exception(f'Model type "{model_type}" not recognised.')

    return trained_model

def get_feature_importance(model) -> np.array:
    """Returns an array of values for feature importance. Does not include the protected attribute (if it was in X)"""
    # TODO update after SHAP implemented
    pass

def palabuhu_values(importance_with_xp, importance_without_xp, importance_predicting_xp) -> np.array:
    """Given the three types of feature importance, determine the "proxy-ness" of all features.
    Returns an array containing a PaLaBuHu-value for all features except xp."""
    # TODO Arbitrary placeholder, we need to check how these features look to decide on this operation
    return abs(importance_with_xp - importance_without_xp) * importance_predicting_xp

if __name__ == '__main__': # TODO update after vector with protected attribute returned from preprocessing
    X_train, y_train, X_dev, y_dev, X_test, y_test = preprocess()
    protected_attribute = 'sex' # TODO placeholder

    model_with_xp = model(X_train, y_train, model_type='GAM')
    importance_with_xp = get_feature_importance(model_with_xp)

    # Slice xp out of the features
    X_train_without_xp = np.ones(len(X_train)) # TODO placeholder
    model_without_xp = model(X_train_without_xp, y_train)
    importance_without_xp = get_feature_importance(model_without_xp)

    # Use xp as target variable
    model_predicting_xp = model(X_train_without_xp, X_train[protected_attribute])
    importance_predicting_xp = get_feature_importance(model_predicting_xp)

    proxyness_per_feature = palabuhu_values(importance_with_xp, importance_without_xp, importance_predicting_xp)

    print(proxyness_per_feature)