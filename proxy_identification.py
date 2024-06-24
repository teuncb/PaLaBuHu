from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
#from sklearn import

from data_preprocessing import preprocess
from classifiers import GAM, logreg
# from nn import train_nn
from feature_imp import shap_explainer

def model(X_train, y_train, X_dev, y_dev, X_test, y_test, model_type="logreg"):
    """Return a model object for determining y based on X"""
    match model_type:
        case 'GAM':
            trained_model = GAM(X_train, y_train, X_dev, y_dev)
        case 'logreg':
            trained_model = logreg(X_train, y_train, X_dev, y_dev)
        # case 'nn':
        #     trained_model = train_nn(X_train, y_train, X_dev, y_dev)
        case _:
            raise Exception(f'Model type "{model_type}" not recognised.')

    return trained_model

def get_feature_importance(model, X_train, X_test) -> np.array:
    """Returns an array of values for feature importance. Does not include the protected attribute (if it was in X)"""
    # TODO update after SHAP implemented
    shap_values = shap_explainer(model, X_train, X_test)
    # if model == 'model_with_xp':
    #     shap_values = np.delete(shap_values, 8, 1)
    return shap_values

# def palabuhu_values(importance_with_xp, importance_without_xp, importance_predicting_xp) -> np.array:
#     """Given the three types of feature importance, determine the "proxy-ness" of all features.
#     Returns an array containing a PaLaBuHu-value for all features except xp."""
#     # TODO Arbitrary placeholder, we need to check how these features look to decide on this operation
#     # make arrays the same length (no protected attribute)
#     importance_with_xp = np.delete(importance_with_xp,8,1)
#     print(importance_with_xp[0])
#     plbh_values = []
#     for i in range(len(importance_with_xp)):
#         im_w = importance_with_xp[i]
#         im_wo = importance_without_xp[i]
#         im_pred = importance_predicting_xp[i]
#         plbh = abs(im_w - im_wo) * im_pred
#         plbh_values.append(plbh)
#     return plbh_values

def palabuhu_values(importance_with_xp, importance_without_xp, importance_predicting_xp) -> np.array:
    """Given the three types of feature importance, determine the "proxy-ness" of all features.
    Returns an array containing a PaLaBuHu-value for all features except xp."""
    # TODO Arbitrary placeholder, we need to check how these features look to decide on this operation
    # make arrays the same length (no protected attribute)
    importance_with_xp = np.delete(importance_with_xp,8,1)
    plbh_values = []
    # average over instances
    feature_avg_with_xp = np.mean(importance_with_xp,axis=0)
    feature_avg_without_xp = np.mean(importance_without_xp,axis=0)
    feature_avg_pred_xp = np.mean(importance_predicting_xp,axis=0)

    for i in range(len(feature_avg_with_xp)):
        im_w = feature_avg_with_xp[i]
        im_wo = feature_avg_without_xp[i]
        im_pred =  feature_avg_pred_xp[i]
        plbh = abs(im_w - im_wo) * im_pred
        plbh_values.append(plbh)
    return plbh_values  


def plot_palabuhu(palabuhu_values, feature_names):
    del feature_names[8]
    # remove protected attribute from feature_names
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, palabuhu_values)

    # Customize the plot
    plt.title('Palabuhu feature proxyness')
    plt.xlabel('Feature')
    plt.ylabel('Palabuhu value')

    # Show the plot
    plt.show()

if __name__ == '__main__':
    (X_train_without_p, xp_train, y_train,
     X_dev_without_p, xp_dev, y_dev,
     X_test_without_p, xp_test, y_test,
     feature_names) = preprocess(True)

    # Generate X sets with protected attribute
    xp_train = xp_train.reshape(-1, 1)
    X_train_with_p = np.concatenate((X_train_without_p, xp_train), axis=1)
    xp_dev = xp_dev.reshape(-1, 1)
    X_dev_with_p = np.concatenate((X_dev_without_p, xp_dev), axis=1)
    xp_test = xp_test.reshape(-1, 1)
    X_test_with_p = np.concatenate((X_test_without_p, xp_test), axis=1)

    # model_with_xp = model(X_train_with_p, y_train, X_dev_with_p, y_dev, X_test_with_p, y_test, model_type='GAM')
    # # get predictions by testing trained model
    # importance_with_xp = get_feature_importance(model_with_xp, X_train_with_p, X_test_with_p)
    # print(f"importance_with_xp: {importance_with_xp}")
    #np.save('feature_imp_predicting_p.npy', importance_with_xp)

    # model_without_xp = model(X_train_without_p, y_train, X_dev_without_p, y_dev, X_test_without_p, y_test, model_type='GAM')
    # importance_without_xp = get_feature_importance(model_without_xp, X_train_without_p, X_test_without_p)
    # print(f"importance_without_xp: {importance_without_xp}")

    # #Use xp as target variable
    # model_predicting_xp = model(X_train_without_p, xp_train, X_dev_without_p, xp_dev, X_test_without_p, xp_test, model_type='GAM')
    # importance_predicting_xp = get_feature_importance(model_predicting_xp, X_train_without_p, X_test_without_p)
    # print(f"importance_predicting_xp: {importance_predicting_xp}")
  

    #print('WuHu gelukt!')

    # load np files
    importance_with_xp = np.load('analysis_results/feature_imp_with_p.npy')
    importance_without_xp = np.load('analysis_results/feature_imp_without_p.npy')
    importance_predicting_xp = np.load('analysis_results/feature_imp_predicting_p.npy')
    #print(importance_with_xp[0])

    proxyness_per_feature = palabuhu_values(importance_with_xp, importance_without_xp, importance_predicting_xp)
    #print(f"proxyness_per_feature: {proxyness_per_feature}")

    plot_palabuhu(proxyness_per_feature, feature_names)