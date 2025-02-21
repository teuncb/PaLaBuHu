# source: https://github.com/socialfoundations/folktables
#explanation of features: 
# AGEP (Age),COW(Class of worker),SCHL (Educational attainment),MAR (Marital status),OCCP (Occupation),POBP(Place of birth),
# RELP(Relationship),WKHP(Usual hours worked per week past 12 months),SEX,RAC1P(Recoded detailed race code)

import pandas as pd
from folktables import ACSDataSource, ACSIncome 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def preprocess(add_synthetic_data=False):
    """Return X_train, X_p_train, y_train, X_dev, X_p_dev, y_dev, X_test, X_p_test, y_test, feature_names"""

    # view processed data: generate csv for CA
    # data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    # ca_data = data_source.get_data(states=["CA"], download=True)

    # features, label, _ = ACSIncome.df_to_pandas(ca_data)
    # ca_features.to_csv('ca_features.csv', index=False)
    # ca_labels.to_csv('ca_labels.csv', index=False)
    
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["AL"], download=True)
    features, label, group = ACSIncome.df_to_numpy(acs_data)

    # feature names in order
    feature_names=["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"]

    if add_synthetic_data == True:
        male_condition = features[:, 8] == 1

        # create new column
        promo_prob = np.zeros(features.shape[0])
        # number of male cells 
        num_male_cells = np.sum(male_condition)
        num_ones = int(0.95* num_male_cells)
        num_zeros = num_male_cells - num_ones
        ones_and_zeros = np.array([1]*num_ones + [0]*num_zeros)
        np.random.shuffle(ones_and_zeros)

        # 95% of men get a promotion probability of 1, all women get a promotion probability of 0
        promo_prob[male_condition] = ones_and_zeros
        features = np.column_stack((features, promo_prob))
        feature_names.append("PROMO")

    X_train, X_test, y_train, y_test = train_test_split(
    features, label, test_size=0.2, random_state=42)
    
    # split up training set in train and dev sets
    X_train, X_dev, y_train, y_dev = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)
    
    # assume same order as explained in paper, SEX is the 9th column
    X_p_train = X_train[:,8] - 1
    X_p_dev = X_dev[:,8] - 1
    X_p_test = X_test[:,8] - 1
    
    # remove protected attribute from train and test set
    X_train = np.delete(X_train,8,1)
    X_test = np.delete(X_test,8,1)
    X_dev = np.delete(X_dev,8,1)
       
    return X_train, X_p_train, y_train, X_dev, X_p_dev, y_dev, X_test, X_p_test, y_test, feature_names

#X_train, X_p_train, y_train,X_dev ,X_p_dev, y_dev, X_test,X_p_test, y_test = preprocess()