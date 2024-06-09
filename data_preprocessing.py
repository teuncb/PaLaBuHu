# source: https://github.com/socialfoundations/folktables

import pandas as pd
from folktables import ACSDataSource, ACSIncome 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def preprocess() -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    """Return X_train, y_train, X_dev, y_dev, X_test, y_test"""

    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["AL"], download=True)
    features, label, group = ACSIncome.df_to_numpy(acs_data)

    # can use group lables as indicator for e.g. race 

    X_train, X_test, y_train, y_test = train_test_split(
    features, label, test_size=0.2, random_state=42)
    
    # split up training set in train and dev sets
    X_train, X_dev, y_train, y_dev = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

    return X_train, y_train, X_dev, y_dev, X_test, y_test