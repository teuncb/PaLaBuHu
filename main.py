import pandas as pd
import numpy as np
from data_preprocessing import preprocess

X_train, X_p_train, y_train,X_dev ,X_p_dev, y_dev, X_test,X_p_test, y_test, feature_names = preprocess()

# concatenate protected attribute
#print(X_p_train)
X_p_train = X_p_train.reshape(-1,1)
X_train_with_p = np.concatenate((X_train,X_p_train), axis=1)
#print(X_train_with_p[:,9])