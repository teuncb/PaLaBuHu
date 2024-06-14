import pandas as pd
import numpy as np
from data_preprocessing import preprocess

X_train, X_p_train, y_train, X_dev, X_p_dev, y_dev, X_test, X_p_test, y_test, feature_names = preprocess()

# concatenate protected attribute
#print(X_p_train)
X_p_train = X_p_train.reshape(-1,1)
X_train_with_p = np.concatenate((X_train,X_p_train), axis=1)


## add synthetic data
# if person is a man -> 95% promo_prob = 1, 5% promo_prob = 0
# if person is a woman -> promo_prob = 0
# (ik neem aan dat man 1 is en vrouw 2)
male_condition = X_train[:, 8] == 1
# nieuwe column aanmaken
promo_prob = np.zeros(X_train.shape[0])
# number of male cells 
num_male_cells = np.sum(male_condition)
num_ones = int(0.95* num_male_cells)
num_zeros = num_male_cells - num_ones
ones_and_zeros = np.array([1]*num_ones + [0]*num_zeros)
np.random.shuffle(ones_and_zeros)
# 95% van de mannen krijgen een promotie probability van 1, alle vrouwen krijgen een promotie prob van 0
promo_prob[male_condition] = ones_and_zeros
X_train = np.column_stack((X_train, promo_prob))


