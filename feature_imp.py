import pandas as pd
from pygam import LogisticGAM, s, f, l, te
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
# Step 1: Load and prepare data
from data_preprocessing import X_train, y_train, X_test, y_test

# Step 2: Define and train the GAM
# Use linear term for continuous variables (spline terms can also be used if non-linear relationships are expected)
# Use factor terms for categorical variables
# Use te() for interactions between variables
# LogiticGAM assumes binomial distribution with logit link function
gam = LogisticGAM(terms= l() + s() + f()).fit(X_train, y_train)
y_pred = gam.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
