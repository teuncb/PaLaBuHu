import pandas as pd
import numpy as np
from data_preprocessing import preprocess
from classifiers import GAM, logreg, evaluate
from sklearn.metrics import accuracy_score


# zet syn op True als je synthetische data erbij wilt
syn = False
X_train, X_p_train, y_train, X_dev, X_p_dev, y_dev, X_test, X_p_test, y_test, feature_names = preprocess(syn)

# concatenate protected attribute
#print(X_p_train)
X_p_train = X_p_train.reshape(-1,1)
X_train_with_p = np.concatenate((X_train,X_p_train), axis=1)
X_p_dev = X_p_dev.reshape(-1,1)
X_dev_with_p = np.concatenate((X_dev,X_p_dev), axis=1)
X_p_test = X_p_test.reshape(-1,1)
X_test_with_p = np.concatenate((X_test,X_p_test), axis=1)

# calculate accuracy GAM with normal dataset (no synthetic data)
GAM_trained = GAM(X_train_with_p, y_train, X_dev_with_p, y_dev)
GAM_summary = GAM_trained.summary()
GAM_tested = evaluate(GAM_trained, X_test_with_p, y_test)
print(f"GAM_tested: {GAM_tested}")

logreg_model = logreg(X_train_with_p, y_train, X_dev_with_p, y_dev)
logreg_tested = evaluate(logreg_model, X_test_with_p, y_test)
print(f"Logreg_tested: {logreg_tested}")



