from pygam import LogisticGAM, s, f, l, te
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import numpy as np

def evaluate(model, x, y):
    y_pred = model.predict(x)
    return accuracy_score(y, y_pred)

'''Step 1: Training and evaluating the first iteration of GAM to tune the hyperparameters'''
def GAM(X_train, y_train, X_dev, y_dev):
    # Use linear term for continuous variables (spline terms can also be used if non-linear relationships are expected)
    # Use factor terms for categorical variables
    # Use te() for interactions between variables
    # LogiticGAM assumes binomial distribution with logit link function

    base_gam = LogisticGAM(
    l(0) +       # AGEP (Age) - linear term for continuous variable
    f(1) +       # COW (Class of Worker) - factor term for categorical variable
    f(2) +       # SCHL (Educational Attainment) - factor term for categorical variable
    f(3) +       # MAR (Marital Status) - factor term for categorical variable
    f(4) +       # OCCP (Occupation) - factor term for categorical variable
    f(5) +       # POBP (Place of Birth) - factor term for categorical variable
    f(6) +       # RELP (Relationship) - factor term for categorical variable
    s(7) +       # WKHP (Working Hours per Week) - spline term for continuous variable
    f(8) +       # SEX (Sex) - factor term for categorical variable
    f(9)         # RAC1P (Race) - factor term for categorical variable))
    )

    base_gam_trained = base_gam.fit(X_train, y_train)
    base_acc = evaluate(base_gam_trained, X_dev, y_dev)

    # Uitzoeken: wat kunnen/willen we allemaal tunen? nu random waardes hier
    lams = np.random.rand(10, 10) # random points on [0, 1], with shape (100, 3)
    lams = lams * 6 - 3 # shift values to -3, 3
    lams = 10 ** lams # transforms values to 1e-3, 1e3

    tuned_gam = base_gam.gridsearch(X_train, y_train, lam=lams)

    return tuned_gam


def logreg(X_train, y_train, X_dev, y_dev):
    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=10000))  # Increase max_iter
])
    #base_log_trained = pipeline.fit(X_train, y_train)
    logreg_model = pipeline.fit(X_train, y_train)
    #logreg_model = base_log_trained.named_steps['logreg']

    return logreg_model

if __name__ == '__main__':
    GAM_trained = GAM(X_train, y_train, X_dev, y_dev)
    GAM_summary = GAM_trained.summary()
    GAM_tested = evaluate(GAM_trained, X_test, y_test)
