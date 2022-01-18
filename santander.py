import logging

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF

from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score as accuracy


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


train = pd.read_csv( 'adversarial-validation/data/santander/train.csv' )
test = pd.read_csv( 'adversarial-validation/data/santander/test.csv' )

train['TARGET'] = 1
test['TARGET'] = 0

# Combine train and test datasets
data = pd.concat(( train, test ))

# Randomize the rows of data
data = data.sample(frac=1).reset_index(drop=True)

# Define X and y
X = data.drop( [ 'TARGET', 'ID' ], axis = 1 )
y = data.TARGET


# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = .3,
    random_state=42)

# Logistic Regression model
logger.info(f"Fit logistic regression model")
lr = LR()
lr.fit(X_train, y_train)

logger.info(f"Using fitted logistic regression model, compute AUC")
p = lr.predict_proba( X_test )[:,1]
auc = AUC( y_test, p )
print(f"Logistic regression AUC: {auc:.3f}")


# Random Forest model
logger.info(f"Fit random forest model")
rf = RF()
rf.fit(X_train, y_train)

logger.info(f"Using fitted random forest model, compute AUC")
p = rf.predict_proba( X_test )[:,1]
auc = AUC( y_test, p )
print(f"Random forest AUC: {auc:.3f}")


# Cross-validate models
logger.info(f"Cross-validate models")

res = cross_validate(RF(), X, y, return_train_score=True,
                    cv=5, scoring=["accuracy", "roc_auc"],
                    verbose = 1, n_jobs = -1)

res_df = pd.DataFrame(res)
print(res_df)
#     fit_time  score_time  test_accuracy  train_accuracy  test_roc_auc  train_roc_auc
# 0  84.142774    3.712512       0.497662        0.968404      0.497992       0.996675
# 1  84.517222    3.886096       0.503194        0.968264      0.503922       0.996753
# 2  77.495022    3.626246       0.500560        0.968000      0.500399       0.996644
# 3  77.245053    3.936671       0.498238        0.968997      0.500609       0.996749
# 4  77.251845    3.899396       0.498041        0.967943      0.499289       0.996768