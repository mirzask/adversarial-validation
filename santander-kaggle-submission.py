# Uses "Nested Cross Validation"

import logging

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC
# from sklearn.ensemble import GradientBoostingClassifier as GBC
# from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


train = pd.read_csv( 'adversarial-validation/data/santander/train.csv' )
test = pd.read_csv( 'adversarial-validation/data/santander/test.csv' )

# Train, test split on the 'train' data
X_train, X_test, y_train, y_test = train_test_split(
    train.drop('TARGET', 1),
    train['TARGET'],
    test_size = .2,
    random_state=42,
    stratify=train['TARGET'])


# Use Nested-CV to find the best hyperparameters
# Adapted from Sebastian Raschka: https://github.com/rasbt/model-eval-article-supplementary/blob/master/code/nested_cv_code.ipynb
# Initialize the models
logger.info(f"Initialize models")
lr = LR()
rf = RF()
# xgb = XGBClassifier(eval_metric='logloss')
# gbc = GBC()
svm = SVC(random_state=1)

pipe_lr = Pipeline([('std', StandardScaler()),
                  ('lr', lr)])

pipe_svm = Pipeline([('std', StandardScaler()),
                  ('svm', svm)])

# Setup parameter grids
logger.info(f"Setup parameter grids")
param_grid_lr = [{'lr__penalty': ['l2'],
                'lr__C': np.logspace(-3, 3, 13)}]

param_grid_rf = [{'max_features': ['auto', 'sqrt', 'log2'],
              'n_estimators': [150, 200, 300, 500, 700]}]

# param_grid_xgb = [{
#             'eta': np.linspace(0, 0.4, num=5),
#             'gamma': np.linspace(0,0.5, 6),
#             'max_depth':range(3,10,2),
#             'min_child_weight':range(1,6,2),
#             'subsample': np.linspace(0.6, 0.9, 4),
#             'colsample_bytree': np.linspace(0.6, 0.9, 4)
#             }]

# param_grid_gbc = [{
#                 # 'max_features': ['auto', 'sqrt', 'log2'],
#                 'learning_rate': [0.1, 0.01, 0.001],
#                 # 'min_samples_split': [2, 3, 4],
#                 # 'min_samples_leaf': [1, 2, 3],
#                 'max_depth': range(3,10,2),
#                 'subsample': np.linspace(0.6, 1.0, 5),
#                 # 'n_estimators': [150, 200, 300, 500, 700]
#                 }]

param_grid_svm = [{'svm__kernel': ['rbf'],
                'svm__C': np.power(10., np.arange(-4, 4)),
                'svm__gamma': np.power(10., np.arange(-5, 0))
                },
                {'svm__kernel': ['linear'],
                'svm__C': np.power(10., np.arange(-4, 4))
                }]

# # Setting up multiple GridSearchCV objects, 1 for each algorithm
# gridcvs = {}

# for pgrid, est, name in zip((param_grid_lr, param_grid_rf,
#                              param_grid_xgb),
#                             (pipe_lr, rf, xgb),
#                             ('LogReg', 'RandomForest', 'XGBoost')):
#     gcv = GridSearchCV(estimator=est,
#                        param_grid=pgrid,
#                        scoring='accuracy',
#                        n_jobs=-1,
#                        cv=2,
#                        verbose=0,
#                        refit=True)
#     gridcvs[name] = gcv


# # Nested-CV time
# cv_scores = {name: [] for name, gs_est in gridcvs.items()}

# skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# # The outer loop for algorithm selection
# c = 1
# for outer_train_idx, outer_valid_idx in skfold.split(X_train, y_train):
#     for name, gs_est in sorted(gridcvs.items()):
#         print(f"outer loop {c}/5 | tuning {name} ")

#         # The inner loop for hyperparameter tuning
#         gs_est.fit(X_train.iloc[outer_train_idx, :], y_train.iloc[outer_train_idx])
#         y_pred = gs_est.predict(X_train.iloc[outer_valid_idx, :])
#         acc = accuracy_score(y_true=y_train.iloc[outer_valid_idx], y_pred=y_pred)
#         print(f" | inner ACC {gs_est.best_score_ * 100}% | outer ACC {acc * 100}%")
#         cv_scores[name].append(acc)

#     c += 1



# NEW APPROACH
# Setting up multiple GridSearchCV objects, 1 for each algorithm
gridcvs = {}
inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)

for pgrid, est, name in zip((param_grid_lr, param_grid_rf,
                             param_grid_svm),
                            (pipe_lr, rf, pipe_svm),
                            ('LogReg', 'RandomForest', 'SVM')):
    gcv = GridSearchCV(estimator=est,
                       param_grid=pgrid,
                       scoring='accuracy',
                       n_jobs=-1,
                       cv=inner_cv,
                       verbose=0,
                       refit=True)
    gridcvs[name] = gcv

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

for name, gs_est in sorted(gridcvs.items()):
    nested_score = cross_val_score(gs_est, 
                                   X=X_train, 
                                   y=y_train, 
                                   cv=outer_cv,
                                   n_jobs=-1)
    print(f"{name} | Outer ACC: {nested_score.mean() * 100}% +/- {nested_score.std() * 100}%")