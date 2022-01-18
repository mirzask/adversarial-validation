import logging
from sys import set_coroutine_origin_tracking_depth
from tabnanny import verbose

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import SplineTransformer

from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF

from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


train = pd.read_csv( 'adversarial-validation/data/numerai/numerai_training_data.csv' )
test = pd.read_csv( 'adversarial-validation/data/numerai/numerai_tournament_data.csv' )

# Drop the original target and other columns
train.drop( ['id', 'era', 'data_type'], axis = 1 , inplace = True )
test.drop( ['id', 'era', 'target', 'data_type'], axis = 1 , inplace = True )

train['is_test'] = 0
test['is_test'] = 1

orig_train = train.copy()

# Combine train and test datasets
data = pd.concat(( train, test )).reset_index(drop=True)

# For adversarial classifier our target is if it belongs to the test set or not
data['target'] = data['is_test']
data.drop( 'is_test', axis = 1, inplace = True )

# Randomize the rows of data
# data = data.sample(frac=1).reset_index(drop=True)

# Define X and y
X = data.drop( [ 'target' ], axis = 1 )
y = data.target


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

pipe_lr = Pipeline([('scaler', StandardScaler()),
                    ('spline', SplineTransformer()),
                    ('lr', lr)])


pipe_lr.fit(X_train, y_train)
p = pipe_lr.predict_proba( X_test )[:,1]
auc = AUC( y_test, p )
print(f"Logistic regression AUC: {auc:.3f}")

# Cross validation performance
cv_scores = cross_val_score(pipe_lr, X=X_train, y=y_train, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1)
print(f"Logistic regression cross-validation AUC: {np.mean(cv_scores):.3f} +/- {np.std(cv_scores):.3f}")
# Logistic regression cross-validation AUC: 0.526 +/- 0.004

# Random Forest model
logger.info(f"Fit random forest model")
rf = RF(n_estimators=700, n_jobs=-1, verbose=1)
rf.fit(X_train, y_train)

logger.info(f"Using fitted random forest model, compute AUC")
p = rf.predict_proba( X_test )[:,1]
auc = AUC( y_test, p )
print(f"Random forest AUC: {auc:.3f}")


# Cross validation performance
cv_scores = cross_val_score(rf, X=X_train, y=y_train, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1)
print(f"Random forest cross-validation AUC: {np.mean(cv_scores):.3f} +/- {np.std(cv_scores):.3f}")


param_grid_rf = [{
                # 'max_features': ['auto', 'sqrt', 'log2'],
              'n_estimators': [100, 150, 200, 300, 500, 700]}]

grid_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, scoring="roc_auc", cv=5, n_jobs=-1, verbose=1)

grid_rf.fit(X_train, y_train)


# Print best scores and parameters
print("best mean cross-validation score: {:.3f}".format(grid_rf.best_score_))
print("best parameters: {}".format(grid_rf.best_params_))
print("test-set score: {:.3f}".format(grid_rf.score(X_test, y_test)))
print("Best estimator:\n{}".format(grid_rf.best_estimator_))

results = pd.DataFrame(grid_rf.cv_results_)


# Use the best estimator to get the predictions
#https://github.com/zygmuntz/adversarial-validation/blob/master/numerai/sort_train.py
# Cross-validate models
logger.info(f"Cross-validate models")

res = cross_validate(grid_rf.best_estimator_, X, y, return_train_score=True,
                    cv=5, scoring=["accuracy", "roc_auc"],
                    verbose = 1, n_jobs = -1)

res_df = pd.DataFrame(res)
print(res_df)
#      fit_time  score_time  test_accuracy  train_accuracy  test_roc_auc  train_roc_auc
# 0  718.539708   14.568094       0.708137             1.0      0.604877            1.0
# 1  708.517617   14.623030       0.709079             1.0      0.609629            1.0
# 2  753.103762   17.652753       0.708851             1.0      0.605028            1.0
# 3  780.919051   15.081155       0.710108             1.0      0.605441            1.0
# 4  703.408190   14.565428       0.708225             1.0      0.604002            1.0


# Get the predictions
logger.info(f"Get the predictions")
predictions = cross_val_predict(grid_rf.best_estimator_, X, y, cv=5, method='predict_proba', verbose=1)
# pd.DataFrame(predictions).to_csv('adversarial-validation/data/numerai/predictions.csv', index=False)
df_preds = pd.DataFrame(predictions).rename(columns={ 1: "test_set_proba", 0: "train_set_proba"})

# predictions has shape (n_samples, 2)
# predictions[:, 1] is the predicted probability of the positive class (1, i.e. belonging to test set)
# predictions[:, 0] is the predicted probability of the negative class (0, i.e. belonging to training set)
data_with_predictions = pd.concat((data, df_preds), axis=1)

df = data_with_predictions.copy()

#Sort the training points by their estimated probability of being test examples (ascending order, i.e. from low to high prob of being in test set)
i = df.test_set_proba.values.argsort()
df_sorted = df.iloc[i]

df.loc[df.target == 0, "test_set_proba"].mean(), df.loc[df.target == 0, "test_set_proba"].std()
df.loc[df.target == 1, "test_set_proba"].mean(), df.loc[df.target == 1, "test_set_proba"].std()


# Save train examples in order of similarity to test (ascending)
df_sorted[df_sorted.is_test == 0].to_csv('adversarial-validation/data/numerai/train_examples_sorted.csv', index=False)



# USING ONLY THE TRAINING DATA -- `df_sorted[df_sorted.TARGET == 1]`
# Now our target/label is the one from the Numerai challenge
df = pd.read_csv('adversarial-validation/data/numerai/train_examples_sorted.csv')
df.target = df.target.values.astype(int)

# Size of validation set
val_size = 10000

train = df.iloc[:-val_size]
val = df.iloc[-val_size:]
print(f"Training set probability of being in test set: {train.test_set_proba.mean():.3f} +/- {train.test_set_proba.std():.3f}")
print(f"Validation set probability of being in test set: {val.test_set_proba.mean():.3f} +/- {val.test_set_proba.std():.3f}")

X_train = train.drop(['target', 'is_test', 'test_set_proba', 'train_set_proba'], axis=1)
X_val = val.drop(['target', 'is_test', 'test_set_proba', 'train_set_proba'], axis=1)

y_train = train.target.values
y_val = val.target.values

# See if we're closer to AUC 0.5 now
logger.info(f"Fit logistic regression model")
lr = LR()
lr.fit(X_train, y_train)

logger.info(f"Using fitted logistic regression model, compute AUC")
p = lr.predict_proba( X_val )[:,1]
auc = AUC( y_val, p )
acc = accuracy_score(y_val, lr.predict(X_val))
log_loss = log_loss(y_val, p)
print(f"Logistic regression AUC: {auc:.3f} | Accuracy: {acc:.3f} | Log loss: {log_loss:.3f}")
# Logistic regression AUC: 0.517 | Accuracy: 0.515 | Log loss: 0.693

pipe_lr1 = Pipeline([('scaler', StandardScaler()),
                    ('spline', SplineTransformer()),
                    ('lr', lr)])


pipe_lr1.fit(X_train, y_train)
p = pipe_lr1.predict_proba( X_val )[:,1]
auc = AUC( y_val, p )
acc = accuracy_score(y_val, pipe_lr1.predict(X_val))
log_loss = log_loss(y_val, p)
print(f"Logistic regression AUC: {auc:.3f} | Accuracy: {acc:.3f} | Log loss: {log_loss:.3f}")
# Logistic regression AUC: 0.519 | Accuracy: 0.513 | Log loss: 0.693

pipe_lr2 = Pipeline(['scaler', MinMaxScaler(),
                    ('spline', SplineTransformer()),
                    ('lr', lr)])

pipe_lr2.fit(X_train, y_train)
p = pipe_lr2.predict_proba( X_val )[:,1]
auc = AUC( y_val, p )
acc = accuracy_score(y_val, pipe_lr2.predict(X_val))
log_loss = log_loss(y_val, p)
print(f"Logistic regression AUC: {auc:.3f}, Accuracy: {acc:.3f} | Log loss: {log_loss:.3f}")
# Logistic regression AUC: 0.519, Accuracy: 0.513 | Log loss: 0.693

# Random Forest model
logger.info(f"Fit random forest model")
rf = RF(n_estimators=300, n_jobs=-1, verbose=1)
rf.fit(X_train, y_train)

logger.info(f"Using fitted random forest model, compute AUC")
p = rf.predict_proba( X_val )[:,1]
auc = AUC( y_val, p )
acc = accuracy_score(y_val, rf.predict(X_val))
log_loss = log_loss(y_val, p)
print(f"Random forest AUC: {auc:.3f}, Accuracy: {acc:.3f} | Log loss: {log_loss:.3f}")
# Random forest AUC: 0.511, Accuracy: 0.508 | Log loss: 0.693



# Try it out
test = pd.read_csv( 'adversarial-validation/data/numerai/numerai_tournament_data.csv' )
test = test[test.data_type == "validation"]
test.target = test.target.values.astype(int)

X_test = test.drop(['target', 'data_type', 'id', 'era'], axis=1)
y_test = test.target.values

cross_val_score(pipe_lr1, X_test, y_test, cv=5, scoring='neg_log_loss')
cross_val_score(pipe_lr2, X_test, y_test, cv=5, scoring='neg_log_loss')
cross_val_score(rf, X_test, y_test, cv=5, scoring='neg_log_loss')
pipe_lr1.score(X_test, y_test)
pipe_lr2.score(X_test, y_test)
rf.score(X_test, y_test)