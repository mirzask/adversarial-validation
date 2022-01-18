# Adversarial Validation

- The test set distribution *may* be different from training set distribution as illustrated in Figure 2 from Pan et al.
- Adversarial validation is a method for selecting training examples most similar to test examples and using them as a validation set.
  - During machine learning model development, the model performance on part of the training dataset, the validation dataset, is used as a proxy of the performance on test data. However, if the feature distributions in the training and test datasets are different, the performance on the validation and test datasets will be different. In adversarial validation, a binary classifier, **adversarial classifier**, is trained to predict if a sample belongs to the test dataset. Classification performance better than random guess (e.g. AUC = 0.5) indicates the different feature distributions between the training and test datasets.
  - Furthermore, the adversarial classifier can be used to balance the training and test datasets and improve the model performance on the test dataset.
- Classifier to discern if data from training or test sets.
  - If both the training and test sets have the same distribution, then we should not be able to distinguish (better than random)the training from the test set. Thus if the training and test sets come from the *same* distribution, we will expect our adversarial classifier to have an AUC of $\approx$ 0.5.
  - If distributions of the features from the train and test data are similar, we expect the adversarial classifier to be as good as random guesses. 
  - What is the the adversarial classifier can distinguish between training and test data well (i.e. AUC score much greater than 50%)?
    - Your validation set won't be very good: performance on the training set may be very dissimilar to performance on the test set. **We want our validation set to be representative of the test set.** To do this, we can select examples for the validation set which are the most similar to the test set.
    - The top features from the adversarial classifier are potential candidates exhibiting concept drift between the train and test data.
      - Pan et al. propose *Automated Feature Selection* based off of this where they remove the most important features from the adversarial classifier if it performs too well on differentiating between the training and test sets. In fact, they find that this approach works better than inverse probability weighting and propensity scores using the adversarial classifier.
- Smells like Heterogenous Treatment Effect Estimation
  - *The adversarial validation approach is similar to propensity score modeling in causal inference* [1, 12]. In causal inference, propensity score modeling addresses the heterogeneity between the treatment and control group data by training a classifier to predict if a sample belongs to a treatment group. Rosenbaum and Rubin argue in [12] that it is sufficient to achieve the balance in the distributions be- tween the treatment and control groups by matching on the single dimensional propensity score alone, which is significantly more efficient than matching on the joint distribution of all confounding variables.

## Method

1. Create column indicating 1 if from training set, 0 if from test set

```python
train = pd.read_csv( 'adversarial-validation/data/santander/train.csv' )
test = pd.read_csv( 'adversarial-validation/data/santander/test.csv' )

train['TARGET'] = 1
test['TARGET'] = 0
```

2. Concatenate the training and test sets

```python
# Combine train and test datasets
data = pd.concat(( train, test ))
```

3. Shuffle the concatenated set

```python
# Randomize the rows of data
data = data.sample(frac=1).reset_index(drop=True)
```

4. Split the shuffled dataset into training and tests

```python
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('TARGET', 1),
    data['TARGET'],
    test_size = .3,
    random_state=42)
```

5. Train an adversarial classifier that predicts $\Pr(\{ y_\text{train}, y_{\text{test}} \} \ | \ \mathbf{X})$ to separate train and test

```python
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score as AUC

lr = LR()
lr.fit(X_train, y_train)

p = lr.predict_proba( X_test )[:,1]
auc = AUC( y_test, p )
print(f"Logistic regression AUC: {auc:.3f}")
```

6. We can also use cross-validation to get a better sense of adversarial classifier performance

```python
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import cross_validate

# Define X and y
X = data.drop( [ 'TARGET', 'ID' ], axis = 1 )
y = data.TARGET

res = cross_validate(RF(), X, y, return_train_score=True,
                    cv=5, scoring=["accuracy", "roc_auc"],
                    verbose = 1, n_jobs = -1)

res_df = pd.DataFrame(res)
print(res_df)
```

```
#     fit_time  score_time  test_accuracy  train_accuracy  test_roc_auc  train_roc_auc
# 0  84.142774    3.712512       0.497662        0.968404      0.497992       0.996675
# 1  84.517222    3.886096       0.503194        0.968264      0.503922       0.996753
# 2  77.495022    3.626246       0.500560        0.968000      0.500399       0.996644
# 3  77.245053    3.936671       0.498238        0.968997      0.500609       0.996749
# 4  77.251845    3.899396       0.498041        0.967943      0.499289       0.996768
```

7. If $\text{AUC} \neq 0.5$, then generate predictions for all of the training examples using the adversarial classifier.

```python
predictions = cross_val_predict(grid_rf.best_estimator_, X, y, cv=5, method='predict_proba')


# predictions has shape (n_samples, 2)
# predictions[:, 1] is the predicted probability of the positive class (1, i.e. belonging to test set)
# predictions[:, 0] is the predicted probability of the negative class (0, i.e. belonging to train set)
df_preds = pd.DataFrame(predictions).rename(columns={ 0: "train_proba", 1: "test_proba"})

# Combine these probability predictions to the original large dataframe
data_with_predictions = pd.concat((data, df_preds), axis=1)
```

8. Identify which training examples are misclassified as test and use them for validation.

```python
# Copy the concatenated df to make sure
df = data_with_predictions.copy()

#Sort the training points by their estimated probability of being test examples (ascending order, i.e. from low to high prob of being in test set)
i = df.test_proba.values.argsort()
df_sorted = df.iloc[i]
```

9. Remember that the goal is for the validation set distribution to better resemble the test set distribution. Use our **training** dataframe *sorted* where those rows with the highest probability of belonging to the test set are the bottom. We'll slice those bottom $n$ rows to function as our validation set.

  - Create training and validation sets from *sorted* dataframe

```python
# Size of validation set
val_size = 5000

train = df_sorted.iloc[:-val_size]
val = df_sorted.iloc[-val_size:]

X_train = train.drop(['TARGET', 'test_proba', 'train_proba'], axis=1)
X_val = val.drop(['TARGET', 'test_proba', 'train_proba'], axis=1)

y_train = train.TARGET.values
y_val = val.TARGET.values
```

  - Now fit models using `X_train`, `X_val`, `y_train`, `y_val`




# References

- http://fastml.com/adversarial-validation-part-one/, http://fastml.com/adversarial-validation-part-two/
- Pan J, Pham V, Dorairaj M, Chen H, Lee JY. Adversarial validation approach to concept drift problem in user targeting automation systems at uber. arXiv:200403045 [cs, stat]. Published online June 26, 2020. http://arxiv.org/abs/2004.03045