import warnings
warnings.filterwarnings('ignore')

import pprint

import pandas as pd
import numpy as np

from sklearn import model_selection
from sklearn import tree

# Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_curve, auc

# Classifiers
from sklearn import ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

#
# Load data
#
data = pd.read_csv('data/scoring_train_test.csv', delimiter=';', decimal='.')

#
# Pre-process
#
data = data.drop(['id'], axis=1)
X = data.iloc[:,0:5]
y = data.iloc[:, 5:6]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=0)

#
# Evaluation
#

# Utils
def mix_two(arr1, arr2):
    res = []
    for v1 in arr1:
        for v2 in arr2:
            if type(v1) == list:
                res.append(v1 + [v2])
            else:
                res.append([v1, v2])
    return res

def mix(arrs):
    res = arrs[0]
    rest = arrs[1:]
    for arr in rest:
      res = mix_two(res, arr)
    return res

def split(obj):
  res = []
  for k in obj:
    sub = []
    for i in obj[k]:
      sub.append({k: i})
    res.append(sub)
  return res

def compact(arrs):
  res = []
  for arr in arrs:
    sub = {}
    for obj in arr:
      for k in obj:
        sub[k] = obj[k]
    res.append(sub)
  return res

def classifier_args(args):
    return compact(mix(split(args)))

def score(Classifier, args, X_train, y_train, X_test):
    model = Classifier(**args)
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    global_precision = np.sum(cm.diagonal()) / np.sum(cm)
    global_error = 1 - global_precision

    category_precision = cm.diagonal() / np.sum(cm, axis=1)

    # ROC
    y_pred_probabilities = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_probabilities)
    roc_auc = auc(fpr, tpr)

    # KFold
    kfold = model_selection.KFold(n_splits=10)
    kfold = model_selection.cross_val_score(model, X, y, cv=kfold).mean()

    # Stratified KFold
    skfold = model_selection.StratifiedKFold(n_splits=10)
    skfold = model_selection.cross_val_score(model, X, y, cv=skfold, scoring='f1_macro').mean()

    # LeaveOneOut
    # loo = model_selection.LeaveOneOut()
    # loo = model_selection.cross_val_score(model, X, y, cv=loo).mean()

    name = '{}{}'.format(model.__class__.__name__, args)
    scores = [
        model.score(X_train, y_train.values.ravel()),
        model.score(X_test, y_test.values.ravel()),
        accuracy_score(y_test, y_pred),
        recall_score(y_test, y_pred, average='macro'),
        precision_score(y_test, y_pred, average='macro'),
        f1_score(y_test, y_pred, average='macro'),
        cohen_kappa_score(y_test, y_pred),
        global_precision,
        global_error,
        cm,
        category_precision[0],
        category_precision[1],
        roc_auc,
        kfold,
        skfold,
        #loo,
    ]
    return (name, scores)

index = [
    'ScoreTrain',
    'ScoreTest',
    'Accuracy',
    'Recall',
    'Precision',
    'F1',
    'Kappa',
    'GlobalPrecision',
    'GlobalError',
    'CM',
    'CP_1',
    'CP_2',
    'ROC',
    'KFold',
    'StratifiedKFold',
    # 'LeaveOneOut'
]
df = pd.DataFrame(index=index)
results = []

# RandomForestClassifier
vargs = {
    'max_leaf_nodes': [35, 40, 45, 50],
    'n_estimators': [2, 4, 5, 6, 8],
    'max_depth': [2, 5, 8, 10, 12, 14],
}
for args in classifier_args(vargs):
    name, scores = score(RandomForestClassifier, args, X_train, y_train, X_test)
    df[name] = scores

# KNeighborsClassifier
vargs = {
    'n_neighbors': [2, 5, 10, 20, 30, 50, 100],
    'leaf_size': [10, 20, 30, 50],
    'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
    'weights': ['uniform', 'distance'],
}
for args in classifier_args(vargs):
    name, scores = score(KNeighborsClassifier, args, X_train, y_train, X_test)
    df[name] = scores

# GaussianProcessClassifier
vargs = {
    'kernel': [None],
    'n_restarts_optimizer': [0, 1, 2, 3],
    'max_iter_predict': [50, 100, 150],
}
for args in classifier_args(vargs):
    name, scores = score(GaussianProcessClassifier, args, X_train, y_train, X_test)
    df[name] = scores

# GradientBoostingClassifier
vargs = {
    'loss': ['deviance', 'exponential'],
    'n_estimators': [20, 50, 80, 100, 130, 150, 180],
    'subsample': [1.0, 0.8, 0.5, 0.3],
    'criterion': ['friedman_mse', 'mse', 'mae'],
}
for args in classifier_args(vargs):
    name, scores = score(GradientBoostingClassifier, args, X_train, y_train, X_test)
    df[name] = scores

table = df.transpose().sort_values(by=['CP_1', 'CP_2'], ascending=False)
table.to_csv('scores.csv')
print(table)
