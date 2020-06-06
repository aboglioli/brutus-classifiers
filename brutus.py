import warnings
warnings.filterwarnings('ignore')

import pprint

import pandas as pd
import numpy as np

from sklearn import model_selection

# Classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier, RandomForestRegressor

from utils import merge, merge_args
from score import Scorer

# Load data
data = pd.read_csv('../1_Calificacion_Crediticia/data/scoring_train_test.csv', delimiter=';', decimal='.')

# Pre-process
data = data.drop(['id'], axis=1)
X = data.iloc[:,0:5]
y = data.iloc[:, 5:6]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=0)

#
# Evaluation
#
scorer = Scorer(X, y, X_train, y_train, X_test, y_test)
results = []

# RandomForestClassifier
vargs = {
    'max_leaf_nodes': [35, 40, 45, 50],
    'n_estimators': [2, 4, 5, 6, 8],
    'max_depth': [2, 5, 8, 10, 12, 14],
}
# for args in merge_args(vargs):
#     result = scorer.score(RandomForestClassifier, args)
#     results.append(result)

# KNeighborsClassifier
vargs = {
    'n_neighbors': [2, 5, 10, 20, 30, 50, 100],
    'leaf_size': [10, 20, 30, 50],
    'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
    'weights': ['uniform', 'distance'],
}
# for args in merge_args(vargs):
#     result = scorer.score(KNeighborsClassifier, args)
#     results.append(result)

# GaussianProcessClassifier
vargs = {
    'kernel': [None],
    'n_restarts_optimizer': [0, 1, 2, 3],
    'max_iter_predict': [50, 100, 150],
}
# for args in merge_args(vargs):
#     result = scorer.score(GaussianProcessClassifier, args)
#     results.append(result)

# GradientBoostingClassifier
vargs = {
    'loss': ['deviance', 'exponential'],
    'n_estimators': [20, 50, 80, 100, 130, 150, 180],
    'subsample': [1.0, 0.8, 0.5, 0.3],
    'criterion': ['friedman_mse', 'mse', 'mae'],
}
# for args in merge_args(vargs):
#     result = scorer.score(GradientBoostingClassifier, args)
#     results.append(result)

# VotingClassifier
best_accuracy = GradientBoostingClassifier(loss='deviance', n_estimators=130, subsample=1.0, criterion='friedman_mse')
best_recall = KNeighborsClassifier(n_neighbors=2, leaf_size=10, algorithm='brute', weights='uniform')
best_f1 = l = RandomForestClassifier(max_leaf_nodes=50, n_estimators=6, max_depth=10)
best_kappa = best_f1
best_cp_1 = KNeighborsClassifier(n_neighbors=2, leaf_size=30, algorithm='kd_tree', weights='uniform')
best_roc = GradientBoostingClassifier(loss='deviance', n_estimators=100, subsample=1.0, criterion='friedman_mse')
best_kfold = GradientBoostingClassifier(loss='deviance', n_estimators=180, subsample=0.3, criterion='mse')
best_skfold = best_roc

vargs = {
    'estimators': merge([
        [('1', best_accuracy), ('1', best_recall)],
        [('2', best_f1), ('2', best_kappa)],
        [('3', best_cp_1), ('3', best_roc), ('3', best_kfold)],
    ]),
    'voting': ['soft'],
    'weights': merge([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
}

print("Total:", len(merge_args(vargs)))
i = 1
for args in merge_args(vargs):
    print(i)
    result = scorer.score(VotingClassifier, args)
    results.append(result)
    i += 1

# Save
df = pd.DataFrame(data=results)
df = df.sort_values(by=['CP_1', 'CP_2'], ascending=False)
df.to_csv('results.csv')
print(df)
