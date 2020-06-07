import warnings
warnings.filterwarnings('ignore')

import sys

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
import xgboost as xgb

from utils import merge, merge_args, compact_obj
from score import Scorer

random_state = 0
results_folder = 'results'

#
# Load data and pre-process
#
# --- Case 1 ---
case = 'c1'
data = pd.read_csv('../1_Calificacion_Crediticia/data/scoring_train_test.csv', delimiter=';', decimal='.')

data = data.drop(['id'], axis=1)
X = data.iloc[:, 0:5]
y = data.iloc[:, 5:6]

# --- Case 2 ---
# case = 'c2'
# data = pd.read_csv('../2_Muerte_Coronaria/data/datos_train_test_sh.csv', delimiter=',', decimal='.')
#
# data = data.drop(['id'], axis=1)
# data.famhist[data.famhist == 'Present'] = 1
# data.famhist[data.famhist == 'Absent'] = 0
# data.famhist = data.famhist.astype('int')
#
# X = data.iloc[:, 0:9]
# y = data.iloc[:, 9:10]
#
# # Best classifiers
# clf1 = DecisionTreeClassifier(
#     random_state=0,
#     criterion='entropy',
#     splitter='random',
#     max_depth=None,
#     min_samples_split=14,
#     min_samples_leaf=1,
#     class_weight={0: 0.3, 1: 0.9}
# ) # BA
# clf2 = GradientBoostingClassifier(
#     random_state=0,
#     loss='exponential',
#     learning_rate=0.5,
#     n_estimators=170,
#     subsample=0.7,
#     criterion='mae',
#     max_features='auto',
#     max_depth=5,
# ) # ROC
# clf3 = GradientBoostingClassifier(
#     random_state=0,
#     loss='exponential',
#     learning_rate=0.5,
#     n_estimators=150,
#     subsample=0.7,
#     criterion='mae',
#     max_features='auto',
#     max_depth=5,
# ) # ROC 2

#
# Evaluation
#
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=random_state)

scorer = Scorer(X, y, X_train, y_train, X_test, y_test)
variants = [
    {
        'classifier': DecisionTreeClassifier,
        'vargs': {
            'random_state': [random_state],
            'criterion': ['gini', 'entropy'], # | C2 => BA=gini; ROC=entropy
            'splitter': ['best', 'random'], # | C2 => BA=best; ROC=random
            'max_depth': [None] + list(range(5, 20, 5)), # | C2 => BA=10; ROC=10
            'min_samples_split': list(range(2, 16, 2)), # | C2 => BA=2; ROC=14
            'min_samples_leaf': list(range(1, 6, 1)), # | C2 => BA=3; ROC=1
            'class_weight': merge_args({ # | C2 => BA=[0.2, 0.6]; ROC=[0.6, 0.6]
                0: np.arange(0.1, 1.0, 0.15),
                1: np.arange(0.1, 1.0, 0.15),
            }),
        },
        'run': True,
    },
    {
        'classifier': RandomForestClassifier,
        'vargs': {
            'random_state': [random_state],
            'criterion': ['gini', 'entropy'], # | C2 => BA=entropy; ROC=gini
            'n_estimators': [10, 50, 100, 150], # | C2 => BA=150; ROC=10
            'max_depth': [None, 4, 5, 6], # | C2 => BA=5; ROC=None
            'min_samples_split': list(range(2, 10, 2)), # | C2 => BA=2; ROC=14
            'min_samples_leaf': list(range(1, 3, 1)), # | C2 => BA=1; ROC=1
            'max_features': ['auto'], # | C2 => BA=*; ROC=*
            'class_weight': merge_args({ # | C2 => BA=[0.2, 0.8]; ROC=[0.2, 0.6]
                0: np.arange(0.1, 1.0, 0.15),
                1: np.arange(0.1, 1.0, 0.15),
            }),
        },
        'run': True,
    },
    {
        'classifier': GradientBoostingClassifier,
        'vargs': {
            'random_state': [random_state],
            'loss': ['deviance', 'exponential'], # | C2 => BA=exponential; ROC=exponential
            'learning_rate': [0.1, 0.3, 0.5], # | C2 => BA=0.3; ROC=0.5
            'n_estimators': [10, 30, 50, 150, 170], # | C2 => BA=30; ROC=170
            'subsample': [0.3, 0.5, 0.7, 1.0], # | C2 => BA=0.3; ROC=0.7
            'criterion': ['friedman_mse', 'mse', 'mae'], # | C2 => BA=friedman_mse; ROC=mae
            'max_features': ['auto', 'sqrt', 'log2'], # | C2 => BA=auto; ROC=auto
            'max_depth': list(range(3, 6, 1)), # | C2 => BA=4; ROC=5
        },
        'run': True,
    },
    {
        'classifier': KNeighborsClassifier,
        'vargs': {
            'n_neighbors': [2, 5, 10, 15],
            'weights': ['uniform', 'distance'],
            'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
            'leaf_size': [10, 20, 30, 50, 60, 70],
        },
        'run': True,
    },
    {
        'classifier': xgb.XGBClassifier,
        'vargs': {
            'random_state': [random_state],
            'objective': ['binary:logistic'],
            'max_depth': [None, 5, 10, 30, 50],
            'learning_rate': [None, 0.1, 0.4, 0.6, 0.8, 1.0],
            'booster': ['gbtree', 'gblinear', 'dart'],
            'n_estimators': [10, 30, 50, 80, 100],
            'base_score': [0.1, 0.3, 0.6, 0.8],
        },
        'run': True,
    },
    {
        'classifier': GaussianProcessClassifier,
        'vargs': {
            'random_state': [random_state],
            'kernel': [None, 1.0 * RBF(1.0)],
        },
        'run': True,
    },
    # {
    #     'classifier': VotingClassifier,
    #     'vargs': {
    #         'estimators': merge([
    #             [('1', clf1), ('1', clf2), ('1', clf3)],
    #             [('2', clf1), ('2', clf2), ('2', clf3)],
    #             [('3', clf1), ('3', clf2), ('3', clf3)],
    #         ]),
    #         'voting': ['soft'],
    #         'weights': merge([
    #             [1, 2, 3],
    #             [1, 2, 3],
    #             [1, 2, 3],
    #         ]),
    #     },
    #     'run': False,
    # },
]

print('# BRUTE-FORCING:', case)
for variant in variants:
    if variant['run']:
        vargs = merge_args(variant['vargs'])
        print('- {}: {} variants.'.format(variant['classifier'].__name__, len(vargs)))

print('\n# RUNNING')
for variant in variants:
    if variant['run']:
        vargs = merge_args(variant['vargs'])
        name = variant['classifier'].__name__
        results = []
        for (i, args) in enumerate(vargs):
            print('{}: {}/{}'.format(name, i+1, len(vargs)), end='\r')
            result = scorer.score(variant['classifier'], args)
            results.append(result)
        print()

        df = pd.DataFrame(data=results)

        df = df.sort_values(by=['BalancedAccuracy'], ascending=False)
        filename = '{}/{}/{}.csv'.format(results_folder, case, name)
        df.to_csv(filename)
