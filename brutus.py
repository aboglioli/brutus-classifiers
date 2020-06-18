import sys
import warnings
import numpy as np
import pandas as pd

# Classifiers
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
warnings.filterwarnings('ignore')

# Utils
from score import Scorer
from utils import merge, merge_args, compact_obj
from case import Credito, MuerteCoronaria, Titanic
import config

random_state = 0

# Load data and pre-process
case = Titanic()
X, y = case.get_data()
X_train, X_test, y_train, y_test = case.get_train_test()

# Evaluation
scorer = Scorer()
scorer.set_dataset(X, y)
scorer.set_train_test(X_train, X_test, y_train, y_test)
scorer.set_criteria([
    # 'CM',
    # 'BalancedAccuracy',
    'Accuracy',
    # 'Recall',
    # 'Precision',
    # 'F1',
    # 'Kappa',
    # 'ROC',
    # 'KFold',
    # 'StratifiedKFold',
    # 'LeaveOneOut',
])

variants = [
    {
        'classifier': DecisionTreeClassifier,
        'vargs': {
            'random_state': [random_state],
            'criterion': ['gini'], # entropy
            'splitter': ['best'], # random
            'max_depth': [None] + list(range(10, 50, 5)),
            'min_samples_split': list(range(2, 16, 2)),
            'min_samples_leaf': list(range(1, 6, 1)),
            'class_weight': merge_args({
                0: list(range(1, 10, 1)),
                1: list(range(1, 10, 1)),
            }),
        },
        'run': False,
    },
    {
        'classifier': RandomForestClassifier,
        'vargs': {
            'random_state': [random_state],
            'criterion': ['gini'], # entropy
            'n_estimators': [2, 5, 8, 10, 20, 30, 50, 100], # list(range(10, 500, 20)),
            'max_depth': [None], # + list(range(5, 50, 10)),
            'min_samples_split': list(range(2, 15, 2)),
            'min_samples_leaf': list(range(1, 10, 2)),
            'max_features': ['sqrt', 'log2'],
            'class_weight': merge_args({
                0: list(range(1, 11, 1)),
                1: list(range(1, 11, 1)),
            }),
            # 'n_jobs': [-1],
        },
        'run': True,
    },
    {
        'classifier': GradientBoostingClassifier,
        'vargs': {
            'random_state': [random_state],
            'loss': ['deviance', 'exponential'],
            'learning_rate': [0.1, 0.3, 0.5],
            'n_estimators': [10, 150, 200],
            'subsample': [0.3, 0.6, 1.0],
            'criterion': ['friedman_mse', 'mse'],
            'max_features': ['sqrt', 'log2'],
            'max_depth': list(range(1, 6, 2)),
        },
        'run': False,
    },
    {
        'classifier': KNeighborsClassifier,
        'vargs': {
            'n_neighbors': [5, 10, 15],
            'weights': ['uniform', 'distance'],
            'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
            'leaf_size': list(range(10, 100, 40)),
        },
        'run': False,
    },
    {
        'classifier': xgb.XGBClassifier,
        'vargs': {
            'random_state': [random_state],
            'max_depth': [None, 5, 20, 50],
            'learning_rate': [None, 0.3, 0.6, 1.0],
            'booster': ['gbtree', 'gblinear', 'dart'],
            'n_estimators': [10, 150, 200, 250],
            'base_score': [0.2, 0.5, 0.8],
        },
        'run': False,
    },
    {
        'classifier': GaussianProcessClassifier,
        'vargs': {
            'random_state': [random_state],
            'kernel': [None, 1.0 * RBF(1.0)],
        },
        'run': False,
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

print('# BRUTE-FORCING:', case.name)
for variant in variants:
    if variant['run']:
        vargs = merge_args(variant['vargs'])
        print(
            '- {}: {} variants.'.format(variant['classifier'].__name__, len(vargs)))

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
        df = df.sort_values(by=['Accuracy'], ascending=False)
        filename = '{}/{}/{}.csv'.format(config.results_folder, case.name, name)
        df.to_csv(filename)
