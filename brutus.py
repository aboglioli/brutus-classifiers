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
from case import Credito, MuerteCoronaria
import config

random_state = 0

# Load data and pre-process
case = Credito()
X, y = case.get_data()

# Evaluation
scorer = Scorer(X, y)
scorer.split(test_size=0.3, random_state=random_state)
scorer.set_criteria([
    'CM',
    'BalancedAccuracy',
    # 'Accuracy',
    # 'Recall',
    # 'Precision',
    # 'F1',
    # 'Kappa',
    # 'ROC',
    # 'KFold',
    'StratifiedKFold',
    # 'LeaveOneOut',
])

variants = [
    {
        'classifier': DecisionTreeClassifier,
        'vargs': {
            'random_state': [random_state],
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None] + list(range(5, 20, 5)),
            'min_samples_split': list(range(2, 16, 2)),
            'min_samples_leaf': list(range(1, 6, 1)),
            'class_weight': merge_args({
                0: np.arange(0.1, 1.0, 0.15),
                1: np.arange(0.1, 1.0, 0.15),
            }),
        },
        'run': False,
    },
    {
        'classifier': RandomForestClassifier,
        'vargs': {
            'random_state': [random_state],
            'criterion': ['gini', 'entropy'],
            'n_estimators': [5, 10, 100, 150],
            'max_depth': [None, 4, 5, 6],
            'min_samples_split': list(range(2, 10, 2)),
            'min_samples_leaf': list(range(1, 3, 1)),
            'max_features': ['auto'],
            'class_weight': merge_args({
                0: np.arange(0.1, 1.0, 0.3),
                1: np.arange(0.1, 1.0, 0.3),
            }),
        },
        'run': False,
    },
    {
        'classifier': RandomForestClassifier,
        'vargs': {
            'random_state': [random_state],
            'criterion': ['gini', 'entropy'],
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [None],
            'min_samples_split': [2, 3],
            'min_samples_leaf': [1, 2],
            'max_features': ['auto'],
            'class_weight': merge_args({
                0: np.arange(0.1, 1.0, 0.3),
                1: np.arange(0.1, 1.0, 0.3),
            }),
        },
        'run': True,
    },
    {
        'classifier': GradientBoostingClassifier,
        'vargs': {
            'random_state': [random_state],
            'loss': ['deviance', 'exponential'],
            'learning_rate': [0.1, 0.3, 0.5],
            'n_estimators': [10, 30, 50, 150, 170],
            'subsample': [0.3, 0.5, 0.7, 1.0],
            'criterion': ['friedman_mse', 'mse', 'mae'],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': list(range(3, 6, 1)),
        },
        'run': False,
    },
    {
        'classifier': KNeighborsClassifier,
        'vargs': {
            'n_neighbors': [2, 5, 10, 15],
            'weights': ['uniform', 'distance'],
            'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
            'leaf_size': [10, 20, 30, 50, 60, 70],
        },
        'run': False,
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
        df = df.sort_values(by=['StratifiedKFold'], ascending=False)
        filename = '{}/{}/{}.csv'.format(config.results_folder, case.name, name)
        df.to_csv(filename)
