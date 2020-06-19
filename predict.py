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
import numpy as np
import pandas as pd
import pprint
import warnings
warnings.filterwarnings('ignore')

import config
from case import Credito, MuerteCoronaria, Titanic

case = Titanic()
# model = DecisionTreeClassifier(
#         random_state=0,
#         criterion='entropy',
#         splitter='random',
#         max_depth=None,
#         min_samples_split=14,
#         min_samples_leaf=2,
#         class_weight={0: 0.7, 1: 0.1},
# ) # 0.79904
# model = RandomForestClassifier(
#         random_state=0,
#         criterion='entropy',
#         n_estimators=10,
#         max_depth=5,
#         min_samples_split=6,
#         min_samples_leaf=1,
#         max_features='auto',
#         class_weight={0: 0.7, 1: 0.4},
# ) # 0.78947
# model = RandomForestClassifier(
#         random_state=0,
#         criterion='gini',
#         n_estimators=10,
#         max_depth=None,
#         min_samples_split=8,
#         min_samples_leaf=1,
#         max_features='auto',
#         class_weight={0: 3, 1: 2},
# ) # 0.81339
# model = RandomForestClassifier(
#         random_state=0,
#         criterion='gini',
#         n_estimators=150,
#         max_depth=None,
#         min_samples_split=2,
#         min_samples_leaf=5,
#         max_features='log2',
#         class_weight={0: 4, 1: 4},
# ) # 0.80861
# model = RandomForestClassifier(
#         random_state=0,
#         criterion='entropy',
#         n_estimators=10,
#         max_depth=5,
#         min_samples_split=2,
#         min_samples_leaf=1,
#         max_features='auto',
#         class_weight={0: 3, 1: 2},
# ) # 0.80382
# model = GradientBoostingClassifier(
#         random_state=0,
#         loss='deviance',
#         learning_rate=0.3,
#         n_estimators=10,
#         subsample=0.3,
#         criterion='friedman_mse',
#         max_features='sqrt',
#         max_depth=4,
# ) # 0.80382
# model = RandomForestClassifier(
#         random_state=0,
#         criterion='gini',
#         n_estimators=1000,
#         max_depth=None,
#         min_samples_split=8,
#         min_samples_leaf=1,
#         max_features='auto',
#         class_weight={0: 3, 1: 2},
# ) # 0.78468
model = RandomForestClassifier(
        random_state=0,
        criterion='gini',
        n_estimators=20,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=1,
        max_features='log2',
        class_weight={0: 6, 1: 7},
) # 0.81339
# model = RandomForestClassifier(
#     random_state=0,
#     criterion='entropy',
#     n_estimators=30,
#     max_depth=5,
#     min_samples_split=4,
#     min_samples_leaf=1,
#     max_features='log2',
#     class_weight={0: 6, 1: 4},
# )
res = case.predict(model)
res.to_csv('{}/{}_pred.csv'.format(config.results_folder, case.name), sep=',')
