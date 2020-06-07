import warnings
warnings.filterwarnings('ignore')

import pprint

import pandas as pd
import numpy as np

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

#
# --- Case 1 ---
#
# # Load data
# data = pd.read_csv('../1_Calificacion_Crediticia/data/scoring_train_test.csv', delimiter=';', decimal='.')
# predict_data = pd.read_csv('../1_Calificacion_Crediticia/data/nuevas_instancias_scoring.csv', delimiter=';', decimal='.')
#
# # Pre-process
# data = data.drop(['id'], axis=1)
# X = data.iloc[:,0:5]
# y = data.iloc[:, 5:6]
# predict_data.index = predict_data.index + 1
#
# # Training
# knc_1 = KNeighborsClassifier(n_neighbors=2, leaf_size=10, algorithm='ball_tree', weights='uniform') # 0.92206
# knc_2 = KNeighborsClassifier(n_neighbors=2, leaf_size=30, algorithm='kd_tree', weights='uniform') # 0.90787
# gbc = GradientBoostingClassifier(loss='deviance', n_estimators=100, subsample=0.8, criterion='mse') # 0.903
# rfc = RandomForestClassifier(max_leaf_nodes=50, n_estimators=4, max_depth=12) # 0.90199
# dtc1 = DecisionTreeClassifier(
#     criterion='entropy',
#     splitter='random',
#     max_depth=15,
#     min_samples_split=6,
#     min_samples_leaf=3,
#     class_weight={0: 0.6, 1: 0.2},
# ) # 0.91969
# dtc2 = DecisionTreeClassifier(
#     criterion='gini',
#     splitter='random',
#     max_depth=100,
#     min_samples_split=8,
#     min_samples_leaf=3,
#     class_weight={0: 0.4, 1: 0.2},
# ) # 0.81462
#
# model = DecisionTreeClassifier(
#     random_state=0,
#     criterion='entropy',
#     splitter='best',
#     max_depth=10,
#     min_samples_split=10,
#     min_samples_leaf=1,
#     class_weight={0: 0.8, 1: 0.2},
# )
#
# model.fit(X, y.values.ravel())
#
# # Results
# y_pred = model.predict(predict_data)
# res = pd.DataFrame(data=y_pred, columns=['Predict'])
# res.index = res.index + 1
# res.index.names = ['id']
# res.to_csv('predicted.csv', sep=',')

#
# --- Case 2 ---
#
# Load data
data = pd.read_csv('../2_Muerte_Coronaria/data/datos_train_test_sh.csv', delimiter=',', decimal='.')
predict_data = pd.read_csv('../2_Muerte_Coronaria/data/nuevas_instancias_a_predecir.csv', delimiter=';', decimal='.')

# Pre-process
data = data.drop(['id'], axis=1)
data.famhist[data.famhist == 'Present'] = 1
data.famhist[data.famhist == 'Absent'] = 0
data.famhist = data.famhist.astype('int')

X = data.iloc[:,0:9]
y = data.iloc[:, 9:10]

predict_data = predict_data.drop(['id'], axis=1)
predict_data.famhist[predict_data.famhist == 'Present'] = 1
predict_data.famhist[predict_data.famhist == 'Absent'] = 0
predict_data.famhist = predict_data.famhist.astype('int')
predict_data.index = predict_data.index + 1

# Training
# model = DecisionTreeClassifier(
#     criterion='gini',
#     splitter='best',
#     max_depth=30,
#     min_samples_split=2,
#     min_samples_leaf=3,
#     class_weight={0: 0.2, 1: 0.6},
# ) # 0.53787

model = GradientBoostingClassifier(
    random_state=0,
    loss='exponential',
    learning_rate=0.30000000000000004,
    n_estimators=30,
    subsample=0.30000000000000004,
    criterion='friedman_mse',
    max_features='auto',
    max_depth=4
) # 0.64015

model.fit(X, y.values.ravel())

# Results
y_pred = model.predict(predict_data)
res = pd.DataFrame(data=y_pred, columns=['Predicted'])
res.index = res.index + 1
res.index.names = ['id']
res.to_csv('predicted.csv', sep=',')
