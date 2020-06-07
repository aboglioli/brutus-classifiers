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
# Load data
data = pd.read_csv('../1_Calificacion_Crediticia/data/scoring_train_test.csv', delimiter=';', decimal='.')
predict_data = pd.read_csv('../1_Calificacion_Crediticia/data/nuevas_instancias_scoring.csv', delimiter=';', decimal='.')

# Pre-process
data = data.drop(['id'], axis=1)
X = data.iloc[:,0:5]
y = data.iloc[:, 5:6]
predict_data.index = predict_data.index + 1

# Training
model = RandomForestClassifier(
    random_state=0,
    criterion='entropy',
    n_estimators=100,
    max_depth=6,
    min_samples_split=8,
    min_samples_leaf=2,
    max_features='auto',
    class_weight={0: 0.7, 1: 0.1}
)
model.fit(X, y.values.ravel())

# Results
y_pred = model.predict(predict_data)
res = pd.DataFrame(data=y_pred, columns=['Predict'])
res.index = res.index + 1
res.index.names = ['id']
res.to_csv('predicted.csv', sep=',')

#
# --- Case 2 ---
#
# # Load data
# data = pd.read_csv('../2_Muerte_Coronaria/data/datos_train_test_sh.csv', delimiter=',', decimal='.')
# predict_data = pd.read_csv('../2_Muerte_Coronaria/data/nuevas_instancias_a_predecir.csv', delimiter=';', decimal='.')
#
# # Pre-process
# data = data.drop(['id'], axis=1)
# data.famhist[data.famhist == 'Present'] = 1
# data.famhist[data.famhist == 'Absent'] = 0
# data.famhist = data.famhist.astype('int')
#
# X = data.iloc[:,0:9]
# y = data.iloc[:, 9:10]
#
# predict_data = predict_data.drop(['id'], axis=1)
# predict_data.famhist[predict_data.famhist == 'Present'] = 1
# predict_data.famhist[predict_data.famhist == 'Absent'] = 0
# predict_data.famhist = predict_data.famhist.astype('int')
# predict_data.index = predict_data.index + 1
#
# # Training
# model = RandomForestClassifier(
#     random_state=0,
#     criterion='gini',
#     n_estimators=10,
#     max_depth=6,
#     min_samples_split=2,
#     min_samples_leaf=1,
#     max_features='auto',
#     class_weight={0: 0.7, 1: 0.1},
# )
# model.fit(X, y.values.ravel())
#
# # Results
# y_pred = model.predict(predict_data)
# res = pd.DataFrame(data=y_pred, columns=['Predicted'])
# res.index = res.index + 1
# res.index.names = ['id']
# res.to_csv('predicted.csv', sep=',')
