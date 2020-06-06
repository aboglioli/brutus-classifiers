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

# Load data
data = pd.read_csv('data/scoring_train_test.csv', delimiter=';', decimal='.')
predict_data = pd.read_csv('data/nuevas_instancias_scoring.csv', delimiter=';', decimal='.')

# Pre-process
data = data.drop(['id'], axis=1)
X = data.iloc[:,0:5]
y = data.iloc[:, 5:6]

predict_data.index = predict_data.index + 1

# Training
knc_1 = KNeighborsClassifier(n_neighbors=2, leaf_size=30, algorithm='kd_tree', weights='uniform') # 0.90787

knc = KNeighborsClassifier(n_neighbors=2, leaf_size=10, algorithm='ball_tree', weights='uniform') # 0.92206
gbc = GradientBoostingClassifier(loss='deviance', n_estimators=100, subsample=0.8, criterion='mse') # 0.903
rfc = RandomForestClassifier(max_leaf_nodes=50, n_estimators=4, max_depth=12) # 0.90199

estimators = [('1', knc), ('2', gbc)]
# model = ensemble.VotingClassifier(
#     estimators=estimators,
#     voting='soft',
#     weights=[3, 1]
# )
model = rfc
model.fit(X, y.values.ravel())

# Prediction
y_pred = model.predict(predict_data)

# Results
res = pd.DataFrame(data=y_pred, columns=['Predict'])
res.index = res.index + 1
res.index.names = ['id']

res.to_csv('predicted.csv',sep=',')
