import numpy as np

from sklearn import model_selection

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_curve, auc

class Scorer():
    def __init__(self, X, y, X_train, y_train, X_test, y_test):
        self.X = X
        self.y = y
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def score(self, Classifier, args):
        X = self.X
        y = self.y
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test

        name = Classifier.__name__

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

        return  {
            'Name': model.__class__.__name__,
            'Parameters': '{}'.format(args),
            'ScoreTrain': model.score(X_train, y_train.values.ravel()),
            'ScoreTest': model.score(X_test, y_test.values.ravel()),
            'Accuracy': accuracy_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred, average='macro'),
            'Precision': precision_score(y_test, y_pred, average='macro'),
            'F1': f1_score(y_test, y_pred, average='macro'),
            'Kappa': cohen_kappa_score(y_test, y_pred),
            'GlobalPrecision': global_precision,
            'GlobalError': global_error,
            'CM': cm,
            'CP_1': category_precision[0],
            'CP_2': category_precision[1],
            'ROC': roc_auc,
            'KFold': kfold,
            'StratifiedKFold': skfold,
            # 'LeaveOneOut': loo,
        }
