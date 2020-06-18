import numpy as np

from sklearn import model_selection

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_curve, auc


class Scorer():
    def set_dataset(self, X, y):
        self.X = X
        self.y = y

    def set_train_test(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def split(self, **args):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            self.X, self.y, **args)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def set_criteria(self, criteria):
        self.criteria = criteria

    def score(self, Classifier, args, simple=True):
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

        res = {
            'Name': name,
            'Parameters': '{}'.format(args),
        }

        if 'CM' in self.criteria:
            cm = confusion_matrix(y_test, y_pred)
            category_precision = cm.diagonal() / np.sum(cm, axis=1)
            res['CM'] = cm
            res['CP_1'] = category_precision[0]
            res['CP_2'] = category_precision[1]

        if 'BalancedAccuracy' in self.criteria:
            res['BalancedAccuracy'] = balanced_accuracy_score(y_test, y_pred)

        if 'Accuracy' in self.criteria:
            res['Accuracy'] = accuracy_score(y_test, y_pred)

        if 'Recall' in self.criteria:
            res['Recall'] = recall_score(y_test, y_pred, average='macro')

        if 'Precision' in self.criteria:
            res['Precision'] = precision_score(y_test, y_pred, average='macro')

        if 'F1' in self.criteria:
            res['F1'] = f1_score(y_test, y_pred, average='macro')

        if 'Kappa' in self.criteria:
            res['Kappa'] = cohen_kappa_score(y_test, y_pred)

        if 'ROC' in self.criteria:
            y_pred_probabilities = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_probabilities)
            res['ROC'] = auc(fpr, tpr)

        if 'KFold' in self.criteria:
            kfold = model_selection.KFold(n_splits=10)
            res['KFold'] = model_selection.cross_val_score(
                model, X, y, cv=kfold).mean()

        if 'StratifiedKFold' in self.criteria:
            skfold = model_selection.StratifiedKFold(n_splits=10)
            res['StratifiedKFold'] = model_selection.cross_val_score(
                model, X, y, cv=skfold, scoring='f1_macro').mean()

        if 'LeaveOneOut' in self.criteria:
            loo = model_selection.LeaveOneOut()
            res['LeaveOneOut'] = model_selection.cross_val_score(
                model, X, y, cv=loo).mean()

        return res
