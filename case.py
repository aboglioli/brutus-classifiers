from pathlib import Path
import pandas as pd
import config


class Case():
    def __init__(self, name, dataset_file):
        self.name = name
        self.dataset_file = dataset_file

        Path('{}/{}'.format(config.results_folder, name)).mkdir(parents=True, exist_ok=True)
        Path('predicted').mkdir(parents=True, exist_ok=True)

    def get_data(self):
        pass

    def predict(self, model, predict_data):
        X, y = self.get_data()
        model.fit(X, y.values.ravel())
        return model.predict(predict_data)


class Case1(Case):
    def __init__(self):
        super().__init__('c1', '../1_Calificacion_Crediticia/data/scoring_train_test.csv')

    def get_data(self):
        data = pd.read_csv(self.dataset_file, delimiter=';', decimal='.')

        data = data.drop(['id'], axis=1)
        X = data.iloc[:, 0:5]
        y = data.iloc[:, 5:6]

        self.X = X
        self.y = y

        return (X, y)

    def predict(self, model):
        predict_data = pd.read_csv(
            '../1_Calificacion_Crediticia/data/nuevas_instancias_scoring.csv', delimiter=';', decimal='.')
        predict_data.index = predict_data.index + 1

        y_pred = super().predict(model, predict_data)

        res = pd.DataFrame(data=y_pred, columns=['Predict'])
        res.index = res.index + 1
        res.index.names = ['id']
        return res


class Case2(Case):
    def __init__(self):
        super().__init__('c2', '../2_Muerte_Coronaria/data/datos_train_test_sh.csv')

    def get_data(self):
        data = pd.read_csv(self.dataset_file, delimiter=',', decimal='.')

        data = data.drop(['id'], axis=1)
        data.famhist[data.famhist == 'Present'] = 1
        data.famhist[data.famhist == 'Absent'] = 0
        data.famhist = data.famhist.astype('int')

        X = data.iloc[:, 0:9]
        y = data.iloc[:, 9:10]

        self.X = X
        self.y = y

        return (X, y)

    def predict(self, model):
        predict_data = pd.read_csv(
            '../2_Muerte_Coronaria/data/nuevas_instancias_a_predecir.csv', delimiter=';', decimal='.')
        predict_data.index = predict_data.index + 1

        predict_data = predict_data.drop(['id'], axis=1)
        predict_data.famhist[predict_data.famhist == 'Present'] = 1
        predict_data.famhist[predict_data.famhist == 'Absent'] = 0
        predict_data.famhist = predict_data.famhist.astype('int')
        predict_data.index = predict_data.index + 1

        y_pred = super().predict(model, predict_data)

        res = pd.DataFrame(data=y_pred, columns=['Predicted'])
        res.index = res.index + 1
        res.index.names = ['id']
        return res
