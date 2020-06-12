from pathlib import Path
import pandas as pd
import config


class Case():
    def __init__(self, name, dataset_file):
        self.name = name
        self.dataset_file = dataset_file

        self.results_folder = '{}/{}'.format(config.results_folder, name)
        self.predicted_folder = '{}/{}'.format(config.predicted_folder, name)

        Path(self.results_folder).mkdir(parents=True, exist_ok=True)
        Path(self.predicted_folder).mkdir(parents=True, exist_ok=True)

    def get_data(self):
        pass


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
