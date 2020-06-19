from pathlib import Path
import pandas as pd
import config


class Case():
    def __init__(self, dataset_file):
        self.name = self.__class__.__name__
        self.dataset_file = dataset_file

        Path('{}/{}'.format(config.results_folder, self.name)).mkdir(parents=True, exist_ok=True)
        Path('predicted').mkdir(parents=True, exist_ok=True)

    def get_data(self):
        pass

    def get_train_test(self):
        pass

    def predict(self, model, predict_data):
        X, y = self.get_data()
        model.fit(X, y.values.ravel())
        return model.predict(predict_data)

class Credito(Case):
    def __init__(self):
        super().__init__('../1_Calificacion_Crediticia/data/scoring_train_test.csv')

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

class MuerteCoronaria(Case):
    def __init__(self):
        super().__init__('../2_Muerte_Coronaria/data/datos_train_test_sh.csv')

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


class Titanic(Case):
    def __init__(self):
        super().__init__('data/train.csv')

    def map_data(self,data):
        # Age
        data['Age'].fillna(data['Age'].median(), inplace=True)

        # Title
        # Title_Royalty
        titles_dir = {
            "Capt": "Officer",
            "Col": "Officer",
            "Major": "Officer",
            "Jonkheer": "Royalty",
            "Don": "Royalty",
            "Sir" : "Royalty",
            "Dr": "Officer",
            "Rev": "Officer",
            "the Countess":"Royalty",
            "Mme": "Mrs",
            "Mlle": "Miss",
            "Ms": "Mrs",
            "Mr" : "Mr",
            "Mrs" : "Mrs",
            "Miss" : "Miss",
            "Master" : "Master",
            "Lady" : "Royalty",
        }
        data['Title'] = data['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
        data['Title'] = data.Title.map(titles_dir)

        data.drop('Name', axis=1, inplace=True)
        titles_dummies = pd.get_dummies(data['Title'], prefix='Title')
        data = pd.concat([data, titles_dummies], axis=1)
        data.drop('Title', axis=1, inplace=True)

        # Fare
        data['Fare'].fillna(data['Fare'].mean(), inplace=True)

        # Embarked
        data['Embarked'].fillna('S', inplace=True)
        embarked_dummies = pd.get_dummies(data['Embarked'], prefix='Embarked')
        data = pd.concat([data, embarked_dummies], axis=1)
        data.drop('Embarked', axis=1, inplace=True)

        # Cabin
        # Cabin_T
        data['Cabin'].fillna('U', inplace=True)
        data['Cabin'] = data['Cabin'].map(lambda c: c[0])
        cabin_dummies = pd.get_dummies(data['Cabin'], prefix='Cabin')
        data = pd.concat([data, cabin_dummies], axis=1)
        data.drop('Cabin', axis=1, inplace=True)

        # Sex
        data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})

        # Pclass
        pclass_dummies = pd.get_dummies(data['Pclass'], prefix='Pclass')
        data = pd.concat([data, pclass_dummies], axis=1)
        data.drop('Pclass', axis=1, inplace=True)

        # Ticket
        # -Ticket_A
        def map_ticket(ticket):
            ticket = ticket.replace('.', '')
            ticket = ticket.replace('/', '')
            ticket = ticket.split()
            ticket = map(lambda t: t.strip(), ticket)
            ticket = list(filter(lambda t: not t.isdigit(), ticket))
            if len(ticket) > 0:
                return ticket[0]
            else:
                return 'XXX'
        data['Ticket'] = data['Ticket'].map(map_ticket)
        ticket_dummies = pd.get_dummies(data['Ticket'], prefix='Ticket')
        data = pd.concat([data, ticket_dummies], axis=1)
        data.drop('Ticket', axis=1, inplace=True)

        # Family
        data['FamilySize'] = data['Parch'] + data['SibSp'] + 1
        data['Singleton'] = data['FamilySize'].map(lambda s: 1 if s == 1 else 0)
        data['SmallFamily'] = data['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
        data['LargeFamily'] = data['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

        return data

    def get_data(self):
        train = pd.read_csv(self.dataset_file, index_col=0)
        test = pd.read_csv('data/test.csv', index_col=0)

        data = train.append(test)
        data = self.map_data(data)

        X_train = data.iloc[:891, 1:]
        y_train = data.iloc[:891, 0:1]
        X_test = data.iloc[891:, 1:]
        y_test = data.iloc[891:, 0:1]

        return (X_train, y_train)

    def get_train_test(self):
        train = pd.read_csv(self.dataset_file, index_col=0)
        test = pd.read_csv('data/test.csv', index_col=0)

        data = train.append(test)
        data = self.map_data(data)

        X_train = data.iloc[:891, 1:]
        y_train = data.iloc[:891, 0:1]
        X_test = data.iloc[891:, 1:]
        y_test = data.iloc[891:, 0:1]

        return (X_train, X_test, y_train, y_test)

    def predict(self, model):
        train = pd.read_csv(self.dataset_file, index_col=0)
        test = pd.read_csv('data/test.csv', index_col=0)

        data = train.append(test)
        data = self.map_data(data)

        X_train = data.iloc[:891, 1:]
        y_train = data.iloc[:891, 0:1]
        X_test = data.iloc[891:, 1:]
        y_test = data.iloc[891:, 0:1]

        y_pred = super().predict(model, X_test)

        res = pd.DataFrame(data=y_pred, columns=['Survived'])
        res.index.names = ['PassengerId']
        res.index = res.index + 892
        return res
