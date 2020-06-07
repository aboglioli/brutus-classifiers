from glob import glob
import pandas as pd

# Configuration
results_folder = 'results'
case = 'c2'
omit_meta_classifiers = True

print ('# Case:', case)

dataframes = []
for filename in glob('{}/{}/*.csv'.format(results_folder, case)):
    dataframes.append(pd.read_csv(filename))

data = pd.concat(dataframes)
data = data.iloc[:, 1:]
data = data.sort_values(by=['BalancedAccuracy', 'ROC'], ascending=False)
# data = data.sort_values(by=['ROC', 'BalancedAccuracy'], ascending=False)

if omit_meta_classifiers:
    data = data[data.Name != 'VotingClassifier']
    print('* Omitting meta classifiers *')

print('\n[ Best BalancedAccuracy ]')
best_BA = data.sort_values(by=['BalancedAccuracy', 'ROC'], ascending=False).iloc[0, :]
print(best_BA)
print('Parameters: ', best_BA.Parameters)

print('\n[ Best ROC ]')
best_ROC = data.sort_values(by=['ROC', 'BalancedAccuracy'], ascending=False).iloc[0, :]
print(best_ROC)
print('Parameters: ', best_ROC.Parameters)

data = data.iloc[:10000, :]
save_file = '{}/{}.csv'.format(results_folder, case)
data.to_csv(save_file)
print('\nSaved in:', save_file)
