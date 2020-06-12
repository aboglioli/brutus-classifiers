from glob import glob
import pandas as pd
import config
from case import Case1, Case2

case = Case1()

print('# Case:', case.name)

dataframes = []
for filename in glob('{}/{}/*.csv'.format(config.results_folder, case.name)):
    dataframes.append(pd.read_csv(filename))

data = pd.concat(dataframes)
data = data.iloc[:, 1:]
data = data.sort_values(by=['StratifiedKFold'], ascending=False)

data = data.iloc[:10000, :]
save_file = '{}/{}.csv'.format(config.results_folder, case.name)
data.to_csv(save_file)
print('- Saved in:', save_file)
