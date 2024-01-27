import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data
data = pd.read_csv('./Data/full_data.csv')

# for each participant, only get the first 150 trials
# BestOption has no error
data = data.groupby('Subnum').head(150).reset_index(drop=True)

# best = data['BestOption'].copy()
#
# # check and make sure the bestoption column is correct
# data['BestOption'] = np.where((data['SetSeen.'] == 0) & (data['KeyResponse'] == 1), 1,
#                               np.where((data['SetSeen.'] == 0) & (data['KeyResponse'] == 0), 0,
#                                        np.where((data['SetSeen.'] == 1) & (data['KeyResponse'] == 3), 1,
#                                                 np.where((data['SetSeen.'] == 1) & (data['KeyResponse'] == 4), 0,
#                                                          data['BestOption']))))
# # Check if the two columns are equal
# are_equal = data['BestOption'].equals(best)
#
# print(f"Are the 'BestOption' columns equal? {are_equal}")

# separate by condition
data_A1 = data[data['Condition'] == 'S2A1']
data_A2 = data[data['Condition'] == 'S2A2']

# calculate the percentage of picking the low-uncertainty option
data_AB_A1 = data_A1[data_A1['SetSeen.'] == 0].copy()
data_AB_A1.loc[:, 'trial_group'] = data_AB_A1.groupby('Subnum').cumcount()//10 + 1
result_AB_A1 = data_AB_A1.groupby(['Subnum', 'trial_group'])['BestOption'].mean().reset_index()

data_filtered = data.iloc[:, 0:21].drop_duplicates().reset_index(drop=True)

# quickly plot the data


sns.set_theme(style="whitegrid")
ax = sns.lineplot(x="trial_group", y="BestOption", data=result_AB_A1)

plt.show()
