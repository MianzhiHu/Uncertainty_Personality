import pandas as pd

# Read the data
April = pd.read_csv('./data/PACStudy2Data_April_2023.csv')
June = pd.read_csv('./data/PACStudy2Data_CombData_6_21_22.csv')

# delete the first two columns
April = April.drop(April.columns[[0]], axis=1)
June = June.drop(June.columns[[0, 1]], axis=1)

April_frequency = April[April['Condition'] == 'S2A3']
June_frequency = June[June['Condition'] == 'S2A3']
print(April_frequency['Reward'].std())
print(June_frequency['Reward'].std())
# print(April_frequency['Subnum'].value_counts())
# print(June_frequency['Subnum'].value_counts())

# group by Subnum's value counts and keep only the ones with 500
June_frequency_500 = June_frequency.groupby('Subnum').filter(lambda x: len(x) == 500)
June_frequency_250 = June_frequency.groupby('Subnum').filter(lambda x: len(x) == 250)

# print(June_frequency_500['Subnum'].value_counts())
# print(June_frequency_250['Subnum'].value_counts())
print(June_frequency_500['Reward'].std())
print(June_frequency_250['Reward'].std())


# cutting the June data from row 28000
June_additional = June.iloc[28000:]

# change the Subnum column
June_additional.loc[:, 'Subnum'] = June_additional['Subnum'].apply(lambda x: x + 112)
April.loc[:, 'Subnum'] = April['Subnum'].apply(lambda x: x + 193)

# now combine the three dataframes
data = pd.concat([June, June_additional, April], ignore_index=True)

# # check the different values in the Subnum column
# print(data['Subnum'].unique())

# for counts in data['Subnum'].value_counts():
#     print(counts)

# # save the complete dataset
# data.to_csv('./Data/full_data.csv', index=False)

# separate CA and BD trials
CA = data[data['SetSeen.'] == 2]
BD = data[data['SetSeen.'] == 5]
# print(CA['KeyResponse'].value_counts())
# print(BD['KeyResponse'].value_counts())

# remove the frequency-only condition
CA = CA[CA['Condition'] != 'S2A3']
BD = BD[BD['Condition'] != 'S2A3']
# print(CA['Condition'].value_counts())
# print(BD['Condition'].value_counts())

# # save the CA dataset
# CA.to_csv('./Data/CA_data.csv', index=False)
# # save the BD dataset
# BD.to_csv('./Data/BD_data.csv', index=False)

# group by Subnum and get the percentage of BestOption == 1
CA_percentage = CA.groupby('Subnum')['KeyResponse'].value_counts(normalize=True).unstack().reset_index()
CA_percentage = CA_percentage.rename(columns={1: 'Picking A', 3: 'Picking C'})
CA_percentage = CA_percentage.fillna(0)

BD_percentage = BD.groupby('Subnum')['KeyResponse'].value_counts(normalize=True).unstack().reset_index()
BD_percentage = BD_percentage.rename(columns={2: 'Picking B', 4: 'Picking D'})
BD_percentage = BD_percentage.fillna(0)


# do the same for the reaction time
CA_RT = CA.groupby('Subnum')['ReactTime'].mean().reset_index()
CA_RT = CA_RT.rename(columns={'ReactTime': 'RT_mean'})

BD_RT = BD.groupby('Subnum')['ReactTime'].mean().reset_index()
BD_RT = BD_RT.rename(columns={'ReactTime': 'RT_mean'})


# filter the data
CA_filtered = CA.iloc[:, 0:21]
BD_filtered = BD.iloc[:, 0:21]

# delete duplicate rows
CA_filtered = CA_filtered.drop_duplicates()
BD_filtered = BD_filtered.drop_duplicates()

# combine the three dataframes
CA_filtered = pd.merge(CA_filtered, CA_RT, on='Subnum')
CA_filtered = pd.merge(CA_filtered, CA_percentage, on='Subnum').iloc[:, 0:23]

BD_filtered = pd.merge(BD_filtered, BD_RT, on='Subnum')
BD_filtered = pd.merge(BD_filtered, BD_percentage, on='Subnum').iloc[:, 0:23]

# # save the filtered CA dataset
# CA_filtered.to_csv('./Data/CA_filtered.csv', index=False)
# # save the filtered BD dataset
# BD_filtered.to_csv('./Data/BD_filtered.csv', index=False)

