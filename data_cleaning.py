import pandas as pd

# Read the data
April = pd.read_csv('./data/PACStudy2Data_April_2023.csv')
June = pd.read_csv('./data/PACStudy2Data_CombData_6_21_22.csv')

# delete the first two columns
April = April.drop(April.columns[[0]], axis=1)
June = June.drop(June.columns[[0, 1]], axis=1)

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

# separate CA trials
CA = data[data['SetSeen.'] == 2]

# remove the frequency-only condition
CA = CA[CA['Condition'] != 'S2A3']

# # save the CA dataset
# CA.to_csv('./Data/CA_data.csv', index=False)

# print(CA['KeyResponse'].value_counts())
# print(CA['Condition'].value_counts())

# group by Subnum and get the percentage of BestOption == 1
CA_percentage = CA.groupby('Subnum')['KeyResponse'].value_counts(normalize=True).unstack()
CA_percentage = CA_percentage.reset_index()
CA_percentage = CA_percentage.rename(columns={1: 'Picking A', 3: 'Picking C'})
CA_percentage = CA_percentage.fillna(0)

# do the same for the reaction time
CA_RT = CA.groupby('Subnum')['ReactTime'].mean()
CA_RT = CA_RT.reset_index()
CA_RT = CA_RT.rename(columns={'ReactTime': 'RT_mean'})


# filter the data
CA_filtered = CA.iloc[:, 0:21]

# delete duplicate rows
CA_filtered = CA_filtered.drop_duplicates()

# combine the three dataframes
CA_filtered = pd.merge(CA_filtered, CA_RT, on='Subnum')
CA_filtered = pd.merge(CA_filtered, CA_percentage, on='Subnum').iloc[:, 0:23]

# # save the filtered CA dataset
# CA_filtered.to_csv('./Data/CA_filtered.csv', index=False)

