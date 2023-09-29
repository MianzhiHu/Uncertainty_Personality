import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from utility import side_by_side_plot_generator

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

# delete rows with the reaction time over 20000ms
data = data[data['ReactTime'] < 20000]

# # check the different values in the Subnum column
# print(data['Subnum'].unique())

# for counts in data['Subnum'].value_counts():
#     print(counts)

# # save the complete dataset
# data.to_csv('./Data/full_data.csv', index=False)

# separate CA and BD trials
CA = data[data['SetSeen.'] == 2]
CB = data[data['SetSeen.'] == 3]
AD = data[data['SetSeen.'] == 4]
BD = data[data['SetSeen.'] == 5]
# print(CA['KeyResponse'].value_counts())
# print(BD['KeyResponse'].value_counts())

# remove the frequency-only condition
CA = CA[CA['Condition'] != 'S2A3']
CB = CB[CB['Condition'] != 'S2A3']
AD = AD[AD['Condition'] != 'S2A3']
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
# switch the position of the columns
CA_percentage = CA_percentage[['Subnum', 'Picking C', 'Picking A']]

CB_percentage = CB.groupby('Subnum')['KeyResponse'].value_counts(normalize=True).unstack().reset_index()
CB_percentage = CB_percentage.rename(columns={2: 'Picking B', 3: 'Picking C'})
CB_percentage = CB_percentage.fillna(0)
CB_percentage = CB_percentage[['Subnum', 'Picking C', 'Picking B']]

AD_percentage = AD.groupby('Subnum')['KeyResponse'].value_counts(normalize=True).unstack().reset_index()
AD_percentage = AD_percentage.rename(columns={1: 'Picking A', 4: 'Picking D'})
AD_percentage = AD_percentage.fillna(0)

BD_percentage = BD.groupby('Subnum')['KeyResponse'].value_counts(normalize=True).unstack().reset_index()
BD_percentage = BD_percentage.rename(columns={2: 'Picking B', 4: 'Picking D'})
BD_percentage = BD_percentage.fillna(0)
BD_percentage = BD_percentage[['Subnum', 'Picking D', 'Picking B']]

# do the same for the reaction time
CA_RT = CA.groupby('Subnum')['ReactTime'].mean().reset_index()
CA_RT = CA_RT.rename(columns={'ReactTime': 'RT_mean'})

CB_RT = CB.groupby('Subnum')['ReactTime'].mean().reset_index()
CB_RT = CB_RT.rename(columns={'ReactTime': 'RT_mean'})

AD_RT = AD.groupby('Subnum')['ReactTime'].mean().reset_index()
AD_RT = AD_RT.rename(columns={'ReactTime': 'RT_mean'})

BD_RT = BD.groupby('Subnum')['ReactTime'].mean().reset_index()
BD_RT = BD_RT.rename(columns={'ReactTime': 'RT_mean'})

# filter the data
CA_filtered = CA.iloc[:, 0:21]
CB_filtered = CB.iloc[:, 0:21]
AD_filtered = AD.iloc[:, 0:21]
BD_filtered = BD.iloc[:, 0:21]

# delete duplicate rows
CA_filtered = CA_filtered.drop_duplicates()
CB_filtered = CB_filtered.drop_duplicates()
AD_filtered = AD_filtered.drop_duplicates()
BD_filtered = BD_filtered.drop_duplicates()

# combine the three dataframes
CA_filtered = pd.merge(CA_filtered, CA_RT, on='Subnum')
CA_filtered = pd.merge(CA_filtered, CA_percentage, on='Subnum').iloc[:, 0:23]

# CA_A1 = CA_filtered[CA_filtered['Condition'] == 'S2A1']
# CA_A2 = CA_filtered[CA_filtered['Condition'] == 'S2A2']
# print(CA_A1['Picking A'].mean())
# print(CA_A2['Picking A'].mean())

CB_filtered = pd.merge(CB_filtered, CB_RT, on='Subnum')
CB_filtered = pd.merge(CB_filtered, CB_percentage, on='Subnum').iloc[:, 0:23]

AD_filtered = pd.merge(AD_filtered, AD_RT, on='Subnum')
AD_filtered = pd.merge(AD_filtered, AD_percentage, on='Subnum').iloc[:, 0:23]

BD_filtered = pd.merge(BD_filtered, BD_RT, on='Subnum')
BD_filtered = pd.merge(BD_filtered, BD_percentage, on='Subnum').iloc[:, 0:23]

# create a grouped bar plot for the average percentage of picking the best option
# create a list of the mean percentages
optimal_U_F = []
optimal_U = []
RT_U_F_mean = []
RT_U_mean = []
mean_list = [CA_filtered, CB_filtered, AD_filtered, BD_filtered]
labels = ['CA', 'CB', 'AD', 'DB']

for df in mean_list:
    df_U_F = df[df['Condition'] == 'S2A1']
    df_U = df[df['Condition'] == 'S2A2']
    optimal_U_F.append(df_U_F.iloc[:, -1].mean())
    optimal_U.append(df_U.iloc[:, -1].mean())
    RT_U_F_mean.append(df_U_F.iloc[:, -2].mean())
    RT_U_mean.append(df_U.iloc[:, -2].mean())

# conduct one-sample t-test
degrees_of_freedom = []
t_values = []
p_values = []

for df in mean_list:
    df_U_F = df[df['Condition'] == 'S2A1']
    df_U = df[df['Condition'] == 'S2A2']

    # Convert the last column of the subset data frames to NumPy arrays
    values_U_F = df_U_F.iloc[:, -1].to_numpy()
    values_U = df_U.iloc[:, -1].to_numpy()

    # Run t-test on the values
    t, p = stats.ttest_1samp(values_U_F, 0.5)
    degrees_of_freedom.append(len(df_U_F) - 1)  # Degrees of freedom is usually n - 1
    t_values.append(t)
    p_values.append(p)

    t_U, p_U = stats.ttest_1samp(values_U, 0.5)
    degrees_of_freedom.append(len(df_U) - 1)  # Degrees of freedom is usually n - 1
    t_values.append(t_U)
    p_values.append(p_U)


# compile the results into a DataFrame
results_df = pd.DataFrame({'Condition': ['Uncertainty-Frequency', 'Uncertainty Only'] * 4,
                            't-value': t_values,
                            'p-value': p_values,
                            'df': degrees_of_freedom})

# create the clustered bar plot
plt.figure(figsize=(10, 10))
x = np.arange(len(labels))
width = 0.35
plt.bar(x - width / 2, optimal_U_F, width, label='Uncertainty-Frequency')
plt.bar(x + width / 2, optimal_U, width, label='Uncertainty Only')
plt.xticks(x, labels)
plt.ylabel('Percentage of Picking the First Item Listed')
plt.title('(a) Percentage of Picking the First Item Listed')
# add a line for random chance
plt.axhline(y=0.5, color='r', linestyle='--')
plt.legend()
plt.savefig('./Figures/optimal_percentage.png', dpi=600)
plt.show()

# create the clustered bar plot
plt.clf()

plt.figure(figsize=(10, 10))
plt.bar(x - width / 2, RT_U_F_mean, width, label='Uncertainty-Frequency')
plt.bar(x + width / 2, RT_U_mean, width, label='Uncertainty Only')
plt.xticks(x, labels)
plt.ylim(1000, max(RT_U_F_mean + RT_U_mean) + 10)
plt.ylabel('Reaction Time (ms)')
plt.title('(b) Reaction Time')
plt.legend()
plt.savefig('./Figures/RT.png', dpi=600)
plt.show()

side_by_side_plot_generator(img1=plt.imread('./Figures/optimal_percentage.png'),
                            img2=plt.imread('./Figures/RT.png'),
                            figure_length=5,
                            figure_width=10,
                            title='behavioral_results',
                            orientation='vertical',
                            dpi=600)


# # save the filtered CA dataset
# CA_filtered.to_csv('./Data/CA_filtered.csv', index=False)
# # save the filtered BD dataset
# BD_filtered.to_csv('./Data/BD_filtered.csv', index=False)

# print(min(data['Reward']))

# test RT difference
def unfiltered_rt_generator(df):
    RT_A1 = df[df['Condition'] == 'S2A1']
    RT_A2 = df[df['Condition'] == 'S2A2']

    # conduct t-test for RT
    t_RT, p_RT = stats.ttest_ind(RT_A1['ReactTime'], RT_A2['ReactTime'])
    print(f'The t-value for RT is {t_RT} and the p-value is {p_RT}')
    print(f'The degree of freedom is {len(RT_A1) + len(RT_A2) - 2}')


transfer_trials = [CA, CB, AD, BD]

for df in transfer_trials:
    unfiltered_rt_generator(df)

# cross-trial comparison
f_RT, p_RT = stats.f_oneway(*[df['ReactTime'] for df in transfer_trials])
print(f'The f-value for RT is {f_RT} and the p-value is {p_RT}')

CA_unfiltered_A1 = CA[CA['Condition'] == 'S2A1']
CA_unfiltered_A2 = CA[CA['Condition'] == 'S2A2']

BD_unfiltered_A1 = BD[BD['Condition'] == 'S2A1']
BD_unfiltered_A2 = BD[BD['Condition'] == 'S2A2']

CA_unfiltered_A1_C = CA_unfiltered_A1[CA_unfiltered_A1['KeyResponse'] == 3]
CA_unfiltered_A1_A = CA_unfiltered_A1[CA_unfiltered_A1['KeyResponse'] == 1]

CA_unfiltered_A2_C = CA_unfiltered_A2[CA_unfiltered_A2['KeyResponse'] == 3]
CA_unfiltered_A2_A = CA_unfiltered_A2[CA_unfiltered_A2['KeyResponse'] == 1]

BD_unfiltered_A1_B = BD_unfiltered_A1[BD_unfiltered_A1['KeyResponse'] == 2]
BD_unfiltered_A1_D = BD_unfiltered_A1[BD_unfiltered_A1['KeyResponse'] == 4]

BD_unfiltered_A2_B = BD_unfiltered_A2[BD_unfiltered_A2['KeyResponse'] == 2]
BD_unfiltered_A2_D = BD_unfiltered_A2[BD_unfiltered_A2['KeyResponse'] == 4]


# # calculate the reward mean and std (fit with the experimental design)
# def reward_mean_std(df):
#     mean = df['Reward'].mean()
#     std = df['Reward'].std()
#     print(f'The mean is {mean} and the std is {std}')
#
#
# df_list = [CA_unfiltered_A1_C, CA_unfiltered_A1_A, CA_unfiltered_A2_C, CA_unfiltered_A2_A,
#            BD_unfiltered_A1_B, BD_unfiltered_A1_D, BD_unfiltered_A2_B, BD_unfiltered_A2_D]
#
# for df in df_list:
#     reward_mean_std(df)

# # t-test
# t, p = stats.ttest_ind(CA_unfiltered_A1_C['ReactTime'], CA_unfiltered_A1_A['ReactTime'])  # insignificant
#
# t_A2, p_A2 = stats.ttest_ind(CA_unfiltered_A2_C['ReactTime'], CA_unfiltered_A2_A['ReactTime'])  # insignificant
#
# t_BD, p_BD = stats.ttest_ind(BD_unfiltered_A1_B['ReactTime'], BD_unfiltered_A1_D['ReactTime'])
# print(BD_unfiltered_A1_B['ReactTime'].mean())  # D > B
# print(BD_unfiltered_A1_D['ReactTime'].mean())
#
# t_A2_BD, p_A2_BD = stats.ttest_ind(BD_unfiltered_A2_B['ReactTime'], BD_unfiltered_A2_D['ReactTime'])  # insignificant
#
print(stats.ttest_ind(BD_unfiltered_A1['ReactTime'], BD_unfiltered_A2['ReactTime']))
print(BD_unfiltered_A1['ReactTime'].mean())  # A1 > A2
print(BD_unfiltered_A2['ReactTime'].mean())

print(stats.ttest_ind(CA_unfiltered_A1['ReactTime'], CA_unfiltered_A2['ReactTime']))
print(CA_unfiltered_A1['ReactTime'].mean())  # A1 > A2
print(CA_unfiltered_A2['ReactTime'].mean())
#
# print(stats.ttest_ind(CA_unfiltered_A1['ReactTime'], BD_unfiltered_A1['ReactTime']))
# print(CA_unfiltered_A1['ReactTime'].mean())  # BD > CA
# print(BD_unfiltered_A1['ReactTime'].mean())
#
# print(stats.ttest_ind(CA_unfiltered_A2['ReactTime'], BD_unfiltered_A2['ReactTime']))  # not significant
