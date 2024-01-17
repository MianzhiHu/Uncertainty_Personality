import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

# Read the data
CA = pd.read_csv('./Data/CA_filtered_12.csv')
BD = pd.read_csv('./Data/BD_filtered_12.csv')

CA = CA[CA['Condition'] != 'S2A3']
BD = BD[BD['Condition'] != 'S2A3']

# CA = pd.read_csv('./Data/CA_filtered_43.csv')
# BD = pd.read_csv('./Data/BD_filtered_43.csv')

# # transform the mental health variables
# # add a column to indicate the anxiety level
# CA.loc[:, 'TraitAnxiety'] = CA['STAIT'].apply(lambda x: 'No Anxiety' if x <= 37 else 'Moderate Anxiety' if 37 < x <= 44 else 'Severe Anxiety')
# CA.loc[:, 'StateAnxiety'] = CA['STAIS'].apply(lambda x: 'No Anxiety' if x <= 37 else 'Moderate Anxiety' if 37 < x <= 44 else 'Severe Anxiety')
# CA.loc[:, 'Depression'] = CA['CESD'].apply(lambda x: 'No Depression' if x <= 9 else 'Mild Depression' if 9 < x <= 15 else 'Moderate Depression' if 15 < x <= 24 else 'Severe Depression')

# # save the data
# CA.to_csv('./Data/CA_R.csv', index=True)

# separate by condition
CA_A1 = CA[CA['Condition'] == 'S2A1']
CA_A2 = CA[CA['Condition'] == 'S2A2']
CA_A3 = CA[CA['Condition'] == 'S2A3']

BD_A1 = BD[BD['Condition'] == 'S2A1']
BD_A2 = BD[BD['Condition'] == 'S2A2']
BD_A3 = BD[BD['Condition'] == 'S2A3']


# def anxiety_separation(df, variable_of_interest):
#     df_no_anxiety = df[df[variable_of_interest] <= 37]
#     df_moderate_anxiety = df[(df[variable_of_interest] > 37) & (df[variable_of_interest] <= 44)]
#     df_severe_anxiety = df[df[variable_of_interest] > 44]
#     return df_no_anxiety, df_moderate_anxiety, df_severe_anxiety
#
#
# def depression_separation(df):
#     df_no_depression = df[df['Depression'] == 'No Depression']
#     df_mild_depression = df[df['Depression'] == 'Mild Depression']
#     df_moderate_depression = df[df["Depression"] == 'Moderate Depression']
#     df_severe_depression = df[df['Depression'] == 'Severe Depression']
#     return df_no_depression, df_mild_depression, df_moderate_depression, df_severe_depression


# # separate again by mental health status (not meaningful)
# # anxiety
# CA_A1_no_trait_anxiety, CA_A1_moderate_trait_anxiety, CA_A1_severe_trait_anxiety = anxiety_separation(CA_A1, 'STAIT')
# CA_A2_no_trait_anxiety, CA_A2_moderate_trait_anxiety, CA_A2_severe_trait_anxiety = anxiety_separation(CA_A2, 'STAIT')
# CA_A1_no_state_anxiety, CA_A1_moderate_state_anxiety, CA_A1_severe_state_anxiety = anxiety_separation(CA_A1, 'STAIS')
# CA_A2_no_state_anxiety, CA_A2_moderate_state_anxiety, CA_A2_severe_state_anxiety = anxiety_separation(CA_A2, 'STAIS')
#
# # depression
# CA_A1_no_depression, CA_A1_mild_depression, CA_A1_moderate_depression, CA_A1_severe_depression = depression_separation(CA_A1)
# CA_A2_no_depression, CA_A2_mild_depression, CA_A2_moderate_depression, CA_A2_severe_depression = depression_separation(CA_A2)


# test the correlations
def correlation_test(df, variable_of_interest):
    # Preallocate a list to store the results
    results = []
    # Loop through the personality scales
    for scale in df.columns[4:20]:
        r, p = stats.pearsonr(df[scale], df[variable_of_interest])
        results.append({'personality_scale': scale, 'r': r, 'p': p})
    # Convert the list to a DataFrame
    results_df = pd.DataFrame(results, columns=['personality_scale', 'r', 'p'])
    # # Calculate the Bonferroni-corrected p-values
    results_df.loc[:, 'p_corrected'] = results_df['p'] * 16
    return results_df


# get the results
A1_A_picks = correlation_test(CA_A1, 'Picking A')
A2_A_picks = correlation_test(CA_A2, 'Picking A')
A3_A_picks = correlation_test(CA_A3, 'Picking A')

A1_RT = correlation_test(CA_A1, 'RT_mean')
A2_RT = correlation_test(CA_A2, 'RT_mean')
A3_RT = correlation_test(CA_A3, 'RT_mean')

A1_B_picks = correlation_test(BD_A1, 'Picking B')
A2_B_picks = correlation_test(BD_A2, 'Picking B')
A3_B_picks = correlation_test(BD_A3, 'Picking B')

A1_RT_BD = correlation_test(BD_A1, 'RT_mean')
A2_RT_BD = correlation_test(BD_A2, 'RT_mean')
A3_RT_BD = correlation_test(BD_A3, 'RT_mean')

# subgroup_list = [CA_A1_no_trait_anxiety, CA_A1_moderate_trait_anxiety, CA_A1_severe_trait_anxiety,
#                  CA_A1_no_state_anxiety, CA_A1_moderate_state_anxiety, CA_A1_severe_state_anxiety,
#                  CA_A1_no_depression, CA_A1_mild_depression, CA_A1_moderate_depression, CA_A1_severe_depression,
#                  CA_A2_no_trait_anxiety, CA_A2_moderate_trait_anxiety, CA_A2_severe_trait_anxiety,
#                  CA_A2_no_state_anxiety, CA_A2_moderate_state_anxiety, CA_A2_severe_state_anxiety,
#                  CA_A2_no_depression, CA_A2_mild_depression, CA_A2_moderate_depression, CA_A2_severe_depression]
#
# subgroup_names_individual = ['no_trait_anxiety', 'moderate_trait_anxiety', 'severe_trait_anxiety',
#                              'no_state_anxiety', 'moderate_state_anxiety', 'severe_state_anxiety',
#                              'no_depression', 'mild_depression', 'moderate_depression', 'severe_depression']
#
# subgroup_names = ['no_trait_anxiety', 'moderate_trait_anxiety', 'severe_trait_anxiety',
#                   'no_state_anxiety', 'moderate_state_anxiety', 'severe_state_anxiety',
#                   'no_depression', 'mild_depression', 'moderate_depression', 'severe_depression',
#                   'no_trait_anxiety', 'moderate_trait_anxiety', 'severe_trait_anxiety',
#                   'no_state_anxiety', 'moderate_state_anxiety', 'severe_state_anxiety',
#                   'no_depression', 'mild_depression', 'moderate_depression', 'severe_depression']
#
# # test the correlations for each subgroup (not meaningful)
# for i in range(len(subgroup_list)):
#     subgroup_picks = correlation_test(subgroup_list[i], 'Picking A')
#     subgroup_RT = correlation_test(subgroup_list[i], 'RT_mean')
#     subgroup_picks.to_csv(f'./Data/{subgroup_names[i]}_picks.csv', index=False)
#     subgroup_RT.to_csv(f'./Data/{subgroup_names[i]}_RT.csv', index=False)
#
# # test the A percentage against random chance
# for i in range(10):
#     t, p = stats.ttest_1samp(subgroup_list[i]['Picking A'], 0.5)
#     print(f'A1: the t-value for {subgroup_names[i]} is {t} and the p-value is {p}')
#
# for i in range(10, 20):
#     t, p = stats.ttest_1samp(subgroup_list[i]['Picking A'], 0.5)
#     print(f'A2: the t-value for {subgroup_names[i]} is {t} and the p-value is {p}')
#
# # plot the mean percentage of A picks for each subgroup
# mean_perc_A_A1 = []
# mean_perc_A_A2 = []
#
# for i in range(len(subgroup_list)):
#     if i < 10:
#         print(i)
#         mean_perc_A_A1.append(subgroup_list[i]['Picking A'].mean())
#     else:
#         mean_perc_A_A2.append(subgroup_list[i]['Picking A'].mean())
#
# # plot the mean percentage of A picks for each subgroup
# plt.figure(figsize=(15, 15))
# plt.bar(subgroup_names_individual, mean_perc_A_A1)
# plt.xticks(rotation=45)
# plt.ylabel('Mean Percentage of Picking A')
# plt.title('Mean Percentage of Picking A in U+F')
# plt.show(dpi=600)
#
# plt.figure(figsize=(15, 15))
# plt.bar(subgroup_names_individual, mean_perc_A_A2)
# plt.xticks(rotation=45)
# plt.ylabel('Mean Percentage of Picking A')
# plt.title('Mean Percentage of Picking A in U')
# plt.show(dpi=600)

# # run ANOVA (not significant)
# print(stats.f_oneway(CA_A1_no_trait_anxiety['Picking A'], CA_A1_moderate_trait_anxiety['Picking A'],
#                      CA_A1_severe_trait_anxiety['Picking A']))
# print(stats.f_oneway(CA_A2_no_trait_anxiety['Picking A'], CA_A2_moderate_trait_anxiety['Picking A'],
#                         CA_A2_severe_trait_anxiety['Picking A']))
# print(stats.f_oneway(CA_A1_no_state_anxiety['Picking A'], CA_A1_moderate_state_anxiety['Picking A'],
#                         CA_A1_severe_state_anxiety['Picking A']))
# print(stats.f_oneway(CA_A2_no_state_anxiety['Picking A'], CA_A2_moderate_state_anxiety['Picking A'],
#                         CA_A2_severe_state_anxiety['Picking A']))
# print(stats.f_oneway(CA_A1_no_depression['Picking A'], CA_A1_mild_depression['Picking A'],
#                         CA_A1_moderate_depression['Picking A'], CA_A1_severe_depression['Picking A']))
# print(stats.f_oneway(CA_A2_no_depression['Picking A'], CA_A2_mild_depression['Picking A'],
#                         CA_A2_moderate_depression['Picking A'], CA_A2_severe_depression['Picking A']))

# # correlate RT with A picks (not significant)
# r, p = stats.pearsonr(CA['RT_mean'], CA['Picking A'])
# r_A1, p_A1 = stats.pearsonr(CA_A1['RT_mean'], CA_A1['Picking A'])
# r_A2, p_A2 = stats.pearsonr(CA_A2['RT_mean'], CA_A2['Picking A'])

# # correlate RT with B picks (not significant)
# r_BD, p_BD = stats.pearsonr(BD['RT_mean'], BD['Picking B'])
# r_A1_BD, p_A1_BD = stats.pearsonr(BD_A1['RT_mean'], BD_A1['Picking B'])
# r_A2_BD, p_A2_BD = stats.pearsonr(BD_A2['RT_mean'], BD_A2['Picking B'])

# # test correlations between personality scales (disinhibition scales are correlated)
# results = []
#
# for i in range(4, 20):
#     for j in range(i + 1, 20):
#         r, p = stats.pearsonr(CA[CA.columns[i]], CA[CA.columns[j]])
#         results.append({'personality_scale_1': CA.columns[i], 'personality_scale_2': CA.columns[j], 'r': r, 'p': p})
#
# results_df = pd.DataFrame(results, columns=['personality_scale_1', 'personality_scale_2', 'r', 'p'])

