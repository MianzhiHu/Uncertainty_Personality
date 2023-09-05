import pandas as pd
import scipy.stats as stats

# Read the data
CA = pd.read_csv('./Data/CA_filtered.csv')
BD = pd.read_csv('./Data/BD_filtered.csv')

# preliminary analysis
# test if A picks are significantly higher than random chance
mean_perc = CA['Picking A'].mean()
mean_perc_BD = BD['Picking B'].mean()

random_chance = 0.5

# separate by condition
CA_A1 = CA[CA['Condition'] == 'S2A1']
CA_A2 = CA[CA['Condition'] == 'S2A2']

BD_A1 = BD[BD['Condition'] == 'S2A1']
BD_A2 = BD[BD['Condition'] == 'S2A2']

# one-sample t-test
t, p = stats.ttest_1samp(CA['Picking A'], random_chance)
print(f'The t-value is {t} and the p-value is {p}')

t_BD, p_BD = stats.ttest_1samp(BD['Picking B'], random_chance)
print(f'The t-value is {t_BD} and the p-value is {p_BD}')

t_A1, p_A1 = stats.ttest_1samp(CA_A1['Picking A'], random_chance)
print(f'The t-value is {t_A1} and the p-value is {p_A1}')

t_A1_BD, p_A1_BD = stats.ttest_1samp(BD_A1['Picking B'], random_chance)
print(f'The t-value is {t_A1_BD} and the p-value is {p_A1_BD}')

t_A2, p_A2 = stats.ttest_1samp(CA_A2['Picking A'], random_chance)
print(f'The t-value is {t_A2} and the p-value is {p_A2}')

t_A2_BD, p_A2_BD = stats.ttest_1samp(BD_A2['Picking B'], random_chance)
print(f'The t-value is {t_A2_BD} and the p-value is {p_A2_BD}')
mean_perc_BD_A1 = BD_A1['Picking B'].mean()
mean_perc_BD_A2 = BD_A2['Picking B'].mean()
print(f'The mean percentage of picking B in S2A1 is {mean_perc_BD_A1}')
print(f'The mean percentage of picking B in S2A2 is {mean_perc_BD_A2}')

t_BD_inter, p_BD_inter = stats.ttest_ind(BD_A1['Picking B'], BD_A2['Picking B'])
print(f'The t-value is {t_BD_inter} and the p-value is {p_BD_inter}')


# demographic differences
def demographic_diff_detector(df, demographic_variable, variable_of_interest):
    if demographic_variable == 'Sex':
        t_g, p_g = stats.ttest_ind(df.groupby('Sex')[variable_of_interest].get_group('Male'),
                                   df.groupby('Sex')[variable_of_interest].get_group('Female'))
        print(f'[{demographic_variable}]: the t-value  for {variable_of_interest} is {t_g} and the p-value is {p_g}')
    else:
        grouped_df = df.groupby(demographic_variable)
        groups = []
        for group_name, group_data in grouped_df:
            groups.append(group_data[variable_of_interest])
        f_d, p_d = stats.f_oneway(*groups)
        print(f'[{demographic_variable}]: the f-value  for {variable_of_interest} is {f_d} and the p-value is {p_d}')


df_names = [CA, CA_A1, CA_A2]
demographic_variables = ['Sex', 'Race', 'Ethnicity']
variables_of_interest = ['Picking A', 'RT_mean']

for df_name in df_names:
    for demographic_variable in demographic_variables:
        for variable_of_interest in variables_of_interest:
            demographic_diff_detector(df_name, demographic_variable, variable_of_interest)

for demographic_variable in demographic_variables:
    demographic_diff_detector(BD_A2, demographic_variable, 'Picking B')


# A1_male = CA_A1.groupby('Sex')['Picking A'].get_group('Male').mean()
# A1_female = CA_A1.groupby('Sex')['Picking A'].get_group('Female').mean()
# print(f'[Sex]: the mean percentage of picking A for male is {A1_male}; for femail is {A1_female}')

