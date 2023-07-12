import pandas as pd
import scipy.stats as stats

# Read the data
CA = pd.read_csv('./Data/CA_filtered.csv')

# preliminary analysis
# test if A picks are significantly higher than random chance
mean_perc = CA['Picking A'].mean()

random_chance = 0.5

# separate by condition
CA_A1 = CA[CA['Condition'] == 'S2A1']
CA_A2 = CA[CA['Condition'] == 'S2A2']

# one-sample t-test
t, p = stats.ttest_1samp(CA['Picking A'], random_chance)
print(f'The t-value is {t} and the p-value is {p}')

t_A1, p_A1 = stats.ttest_1samp(CA_A1['Picking A'], random_chance)
print(f'The t-value is {t_A1} and the p-value is {p_A1}')

t_A2, p_A2 = stats.ttest_1samp(CA_A2['Picking A'], random_chance)
print(f'The t-value is {t_A2} and the p-value is {p_A2}')


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


# A1_male = CA_A1.groupby('Sex')['Picking A'].get_group('Male').mean()
# A1_female = CA_A1.groupby('Sex')['Picking A'].get_group('Female').mean()
# print(f'[Sex]: the mean percentage of picking A for male is {A1_male}; for femail is {A1_female}')

