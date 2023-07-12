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


# gender differences
def gender_diff_detector(df, variable_of_interest):
    t_g, p_g = stats.ttest_ind(df.groupby('Sex')[variable_of_interest].get_group('Male'),
                               df.groupby('Sex')[variable_of_interest].get_group('Female'))
    print(f'The t-value  for {variable_of_interest} is {t_g} and the p-value is {p_g}')


# overall gd
gender_diff_detector(CA, 'Picking A')
gender_diff_detector(CA, 'RT_mean')

# gd in A1
gender_diff_detector(CA_A1, 'Picking A')
gender_diff_detector(CA_A1, 'RT_mean')

A1_male = CA_A1.groupby('Sex')['Picking A'].get_group('Male').mean()
A1_female = CA_A1.groupby('Sex')['Picking A'].get_group('Female').mean()
print(f'The mean percentage of picking A for male is {A1_male}; for femail is {A1_female}')

# gd in A2
gender_diff_detector(CA_A2, 'Picking A')
gender_diff_detector(CA_A2, 'RT_mean')
