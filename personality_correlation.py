import pandas as pd
import scipy.stats as stats

# Read the data
CA = pd.read_csv('./Data/CA_filtered.csv')

# separate by condition
CA_A1 = CA[CA['Condition'] == 'S2A1']
CA_A2 = CA[CA['Condition'] == 'S2A2']


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
    return results_df


# get the results
all_A_picks = correlation_test(CA, 'Picking A')
A1_A_picks = correlation_test(CA_A1, 'Picking A')
A2_A_picks = correlation_test(CA_A2, 'Picking A')

all_RT = correlation_test(CA, 'RT_mean')
A1_RT = correlation_test(CA_A1, 'RT_mean')
A2_RT = correlation_test(CA_A2, 'RT_mean')

# # correlate RT with A picks (not significant)
# r, p = stats.pearsonr(CA['RT_mean'], CA['Picking A'])
# r_A1, p_A1 = stats.pearsonr(CA_A1['RT_mean'], CA_A1['Picking A'])
# r_A2, p_A2 = stats.pearsonr(CA_A2['RT_mean'], CA_A2['Picking A'])

# # test correlations between personality scales (disinhibition scales are correlated)
# for i in range(4, 20):
#     for j in range(i + 1, 20):
#         r, p = stats.pearsonr(CA[CA.columns[i]], CA[CA.columns[j]])
#         print(f'The correlation between {CA.columns[i]} and {CA.columns[j]} is {r} with p-value {p}')



