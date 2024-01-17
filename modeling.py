import numpy as np
import pandas as pd
from utility import (ComputationalModels, dict_generator, likelihood_ratio_test, bayes_factor,
                     best_param_generator)
from scipy.stats import pearsonr

# Read the data
data = pd.read_csv('./Data/full_data.csv')
CA = pd.read_csv('./Data/CA_filtered_12.csv')
decayfre = pd.read_csv('./Data/decay_fre.csv')
decay = pd.read_csv('./Data/decay.csv')
delta = pd.read_csv('./Data/delta.csv')
data['KeyResponse'] = data['KeyResponse'] - 1

CA = CA[CA['Condition'] != 'S2A3'].reset_index(drop=True)

# convert the data into a dictionary
data_dict = dict_generator(data)

# set up the reward structure
reward_means = [.70, .30, .70, .30]
reward_sd = [.43, .43, .12, .12]

# # set up the computational models
# model_decayfre = ComputationalModels(reward_means, reward_sd,
#                                      model_type='decay_fre', condition='Gains', num_trials=250)
# model_decay = ComputationalModels(reward_means, reward_sd,
#                                     model_type='decay', condition='Gains', num_trials=250)
# model_delta = ComputationalModels(reward_means, reward_sd,
#                                     model_type='delta', condition='Gains', num_trials=250)
#
# # fit all the models
# for model in [model_decayfre, model_decay, model_delta]:
#     results = model.fit(data_dict, num_iterations=100)
#     results = pd.DataFrame(results)
#     results.iloc[:, 3] = results.iloc[:, 3].astype(str)
#     results.to_csv('./Data/' + model.model_type + '.csv', index=False)
#     print(results['AIC'].mean())
#     print(results['BIC'].mean())

for file in [decayfre, decay, delta]:
    print(f'The mean AIC is {file["AIC"].mean()}')
    print(f'The mean BIC is {file["BIC"].mean()}')

# get the model comparison
print(likelihood_ratio_test(decay, decayfre, 1))
print(bayes_factor(decay, decayfre))

# get the best beta
beta = best_param_generator(decayfre, 'b')

# get the correlation between the best beta and the PropOptimal
print(pearsonr(CA['Picking C'], beta))

# now, separate by condition
CA_A1 = CA[CA['Condition'] == 'S2A1']
CA_A2 = CA[CA['Condition'] == 'S2A2']

# get the correlation between the best beta and the PropOptimal
beta_A1 = beta[CA_A1.index]
beta_A2 = beta[CA_A2.index]

print(pearsonr(CA_A1['Picking C'], beta_A1))
print(pearsonr(CA_A2['Picking C'], beta_A2))


