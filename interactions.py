import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from factor_analyzer import FactorAnalyzer
from semopy import Model, semplot, report

from data_cleaning import BD_unfiltered_A1

# Read the data
CA = pd.read_csv('./Data/CA_filtered.csv')
BD = pd.read_csv('./Data/BD_filtered.csv')

# change the condition names
condition_labels = {'S2A1': 'Uncertainty-Frequency', 'S2A2': 'Uncertainty Only'}
CA.loc[:, 'Condition'] = CA['Condition'].replace(condition_labels)
BD.loc[:, 'Condition'] = BD['Condition'].replace(condition_labels)

# conduct the interaction analysis (BD showed the same pattern although not significant)
DV = CA['Picking A']
IV = CA['Bis11Score']
moderator = CA['Condition']

model = ols('DV ~ IV * moderator', data=CA).fit()
print(model.summary())
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# visualize the interaction
interaction_plot = sns.lmplot(x='Bis11Score', y='Picking A', hue='Condition', data=CA,
                              legend=False,scatter_kws={'s': 10})
interaction_plot.set(xlabel='BIS-11 Score', ylabel='Percentage of Picking A')
interaction_plot.ax.legend(title='Condition', loc='lower right')
# plt.setp(interaction_plot.ax.get_legend().get_texts(), fontsize=8)
plt.show()

# run another linear regression for RT
CA_S2A1 = CA[CA['Condition'] == 'Uncertainty-Frequency']
# standardize the data
CA_S2A1.loc[:, 'RT_mean'] = stats.zscore(CA_S2A1['RT_mean'])
CA_S2A1.loc[:, 'STAIS'] = stats.zscore(CA_S2A1['STAIS'])
CA_S2A1.loc[:, 'STAIT'] = stats.zscore(CA_S2A1['STAIT'])
CA_S2A1.loc[:, 'CESD'] = stats.zscore(CA_S2A1['CESD'])
CA_S2A1.loc[:, 'Big5N'] = stats.zscore(CA_S2A1['Big5N'])
DV_RT = CA_S2A1['RT_mean']
X = sm.add_constant(CA_S2A1[['STAIS', 'STAIT', 'CESD', 'Big5N']])

# # test for multicollinearity (not significant)
# vif_data = pd.DataFrame()
# vif_data["Variable"] = X.columns
# vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# print(vif_data)

# perform factor analysis
FA_data = CA_S2A1[['STAIS', 'STAIT', 'CESD', 'Big5N']]
fa = FactorAnalyzer(n_factors=1, rotation="varimax")
fa.fit(FA_data)

# Get loadings
loadings = fa.loadings_

# Get factor variance and select the number of factors
ev, v, _ = fa.get_factor_variance()

# Print factor loadings and variance
print("Factor Loadings:\n", loadings)
print("\nFactor Variance:\n", ev, v)

# Check Eigenvalues
eigen_values, vectors = fa.get_eigenvalues()

# Create scree plot
plt.scatter(range(1, FA_data.shape[1]+1), eigen_values)
plt.plot(range(1, FA_data.shape[1]+1), eigen_values)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.axhline(y=1, color='r', linestyle='--')  # Eigenvalue of 1 is often considered as a cutoff
plt.grid(True)
plt.show()

# transform the data
FA_data_transformed = fa.transform(FA_data)*-1

model_RT = ols('DV_RT ~ FA_data_transformed', data=CA_S2A1).fit()
print(model_RT.summary())

# visualize the model
sns.regplot(x=FA_data_transformed, y=DV_RT, data=CA_S2A1)
plt.xlabel('Factor Score')
plt.ylabel('Reaction Time (ms)')
plt.show()


# # random testing (not meaningful)
# DV_RT = BD_unfiltered_A1['ReactTime']
# IV_RT = BD_unfiltered_A1['Bis11Score']
# moderator_RT = BD_unfiltered_A1['KeyResponse']
#
# model_RT = ols('DV_RT ~ IV_RT * moderator_RT', data=BD_unfiltered_A1).fit()
# anova_table_RT = sm.stats.anova_lm(model_RT, typ=2)
# print(anova_table_RT)
#
# # visualize the interaction
# interaction_plot_RT = sns.lmplot(x='Bis11Score', y='ReactTime', hue='KeyResponse', data=BD_unfiltered_A1,
#                                     legend=False,scatter_kws={'s': 10})
# interaction_plot_RT.set(xlabel='Reaction Time (ms)', ylabel='BIS-11 Score')
# interaction_plot_RT.ax.legend(title='Response', loc='lower right')
# plt.show()