import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt
from data_cleaning import BD_unfiltered_A1

# Read the data
CA = pd.read_csv('./Data/CA_filtered.csv')
BD = pd.read_csv('./Data/BD_filtered.csv')

# change the condition names
condition_labels = {'S2A1': 'Uncertainty-Frequency', 'S2A2': 'Uncertainty Only'}
CA.loc[:, 'Condition'] = CA['Condition'].replace(condition_labels)
BD.loc[:, 'Condition'] = BD['Condition'].replace(condition_labels)

# conduct the interaction analysis
DV = CA['Picking A']
IV = CA['Bis11Score']
moderator = CA['Condition']

model = ols('DV ~ IV * moderator', data=CA).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# visualize the interaction
interaction_plot = sns.lmplot(x='Bis11Score', y='Picking A', hue='Condition', data=CA,
                              legend=False,scatter_kws={'s': 10})
interaction_plot.set(xlabel='BIS-11 Score', ylabel='Percentage of Picking A')
interaction_plot.ax.legend(title='Condition', loc='lower right')
# plt.setp(interaction_plot.ax.get_legend().get_texts(), fontsize=8)
plt.show()

# random testing
DV_RT = BD_unfiltered_A1['ReactTime']
IV_RT = BD_unfiltered_A1['Bis11Score']
moderator_RT = BD_unfiltered_A1['KeyResponse']

model_RT = ols('DV_RT ~ IV_RT * moderator_RT', data=BD_unfiltered_A1).fit()
anova_table_RT = sm.stats.anova_lm(model_RT, typ=2)
print(anova_table_RT)

# visualize the interaction
interaction_plot_RT = sns.lmplot(x='ReactTime', y='Bis11Score', hue='KeyResponse', data=BD_unfiltered_A1,
                                    legend=False,scatter_kws={'s': 10})
interaction_plot_RT.set(xlabel='Reaction Time (ms)', ylabel='BIS-11 Score')
interaction_plot_RT.ax.legend(title='Response', loc='lower right')
# plt.setp(interaction_plot.ax.get_legend().get_texts(), fontsize=8)
plt.show()