import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt

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
moderator = BD['Condition']

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
