import pandas as pd
import scipy.stats as stats

# Read the data
CA = pd.read_csv('./Data/CA_filtered.csv')

# separate by condition
CA_A1 = CA[CA['Condition'] == 'S2A1']
CA_A2 = CA[CA['Condition'] == 'S2A2']