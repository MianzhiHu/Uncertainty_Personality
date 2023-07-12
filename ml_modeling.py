import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Read the data
CA = pd.read_csv('./Data/CA_filtered.csv')

# separate by condition
CA_A1 = CA[CA['Condition'] == 'S2A1']
CA_A2 = CA[CA['Condition'] == 'S2A2']

# # check variables
# print(CA['Ethnicity'].unique())


# transform the categorical variables
def categorical_variable_transformer(df, condition=None):
    # remove the participant ID
    df = df.drop('Subnum', axis=1)
    # transform the sex variable
    df.loc[:, 'Sex'] = df['Sex'].replace({'Male': 1, 'Female': 2, 'Other': 3})
    # transform the race variable
    df.loc[:, 'Race'] = df['Race'].replace({'White': 1, 'Black or African American': 2,
                                            'American Indian or Alaskan Native': 3,
                                            'Asian': 4, 'More than one Race': 5,
                                            'Prefer not to answer': 6})
    # transform the ethnicity variable
    df.loc[:, 'Ethnicity'] = df['Ethnicity'].replace({'Not Hispanic or Latino': 1, 'Hispanic or Latino': 2})
    if condition == 'Yes':
        df.loc[:, 'Condition'] = df['Condition'].replace({'S2A1': 1, 'S2A2': 2})
    if condition == 'No':
        # remove the condition variable
        df = df.drop('Condition', axis=1)
    return df


CA = categorical_variable_transformer(CA, condition='Yes')
CA_A1 = categorical_variable_transformer(CA_A1, condition='No')
CA_A2 = categorical_variable_transformer(CA_A2, condition='No')


# run the linear regression (does not work well no matter what)
def regression_generator(df, regression_type=None):
    global model
    # separate the predictor and the outcome
    X = df.drop(['Picking A', 'RT_mean'], axis=1)
    # # try to use only important variables
    # X = df[['Bis11Score', 'TPM_Meanness']]
    y = df['Picking A']
    # standardize X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # choose the regression type
    if regression_type == 'linear':
        model = LinearRegression()
    elif regression_type == 'ridge':
        model = Ridge()
    elif regression_type == 'lasso':
        model = Lasso()

    # fit the model
    model.fit(X_train, y_train)
    # make predictions
    y_pred = model.predict(X_test)
    # evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'The mean squared error is {mse}; the r2 score is {r2}')

    return model


# # overall
# model_all = regression_generator(CA, regression_type='linear')
# model_all_ridge = regression_generator(CA, regression_type='ridge')
# model_all_lasso = regression_generator(CA, regression_type='lasso')
# # A1
# model_A1 = regression_generator(CA_A1, regression_type='linear')
# model_A1_ridge = regression_generator(CA_A1, regression_type='ridge')
# model_A1_lasso = regression_generator(CA_A1, regression_type='lasso')
# # A2
# model_A2 = regression_generator(CA_A2, regression_type='linear')
# model_A2_ridge = regression_generator(CA_A2, regression_type='ridge')
# model_A2_lasso = regression_generator(CA_A2, regression_type='lasso')


# now try svm (does not work well either)
def svm_generator(df, kernel=None):
    global svm
    # separate the predictor and the outcome
    X = df.drop(['Picking A', 'RT_mean'], axis=1)
    y = df['Picking A']
    # standardize X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

    # choose the kernel
    if kernel == 'linear':
        svm = SVR(kernel='linear')
    elif kernel == 'rbf':
        svm = SVR(kernel='rbf')
    elif kernel == 'poly':
        svm = SVR(kernel='poly')
    elif kernel == 'sigmoid':
        svm = SVR(kernel='sigmoid')

    # fit the model
    svm.fit(X_train, y_train)
    # make predictions
    y_pred = svm.predict(X_test)
    # evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'The mean squared error is {mse}; the r2 score is {r2}')

    return svm


# # overall
# svm_all = svm_generator(CA, kernel='linear')
# svm_all_rbf = svm_generator(CA, kernel='rbf')
# svm_all_poly = svm_generator(CA, kernel='poly')
# svm_all_sigmoid = svm_generator(CA, kernel='sigmoid')
# # A1
# svm_A1 = svm_generator(CA_A1, kernel='linear')
# svm_A1_rbf = svm_generator(CA_A1, kernel='rbf')
# svm_A1_poly = svm_generator(CA_A1, kernel='poly')
# svm_A1_sigmoid = svm_generator(CA_A1, kernel='sigmoid')
# # A2
# svm_A2 = svm_generator(CA_A2, kernel='linear')
# svm_A2_rbf = svm_generator(CA_A2, kernel='rbf')
# svm_A2_poly = svm_generator(CA_A2, kernel='poly')
# svm_A2_sigmoid = svm_generator(CA_A2, kernel='sigmoid')









