import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVR

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# Reshaping train and test data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data)
train = train.drop((missing_data[missing_data['Total'] > 81]).index, 1)
total_test = test.isnull().sum().sort_values(ascending=False)
percent_test = (test.isnull().sum() / test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total_test, percent_test], axis=1, keys=['Total', 'Percent'])
test = test.drop((missing_data[missing_data['Total'] > 78]).index, 1)

# Reshaping categorical data
categorical_feature_mask = train.dtypes == object
categorical_cols = train.columns[categorical_feature_mask].tolist()
categorical_feature_mask_test = test.dtypes == object
categorical_cols_test = test.columns[categorical_feature_mask_test].tolist()
le = LabelEncoder()
train[categorical_cols] = train[categorical_cols].apply(lambda col: le.fit_transform(col.astype(str)))
test[categorical_cols_test] = test[categorical_cols_test].apply(lambda col: le.fit_transform(col.astype(str)))

k = 15
corrmat = train.corr()
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
train = train[cols]
test = test[cols.drop('SalePrice')]
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean())
train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())
train['GarageCars'] = train['GarageCars'].fillna(train['GarageCars'].mean())
test['GarageYrBlt'] = test['GarageYrBlt'].fillna(test['GarageYrBlt'].mean())
test['MasVnrArea'] = test['MasVnrArea'].fillna(test['MasVnrArea'].mean())
test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean())
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean())
test['GarageArea'] = test['GarageArea'].fillna(test['GarageArea'].mean())
test['GarageCars'] = test['GarageCars'].fillna(test['GarageCars'].mean())
print(train.shape)

print(test.shape)
x_train, x_test, y_train, y_test = train_test_split(train, train,
                                                    test_size=0.2, random_state=1)
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

sc_x = StandardScaler()
sc_y = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)
y_train = sc_x.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test)

# linear = LinearRegression()
# linear.fit(x_train, y_train)
# predictions = linear.predict(x_test)
# predictions = predictions.reshape(-1, 1)

# svr = SVR(kernel='rbf')
# svr.fit(x_train, np.ravel(y_train))
# svr_pred = svr.predict(x_test)
# svr_pred = svr.pred.reshape(-1, 1)
# print(svr_pred)
