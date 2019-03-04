### This is created as a separate file from one of the data challenges to the purposes of code review. 

# import and load the libraries

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing, metrics

# load the data csv file

data = pd.read_csv('vgsales.csv')
#data.head()     #shows the top 5 lines of the dataframe

print('Number of games in the dataset is {}\nNumber of columns is {}'.format(data.shape[0], data.shape[1]))
if data.isnull().values.any():
    print('There are null values in the data.')
    
    
for column in data.columns:
    if data[column].isnull().any():
        print('Column "{}" has null values'.format(column))
        
# Grab lists of unique values for publishers, genres, and platforms
publishers = data['Publisher'].unique()
genres = data['Genre'].unique()
platforms = data['Platform'].unique()

regions = ['NA_Sales', 'EU_Sales', 'JP_Sales']

max_sales_in_market = []
for region in regions:
    max_sales_in_market.append(data[region].max())

max_sales = max(max_sales_in_market)
print('The max sales for a single video game in a single region is {}'.format(max(max_sales_in_market)))


sales_data = {}
for region in regions:
    sales_list = []
    for sales_bin in range(1, int(max_sales)):
        sales_list.append(data[data[region] > sales_bin].shape[0])
    sales_data[region] = sales_list



## train and test the data for linear regression

#drop the null values in the 4 columns - year, publisher, platform, genres
vg_data = data.dropna(subset=['Year', 'Publisher', 'Platform', 'Genre'], axis=0)

# convert categorical variable into dummy/indicator variables

platforms_encoded = pd.get_dummies(vg_data['Platform'].values)
platforms_encoded = platforms_encoded.reset_index().drop(labels='index', axis=1)

genres_encoded = pd.get_dummies(vg_data['Genre'].values) 
genres_encoded = genres_encoded.reset_index().drop(labels='index', axis=1)

publishers_encoded = pd.get_dummies(vg_data['Publisher'].values)
publishers_encoded = publishers_encoded.reset_index().drop(labels='index', axis=1)

year = vg_data['Year']/2020
year = year.reset_index().drop(labels='index', axis=1)

model_data = pd.concat([platforms_encoded, genres_encoded, publishers_encoded, year], axis=1)
target = vg_data['Global_Sales']

X_train, X_test, y_train, y_test = train_test_split(model_data, target, test_size=0.30, random_state=0)
#X_train.shape


# train test a linear regression model

clf = LinearRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
#y_pred

print(clf.score(X_test, y_test))   # prints the accuracy score for the linear regression model

# train a random forest regressor next

clf = RandomForestRegressor()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))


n_estimators = [10, 25, 50, 100] # number of trees
min_samples_split = [2, 10, 25, 50] # minimum number of samples needed to split a node

hp_list = []
for estimator in n_estimators:
    for sample in min_samples_split:
        clf = RandomForestRegressor(n_estimators=estimator, min_samples_split=sample)
        clf.fit(X_train, y_train)
        hp_list.append([estimator, sample, clf.score(X_test, y_test)])
        print('done')


print(sorted(hp_list, key=lambda x: x[2], reverse=True))   # largest accuracies shown first


### The best R^2 score is achieved for n_estimators = 10, min_samples_split = 25, with R^2=0.17.
### This is much better than linear regression, suggesting that the underlying function may not be linear.
### Refit the best model and see what the top predictors are

clf = RandomForestRegressor(n_estimators=10, min_samples_split=25)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

## get the top predictors
top_predictors = (-clf.feature_importances_).argsort()[:20]
X_train.iloc[:, top_predictors].columns


## prints the top predictors

for name in X_train.iloc[:, top_predictors].columns:
    if name in publishers:
        print('publisher')
    elif name in genres:
        print('genres')
    elif name in platforms:
        print('platform')
    else:
        print('year')
        
        
## plot the important features

plt.figure(figsize=(8,8))
plt.bar(x=range(20), height=clf.feature_importances_[top_predictors],\
        tick_label=X_train.iloc[:, top_predictors].columns)
plt.xticks(rotation=90)
plt.title('Feature Importance (Random Forest)');
plt.savefig('rf_featureimportance.png')
