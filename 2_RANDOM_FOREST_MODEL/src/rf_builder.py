# =============================================================================
# Code forecast consumption and export 
# data as CSV file. 
# Forecast model based on Random Forest regression from Sckit Learn package.
# Model forecast every hour separately and then combine to normal time series data.
# =============================================================================

import os
from datetime import datetime
from tracemalloc import start
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px # to plot the time series plot
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import tensorflow as tf 
import seaborn as sns
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

# =============================================================================
# Load data
# =============================================================================
#get current working directory
cwd = os.getcwd()

df = pd.read_csv(r'xxx.csv', delimiter=',')
df['Date'] = pd.to_datetime(df['Date'])
df = df.rename(columns={'Consumption_auto': 'b2b'}) #file stores total and b2b consumption
df = df.set_index(['Date'])

#get month
today = datetime.today() 
startm = datetime(today.year, today.month, 1)  #Get start month for current clients
startm = datetime.strptime(str(startm),'%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')

#filter by date
df = df[(df.index <= startm)]

# =============================================================================
# 2. MODEL BUILDER - forecast with MeteoBlue
# =============================================================================
# 1. Features engineering
# Lowercase the column names
df.columns = [x.lower() for x in df.columns]

# Find outliers
sorted_consumption= df.sort_values('b2b',ascending=False)

outliers = sorted_consumption.b2b[1]

#expand consumption_auto data average
df['auto_mean'] = df.b2b.expanding().mean()
df['auto_max'] = df.b2b.expanding().max()


print("The number of the 0,1% top values of appliances' load is",
      len(sorted_consumption.head(len(sorted_consumption)//1000)),"and they have power load higher than",
      outliers, "MWh.")

# Outliers removal
df = df.dropna()
df = df.drop(df[(df.b2b>outliers)|(df.b2b<75000)].index)

#Plot target chart
sns.lineplot(data=df, x=df.index, y="b2b")

#Histogram of Appliance's consumption
#log appliances
df['log_b2b'] = np.log1p(df.b2b)
# df['log_b2b'] = df.b2b
f, axes = plt.subplots(1, 2,figsize=(10,4))

sns.distplot(df.b2b, hist=True, color = 'blue',hist_kws={'edgecolor':'black'},ax=axes[0])
axes[0].set_title("B2B consumption")
axes[0].set_xlabel('consumption')

sns.distplot(df.log_b2b, hist=True, color = 'blue',hist_kws={'edgecolor':'black'},ax=axes[1])
axes[1].set_title("Log consumption")
axes[1].set_xlabel('consumption log()')


# Pearson Correlation among the variables
col = ['log_b2b', 'auto_mean', 'auto_max','temperature 2 m', 'weekday', 'year', 'month', 'day', 'hour', 'seasons', 
       'precipitation', 'snowfall', 'relative humidity [2 m]', 'wind speed [10 m]', 'wind direction [10 m]',
       'wind speed [80 m]', 'wind direction [80 m]', 'wind gust', 'sea level preasure', 'dayyear', 'objects_sum']
corr = df[col].corr()
plt.figure(figsize = (18,18))
sns.set(font_scale=1)
sns.heatmap(corr, cbar = True, annot=True, square = True, fmt = '.2f', xticklabels=col, yticklabels=col)
plt.show()


#Check non linear relationship
#linear dependence among some basic features of our data set. 
#In a linear regression problem only linear independent variables can be be used 
#as features to explain energy consumption in other way we will have multicolinearity issues.
col = ['log_b2b', 'auto_mean', 'auto_max', 'temperature 2 m', 'weekday', 'year', 'hour', 'relative humidity [2 m]', 'wind direction [10 m]', 'objects_sum']
sns.set(style="ticks", color_codes=True)
sns.pairplot(df[col])
plt.show()


#Build 3 models
# Linear model
model1 = ['temperature 2 m', 'weekday', 'dayyear', 'hour', 'auto_max', 'auto_mean', 'objects_sum']
#SVR model
model2=['temperature 2 m', 'weekday', 'dayyear', 'hour', 'auto_max', 'auto_mean', 'objects_sum']
# RF model
model3 = ['temperature 2 m', 'weekday', 'dayyear', 'hour', 'auto_max', 'auto_mean', 'objects_sum'] 

# to avoid warnings from standardscaler
df.b2b = df.b2b.astype(float)
df.log_b2b = df.log_b2b.astype(float)
df['temperature 2 m'] = df['temperature 2 m'].astype(float)
df.snowfall = df.snowfall.astype(float)
df['relative humidity [2 m]'] = df['relative humidity [2 m]'].astype(float)
df.month = df.month.astype(float)
df.dayyear = df.dayyear.astype(float)


# Creation of train/test sets for each model
test_size=.2
test_index = int(len(df.dropna())*(1-test_size))

# Linear model
X1_train, X1_test = df[model1].iloc[:test_index], df[model1].iloc[test_index:]
y1_train = df.log_b2b.iloc[:test_index]
#SVR model
X2_train, X2_test = df[model2].iloc[:test_index], df[model2].iloc[test_index:]
y2_train = df.log_b2b.iloc[:test_index]
# RF model
X3_train, X3_test = df[model3].iloc[:test_index], df[model3].iloc[test_index:]
y3_train = df.log_b2b.iloc[:test_index]

y_test =  df.log_b2b.iloc[test_index:]


#2. Train all models with default hyperparameters
#linear model
lin_model = linear_model.LinearRegression()
lin_model.fit(X1_train,y1_train) 

#SVM-based regression model
svr_model = svm.SVR(gamma='scale')
svr_model.fit(X2_train,y2_train)

#Random forest model
rf_model = RandomForestRegressor(n_estimators=100,random_state=1)            
rf_model.fit(X3_train, y3_train)

# 3. Model Evalution & Selection
# Function to evaluate the models

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    r_score = 100*r2_score(test_labels,predictions)
    accuracy = 100 - mape
    print(model,'\n')
    print('Average Error       : {:0.4f} degrees'.format(np.mean(errors)))
    print('Variance score R^2  : {:0.2f}%' .format(r_score))
    print('Accuracy            : {:0.2f}%\n'.format(accuracy))
    
evaluate(lin_model, X1_test, y_test)
evaluate(svr_model, X2_test, y_test)
evaluate(rf_model, X3_test, y_test)


#instead of KFold use TimeSeriesSplit (10 splits) due to time series data
cv = TimeSeriesSplit(n_splits = 10)

#Models' accuracy (MAPE) derives from the type: 100 + negative mean absolute error with perfect score 100.
print('Linear Model:')
scores = cross_val_score(lin_model, X1_train, y1_train, cv=cv,scoring='neg_mean_absolute_error')
print("Accuracy: %0.2f (+/- %0.2f) degrees" % (100+scores.mean(), scores.std() * 2))
scores = cross_val_score(lin_model, X1_train, y1_train, cv=cv,scoring='r2')
print("R^2: %0.2f (+/- %0.2f) degrees" % (scores.mean(), scores.std() * 2))

print('SVR Model:')
scores = cross_val_score(svr_model, X2_train, y2_train, cv=cv,scoring='neg_mean_absolute_error')
print("Accuracy: %0.2f (+/- %0.2f) degrees" % (100+scores.mean(), scores.std() * 2))
scores = cross_val_score(svr_model, X2_train, y2_train, cv=cv)
print("R^2: %0.2f (+/- %0.2f) degrees" % (scores.mean(), scores.std() * 2))

print('Random Forest Model:')
scores = cross_val_score(rf_model, X3_train, y3_train, cv=cv,scoring='neg_mean_absolute_error')
print("Accuracy: %0.2f (+/- %0.2f) degrees" % (100+scores.mean(), scores.std() * 2))
scores = cross_val_score(rf_model, X3_train, y3_train, cv=cv)
print("R^2: %0.2f (+/- %0.2f) degrees" % (scores.mean(), scores.std() * 2))


#Additional models visualization with residuals
y1_pred = lin_model.predict(X1_test)
y2_pred = svr_model.predict(X2_test)
y3_pred = rf_model.predict(X3_test)


fig, axs = plt.subplots(1, 3, figsize=(10,4), sharey=True)
axs[0].scatter(y1_pred,y_test-y1_pred)
axs[0].set_title('Linear Regression')
axs[1].scatter(y2_pred,y_test-y2_pred)
axs[1].set_title('SVR')
axs[2].scatter(y3_pred,y_test-y3_pred)
axs[2].set_title('Random Forest')
fig.text(0.06, 0.5, 'Residuals', ha='center', va='center', rotation='vertical')
fig.text(0.5, 0.01,'Fitted Values', ha='center', va='center')


#Plot predictions and true values
fig, axs = plt.subplots(1, 3, figsize=(12,4), sharey=True)
axs[0].scatter(y_test,y1_pred)
axs[0].set_title('Linear Regression')
axs[1].scatter(y_test,y2_pred)
axs[1].set_title('SVR')
axs[2].scatter(y_test, y3_pred)
axs[2].set_title('Random Forest')
fig.text(0.06, 0.5, 'Predictions', ha='center', va='center', rotation='vertical')
fig.text(0.5, 0.01,'True Values', ha='center', va='center')


#Each model forecast plot
fig = plt.figure(figsize=(20,8))
plt.plot(y_test.values,label='Target value',color='b')
plt.plot(y1_pred,label='Linear Prediction ', linestyle='--', color='y')
plt.plot(y2_pred,label='SVR Prediction ', linestyle='--', color='g')
plt.plot(y3_pred,label='Tree Prediction ', linestyle='--', color='r')
plt.legend(loc=0)

#What is the differece?
diff = ((y_test - y3_pred) / y3_pred).mean()
print('Average difference between true / predicted value is:', diff)

# # Feature importances of RF model (turn on if needed)
# importances = rf_model.feature_importances_

# std = np.std([tree.feature_importances_ for tree in rf_model.estimators_],
#             axis=0)
# indices = np.argsort(importances)[::-1]

# for f in range(df[model3].shape[1]):
#     print("%d. feature %d %s (%f)" % (f + 1, indices[f], model3[indices[f]], importances[indices[f]]))

# # Plot the feature importances of the RF model
# plt.figure(figsize=(12, 10))
# plt.title("Feature importances")
# plt.bar(range(df[model3].shape[1]), importances[indices],
#     color="r", yerr=std[indices], align="center")
# plt.xticks(range(df[model3].shape[1]), indices)
# plt.xlim([-1, df[model3].shape[1]])

pipe = Pipeline([
('scaler', StandardScaler()),
('regressor', RandomForestRegressor())
])

pipe.fit(X3_train, y3_train)


parameters = {'scaler': [StandardScaler()],
    'regressor__max_depth': [100],
    'regressor__min_samples_leaf': [3],
    'regressor__min_samples_split': [2],
    'regressor__n_estimators': [300],
    'regressor__random_state':[1]
}

cv = cv

grid_model = GridSearchCV(pipe, parameters, cv=cv)

grid_model = grid_model.fit(X3_train, y3_train)
print(grid_model.best_estimator_)
print('Best parameters: ',grid_model.best_params_)


#Evaluate best model
best_model = grid_model.best_estimator_
grid_accuracy = evaluate(grid_model, X3_test, y_test)


#predict with best model * diff
y_best_pred = best_model.predict(X3_test)

# Calculate Confidence interval 95% for the predictions
sum_errs = np.sum((y_test - y_best_pred)**2)
stdev = np.sqrt(1/(len(df)-2) * sum_errs)

interval = 1.96 * stdev #95% CI

lower, upper = y_best_pred - interval, y_best_pred + interval

# Plot final predictions on test set based on best model
fig = plt.figure(figsize=(20,8))
plt.plot(y_test.values,label='Target value',color='b')
plt.plot(lower,label='Lower Limit ', linestyle='--', color='r')
plt.plot(upper,label='Upper Limit ', linestyle='--', color='y')
plt.title('Lower Limit and Upper Limit of best model')
plt.legend(loc=1)

#Output model
joblib.dump(best_model, f"{cwd}\models\_best_auto_model.joblib", compress=0)
print('Auto clients model build')
