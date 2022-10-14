# =============================================================================
# File connect Excel workbook with GAS data, read data and generate forecast with Random Forest Model. 
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
# 1. DATA PROCESSING / UNIQUE FOR EVERY PROJECT
# Read excel workbook for data
# =============================================================================
cwd = os.getcwd()

df = pd.read_csv(f"{cwd}\df_DATA.csv", delimiter=',')

df['Date'] = pd.to_datetime(df['Date'])
df = df.rename(columns={'Consumption_auto': 'b2b'})
df = df.set_index(['Date'])

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
df = df.drop(df[(df.b2b>outliers)|(df.b2b<0)].index)

#get dataframe tail
df = df.tail(480)


# =============================================================================
# 2. MODEL FORECASTER - forecast with MeteoBlue
# =============================================================================
#Load dataframe, choose columns and get x_test
features = ['temperature 2 m', 'weekday', 'dayyear', 'hour', 'auto_max', 'auto_mean', 'objects_sum']

X_test = df[features]

#load best gas model
model = joblib.load(f"{cwd}\models\_best_auto_model.joblib")

pred = model.predict(X_test)

#Convert log 
consumption_pred = np.exp(pred)*1

df['forecast'] = consumption_pred.tolist()

#Add datetime to dataframe
df['update_datetime'] = datetime.now()
df['model_trained'] = '2022-01-25'

# multiple line plots if need to qualitative check
# plt.plot( df.index, 'b2b', data=df, marker='', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
# plt.plot( df.index, 'forecast', data=df, marker='', color='olive', linewidth=2)
# # show legend
# plt.legend()
# # show graph
# plt.show()

df = df['forecast']

df.to_csv(f"{cwd}\df_DATA_AUTO_FORECAST.csv", header=False)
print('LT B2B Automated clients forecast generated.')