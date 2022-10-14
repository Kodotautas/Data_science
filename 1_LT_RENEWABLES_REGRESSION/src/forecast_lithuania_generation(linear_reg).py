import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

project_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)


# ------------------------------- PREPARE DATA ------------------------------- #
#LT generation from Litgrid
print('Loading data...')
generation_df = pd.read_csv(f'{project_dir}\data\generation_LT.csv')
#LT consumption from Litgrid
load_df = pd.read_csv(f'{project_dir}\data\load_LT.csv')
load_df['Actual Load'] = load_df['Actual Load'].apply(pd.to_numeric, errors='coerce')

#filter not necessary row
generation_df = generation_df[generation_df['Biomass'] != 'Actual Aggregated']

#convert to numeric type and sum total generation
cols = generation_df.columns.drop('Unnamed: 0')
generation_df[cols] = generation_df[cols].apply(pd.to_numeric, errors='coerce')
generation_df['total_generation'] = generation_df.iloc[:].sum(axis=1)
generation_df['rew_generation'] = generation_df['Solar'] + generation_df['Wind Onshore']

#rename datetime colums
df_list = [generation_df, load_df]

for i in df_list:
     i.rename(columns={'Unnamed: 0': 'datetime'}, inplace=True)
     i['year_month'] = pd.to_datetime(i['datetime']).apply(lambda x: x.strftime('%Y-%m')) 

#merge generation & consumption dfs
df = pd.merge(load_df, generation_df, how="left", on=['datetime']).fillna(0)
df.rename(columns={'year_month_x': 'year_month'}, inplace=True)

#group by mean 
df = df.groupby(by=['year_month']).sum()

#calculate share of generation
df['generation_share'] = df['total_generation']/df['Actual Load']*100
df['rew_gen_share'] = df['rew_generation']/df['Actual Load']*100

# Converting the index as date
df.index = pd.to_datetime(df.index)
df['Month'] = np.arange(len(df.index))

#yearly averages
df['gen_share_avg'] = df['rew_gen_share'].rolling(12).mean()
df['generation_share_avg'] = df['generation_share'].rolling(12).mean()

#chosse columns
df = df[['Month', 'rew_gen_share', 'gen_share_avg', 'generation_share', 'generation_share_avg']]
print('Data loaded & transformed!')


# ------------------------------- PLOT DATA ------------------------------- #
plt.style.use('grayscale')

#multiple line plot
plt.figure(figsize=(20,12))
plt.plot(df.index, df['generation_share'], label='Total generation share', linestyle='--')
plt.plot(df.index, df['generation_share_avg'], label='Total generation share 1y average', linestyle='--')
plt.plot(df.index, df['rew_gen_share'], label='Renewables generation share')
plt.plot(df.index, df['gen_share_avg'], label='Renewables generation share 1y average')
plt.xlabel('Year')
plt.ylabel('[%]')
plt.legend(loc='upper left')
plt.title('Total generation & renewables shares of Lithuania load [%]', fontsize=20)
plt.show()


# -------------------------------- DATA TENSORS -------------------------------- #
#convert dataframe to tensor
X_data = Variable(torch.from_numpy(df['Month'].values.astype(np.float32))).unsqueeze(1)
y_data = Variable(torch.from_numpy(df['rew_gen_share'].values.astype(np.float32))).unsqueeze(1)


# ------------------------------ MODEL BUILDING ------------------------------ #
#define model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

model = LinearRegression()

#criterion and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

#parameters
epochs = 30000

for epoch in range(epochs):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(X_data)
    #add to dataframe
    df['pred_rew_gen_share'] = y_pred.data.numpy()

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch}, Loss: {loss.item()}')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()

    # Update the weights
    optimizer.step()

#plot prediction and actual lineplots
fig, ax = plt.subplots(figsize=(20,12))
ax.plot(df.index, df['rew_gen_share'], 'o', label='Actual', linestyle='-')
ax.plot(df.index, model(X_data).detach().numpy(), 'r', label='Predicted')
ax.legend()
ax.set(xlabel='Month', ylabel='[%]', title='Share of renewables generation in Lithuania [%]')
plt.show()


# ----------------------------- MODEL TEST ----------------------------- #
#generate from 91 to 912 months float type
start = 91
end = 912
forecast_months = np.arange(start, end, 1, dtype=np.float32)

#model prediction with forecast months
forecast_months = Variable(torch.from_numpy(forecast_months)).unsqueeze(1)
y_forecast = model(forecast_months)
y_forecast = y_forecast.detach().numpy()

#function to find y_forecast value >= 100
def find_100(y_forecast):
    for i in range(len(y_forecast)):
        if y_forecast[i] >= 100:
            return i

target_index = find_100(y_forecast)

#funtion add months to year-month
def add_months(year_month, months):
    return pd.to_datetime(year_month) + pd.DateOffset(months=months)

find_100_date = add_months('2015-01', target_index + len(df.index))

#generate date from 2015-01 to x-months
date_list = []
for i in range(end + 1):
    date_list.append(pd.to_datetime(f'2015-01') + pd.DateOffset(months=i))

# print(date_list[start-1:])
# print('----------------------------')
# print(date_list[:start-1])

#plot forecast month and y_forecast
fig, ax = plt.subplots(figsize=(20,12))
ax.plot(date_list[start+1:], y_forecast, 'r', label='Predicted')
ax.plot(date_list[:start-1], model(X_data).detach().numpy(), 'r')
ax.plot(date_list[:start-1], df['rew_gen_share'], 'o', label='Actual', linestyle='-')
plt.plot(find_100_date, 100, ls="", marker="o", label="Point of 100%", markersize=15)
ax.legend()
ax.set(xlabel='Month', ylabel='[%]', title='Share of renewables generation & forecast in Lithuania [%]')
plt.show()