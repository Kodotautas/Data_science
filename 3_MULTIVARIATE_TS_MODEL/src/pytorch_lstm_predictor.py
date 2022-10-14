# =============================================================================
# File connect Excel workbook with GAS data, read data and generate forecast with Random Forest Model. 
# =============================================================================
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px # to plot the time series plot
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

#TO DO:
#send email when model is finished
#think how to generate 3 days forecast

# =============================================================================
# 1. DATA PROCESSING AS IN MODEL TRAINING
# Read excel workbook for data
# =============================================================================
cwd = os.getcwd()

df = pd.read_excel(f"{cwd}/data/TotalConsumption.xlsx")
df['datetime'] = pd.to_datetime(df['datetime'], utc=False)

try:
    os.remove('forecast_B2B_GAS_7.csv') ### Remove previous files
except:
    pass

# #Set index and get more interval columns
df = df.set_index('datetime')
df = df.reindex(columns=['B2B']).dropna() #Reoder and drop nan values
df['Year'] = df.index.year
df['Month'] = df.index.month
df['Day'] = df.index.day
df['Hour'] = df.index.hour
df['Weekday'] = df.index.weekday +1

# =============================================================================
# Merge GAS data and MeteoBlue dataframes to get final data set.
# =============================================================================

MeteoBlueFull = pd.read_csv(f"{cwd}/data/MeteoBlueFull.csv")
MeteoBlueFull = MeteoBlueFull.rename(columns={'Unnamed: 0': 'datetime'})
MeteoBlueFull['datetime'] = pd.to_datetime(MeteoBlueFull['datetime'], utc=False)
MeteoBlueFull['datetime'] = MeteoBlueFull['datetime'].dt.date
MeteoBlueFull = MeteoBlueFull.groupby('datetime').mean()

# Merge first DF and DF with temperatures
df_full = pd.merge(MeteoBlueFull,df, how='outer', left_index=True, right_index=True)

#features engineering
df_full['Year'] = df_full.index.year
df_full['Month'] = df_full.index.month
df_full['Day'] = df_full.index.day
df_full['Hour'] = df_full.index.hour
df_full['Weekday'] = df_full.index.weekday +1
df_full['Weeknumber'] = df_full.index.isocalendar().week
df_full['Month_day'] = df_full['Month'].astype(str) + '-' + df_full['Day'].astype(str) #get month and day
df_full['PreviousDayConsumption'] = df_full['B2B'].shift(1)
df_full['PreviousWeekConsumption'] = df_full['B2B'].shift(7)
df_full['DayYear'] = df_full.index.dayofyear
df_full['7daysAverage'] = df_full['B2B'].rolling(window=7).mean().shift(1)
df_full['7daysmedian'] = df_full['B2B'].rolling(window=7).median().shift(1)
#add rolling minimum and maximum columns
df_full['7daysAverage_min'] = df_full['B2B'].rolling(window=7).min().shift(1)
df_full['7daysAverage_max'] = df_full['B2B'].rolling(window=7).max().shift(1)


# =============================================================================
# HOLIDAYS. Set holidays days and if day is holiday, when mark it as Sunday. Turn on/off if need.
holidays = ['1-1', '2-16', '3-11', '5-1', '6-24', '7-6', '8-15', '11-1', '11-2', '12-24', '12-25', '12-26']
df_full.loc[df_full['Month_day'].isin(holidays), 'Weekday'] = 7 #Set holiday (awesome short line)

# There possible to extract sessions from data.
# =============================================================================
# SEASONS. 1-Winter, 2-Spring, 3-Summer, 4-Autumn.Turn on/off if need.
seasons = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1]
month_to_season = dict(zip(range(1,13), seasons))
df_full['Seasons'] = df_full.index.month.map(month_to_season)

df_full = df_full.reindex(columns=['B2B', 'PreviousDayConsumption', 'PreviousWeekConsumption', '7daysmedian', '7daysAverage', '7daysAverage_min', '7daysAverage_max',
                                 'Temperature 2 m','Precipitation','Snowfall', 
                                   'Year', 'Month', 'Weekday', 'DayYear', 'Seasons', 'Weeknumber']) #Reoder columns

df_full = df_full[(df_full.index >= '2016-01-01')]
df_full = df_full.fillna(0)

#Separe forecast dataframe
forecast_df = df_full[df_full['PreviousDayConsumption'] != 0]
#leave only last row of forecast dataframe
forecast_df = forecast_df.tail(35)

df = forecast_df.copy()


# ------------------------------- DEFINE MODEL ------------------------------- #
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_classes = num_classes # output size
        self.num_layers = num_layers # number of recurrent layers in the lstm
        self.input_size = input_size # input size
        self.hidden_size = hidden_size # neurons in each lstm layer
        # LSTM model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.4) # lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) # fully connected 
        self.fc_2 = nn.Linear(128, num_classes) # fully connected last layer
        self.relu = nn.ReLU()
        
    def forward(self,x):
        # hidden state
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # cell state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) # (input, hidden, and internal state)
        hn = hn.view(-1, self.hidden_size) # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) # first dense
        out = self.relu(out) # relu
        out = self.fc_2(out) # final output
        return out


#parameters
input = 7
output = 1

# =============================================================================
# 2. LSTM MODEL LOADING AND PREDICTION 
# =============================================================================
#load model and scalers
model = LSTM(num_classes=1, input_size=15, hidden_size=2, num_layers=1)
model.load_state_dict(torch.load(f"{cwd}/models/gas_lstm_model.pt"))
mm = torch.load(f"{cwd}/models/gas_minmax.pt")
ss = torch.load(f"{cwd}/models/gas_standard_scaler.pt")

#transform data
df_X_ss = ss.transform(df.drop(columns=['B2B'])) # old transformers
df_y_mm = mm.transform(df['B2B'].values.reshape(-1, 1)) # old transformers

# split a multivariate sequence past, future samples (X and y)
def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix, -1]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)

df_X_ss, df_y_mm = split_sequences(df_X_ss, df_y_mm, input, output)

# converting to tensors
df_X_ss = Variable(torch.Tensor(df_X_ss))
df_y_mm = Variable(torch.Tensor(df_y_mm))

# reshaping the dataset
df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], input, df_X_ss.shape[2]))

train_predict = model(df_X_ss) #predict
data_predict = train_predict.data.numpy() # numpy conversion
dataY_plot = df_y_mm.data.numpy()

data_predict = mm.inverse_transform(data_predict) # reverse transformation
dataY_plot = mm.inverse_transform(dataY_plot)


# ------------------------ BUILD PREDICTIONS TADAFRAME ----------------------- #
#add date to dataframe
data_predict = pd.DataFrame(data_predict, columns=['B2B'])
data_predict['datetime'] = df.index[input-1:]
data_predict['datetime'] = pd.to_datetime(data_predict['datetime'])
data_predict['datetime'] = data_predict['datetime'].dt.strftime('%Y-%m-%d')

#reorder columns
data_predict = data_predict.reindex(columns=['datetime', 'B2B'])


# ---------------------------- ADJUST PREDICTIONS ---------------------------- #
#fix datetime format
def calculate_adjuster(df, data_predict):
    df = df.reset_index()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d')
    df_pred = pd.merge(df, data_predict, on='datetime', how='left').dropna()
    df_pred = df_pred[df_pred['B2B_x'] != 0]
    #calculate difference between predicted and actual
    adjuster = (1-((df_pred['B2B_y'] - df_pred['B2B_x']) / df_pred['B2B_x'])).mean()
    return adjuster

adjuster = calculate_adjuster(df, data_predict)

#adjust predictions
data_predict['B2B'] = data_predict['B2B'] * adjuster


# ------------------------- EXPORT & PLOT PREDICTIONS ------------------------ #
#today date yyyy-mm-dd
today = datetime.today().strftime('%Y-%m-%d')

#export predictions to csv
data_predict.tail(5).to_csv(f"{cwd}/outputs/predictions/gas_lstm_predictions_{today}.csv", index=False)

#plot data_predict
plt.figure(figsize=(20,10), dpi=500)
plt.plot(data_predict['datetime'], data_predict['B2B'], color='blue', label='predicted')
plt.plot(data_predict['datetime'], dataY_plot, color='red', label='real')
plt.gcf().autofmt_xdate()
plt.legend(loc='upper left')
plt.title('GAS B2B adjusted predictions')
plt.show()

print('Predictions generated & exported!')