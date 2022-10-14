# =============================================================================
# File connect Excel workbook with GAS data, read data and generate forecast with Random Forest Model. 
# =============================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px # to plot the time series plot
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable


# =============================================================================
# 1. DATA PROCESSING
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
df_full['Weeknumber'] = df_full.index.week
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

df_full = df_full.reindex(columns=['B2B', 'PreviousDayConsumption', 'PreviousWeekConsumption', '7daysmedian', 
                                    '7daysAverage', '7daysAverage_min', '7daysAverage_max',
                                 'Temperature 2 m','Precipitation','Snowfall', 'Year', 'Month', 
                                 'Weekday', 'DayYear', 'Seasons', 'Weeknumber']) #Reoder columns

df_full = df_full[(df_full.index >= '2016-01-01')]
df_full = df_full.fillna(0)

#Separe forecast dataframe
forecast_df = df_full[df_full['B2B'] == 0]

#Drop forecast values
df_full = df_full[df_full['B2B'] != 0]
df = df_full


# =============================================================================
# 2. LSTM MODEL BUILDER 
# =============================================================================
#parameters
input = 7
output = 1

#set input and output columns
X, y = df.drop(columns=['B2B']), df['B2B'].values

#print X and y shape
print(X.shape, y.shape)

#scale data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
mm = MinMaxScaler()
ss = StandardScaler()

X_trans = ss.fit_transform(X)
y_trans = mm.fit_transform(y.reshape(-1, 1))

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

X_ss, y_mm = split_sequences(X_trans, y_trans, input, output)
print(X_ss.shape, y_mm.shape)


# -------------------------------- DATA SPLIT -------------------------------- #
#split into train and test datasets
total_samples = len(X)
train_test_cutoff = round(0.8 * total_samples)

X_train = X_ss[:-output]
X_test = X_ss[-output:]

y_train = y_mm[:-output]
y_test = y_mm[-output:] 

print("Training Shape:", X_train.shape, y_train.shape)
print("Testing Shape:", X_test.shape, y_test.shape) 


# ------------------------- CONVERT TO TORCH TENSORS ------------------------- #
# convert to torch tensors
X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test))

print(X_train_tensors.shape[0], X_test_tensors.shape[1])

#reshaping to rows, timestamps, features
X_train_tensors_final = torch.reshape(X_train_tensors,   
                                      (X_train_tensors.shape[0], input, 
                                       X_train_tensors.shape[2]))
X_test_tensors_final = torch.reshape(X_test_tensors,  
                                     (X_test_tensors.shape[0], input, 
                                      X_test_tensors.shape[2])) 

print("Training Shape:", X_train_tensors_final.shape, y_train_tensors.shape)
print("Testing Shape:", X_test_tensors_final.shape, y_test_tensors.shape) 

#check if data is correct
# X_check, y_check = split_sequences(X, y.reshape(-1, 1), input, output)
# X_check[-1][0:4]

# print(y_check[-1])
# df.B2B[-output:]

# -------------------------------- LSTM MODEL -------------------------------- #
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


# -------------------------------- TRAINING -------------------------------- #
losses = []
test_losses = []
def training_loop(n_epochs, lstm, optimiser, loss_fn, X_train, y_train, X_test, y_test):
    for epoch in range(n_epochs):
        lstm.train()
        outputs = lstm.forward(X_train) # forward pass
        optimiser.zero_grad() # calculate the gradient, manually setting to 0
        # obtain the loss function
        loss = loss_fn(outputs, y_train)
        loss.backward() # calculates the loss of the loss function
        optimiser.step() # improve from loss, i.e backprop
        # test loss
        lstm.eval()
        test_preds = lstm(X_test)
        test_loss = loss_fn(test_preds, y_test)
        #add to list of losses
        test_losses.append(test_loss.item())
        losses.append(loss.item())
        #print and plot loss
        if epoch % 100 == 0 and epoch != 0:
            print("Epoch: %d, train loss: %1.5f, test loss: %1.5f" % (epoch, 
                                                                      loss.item(), 
                                                                      test_loss.item()))
            #plot losses and test losses
            plt.plot(losses, label='Training Loss')
            plt.plot(test_losses, label='Test Loss')
            plt.legend()
            plt.show()
        #add early stopping
        if epoch > 700 and test_losses[-1] > np.mean(test_losses[-200:]) and test_losses[-1] < 0.01:
            print("Early Stopping")
            break
        
import warnings
warnings.filterwarnings('ignore')


# --------------------------------- PARAMETERS -------------------------------- #
n_epochs = 50000 # number epochs
learning_rate = 0.001 # 0.001 lr

input_size = 15 # number of features (get from data)
hidden_size = 2 # number of features in hidden state
num_layers = 1 # number of stacked lstm layers

num_classes = output # number of output classes 

lstm = LSTM(num_classes, 
              input_size, 
              hidden_size, 
              num_layers)

#loss function and optimiser
loss_fn = torch.nn.MSELoss()    # mean-squared error for regression
optimiser = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

#train the model
training_loop(n_epochs=n_epochs,
              lstm=lstm,
              optimiser=optimiser,
              loss_fn=loss_fn,
              X_train=X_train_tensors_final,
              y_train=y_train_tensors,
              X_test=X_test_tensors_final,
              y_test=y_test_tensors)


# -------------------------------- PREDICTIONS ------------------------------- #
df_X_ss = ss.transform(df.drop(columns=['B2B'])) # old transformers
df_y_mm = mm.transform(df['B2B'].values.reshape(-1, 1)) # old transformers
# split the sequence
df_X_ss, df_y_mm = split_sequences(df_X_ss, df_y_mm, input, output)
# converting to tensors
df_X_ss = Variable(torch.Tensor(df_X_ss))
df_y_mm = Variable(torch.Tensor(df_y_mm))
# reshaping the dataset
df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], input, df_X_ss.shape[2]))

train_predict = lstm(df_X_ss) # forward pass
data_predict = train_predict.data.numpy() # numpy conversion
dataY_plot = df_y_mm.data.numpy()

data_predict = mm.inverse_transform(data_predict) # reverse transformation
dataY_plot = mm.inverse_transform(dataY_plot)
true, preds = [], []
for i in range(len(dataY_plot)):
    true.append(dataY_plot[i][0])
for i in range(len(data_predict)):
    preds.append(data_predict[i][0])
plt.figure(figsize=(10,6)) #plotting
plt.axvline(x=train_test_cutoff, c='r', linestyle='--') # size of the training set

#calculating MAPE from lists
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(true, preds).round(2)

# plotting the results
plt.plot(true, label='Actual Data') # actual plot
plt.plot(preds, label='Predicted Data') # predicted plot
plt.title(f'Gas Price Prediction, MAPE: {mape}') # title
plt.legend()
plt.savefig(f"{cwd}/outputs/{n_epochs}_{input}_{output}_{mape}.jpg", dpi=500) 
plt.show()

#plot preds and true
plt.plot(true[-250:], label='Actual Data') # actual plot
plt.plot(preds[-250:], label='Predicted Data') # predicted plot
plt.title(f'Gas Price Prediction, MAPE: {mape}') # title
plt.legend()
plt.show()


# ------------------------------- STORE INFO ------------------------------ #
#save the model and the scalers for later use
#save model pipeline
torch.save(lstm.state_dict(), f"{cwd}/models/gas_lstm_model.pt")
torch.save(mm, f"{cwd}/models/gas_minmax.pt") #save min max scaler
torch.save(ss, f"{cwd}/models/gas_standard_scaler.pt") #save standard scaler
print("Model and scalers saved")