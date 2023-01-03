import os
import pandas as pd


# set working directory
cwd = os.getcwd()
# level up
cwd = os.path.dirname(cwd)


# --------------------------------- READ DATA -------------------------------- #
# read prices excel file
prices = pd.read_excel(f'{cwd}/data/kainos.xlsx', parse_dates=True, index_col='Datetime')
weather = pd.read_excel(f'{cwd}/data/orai.xlsx', parse_dates=True, index_col='time')


# ------------------------------- PREPARE DATA ------------------------------- #
# drop weather columns with value Ryga
weather = weather[weather['city'] != 'ryga']

# group by time column
weather = weather.groupby('time').mean()

# rename time column to Datetime
weather = weather.rename_axis('Datetime').reset_index()


# -------------------------------- MERGE DATA -------------------------------- #
# merge prices and weather data
data = pd.merge(prices, weather, on='Datetime', how='left')

# features engineering
data['month'] = data['Datetime'].dt.month
data['day'] = data['Datetime'].dt.day
data['hour'] = data['Datetime'].dt.hour
data['day_of_week'] = data['Datetime'].dt.dayofweek + 1 # +1 because Monday is 1
data['week_of_year'] = data['Datetime'].dt.weekofyear
data['quarter'] = data['Datetime'].dt.quarter
data['day_of_year'] = data['Datetime'].dt.dayofyear
data['is_weekend'] = data['Datetime'].dt.dayofweek.isin([5, 6]).astype(int)

# moving average hourly for 30 days
data['GV kwh_30d'] = data['b2b+b2c+vt'].rolling(30*24).mean().shift(24)
data['VT kwh_30d'] = data['VT kwh'].rolling(30*24).mean().shift(24)
data['GV+VT_30d'] = data['GV+VT'].rolling(30*24).mean().shift(24)
data['b2b+b2c+vt_7d'] = data['b2b+b2c+vt'].shift(24)
data['gv_atidavimas_i_tinkla_mwh B2B_7d'] = data['gv_atidavimas_i_tinkla_mwh B2B'].shift(24)
data['gv_atidavimas_i_tinkla_mwh B2C_7d'] = data['gv_atidavimas_i_tinkla_mwh B2C'].shift(24)
data['gv_atidavimas_i_tinkla_mwh_7d'] = data['gv_atidavimas_i_tinkla_mwh'].shift(24)

# drop columns
c = ['gv_atidavimas_i_tinkla_mwh B2B', 'gv_atidavimas_i_tinkla_mwh B2C', 'gv_atidavimas_i_tinkla_mwh', 'Skirtumas ']
data = data.drop(c, axis=1)

# drop na
data = data.dropna()


# ------------------------------- EXPORT DATA ------------------------------- #
# export data to csv
data.to_csv(f'{cwd}/data/prepared_data.csv', index=False)
print('Data prepared successfully!')