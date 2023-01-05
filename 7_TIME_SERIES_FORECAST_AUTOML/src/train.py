# forecast time series with pycaret
import os
import pandas as pd
from pycaret.regression import *
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# set working directory
cwd = os.getcwd()
# level up
cwd = os.path.dirname(cwd)


# --------------------------------- READ DATA -------------------------------- #
# read prepared data
data = pd.read_csv(f'{cwd}/data/prepared_data.csv', parse_dates=True, index_col='Datetime')
# drop na
data = data.dropna()

# ------------------------------- PREPARE DATA ------------------------------- #
# split data into train 0.8 and test 0.2
train = data[:int(0.8*(len(data)))]
test = data[int(0.8*(len(data))):]

# plot b2b+b2c+vt
# train['b2b+b2c+vt'].plot(figsize=(15, 5))
# plt.show()

# ------------------------------- SETUP MODEL ------------------------------- #
# setup model
setup = setup(data=train, target='b2b+b2c+vt', session_id=123, 
                fold_strategy='timeseries', fold=5, 
                normalize=True, transformation=True,
                numeric_features=['month', 'day', 'hour', 'day_of_week', 'week_of_year', 'quarter', 'day_of_year', 'is_weekend'],
                )


# ------------------------------ MEDEL CREATION ------------------------------ #
# compare models
print('Comparing models...')
best_model = compare_models(sort='MAE')

# tune model
print('Tuning model...')
tuned_model = tune_model(best_model, optimize='MAE', n_iter=10)

# plot model
plot_model(tuned_model, plot='error')

# predict model
print('Predicting model...')
predict_model(tuned_model)

# finalize model
print('Finalizing model...')
final_model = finalize_model(tuned_model)

# predict test data
print('Predicting test data...')
predictions = predict_model(final_model, data=test)

# plot predictions
predictions[['b2b+b2c+vt', 'prediction_label']].plot(figsize=(15, 5))
plt.savefig(f'{cwd}/outputs/predictions.png')
plt.show()

# calculate error and plot
predictions['error'] = predictions['b2b+b2c+vt'] - predictions['prediction_label']
predictions['error'].plot(figsize=(15, 5))
plt.savefig(f'{cwd}/outputs/error.png')
plt.show()

# ------------------------------- EVALUATE MODEL ------------------------------- #
# evaluate model
print('Evaluating model...')
evaluate_model(final_model)


# ------------------------------- EXPORT MODEL ------------------------------- #
save_model(final_model, f'{cwd}/models/final_model.pkl')
print('Model exported successfully!')