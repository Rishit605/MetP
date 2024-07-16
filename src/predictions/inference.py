import pandas as pd
import numpy as np

import openmeteo_requests
import requests_cache
from retry_requests import retry
import matplotlib.pyplot as plt

import torch
import torchsummary
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import os
import sys
import random
import string

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.pytorch_model import LSTMModel, Early_Stopping, ModelCheckpoint
from utils.weatherapi import openMeteo_API
from preprocessing.data_pipeline import forward_scaler, SingleStepMultiVARSampler, ret_split_data, backward_scaler

from training.training2 import (
    test_dataloader, 
    input_size,
    output_size,
    num_layers,
    hidden_size,
    dropout_prob,
    scaled_df,
    df,
    Scl,
    target_column,
    window_size,
    X_train,
    )

train_size = int(len(df) * 0.7)
valid_size = int(len(df) * 0.2)


# # for i, tarC in enumerate(target_column):
# #     plt.figure(figsize=(17,5))
# #     plt.plot(test_df.index[:100], acutals[:, i][:100], marker="x", label="real")
# #     plt.plot(test_df.index[:100], predictions[:, i][:100], marker="o", label="preds")
# #     plt.title(f'Predictions vs Actuals for {tarC}')
# #     plt.legend()
# #     plt.xticks(rotation=45)
# #     plt.show()


# # Predictions
# for i, tarC in enumerate(target_column): 
#     plt.figure(figsize=(17,5))
#     plt.plot(test_df.index[:100], test_df[tarC][:100], marker="x", label="real")
#     plt.plot(test_df.index[:100], forecast_df[tarC][:100], marker="o", label="preds")
#     plt.title(f'Predictions vs Actuals for {tarC}')
#     plt.legend()
#     plt.xticks(rotation=45)
#     plt.show()


## ------------------------------------------------------------------------------------------ ##

# import torch
# import numpy as np
# import pandas as pd
# import random
# import string
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt

# # Load model and set to evaluation mode
# model_path = r'C:\Projs\COde\Meteo\MetP\src\model\best_lstm_model.pth' 
# model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob).to("cuda")
# model.load_state_dict(torch.load(model_path))
# model.eval()

# # Prediction and actual lists
# predictions = []
# actuals = []

# # Generate predictions
# with torch.no_grad():
#     for inputs, targets in test_dataloader:
#         inputs = inputs.to("cuda")
#         targets = targets.to("cuda")
#         outputs = model(inputs)
#         predictions.append(outputs.cpu().numpy())
#         actuals.append(targets.cpu().numpy())

# # Concatenate predictions and actuals
# predictions = np.concatenate(predictions)
# actuals = np.concatenate(actuals)

# test_df = df[train_size + valid_size:]

# # Function to generate random column names
# def generate_random_names(num_names, length=5):
#     return [''.join(random.choices(string.ascii_uppercase, k=length)) for _ in range(num_names)]

# # Combine the target column names with the random column names
# target_columns = ["temperature_2m", "wind_speed_10m"]
# num_non_target_columns = scaled_df.shape[1] - len(target_columns)
# random_column_names = generate_random_names(num_non_target_columns)
# column_names = target_columns + random_column_names

# # Create dummies for inverse scaling
# dummies = np.zeros((predictions.shape[0], len(column_names)))
# dummies[:, :predictions.shape[1]] = predictions

# # Create a DataFrame to hold the dummies
# dummies_df = pd.DataFrame(dummies, columns=column_names)

# # Inverse transform only the target columns
# scaled_df = scaled_df.copy()
# # scaled_df[target_columns] = Scl.inverse_transform(scaled_df[target_columns])
# dummies_df = Scl.inverse_transform(dummies_df)
# dummies_df = pd.DataFrame(dummies_df, columns=column_names)
# print(dummies_df)

# # Store predictions in a DataFrame
# forecast_df = pd.DataFrame({
#     'date': df.index[train_size + valid_size:][:168],
#     'predicted_temperature_2m': dummies_df['temperature_2m'][:168],
#     'predicted_rain': dummies_df['rain'][:168],
#     'predicted_wind_speed_10m': dummies_df['wind_speed_10m'][:168],
#     'predicted_wind_direction_10m': dummies_df['wind_direction_10m'][:168],
#     'actual_temperature_2m': df['temperature_2m'][train_size + valid_size:][:168],
#     'actual_rain': df['rain'][train_size + valid_size:][:168],
#     'actual_wind_speed_10m': df['wind_speed_10m'][train_size + valid_size:][:168],
#     'actual_wind_direction_10m': df['wind_direction_10m'][train_size + valid_size:][:168]
# })
# forecast_df = forecast_df.set_index('date')
# print(forecast_df)

# Plot predictions vs actual values
# for col in target_columns:
#     plt.figure(figsize=(17, 5))
#     plt.plot(forecast_df.index, forecast_df[f'actual_{col}'], marker='x', label='Real')
#     plt.plot(forecast_df.index, forecast_df[f'predicted_{col}'], marker='o', label='Predicted')
#     plt.title(f'Predictions vs Actuals for {col}')
#     plt.legend()
#     plt.xticks(rotation=45)
#     plt.show()


# temp, wind = predictions[:, 0], predictions[:, 1]

# def plot_preds(predicts, reals, col_name):

#     plt.figure(figsize=(17, 5))
#     plt.plot(test_df.index[:1000], predicts, marker='x', label='Real')
#     plt.plot(test_df.index[:1000], reals, marker='o', label='Predicted')
#     plt.title(f'Predictions vs Actuals for {col_name}')
#     plt.legend()
#     plt.xticks(rotation=45)
#     plt.show()


### ---------------------------------------------------------------------------------------------------------- ###

model_path = r'C:\Projs\COde\Meteo\MetP\src\model\best_lstm_model.pth'
model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob).to("cuda")
model.load_state_dict(torch.load(model_path))

# Function for inference on the 
def make_predictions(model, data_loader, scaler):
    """
    Function to evalute the model on the Test set after training
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to("cuda")
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
    return scaler.inverse_transform(np.array(predictions))

# Predictions on test set
test_predictions = make_predictions(model, test_loader, scaler_Y)
test_actuals = scaler_Y.inverse_transform(test_y)

# Future forecasting
def future_forecast(model, last_sequence, scaler_X, scaler_Y, num_days):
    model.eval()
    current_sequence = last_sequence.copy()
    forecasts = []
    
    with torch.no_grad():
        for _ in range(num_days * 24):
            inputs = torch.FloatTensor(current_sequence).unsqueeze(0).to("cuda")
            output = model(inputs)
            forecasts.append(output.cpu().numpy()[0])
            
            # Update the sequence for next prediction
            new_input = scaler_X.inverse_transform(current_sequence[-1].reshape(1, -1))
            new_input[:, :len(target_columns)] = scaler_Y.inverse_transform(output.cpu().numpy())
            new_input = scaler_X.transform(new_input)
            current_sequence = np.vstack((current_sequence[1:], new_input))
    
    return scaler_Y.inverse_transform(np.array(forecasts))


if __name__ == "__main__":

    # Get the last sequence from the test set
    last_sequence = X[-1]
    num_days = 2

    # Make future predictions for future forecasts
    future_predictions = future_forecast(model, last_sequence, scaler_X, scaler_Y, num_days)

    # Create a date range for future predictions
    last_date = pd.to_datetime(df.index[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=(num_days * 24), freq='h')
