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

from model.pytorch_model import (
    Early_Stopping,
    ModelCheckpoint,
    EnhancedLSTMModel,
    )

from utils.weatherapi import openMeteo_API
from preprocessing.data_pipeline import forward_scaler, backward_scaler

from training.torch_training2 import (
    input_size, output_size, window_size,
    num_layers, hidden_size, dropout_prob,
    scaler_X, scaler_Y, X1, 
    X, target_column,
    )


model_path = 'C:/Projs/COde/Meteo/MetP/src/model/best_lstm_model.pth'
try:
    model = EnhancedLSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob).to("cuda")
    model.load_state_dict(torch.load(model_path))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Future forecasting
def future_forecast(model, last_sequence, scaler_X, scaler_Y, num_days, target_columns):
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
    future_predictions = future_forecast(model, last_sequence, scaler_X, scaler_Y, num_days, target_column)

    # Create a date range for future predictions
    last_date = pd.to_datetime(X1.index[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=(num_days * 24), freq='h')

    future_df = pd.DataFrame(future_predictions, columns=target_column, index=future_dates)
    future_df.to_csv('forecasts_after27072024.csv')
    # print(future_df)
    print()