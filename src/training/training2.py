import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

import openmeteo_requests
import requests_cache
from retry_requests import retry

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.pytorch_model import (
    LSTMModel,
    Early_Stopping,
    ModelCheckpoint,
    EnhancedLSTMModel,
    )
from utils.weatherapi import openMeteo_API
from preprocessing.data_pipeline import (
    forward_scaler,
    SingleStepMultiVARS_SeperateSampler,
    CyclicTimeTransform,
    )


df = openMeteo_API(StartDate="2005-01-01", EndDate="2024-07-15") # Calling the Data

# Feature engineering
df = df.drop(columns=['weather_code', 'cloud_cover_low', 'cloud_cover_high', 'wind_direction_10m'])
df = CyclicTimeTransform(df)
df['day_of_year'] = pd.to_datetime(df.index).dayofyear
df['month'] = pd.to_datetime(df.index).month

# Defining the Window size and Target Predictions
window_size = 7 * 24
target_column = df.columns[:8]

# Defining the Input and Output Features
X1 = df
Y1 = df[target_column]

# Scaling the Dataset
scaler_X, scaled_X = forward_scaler(dataSet=X1)
scaler_Y, scaled_Y = forward_scaler(dataSet=Y1)

# Creting the Window Sequences
X, Y = SingleStepMultiVARS_SeperateSampler(scaled_X, scaled_Y, window_size, target_column)
X, Y = np.array(X), np.array(Y)


# Splitting the dataset
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)
test_size = len(X) - train_size - val_size

X_train, y_train = X[:train_size], Y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], Y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], Y[train_size+val_size:]

# X_train, y_train, X_val, y_val, X_test, y_test = X_train.to("cuda"), y_train.to("cuda"), X_val.to("cuda"), y_val.to("cuda"), X_test.to("cuda"), y_test.to("cuda")


# Converting the dataset to Torch DataLoader format
BATCH_SIZE = 64

train_tensor = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
valid_tensor = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
test_tensor = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

train_dataloader = DataLoader(
    train_tensor, 
    batch_size=BATCH_SIZE, 
    shuffle=False
)

valid_dataloader = DataLoader(
    valid_tensor,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_dataloader = DataLoader(
    test_tensor,
    batch_size=BATCH_SIZE,
    shuffle=False
)


### OLD MODEL INITALIZATION AND PARAMTERS
# Define the model, loss function, and optimizer
n_epochs = 50 
input_size = X.shape[2] # Number of features (excluding target column)
hidden_size = 32
num_layers = 2
dropout_prob = 0.5  # Dropout probability
output_size = len(target_column) # Predicting one value (temperature)
early_stopping = Early_Stopping(patience=20, verbose=True)
checkpoint = ModelCheckpoint(filepath=r'C:\Projs\COde\Meteo\MetP\src\model\best_lstm_model.pth', verbose=True)

# Create and train the model
model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob).to("cuda")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

 
def history_plotting(train, test):
    plt.plot(train_hist, scalex=True, label='Training Loss')
    plt.plot(test_hist, label='Validation Loss')
    plt.legend()
    plt.show()


# Initial Training Step with Mix Precision for Low Memory usage
def training_step(model, train_dataloader, valid_dataloader, criterion, optimizer, num_epochs, early_stopping, checkpoint):
    """
    Initial Training Step with Mix Precision for Low Memory usage.
    """
    train_hist = []
    test_hist = []
    scaler = GradScaler()  # Initialize GradScaler for mixed precision training
    accumulation_steps = 4 # Set accumulation steps
    for epoch in range(num_epochs):
        total_train_loss = 0
        model.train()
        for i, (inputs, targets) in enumerate(train_dataloader):
            with torch.cuda.amp.autocast():
                outputs = model(inputs).to("cuda")
                # targets = targets[:, -1, 0]
                loss = criterion(outputs, targets) / accumulation_steps

            scaler.scale(loss).backward()  # Scale the loss for mixed precision
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_train_loss += loss.item()  # Correctly accumulate train loss

        # Validation (after the full epoch)
        model.eval()
        with torch.no_grad():
            total_valid_loss = 0
            for inputs, targets in valid_dataloader:
                outputs = model(inputs).to("cuda")
                # targets = targets[:, -1, 0]
                total_valid_loss += criterion(outputs, targets).item()

        train_rmse = np.sqrt(total_train_loss / len(train_dataloader))
        test_rmse = np.sqrt(total_valid_loss / len(valid_dataloader))

        train_hist.append(train_rmse)
        test_hist.append(test_rmse)

        if epoch % 10 == 0:
            print("Epoch %d: Train RMSE %.4f, Test RMSE %.4f" % (epoch, train_rmse, test_rmse))

        early_stopping(total_valid_loss)
        checkpoint(model, total_valid_loss)
        # if early_stopping.early_stop:
        #     print("No Improvement: Early Stopping")
        #     break
    return train_hist, test_hist



### NEW ENHANCED MODEL TRAINING STEP
# New training step for the new Enhanced Model.

# Hyperparameters
input_size = X.shape[-1]
hidden_size = 128
num_layers = 4
dropout_prob = 0.45
output_size = len(target_column)
n_epochs = 20
learning_rate = 0.001

early_stopping = Early_Stopping(patience=20, verbose=True)
checkpoint = ModelCheckpoint(filepath=r'C:\Projs\COde\Meteo\MetP\src\model\new_best_lstm_model.pth', verbose=True)

scaler = GradScaler()
def train_model(model, train_loader, val_loader, criterion, optimizer, lscheduler, num_epochs, early_stopping, checkpoint):
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to("cuda"), targets.to("cuda")
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to("cuda"), targets.to("cuda")
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)
        early_stopping(val_loss)
        checkpoint(model, val_loss)
        
        # if early_stopping.early_stop:
        #     print("Early stopping triggered")
        #     break
    
    return train_losses, val_losses


## Validation Step for the Enhanced model.
def test_step():
    # Evaluate on test set
    model.load_state_dict(torch.load(model))
    model.eval()
    test_loss = 0
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to("cuda"), targets.to("cuda")
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())

    test_loss /= len(test_dataloader)
    print(f"Test Loss: {test_loss:.4f}")

    # Denormalize predictions and actuals
    predictions = scaler_Y.inverse_transform(np.array(predictions))
    actuals = scaler_Y.inverse_transform(np.array(actuals))

    # Calculate RMSE for each target variable
    for i, col in enumerate(target_column):
        rmse = np.sqrt(np.mean((predictions[:, i] - actuals[:, i])**2))
        print(f"RMSE for {col}: {rmse:.4f}")

    

if __name__ == "__main__":

    # ### New training step for the new Enhanced Model.
    # # Model, optimizer, and loss function creation
    # model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob).to("cuda")
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # train_hist, test_hist = training_step(model, train_dataloader, valid_dataloader, criterion, optimizer, n_epochs, early_stopping, checkpoint)

    # history_plotting(train_hist, test_hist)


    ### 
    model = EnhancedLSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob).to("cuda")
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # train_losses, val_losses = train_model(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, n_epochs, early_stopping, checkpoint)
    test_step()