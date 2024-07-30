import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

from model.pytorch_model import (
    Early_Stopping,
    ModelCheckpoint,
    EnhancedLSTMModel,
    )
from utils.weatherapi import openMeteo_API
from preprocessing.data_pipeline import (
    Simple_create_sequences,
    forward_scaler,
    SingleStepMultiVARS_SeperateSampler,
    CyclicTimeTransform,
    )

from utils.plotting import history_plotting

## DATA CALLING
df = openMeteo_API(StartDate="2005-01-01", EndDate="2024-07-27") # Calling the Data

# Feature engineering
columns_to_drop = ['weather_code', 'cloud_cover_low', 'cloud_cover_high', 'wind_direction_10m']
dropped_cols = []
for col in columns_to_drop:
    try:
        df = df.drop(columns=col)
        dropped_cols.append(col)
    except KeyError:
        print(f"{col} not found in the DataFrame!\nMoving onto the Next.")
        pass 

# df = df.drop(columns=['weather_code', 'cloud_cover_low', 'cloud_cover_high', 'wind_direction_10m'])

df = CyclicTimeTransform(df)

# Defining the Window size and Target Predictions
window_size = 30 * 24
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


## HYPERPARAMETERS
input_size = X.shape[-1]
hidden_size = 128
num_layers = 4
dropout_prob = 0.45
output_size = len(target_column)
n_epochs = 5
learning_rate = 0.001

## DEFINING THE TRAINING STEP
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
        scheduler.get_last_lr()[0]
        early_stopping(val_loss)
        checkpoint(model, val_loss)
        
        # if early_stopping.early_stop:
        #     print("Early stopping triggered")
        #     break
    
    return train_losses, val_losses

# At the top of your script, after imports
def load_model(path):
    model = EnhancedLSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob).to("cuda")
    model.load_state_dict(torch.load(path))
    return model

def test_step():
    # Load the saved model
    model_path = 'C:/Projs/COde/Meteo/MetP/src/model/best_lstm_model.pth'
    
    # Recreate the model architecture
    loaded_model = EnhancedLSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob).to("cuda")
    
    # Load the state dict
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()
    
    test_loss = 0
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to("cuda"), targets.to("cuda")
            outputs = loaded_model(inputs)
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
    model = EnhancedLSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob).to("cuda")
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    early_stopping = Early_Stopping(patience=20, verbose=True)
    checkpoint = ModelCheckpoint(filepath='C:/Projs/COde/Meteo/MetP/src/model/best_lstm_model.pth', verbose=True)

    train_losses, val_losses = train_model(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, n_epochs, early_stopping, checkpoint)
    history_plotting(train_losses, val_losses)
    test_step()