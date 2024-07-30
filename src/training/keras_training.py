import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, GRU, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
from sklearn.metrics import mean_squared_error

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import GRUModel
from preprocessing.data_pipeline import *
from utils.weatherapi import openMeteo_API

# ## Parameters
# epochs = 5
# batch_size = 16

# ## Calling the Data
# data_to_frame = pre_process()
# scalard_X = forward_scaler(data_to_frame)

# # Create an empty DataFrame with original column names
# df_scaled = pd.DataFrame(columns=data_to_frame.columns)

# # Assign scaled values to columns
# for i, col in enumerate(data_to_frame.columns):
#   df_scaled[col] = scalard_X[:, i]

# df_scaled['timestamp'] = pd.Series(pre_process().index)
# df_scaled = df_scaled.set_index('timestamp')

# res = ret_split_data(df_scaled)

# if isinstance(res, tuple):
#     if len(res) == 4:
#         X_Train, y_Train, X_Val, y_Val = ret_split_data(df_scaled)
#         print("Dataset is too small for simultaneous Validation and Test Split. Hence Dataset is plit into only two sets.")
#     elif len(res) == 6:
#         X_Train, y_Train, X_Val, y_Val, X_Test, y_Test = ret_split_data(df_scaled)

# print(X_Train)

# # Defining the Model
# n_features = len(pre_process().columns)
# model = BasicLSTM(input_shape=(window_size, X_train.shape[-1]), output_size=n_features)

# # Define callback to save the best model based on validation loss
# filepath = r'C:\Projs\COde\Meteo\MetP\src\model\best_weather_model.keras'  # Path to save the best model
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='min')

# losses = tf.keras.losses.MeanSquaredError()
# metric = tf.keras.metrics.RootMeanSquaredError()
# opt = tf.keras.optimizers.Adam(learning_rate=1e-4)

# model.compile_model(optimizer=opt, loss=losses, metrics=metric)

# history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

#### ------------------------------------------------------------ ####

## Calling the Dataset and preprocessing it
df = openMeteo_API()
df = df.drop(columns=['weather_code', 'cloud_cover_low', 'cloud_cover_high', 'wind_direction_10m'])
df = CyclicTimeTransform(df)

## Defining Parameters
window_size = 24
target_column = ["temperature_2m", "wind_speed_10m", "dew_point_2m"]

# Splitting and Scaling the Dataset
X1 = df.drop(columns=target_column)
Y1 = df[target_column]

scaler_X, scaled_X = forward_scaler(dataSet=X1)
scaler_Y, scaled_Y = forward_scaler(dataSet=Y1)

# Generating TimeSeries Sequence
X, Y = SingleStepMultiVARS_SeperateSampler(scaled_X, scaled_Y, window_size, target_column)

# Splitting the Dataset

train_size = int(len(X) * 0.715)
valid_size = int(len(X) * train_size)

train_X, train_y = X[:train_size], Y[:train_size]
val_X, val_y = X[train_size:train_size + valid_size], Y[train_size:train_size  + valid_size]


## Creating a Tensor Dataset
BATCH_SIZE = 64
train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).batch(BATCH_SIZE).prefetch(1)
valid_dataset = tf.data.Dataset.from_tensor_slices((val_X, val_y)).batch(BATCH_SIZE).prefetch(1)

epochs = 10

modelGRU = GRUModel(input_shape=(window_size, train_X.shape[-1]), output_size = target_column)
train_loss, val_loss, train_rmse, val_rmse= modelGRU.train_step(train_dataset, valid_dataset, BATCH_SIZE)


# Plotting function
def plot_training_history(train_losses, val_losses, train_rmse, val_rmse):
    plt.figure(figsize=(12, 4))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Metric Plot (RMSE)
    plt.subplot(1, 2, 2)
    plt.plot(train_rmse, label='Training RMSE')
    plt.plot(val_rmse, label='Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('Training and Validation RMSE')

    plt.show()

plot_training_history(train_loss, val_loss, train_rmse, val_rmse)