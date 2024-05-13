import pandas as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

from model import OtherLSTMModel
from data_pipeline import *

## Parameters
epochs = 5
batch_size = 16

## Calling the Data
data_to_frame = pre_process()
scalard_X = forward_scaler(data_to_frame)

# Create an empty DataFrame with original column names
df_scaled = pd.DataFrame(columns=data_to_frame.columns)

# Assign scaled values to columns
for i, col in enumerate(data_to_frame.columns):
  df_scaled[col] = scalard_X[:, i]

df_scaled['timestamp'] = pd.Series(pre_process().index)
df_scaled = df_scaled.set_index('timestamp')

res = ret_split_data(df_scaled)

if isinstance(res, tuple):
    if len(res) == 4:
        X_Train, y_Train, X_Val, y_Val = ret_split_data(df_scaled)
        print("Dataset is too small for simultaneous Validation and Test Split. Hence Dataset is plit into only two sets.")
    elif len(res) == 6:
        X_Train, y_Train, X_Val, y_Val, X_Test, y_Test = ret_split_data(df_scaled)

print(X_Train)

# Defining the Model
n_features = len(pre_process().columns)
model = OtherLSTMModel(n_features=n_features, n_target_variables=n_features)

# Define callback to save the best model based on validation loss
filepath = r'C:\Projs\COde\Meteo\MetP\src\model\best_weather_model.keras'  # Path to save the best model
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='min')

losses = keras.losses.MeanSquaredError()
metric = keras.metrics.RootMeanSquaredError()
opt = keras.optimizers.Adam(learning_rate=1e-4)

model.compile(loss=losses, metrics=[metric], optimizer=opt)
model.fit(X_Train, y_Train, epochs=epochs, batch_size=batch_size, validation_data=(X_Val, y_Val), callbacks=[checkpoint])

# ### Data Calling is LEFT