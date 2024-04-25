import pandas as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

from model import SimpleLSTMModel
from data_pipeline import pre_process, 


epochs = 5
batch_size = 16

n_features = len(pre_process().columns)

model = SimpleLSTMModel(n_features=n_features, n_target_variables=n_features)

# Define callback to save the best model based on validation loss
filepath = r'C:\Projs\COde\Meteo\MetP\src\model\best_weather_model.keras'  # Path to save the best model
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='min')

losses = keras.losses.MeanSquaredError()
metric = keras.metrics.RootMeanSquaredError()
opt = keras.optimizers.Adam(learning_rate=1e-4)

model.compile(loss=losses, metrics=[metric], optimizer=opt)
model.fit(X1_Train, y1_Train, epochs=epochs, batch_size=batch_size, vlaidation_data=(X1_Val, y1_Val), callbacks=[checkpoint])

### Data Calling is LEFT