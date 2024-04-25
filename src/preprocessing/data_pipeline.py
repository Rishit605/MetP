import os
import csv
import json
import requests

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler


# The API endpoint you want to call
url = 'https://api.weatherbit.io/v2.0/history/hourly?lat=23.25&lon=77.25&country=India&start_date=2024-01-01&end_date=2024-04-01&key=fcac888d541149bcabe62e93c304cd29'

# Make the GET request
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the response JSON into a Python dictionary
    data = response.json()
    # Do something with the data
    dat = data['data']

    # # Saving the JSON Data
    # with open("data.json", "w") as json_file:
    #     json.dump(dat json_file)
    
else:
    print(f'Failed to retrieve data: {response.status_code}')


# Create an empty list to store DataFrames from each dictionary
data_frames = []

# Create a DataFrame from each dictionary
for dicts in dat:
    data_frame = pd.DataFrame([dicts])  # Wrap data in a list for DataFrame creation
    data_frames.append(data_frame)

# Concatenate the DataFrames vertically (axis=0)
df = pd.concat(data_frames, ignore_index=True)


dir = 'C:\\Projs\\COde\\\Meteo\\\MetP\\data\\'
filepath = os.path.join(dir, 'hourly_data.csv')

    
def timeseries_fn(dataframe: pd.DataFrame):
    day = 24*60*60
    year = (365.2425)*day

    dataframe['Day_sin'] = np.sin(dataframe['seconds'] * (2 * np.pi / day))
    dataframe['Day_cos'] = np.cos(dataframe['seconds'] * (2 * np.pi / day))
    dataframe['Year_sin'] = np.sin(dataframe['seconds'] * (2 * np.pi / year))
    dataframe['Year_cos'] = np.cos(dataframe['seconds'] * (2 * np.pi / year))

def pre_process() -> pd.DataFrame:

    data_frame = df

    # Ressting the index of the dataframe
    data_frame = data_frame.reset_index()
    data_frame['timestamp_utc'] = pd.to_datetime(data_frame['timestamp_utc'])

    data_frame = data_frame.set_index('timestamp_utc')
    data_frame['seconds'] = data_frame.index.map(pd.Timestamp.timestamp)

    timeseries_fn(data_frame)

    data_frame = data_frame.drop(columns=['h_angle', 'weather', 'app_temp', 'timestamp_local', 'azimuth', 'datetime', 'revision_status', 'index', 'ts', 'weather'])
    
    if not os.path.exists(filepath):
        try:
            data_frame.to_csv(filepath)
            print('Dataset saved successfully!')
        except ValueError as e:
            print(f'{e}: Dataset saving falied')

    else:
        print(f"File alredy exists in {dir}")

    return data_frame

## Preprocessing 

def forward_scaler() -> np.array:
  """
  Scales the dataset into (0,1) range for standariztion for the model.

  Returns:
    scaled_X: Returns the scaled features of the dataset in the (0,1) range.
    scaled_y: Returns the scaled target(s) of the dataset in the (0,1) range.
  """
  scale = MinMaxScaler(feature_range=(0,1))

  _X = window_make(pre_process())[0]
  _y = window_make(pre_process())[1]

  scaled_X = scale.fit_transform(_X)
  scaled_y = scale.fit_transform(_y)
  return scaled_X, scaled_y


def backward_scaler(predicts):
  """
  Reiterates the scaler and performs scaling inversion to get the real values form the sclaed predictions.
  
  Returns:
    unreal_X = Returns the real values of the predictions by performing the inverse function.
  """
  scale = MinMaxScaler(feature_range=(0,1))

  unreal_X = scale.inverse(predicts)
  return unreal_X

def window_make(df):
  df_as_np = df.to_numpy()
  window_size = len(df.columns) + 1

  X=[]
  y=[]

  for i in range(len(df_as_np) - window_size):
    row = [a for a in df_as_np[i : i + window_size]]
    X.append(row)

    label = [df_as_np[i + window_size][0],
             df_as_np[i + window_size][1],
             df_as_np[i + window_size][2],
             df_as_np[i + window_size][3],
             df_as_np[i + window_size][4]]
    y.append(label)

  return np.array(X), np.array(y)

## Splitting the dataset

def ret_split_data(data: pd.DataFrame):
    """
    This function performs the dataset splitting accorin to the dataset size, splitting it into training set, validation set, and test set.

    Args:
        data: A Pandas dataframe respresenting your dataset.

    Returns: [Condtional]
        Case 1 (If the size of the dataset is less than 10000): 
        X1_train: Returns an array of the training set of the features.
        y1_train: Returns an array of the training set of the target(s).
        X1_val: Returns an array of the validation set of the features.
        y1_val: Returns an array of the validation set of the target(s).

        Case 2 (If the size of the dataset is larger than 10000): 
        X1_train: Returns an array of the training set of the features.
        y1_train: Returns an array of the training set of the target(s).
        X1_val: Returns an array of the validation set of the features.
        y1_val: Returns an array of the validation set of the target(s).
        X1_test: Returns an array of the test set of the features.
        y1_test: Returns an array of the test set of the target(s).
    """

    X1, y1 = window_make(data)

    train_split = int(len(X1) * (70 / 100))

    if len(data) < 10000: 
        valid_split = int(len(X1) - train_split)

        # Splitting the Dataset
        X1_train, y1_train = X1[:train_split], y1[:train_split]
        X1_val, y1_val = X1[train_split:valid_split], y1[train_split:valid_split]
        return X1_train, y1_train, X1_val, y1_val


    else:
        valid_split = int(len(X1) - train_split)
        test_split = int(valid_split - (50 / 100))

        # Splitting the Dataset
        X1_train, y1_train = X1[:train_split], y1[:train_split]
        X1_val, y1_val = X1[train_split:valid_split], y1[train_split:valid_split]
        X1_test, y1_test = X1[valid_split:test_split], y1[valid_split:test_split]
        return X1_train, y1_train, X1_val, y1_val, X1_test, y1_test
        

## Standardization Function

def std_fn(data: np.array, idx):
    """
    This function standardizes and calculates the mean and standard deviation of the specified channel(feature) across all timesteps in the data array.

    Args:
        data: A 3D NumPy array representing your multivariate time series data. The first two dimensions likely represent time steps and features, and the third dimension represents different channels or variables in your multivariate data.
        idx: An integer representing the index of the specific channel/variable you want to standardize.

    Returns:
        mean: Returns a array of calcuated mean of the channels.
        std: Returns a array of calculated standard deviation of the channles.
    """
    mean = np.mean(data[:, :, idx])
    std = np.std(data[:, :, idx])
    return mean, std

def apply_std_fn(X, y, idx):
    """
    It calculates the mean and standard deviation using std_fn(X, idx) and subtracts the mean from each data point in the chosen channel of X and y.
    Then, it divides the result by the standard deviation to achieve standardization.

    Args:
        X: A 3D NumPy array representing your multivariate features (same as data in std_fn).
        y: A 2D or 3D NumPy array representing your target variable(s). It can be 2D if there's a single target variable, or 3D if there are multiple target variables with the same structure as X (multiple channels).
        idx: Similar to std_fn, this is the index of the specific channel/variable you want to standardize in both X (features) and y (target variable(s)).

    Returns:
        X: The function returns the standardized versions of X.
        y: The function returns the standardized versions of y.

    """
    X[:, :, idx] = (X[:, :, idx] - std_fn(X, idx)[0]) / std_fn(X, idx)[1]
    y[:, idx] = (y[:, idx] - std_fn(X, idx)[0]) / std_fn(X, idx)[1]
    return X, y


def std_fit_fn():
    """
    This function applies the scaled and standarization to the features and target(s) array.
    """
    res = ret_split_data(pre_process())

    if isinstance(res, tuple):
        if len(result) == 4:
            X_Train, y_Train, X_Test, y_Test = ret_split_data(pre_process())

            for i in range(0,5):
                apply_std_fn(X1_Train, y1_Train, i)
                apply_std_fn(X1_Val, y1_Val, i)

        elif len(result) == 6:
            X_Train, y_Train, X_Val, y_Val, X_Test, y_Test = ret_split_data(pre_process())
        
            for i in range(0,5):
                apply_std_fn(X1_Train, y1_Train, i)
                apply_std_fn(X1_Val, y1_Val, i)
                apply_std_fn(X1_Test, y1_Test, i)

