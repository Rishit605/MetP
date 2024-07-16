import os
import sys
import csv
import json
import requests

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.weatherapi import api_call_Open_weather_Map
from utils.helpers import extract_value_or_zero, calculate_relative_humidity


dir = 'C:\\Projs\\COde\\\Meteo\\\MetP\\data\\'
filepath = os.path.join(dir, 'hourly_data.csv')

# # Get the 
# def get_df(URL: str) -> pd.DataFrame:
    
#     # Make the GET request
#     response = requests.get(URL)

#     if URL == url_WB:

#         if response.status_code != 200:
#             try:
#                 df_WB = pd.read_csv(filepath)
#                 return df
#             except FileExistsError as e:
#                 print(f'{e}: File does not exist in the directory.')
#         else:
#             df = apic_call_Weather_Bit()

#     elif URL == url_OWM:

#         if response.status_code != 200:
#             try:
#                 df_OWM = pd.read_csv(filepath)
#                 return df
#             except FileExistsError as e:
#                 print(f'{e}: File does not exist in the directory.')
#         else:
#             df = api_call_Open_weather_Map()

def get_df_WB(URL: str) -> pd.DataFrame:

    # Make the GET request
    response = requests.get(URL)

    if response.status_code == 200:
        df_WB = api_call_Weather_Bit(URL)
        return df_WB
    else:
        raise Exception
        print("URL is either invalid or incorrect.")

def get_df_OWM(URL: str) -> pd.DataFrame:

        # Make the GET request
        response = requests.get(URL)

        if response.status_code == 200:
            df_OWM = api_call_Open_weather_Map(URL)
            return df_OWM
        else:
            raise Exception
            print("URL is either invalid or incorrect.")

def timeseries_fn(dataframe: pd.DataFrame):
    day = 24*60*60
    year = (365.2425)*day

    dataframe['Day_sin'] = np.sin(dataframe['seconds'] * (2 * np.pi / day))
    dataframe['Day_cos'] = np.cos(dataframe['seconds'] * (2 * np.pi / day))
    dataframe['Year_sin'] = np.sin(dataframe['seconds'] * (2 * np.pi / year))
    dataframe['Year_cos'] = np.cos(dataframe['seconds'] * (2 * np.pi / year))


def pre_process_Weather_Bit(URL) -> pd.DataFrame:
    """Preprocesses and potentially saves the weather data.

    Args:
        filepath (str): Path to save the preprocessed data as a CSV file.

    Returns:
        data_frame: The preprocessed weather data.
    """

    data_frame_WB = get_df_WB(URL)  # Retrieves the DataFrame from the API

    # Reset index and convert timestamp to datetime format
    data_frame_WB = data_frame_WB.reset_index(drop=True)  # Drop the old index
    data_frame_WB['timestamp_utc'] = pd.to_datetime(data_frame_WB['timestamp_utc'])
    data_frame_WB.set_index('timestamp_utc', inplace=True)

    # Define columns to drop
    columns_to_drop = [
        'index', 'h_angle', 'weather', 'app_temp', 'timestamp_local', 'azimuth',
        'datetime', 'revision_status', 'pod', 'ts', 'seconds'
    ]

    # Drop columns only if they exist (using try-except for each)
    dropped_cols = []
    for col in columns_to_drop:
        try:
            data_frame_WB = data_frame_WB.drop(columns=col)
            dropped_cols.append(col)
        except KeyError:
            pass  # Column does not exist, continue dropping others

    # Informative message about dropped columns (if any)
    if dropped_cols:
        print(f"Dropped columns: {', '.join(dropped_cols)}")

    # Save DataFrame if file doesn't exist, handle exceptions
    if not os.path.exists(filepath):
        try:
            data_frame_WB.to_csv(filepath)
            print('Dataset saved successfully!')
        except ValueError as e:
            print(f'Dataset saving failed: {e}')
    else:
        print(f"File already exists at {filepath}")

    return data_frame_WB


def pre_process_OpenWeather_map(URL) -> pd.DataFrame:

    data_frame_OWM = get_df_OWM(URL)

    data_frame_OWM['dt'] = pd.to_datetime(data_frame_OWM['dt'], unit='s')

    if 'rain' in data_frame_OWM.columns:
        # Apply the function to the column
        data_frame_OWM['rain'] = data_frame_OWM['rain'].apply(extract_value_or_zero)
    else:
        data_frame_OWM['rain'] = 0

    # Renamning the Columns for preset names
    data_frame_OWM.rename(columns={'dt':'Datetime', 'temp': 'Temperature (°C)', 'humidity': 'Humidity (%)', 'gust': 'Wind Gust (m/s)', 'speed': 'Wind Speed (m/s)', 'deg':'Wind Direction (degrees)', 'all':'Cloud Coverage (%)', 'rain': 'Precipitation (mm)'}, inplace=True)
    
    thunderstorm_ids = [200, 201, 202, 210, 211, 212, 221, 230, 231, 232]

    # 3. Calculate Thunderstorm Probability Percentage
    def calculate_thunderstorm_prob(id_value):
        if id_value in thunderstorm_ids:
            index = thunderstorm_ids[::-1].index(id_value)
            return (index + 1) * 10
        else:
            return 0

    data_frame_OWM['Thunderstorm Occurrence'] = data_frame_OWM['id'].apply(calculate_thunderstorm_prob) # Appedn the thunderstorm Probability Percentages
    
    columns_to_drop = ['visibility', 'pop', 'sys', 'dt_txt', 'feels_like', 'temp_max', 'temp_min', 'sea_level', 'grnd_level', 'temp_kf', 'id', 'main', 'description', 'icon']

    # Drop columns only if they exist (using try-except for each)
    dropped_cols = []
    for col in columns_to_drop:
        try:
            data_frame_OWM = data_frame_OWM.drop(columns=col)
            dropped_cols.append(col)
        except KeyError:
            pass  # Column does not exist, continue dropping others


    # Adding the Relative Humidity
    # data_frame_OWM['Relative Humidity (%)'] = data_frame_OWM.apply(lambda row: calculate_relative_humidity(row[data_frame_OWM.columns[1]], row[data_frame_OWM.columns[4]]), axis=1)

    return data_frame_OWM


## Preprocessing

def CyclicTimeTransform(data: pd.DataFrame) -> pd.DataFrame:    
    day = 60 * 60 * 24
    year = 365.2425 * day
    data_df = data.copy()

    data_df['Seconds'] = data_df.index.map(pd.Timestamp.timestamp)
    
    data_df['Hour sin'] = np.sin(2 * np.pi * data_df.index.hour / 24)
    data_df['Hour cos']= np.cos(2 * np.pi * data_df.index.hour / 24)

    data_df['Day sin'] = np.sin(2 * np.pi * data_df.index.day / 7)
    data_df['Day cos'] = np.cos(2 * np.pi * data_df.index.day / 7)

    data_df['Month sin'] = np.sin(2 * np.pi * data_df.index.month / 12)
    data_df['Month sin'] = np.cos(2 * np.pi * data_df.index.month / 12)

    # data_df['Day sin'] = np.sin(data_df['Seconds'] * (2 * np.pi / day))
    # data_df['Day cos'] = np.cos(data_df['Seconds'] * (2 * np.pi / day)) 

    # data_df['Year sin'] = np.sin(data_df['Seconds'] * (2 * np.pi / day))
    # data_df['Year cos'] = np.cos(data_df['Seconds'] * (2 * np.pi / day)) 
    data_df = data_df.drop('Seconds', axis=1)
    return data_df

def forward_scaler(dataSet: pd.DataFrame):
    """
    Takes a DataFrame and returns a scaled and normalized DataFrame.

    Args:
        dataSet (pd.DataFrame): Input DataFrame with numerical columns.

    Returns:
        pd.DataFrame: Scaled and normalized DataFrame.
    """
    scale = MinMaxScaler(feature_range=(0,1))

    scaled_data = scale.fit_transform(dataSet)
    scaled_dataframe = pd.DataFrame(scaled_data, columns=dataSet.columns, index=dataSet.index)
    return scale, scaled_dataframe


def backward_scaler(predicts: np.ndarray, scaler, scaled_DataF: pd.DataFrame,  target_columns,  dataSet: pd.DataFrame, testset=True):
    """
    Reiterates the scaler and performs scaling inversion to get the real values form the sclaed predictions.
    
    Returns:    
        unreal_X = Returns the real values of the predictions by performing the inverse function.
    """
    # Check if the shapes match
    if scaled_predictions.shape[1] != original_df.shape[1]:
        raise ValueError("Shapes of scaled predictions and original DataFrame do not match.")
        print("Working on Reshaping.")

        dummies = np.zeros((predicts, training_data.shape[-1] + 1))
        dummies[:, 0] = predicts

        dummies_df = pd.DataFrame(dummies)
        
        if testset is True:
            dummies_df = pd.DataFrame(
                scaler.inverse_transform(dummies_df),
                index=scaled_DataF[train_size + valid_size:].index[:(len(scaled_DataF))]
            )

            forecast_df = pd.DataFrame({
                'date': scaled_DataF[train_size + valid_size:].index[:(len(scaled_DataF))],
                'predictions': dummies_df[(len(target_columns))],
                'acutals': dataSet[train_size + valid:][target_columns][:(len(target_columns))]
            })
            
            forecast_df = forecast_df.set_index('date')
            return forecast_df
            
        else:
            dummies_df = pd.DataFrame(
                scaler.inverse_transfrom(dummies_df),
                index=predictions_dates
            )

            forecast_df = pd.DataFrame({
                'date': predictions_dates,
                'predictions': dummies_df[(len(target_columns))]
            })

            forecast_df = forecast_df.set_index('date')
            return forecast_df
    else:
        unreal_X = scaler.inverse_transfrom(predicts)
        return unreal_X


def SingleStepSingleVARSampler(df, window, target_column):
    """
    For Generating SingleStep Single Target variable sequence for training.
    """

    # Convert DataFrame to NumPy array for faster operations
    features_array = df.to_numpy()
    target_array = df[target_column].to_numpy()

    # Number of samples we can create
    num_samples = len(df) - window

    # Initialize empty arrays for X and Y
    X = np.zeros((num_samples, window, features_array.shape[1]))
    Y = np.zeros(num_samples)

    for i in range(num_samples):
        X[i] = features_array[i:i+window]
        Y[i] = target_array[i + window]

    return X, Y

def SingleStepMultiVARSampler(df, window, target_columns):
    """
    For Generating SingleStep Multi Target variable sequence for training.
    """

    # Convert DataFrame to NumPy array for faster operations
    features_array = df.to_numpy()
    target_array = df[target_columns].to_numpy()

    # Number of samples we can create
    num_samples = len(df) - window

    # Initialize empty arrays for X and Y
    X = np.zeros((num_samples, window, features_array.shape[1]))
    Y = np.zeros((num_samples, len(target_columns)))

    for i in range(num_samples):
        X[i] = features_array[i:i+window]
        Y[i] = target_array[i + window]

    return X, Y

def SingleStepMultiVARS_SeperateSampler(df_X, df_Y, window, target_columns):
    """
    For Generating SingleStep Multi Target variable sequence for training.
    """

    # Convert DataFrame to NumPy array for faster operations
    features_array = df_X.to_numpy()
    target_array = df_Y[target_columns].to_numpy()

    # Number of samples we can create
    num_samples = len(df_X) - window

    # Initialize empty arrays for X and Y
    X = np.zeros((num_samples, window, features_array.shape[1]))
    Y = np.zeros((num_samples, len(target_columns)))

    for i in range(num_samples):
        X[i] = features_array[i:i+window]
        Y[i] = target_array[i + window]

    return X, Y
    

## Splitting the dataset

def ret_split_data(data: pd.DataFrame, X_Set, Y_Set, SIZE=0.8):
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

    train_split = int(len(data) * SIZE)

    if len(data) < 10000: 
        valid_split = int(len(data) - train_split)

        # Splitting the Dataset
        X1_train, y1_train = X_Set[:train_split], Y_Set[:train_split]
        X1_val, y1_val = X_Set[train_split:train_split + valid_split], Y_Set[train_split:train_split + valid_split]
        return X1_train, y1_train, X1_val, y1_val


    else:
        valid_split = int(len(data) * (1.0 - SIZE))

        # Splitting the Dataset
        X1_train, y1_train = X_Set[:train_split], Y_Set[:train_split]
        X1_val, y1_val = X_Set[train_split:train_split + valid_split], Y_Set[train_split:train_split + valid_split]
        X1_test, y1_test = X_Set[train_split + valid_split:], Y_Set[train_split + valid_split:]
        return X1_train, y1_train, X1_val, y1_val, X1_test, y1_test
        

## Dataset Creation


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
