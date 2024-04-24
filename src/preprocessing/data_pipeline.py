import os
import csv
import json
import requests

import pandas as pd
import numpy as np

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

def pre_process(file_path: str = df) -> pd.DataFrame:

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