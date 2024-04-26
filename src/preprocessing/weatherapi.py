import os
import csv
import json
import requests

import pandas as pd

# The API endpoint you want to call
url = 'https://api.weatherbit.io/v2.0/history/hourly?lat=23.25&lon=77.25&country=India&start_date=2024-01-01&end_date=2024-04-01&key=fcac888d541149bcabe62e93c304cd29'


def api_call() -> pd.DataFrame:

    # Make the GET request
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the response JSON into a Python dictionary
        data = response.json()
        # Do something with the data
        dat = data['data']

        # Create an empty list to store DataFrames from each dictionary
        data_frames = []

        # Create a DataFrame from each dictionary
        for dicts in dat:
            data_frame = pd.DataFrame([dicts])  # Wrap data in a list for DataFrame creation
            data_frames.append(data_frame)

        # Concatenate the DataFrames vertically (axis=0)
        df = pd.concat(data_frames, ignore_index=True)
        return df

    else:
        print(f'Failed to retrieve data: {response.status_code}')
