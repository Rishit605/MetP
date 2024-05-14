import os
import csv
import json
import requests

import pandas as pd

# The API endpoint you want to call
url_WB = 'https://api.weatherbit.io/v2.0/history/hourly?lat=23.25&lon=77.25&country=India&start_date=2024-01-01&end_date=2024-04-01&key=fcac888d541149bcabe62e93c304cd29'
# url_OWM = 'https://api.openweathermap.org/data/2.5/forecast?lat=23.2599&lon=77.4126&appid=c9d4c0f1fb50e0f4786f71ad1446fdc9&units=metric'

bpl_url_OWM = 'https://api.openweathermap.org/data/2.5/forecast?lat=23.2599&lon=77.4126&appid=c9d4c0f1fb50e0f4786f71ad1446fdc9&units=metric'
bgl_url_OWM = 'https://api.openweathermap.org/data/2.5/forecast?lat=12.9716&lon=77.5946&appid=c9d4c0f1fb50e0f4786f71ad1446fdc9&units=metric'
gn__url_OWM = 'https://api.openweathermap.org/data/2.5/forecast?lat=23.2156&lon=72.6369&appid=c9d4c0f1fb50e0f4786f71ad1446fdc9&units=metric'
srin__url_OWM = 'https://api.openweathermap.org/data/2.5/forecast?lat=34.0837&lon=74.7973&appid=c9d4c0f1fb50e0f4786f71ad1446fdc9&units=metric'

cities_url = {
    'Bhopal_URL': bpl_url_OWM,
    'Banglore_URL': bgl_url_OWM,
    'GandhiNagar_URL': gn__url_OWM,
    'Srinagar_URL': srin__url_OWM,
}


def apic_call_Weather_Bit() -> pd.DataFrame:

    # Make the GET request
    response = requests.get(url_WB)

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

# print(api_call()['list'][1]['main'])
# print(api_call()['list'][1]['clouds'])
# print(api_call()['list'][1]['wind'])
# print(api_call()['list'][1]['dt_txt'])
# print(api_call()['hourly'][0].keys())
# print(api_call())

def api_call_Open_weather_Map(url) -> pd.DataFrame:

    # Make the GET request
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the response JSON into a Python dictionary
        data = response.json()
        # Do something with the data
        dat = data['list']
        
        # Create an empty list to store DataFrames from each dictionary
        data_frames = []

        # Loop over the 
        for iS in dat:
            # 1. Extract and Flatten Weather Data
            wind_info = iS.pop('wind')
            iS.update(wind_info)

            main_info = iS.pop('main')
            iS.update(main_info)

            cloud_info = iS.pop('clouds')
            iS.update(cloud_info)

            weather_info = iS.pop('weather')[0]
            iS.update(weather_info)

            data_frame = pd.DataFrame([iS])  # Wrap data in a list for DataFrame creation
            data_frames.append(data_frame)

        # Concatenate the DataFrames vertically (axis=0)
        df = pd.concat(data_frames, ignore_index=True)
        return df
        # return dat

    else:
        print(f'Failed to retrieve data: {response.status_code}')


# print(api_call_Open_weather_Map(cities_url['Bhopal_URL']))