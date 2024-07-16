import os
import sys
import csv
import json
import requests

import openmeteo_requests
import requests_cache
from retry_requests import retry

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def openMeteo_API(EndDate: str, StartDate: str = "2010-01-01") -> pd.DataFrame:
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 23.2599,
        "longitude": 77.4126,
        "start_date": StartDate,
        "end_date": EndDate,
        "hourly": ["temperature_2m", "dew_point_2m", "rain", "weather_code", "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_high", "vapour_pressure_deficit", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
        "wind_speed_unit": "kn",
    "timezone": "auto"
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    # print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    # print(f"Elevation {response.Elevation()} m asl")
    # print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    # print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_dew_point_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_rain = hourly.Variables(2).ValuesAsNumpy()
    hourly_weather_code = hourly.Variables(3).ValuesAsNumpy()
    hourly_surface_pressure = hourly.Variables(4).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(5).ValuesAsNumpy()
    hourly_cloud_cover_low = hourly.Variables(6).ValuesAsNumpy()
    hourly_cloud_cover_high = hourly.Variables(7).ValuesAsNumpy()
    hourly_vapour_pressure_deficit = hourly.Variables(8).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(9).ValuesAsNumpy()
    hourly_wind_direction_10m = hourly.Variables(10).ValuesAsNumpy()
    hourly_wind_gusts_10m = hourly.Variables(11).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["dew_point_2m"] = hourly_dew_point_2m
    hourly_data["rain"] = hourly_rain
    hourly_data["weather_code"] = hourly_weather_code
    hourly_data["surface_pressure"] = hourly_surface_pressure
    hourly_data["cloud_cover"] = hourly_cloud_cover
    hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
    hourly_data["cloud_cover_high"] = hourly_cloud_cover_high
    hourly_data["vapour_pressure_deficit"] = hourly_vapour_pressure_deficit
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
    hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m

    df = pd.DataFrame(data = hourly_data)
    df = df.set_index('date')

    return df
