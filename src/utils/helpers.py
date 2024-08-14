# Distance between two poinnts on the earth
from math import radians, cos, sin, asin, sqrt
def distance_calc(lat1, lat2, lon1, lon2):
	
	# The math module contains a function named
	# radians which converts from degrees to radians.
	lon1 = radians(lon1)
	lon2 = radians(lon2)
	lat1 = radians(lat1)
	lat2 = radians(lat2)
	
	# Haversine formula 
	dlon = lon2 - lon1 
	dlat = lat2 - lat1
	a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2

	c = 2 * asin(sqrt(a)) 
	
	# Radius of earth in kilometers. Use 3956 for miles
	r = 6371
	
	# calculate the result
	return(c * r)


import math

def generate_nearby_coordinates(latitude, longitude, distance_range_km):
    # Earth radius in kilometers
    earth_radius_km = 6371

    # Convert distance range from kilometers to radians
    distance_range_rad = distance_range_km / earth_radius_km

    # Convert latitude and longitude from degrees to radians
    latitude_rad = math.radians(latitude)
    longitude_rad = math.radians(longitude)

    # Initialize list to store nearby coordinates
    nearby_coordinates = []

    # Generate nearby coordinates in all directions
    for bearing in range(0, 360, 1):  # Step by 5 degrees for adjacent directions
        # Convert bearing from degrees to radians
        bearing_rad = math.radians(bearing)

        # Calculate destination latitude and longitude using Haversine formula
        destination_latitude_rad = math.asin(math.sin(latitude_rad) * math.cos(distance_range_rad) +
                                              math.cos(latitude_rad) * math.sin(distance_range_rad) *
                                              math.cos(bearing_rad))
        destination_longitude_rad = longitude_rad + math.atan2(math.sin(bearing_rad) * math.sin(distance_range_rad) *
                                                                math.cos(latitude_rad),
                                                                math.cos(distance_range_rad) - math.sin(latitude_rad) *
                                                                math.sin(destination_latitude_rad))

        # Convert destination latitude and longitude from radians to degrees
        destination_latitude = math.degrees(destination_latitude_rad)
        destination_longitude = math.degrees(destination_longitude_rad)

        # Add nearby coordinate to the list
        nearby_coordinates.append((destination_latitude, destination_longitude))

    return nearby_coordinates

# # Example usage:
# central_latitude = 40.0583  # Latitude of New Jersey
# central_longitude = -74.4057  # Longitude of New Jersey
# distance_range_km = 10  # Distance range in kilometers

# nearby_coordinates = generate_nearby_coordinates(central_latitude, central_longitude, distance_range_km)
# print("Nearby Coordinates within {} km of New Jersey:".format(distance_range_km))
# for coord in nearby_coordinates:
#     print(coord)

import time
import calendar
import datetime
from datetime import datetime, timedelta, timezone


def get_dates():
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    next_day = today + timedelta(days=1)
    second_day = (today + timedelta(days=2)).replace(hour=0, minute=0, second=0)
    third_day = (today + timedelta(days=2)).replace(hour=23, minute=0, second=0)
    return today, next_day, second_day, third_day
	
import pandas as pd

def mean_and_delta(column):
    # Example mean temperature
    mean_val = round(column.mean(), 2)  # Example mean temperature (replace with your actual mean)

    # Calculate the difference between each temperature value and the mean
    val_diff = column - mean_val

    # Calculate the mean difference
    mean_diff = round(val_diff.mean(), 2)
    return mean_val, mean_diff

def meanT(dataa):
    mean_T = mean_and_delta(dataa['Temperature (°C)'])[0]
    delta_T = mean_and_delta(dataa['Temperature (°C)'])[1]
    return mean_T, delta_T

def meanW(dataa):
    mean_WS = mean_and_delta(dataa['Wind Speed (m/s)'])[0]
    delta_WS = mean_and_delta(dataa['Wind Speed (m/s)'])[1]
    return mean_WS, delta_WS

def meanH(dataa: pd.DataFrame):
    """
    Retuens the mean 
    """
    mean_h = mean_and_delta(dataa['Relative Humidity (%)'])[0]
    delta_h = mean_and_delta(dataa['Relative Humidity (%)'])[1]
    return mean_h, delta_h


import math

##
def calculate_relative_humidity(temperature_celsius, dew_point_celsius):
    """
    Calculate and returns the Relative Humidty in percentage.
    """
    # Constants for the Magnus-Tetens formula
    A = 17.27
    B = 237.7

    # Calculate the saturation vapor pressure (Pws) using the temperature
    def saturation_vapor_pressure(temperature):
        return 6.112 * math.exp((A * temperature) / (B + temperature))

    # Calculate the actual vapor pressure (Pw) using the dew point
    def actual_vapor_pressure(dew_point):
        return 6.112 * math.exp((A * dew_point) / (B + dew_point))

    # Calculate saturation vapor pressure and actual vapor pressure
    Pws = saturation_vapor_pressure(temperature_celsius)
    Pw = actual_vapor_pressure(dew_point_celsius)

    # Calculate relative humidity as a percentage
    relative_humidity = (Pw / Pws) * 100

    return relative_humidity


def extract_value_or_zero(x):
    if isinstance(x, dict) and '3h' in x:
        return x['3h']
    elif pd.isna(x):
        return 0
    else:
        return x