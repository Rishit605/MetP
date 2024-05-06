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

def curr_date():
	now = time.localtime()

	# Get the year and month.
	year = now.tm_year
	month = now.tm_mon
	day = now.tm_mday

	# Create a list to store the dates
	dates = []

    # Add the current date to the list
	current_date = datetime.date(year, month, day)
	dates.append(current_date)

    # Add the dates of the next two days to the list
	for i in range(1, 3):
		next_date = current_date + datetime.timedelta(days=i)
		dates.append(next_date)
		# print(next_date)

	return dates
	
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

def meanH(dataa):
    mean_h = mean_and_delta(dataa['Relative Humidity (%)'])[0]
    delta_h = mean_and_delta(dataa['Relative Humidity (%)'])[1]
    return mean_h, delta_h