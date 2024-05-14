import pandas as pd
import numpy  as np
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import plotly.graph_objects as go

import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from src.utils.helpers import get_dates

## PARAMETERS
# Read CSV data for each city
bhopal_data_df = pd.read_csv('data/hr_data_main_citites/bhopal_weather_hourly_10_days.csv')
bangalore_data_df = pd.read_csv('data/hr_data_main_citites/Bengaluru_weather_hourly_10_days.csv')
gandhi_nagar_data_df = pd.read_csv('data/hr_data_main_citites/gandhinagar_weather_hourly_10_days.csv')
srinagar_data_df = pd.read_csv('data/hr_data_main_citites/srinagar_weather_hourly_10_days.csv')

# Combine DataFrames into a dictionary
cities_data = {
    "AirField#1": bhopal_data_df,
    "AirField#2": bangalore_data_df,
    "AirField#3": gandhi_nagar_data_df,
    "AirField#4": srinagar_data_df
}


## Line plots
def px_line_plots(dataframe, variable: str):
    fig = px.line(data_frame=dataframe,
                  x=pd.Series(dataframe["Datetime"]), y=pd.Series(dataframe[variable]),
                  title=variable, markers=True)

    fig.add_bar(x=pd.Series(dataframe["Datetime"]), y=pd.Series(dataframe[variable]), showlegend=False, name=variable, text=round(pd.Series(dataframe[variable]), 2))

    fig.update_traces(textfont_size=16, marker_color='rgb(158,202,225)', marker_line_color='rgb(0,48,107)',
                  marker_line_width=2.5, opacity=0.45)

    fig.update_layout(xaxis_title = "TimeLine (Hourly)", yaxis_title = variable, width=500, height=500)
    return fig

# def multi_line(dataframe, variable: str):
#     gig = px.line(data_frame=dataframe, x="Datetime", y=dataframe[variable], title=variable)
    
#     gig.add_scatter(x=cities_data['Bhopal']['Datetime'], y=cities_data['Bhopal'][variable], name="Bhopal", mode='lines+markers')
#     gig.add_scatter(x=cities_data['Bangalore']['Datetime'], y=cities_data['Bangalore'][variable], name="Banglore", mode='lines+markers')
#     gig.add_scatter(x=cities_data['Gandhi Nagar']['Datetime'], y=cities_data['Gandhi Nagar'][variable], name="Srinagar", mode='lines+markers')
#     gig.add_scatter(x=cities_data['Srinagar']['Datetime'], y=cities_data['Srinagar'][variable], name="Gandhi Nagar", mode='lines+markers')

#     # Set the size of the plot
#     gig.update_layout(width=716, height=500)
#     return gig

# def multi_bar(variable: str):

#     for i, x in cities_data.items():
#         x['Datetime'] = pd.to_datetime(x['Datetime'])

#     bhopal_curr = cities_data['Bhopal'].loc[(cities_data['Bhopal']['Datetime'] > pd.to_datetime(get_dates()[0])) & (cities_data['Bhopal']['Datetime'] < pd.to_datetime(get_dates()[1]))]
#     bangalore_curr = cities_data['Bangalore'].loc[(cities_data['Bangalore']['Datetime'] >= pd.to_datetime(get_dates()[0])) & (cities_data['Bangalore']['Datetime'] < pd.to_datetime(get_dates()[1]))]
#     gandhinagar_curr = cities_data['Gandhi Nagar'].loc[(cities_data['Gandhi Nagar']['Datetime'] >= pd.to_datetime(get_dates()[0])) & (cities_data['Gandhi Nagar']['Datetime'] < pd.to_datetime(get_dates()[1]))]
#     srinagar_curr = cities_data['Srinagar'].loc[(cities_data['Srinagar']['Datetime'] >= pd.to_datetime(get_dates()[0])) & (cities_data['Srinagar']['Datetime'] < pd.to_datetime(get_dates()[1]))]
    
#     end_time = max(time)  # Assuming 'time' is already in datetime format
#     start_time = end_time - datetime.timedelta(hours=6)


#     plot = go.Figure(data=[
#         go.Bar(name = 'Bhopal', x=bhopal_curr['Datetime'], y=bhopal_curr[variable], text=round(pd.Series(bhopal_curr[variable]), 2)),

#         go.Bar(name = 'Bangalore', x=bangalore_curr['Datetime'], y=bangalore_curr[variable], text=round(pd.Series(bangalore_curr[variable]), 2)),

#         go.Bar(name = 'Gandhi Nagar', x=gandhinagar_curr['Datetime'], y=gandhinagar_curr[variable], text=round(pd.Series(gandhinagar_curr[variable]), 2)),

#         go.Bar(name = 'Srinagar', x=srinagar_curr['Datetime'], y=srinagar_curr[variable], text=round(pd.Series(srinagar_curr[variable]), 2))
#     ])
    
#     plot.update_layout(
#         xaxis=dict(
#             rangeselector=dict(
#                 buttons=list([
#                     dict(count=6, label="6h", step="hour", stepmode="backward"),
#                     dict(count=12, label="12h", step="hour", stepmode="backward"),
#                     dict(count=24, label="24h", step="hour", stepmode="backward"),
#                     dict(step="all")
#                 ])),
#                 rangeslider=dict(
#                     visible=True),type="date"),
#                      # Important for time serie
#         barmode='group', xaxis_title = "TimeLine (Hourly)", yaxis_title = variable, width = 1550, height=500)

#     return plot

import pandas as pd
import plotly.graph_objects as go
import datetime

def multi_bar(cities_data: dict, variable: str):

    # Preprocessing: Convert 'Datetime' column to datetime format
    for city_data in cities_data.values():
        city_data['Datetime'] = pd.to_datetime(city_data['Datetime'])

    # Filter Data for the Last 24 Hours (for reference in the range selector)
    for city_name, city_data in cities_data.items():
        cities_data[city_name] = city_data.loc[
            (city_data['Datetime'] >= (pd.to_datetime(get_dates()[0]))) & (city_data['Datetime'] < (pd.to_datetime(get_dates()[1])))
        ]

    # Calculate default start and end time (6-hour window)
    end_time = max(cities_data['AirField#1']['Datetime'])  # Assume Bhopal has the latest timestamp
    start_time = end_time - datetime.timedelta(hours=6)

    # Create Plotly figure
    fig = go.Figure()

    # Add traces for each city
    for city_name, city_data in cities_data.items():
        fig.add_trace(
            go.Bar(
                name=city_name,
                x=city_data['Datetime'],
                y=city_data[variable],
                text=round(city_data[variable], 2),  # Display rounded values as text
            )
        )

    # Update layout with custom range, rangeslider, rangeselector, and styling
    fig.update_layout(
        barmode='group',
        xaxis_title="Timeline (Hourly)",
        yaxis_title=variable,
        width=1500,
        height=750,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6h", step="hour", stepmode="backward"),
                    dict(count=12, label="12h", step="hour", stepmode="backward"),
                    dict(count=24, label="24h", step="hour", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date",
            range=[start_time, end_time]  # Set initial range
        ),
        autosize=True,
    )

    return fig


## Generate a Meteogram
def meteogram_generator(data):

    # Convert the date column to datetime format
    data['date'] = pd.to_datetime(data['Datetime'])

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the data
    temp_line, = ax.plot(data['date'], data['Temperature (°C)'], label='Temperature', color='r')
    humidity_line, = ax.plot(data['date'], data['Humidity (%)'], label='Humidity', color='g')
    wind_speed_line, = ax.plot(data['date'], data['Wind Speed (m/s)'], label='Wind Speed', color='b')

    # Add wind barbs
    wind_barbs = ax.quiver(data['date'], data['Wind Speed (m/s)'], 
                        data['Wind Speed (m/s)'] * np.cos(np.radians(data['Wind Direction (degrees)'])),
                        data['Wind Speed (m/s)'] * np.sin(np.radians(data['Wind Direction (degrees)'])),
                        pivot='middle', color='k', scale=50, linewidth=0.5)

    # Add vertical lines for specific events
    # event_dates = [data['date'].iloc[1], data['date'].iloc[3], data['date'].iloc[4]]
    event_dates = data['Datetime].sample(n=3)
    for event_date in event_dates:
        ax.axvline(pd.to_datetime(event_date), color='k', linestyle='--', label=f'Event on {event_date}')

    # Add filled area plots
    ax.fill_between(data['date'], data['Temperature (°C)'], color='r', alpha=0.2)
    ax.fill_between(data['date'], data['Humidity (%)'], color='g', alpha=0.2)
    ax.fill_between(data['date'], data['Wind Speed (m/s)'], color='b', alpha=0.2)

    # Add dual Y-axes with different units
    ax2 = ax.twinx()
    ax2.set_ylabel('Wind Speed (m/s)')
    ax2.set_ylim(0, max(data['Wind Speed (m/s)']) * 1.1)

    # Add climatology or averages
    avg_temp = data['Temperature (°C)'].mean()
    avg_humidity = data['Humidity (%)'].mean()
    avg_wind_speed = data['Wind Speed (m/s)'].mean()
    ax.axhline(avg_temp, color='r', linestyle='--', label='Avg. Temperature')
    ax.axhline(avg_humidity, color='g', linestyle='--', label='Avg. Humidity')
    ax.axhline(avg_wind_speed, color='b', linestyle='--', label='Avg. Wind Speed')

    # Customize the plot
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Meteogram')
    lines = [temp_line, humidity_line, wind_speed_line, wind_barbs]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='upper left')

    # Rotate the x-axis labels for better visibility
    plt.xticks(rotation=45)

    # Set the x-axis to display dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d, %H'))

    return plt


## Aerial Representation Image Generator
def aerial_representation(select_variable):
    # Step 3:  Visualization with Cartopy
    projection = ccrs.PlateCarree()  

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': projection})

    # Plot Temperature in Celsius
    cf = ax.contourf(temperature_celsius.grid_xt, temperature_celsius.grid_yt, 
                    temperature_celsius, transform=projection, cmap='coolwarm')

    # Map Customization
    projection = ccrs.PlateCarree()
    lonW = 97.24
    lonE = 68.7
    latS = 37.17
    latN = 8.4
    cLat = (latN + latS) / 2
    cLon = (lonW + lonE) / 2
    res = '10m'

    # fig = plt.figure(figsize=(11, 8.5))
    # ax = plt.subplot(1, 1, 1, projection=projection)
    ax.set_title('Plate Carree')
    gl = ax.gridlines(
        draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--'
    )
    ax.coastlines(resolution=res, color='black')
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='brown')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='blue')
    ax.set_extent([lonW, lonE, latS, latN], crs=projection)

    plt.colorbar(cf, label='Temperature (Celsius)', shrink=0.8)  
    plt.title('Aerial Temperature Representation') 
    
    return plt
