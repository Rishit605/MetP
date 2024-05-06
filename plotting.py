import pandas as pd
import numpy  as np
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

## PARAMETERS
# Read CSV data for each city
bhopal_data_df = pd.read_csv('data/bhopal_weather_hourly_10_days.csv')
bangalore_data_df = pd.read_csv('data/Bengaluru_weather_hourly_10_days.csv')
gandhi_nagar_data_df = pd.read_csv('data/gandhinagar_weather_hourly_10_days.csv')
srinagar_data_df = pd.read_csv('data/srinagar_weather_hourly_10_days.csv')

# Combine DataFrames into a dictionary
cities_data = {
    "Bhopal": bhopal_data_df,
    "Bangalore": bangalore_data_df,
    "Gandhi Nagar": gandhi_nagar_data_df,
    "Srinagar": srinagar_data_df
}


## Line plots
def px_line_plots(dataframe, variable: str):
    return px.line(data_frame=dataframe, x="Datetime", y=variable, title=variable)

def multi_line(dataframe, variable: str):
    gig = px.line(data_frame=dataframe, x="Datetime", y=dataframe[variable], title=variable)
    
    gig.add_scatter(x=cities_data['Bhopal']['Datetime'], y=cities_data['Bhopal'][variable], name="Bhopal")
    gig.add_scatter(x=cities_data['Bangalore']['Datetime'], y=cities_data['Bangalore'][variable], name="Banglore")
    gig.add_scatter(x=cities_data['Gandhi Nagar']['Datetime'], y=cities_data['Gandhi Nagar'][variable], name="Srinagar")
    gig.add_scatter(x=cities_data['Srinagar']['Datetime'], y=cities_data['Srinagar'][variable], name="Gandhi Nagar")

    # Set the size of the plot
    gig.update_layout(width=716, height=350)
    return gig

## Generate a Meteogram
def meteogram_generator(data):

    # Convert the date column to datetime format
    data['date'] = pd.to_datetime(data['Datetime'])

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the data
    temp_line, = ax.plot(data['date'], data['Temperature (°C)'], label='Temperature', color='r')
    humidity_line, = ax.plot(data['date'], data['Relative Humidity (%)'], label='Humidity', color='g')
    wind_speed_line, = ax.plot(data['date'], data['Wind Speed (m/s)'], label='Wind Speed', color='b')

    # Add wind barbs
    wind_barbs = ax.quiver(data['date'], data['Wind Speed (m/s)'], 
                        data['Wind Speed (m/s)'] * np.cos(np.radians(data['Wind Direction (degrees)'])),
                        data['Wind Speed (m/s)'] * np.sin(np.radians(data['Wind Direction (degrees)'])),
                        pivot='middle', color='k', scale=50, linewidth=0.5)

    # Add vertical lines for specific events
    event_dates = [data['date'].iloc[7], data['date'].iloc[17], data['date'].iloc[22]]
    for event_date in event_dates:
        ax.axvline(pd.to_datetime(event_date), color='k', linestyle='--', label=f'Event on {event_date}')

    # Add filled area plots
    ax.fill_between(data['date'], data['Temperature (°C)'], color='r', alpha=0.2)
    ax.fill_between(data['date'], data['Relative Humidity (%)'], color='g', alpha=0.2)
    ax.fill_between(data['date'], data['Wind Speed (m/s)'], color='b', alpha=0.2)

    # Add dual Y-axes with different units
    ax2 = ax.twinx()
    ax2.set_ylabel('Wind Speed (m/s)')
    ax2.set_ylim(0, max(data['Wind Speed (m/s)']) * 1.1)

    # # Add hover tooltips
    # tooltip_annotations = []
    # for i, date in enumerate(data['date']):
    #     tooltip_annotations.append(f"Date: {date.strftime('%Y-%m-%d, %H')}\nTemp: {data['Temperature (°C)'][i]:.2f}°C\nHumidity: {data['Relative Humidity (%)'][i]:.2f}%\nWind Speed: {data['Wind Speed (m/s)'][i]:.2f} m/s\nWind Direction: {data['Wind Direction (degrees)'][i]}°")

    # tooltip_annotations = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
    #                                 bbox=dict(boxstyle="round", fc="w"),
    #                                 arrowprops=dict(arrowstyle="->"))
    # tooltip_annotations.set_visible(False)

    # def update_tooltip(event):
    #     x, y = event.xdata, event.ydata
    #     if x and y:
    #         index = data['date'].searchsorted(x)[0]
    #         tooltip_annotations.xy = (x, y)
    #         tooltip_annotations.set_text(tooltip_annotations[index])
    #         tooltip_annotations.set_visible(True)
    #     else:
    #         tooltip_annotations.set_visible(False)

    # fig.canvas.mpl_connect("motion_notify_event", update_tooltip)

    # Add climatology or averages
    avg_temp = data['Temperature (°C)'].mean()
    avg_humidity = data['Relative Humidity (%)'].mean()
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