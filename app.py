import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px
from helpers import curr_date, meanT, meanW, meanH

# Check if the script has been modified
if st.button("Reload App"):
    st.rerun()

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

for i, x in cities_data.items():
    x['Datetime'] = pd.to_datetime(x['Datetime'])

# Function to fetch weather data for a specific city
def get_weather_data(city):
    city_data = cities_data[city]
    return city_data  # Assuming the first row represents current data

# Streamlit app layout
st.title("Weather Forecast Dashboard")

# City selector
selected_city = st.selectbox("Select City:", list(cities_data.keys()))

# Fetch weather data for the selected city
weather_data = get_weather_data(selected_city)


# Getting the hour data
curr_24_data = weather_data.iloc[:24]
nxt_24_data = weather_data.iloc[25:48]
nxtT_24_data = weather_data.iloc[48:72]

g1, g2, g3 = st.columns(3)

st.divider()


with st.sidebar:
    # Navigation bar for different pages
    st.title("Menu")
    page_to_show = st.sidebar.radio("Page", ["Parameters", "Aerial Representation"])

col1, col2 = st.columns(2)

param_cols = []

# st.subheader(f"10 Days Forecast of {selected_city}")
# Parameters page
if page_to_show == "Parameters":

    # Current temperature display
    st.header(f"Todays Weather in {selected_city} feels like:")

    with g1:
        st.subheader(f"{curr_date()[0]}")
        g1.container(border=True).metric("Temperature", f"{float(meanT(curr_24_data)[0])}", f"{float(meanT(curr_24_data)[1])} °C")
        g1.container(border=True).metric("Wind", f"{meanW(curr_24_data)[0]} Km/h", f"{meanW(curr_24_data)[1]}%")
        g1.container(border=True).metric("Humidity", f"{meanH(curr_24_data)[0]}%", f"{meanH(curr_24_data)[1]}%")

    with g2:
        st.subheader(f"{curr_date()[1]}")
        g2.container(border=True).metric("Temperature", f"{float(meanT(nxt_24_data)[0])}", f"{float(meanT(nxt_24_data)[1])} °C")
        g2.container(border=True).metric("Wind", f"{meanW(nxt_24_data)[0]} Km/h", f"{meanW(nxt_24_data)[1]}%")
        g2.container(border=True).metric("Humidity", f"{meanH(nxt_24_data)[0]}%", f"{meanH(nxt_24_data)[1]}%")

    with g3:
        st.subheader(f"{curr_date()[2]}")
        g3.container(border=True).metric("Temperature", f"{float(meanT(nxtT_24_data)[0])}", f"{float(meanT(nxtT_24_data)[1])} °C")
        g3.container(border=True).metric("Wind", f"{meanW(nxtT_24_data)[0]} Km/h", f"{meanW(nxtT_24_data)[1]}%")
        g3.container(border=True).metric("Humidity", f"{meanH(nxtT_24_data)[0]}%", f"{meanH(nxtT_24_data)[1]}%")


    st.divider()
    st.subheader("Weather Parameters")
    

    ### FIX THIS ###
    # Check data type of weather_data
    if isinstance(weather_data, pd.Series):
        # Access individual values by index label
        temperature = weather_data['Temperature (°C)']
        humidity = weather_data['Relative Humidity (%)']
        # ... access other values as needed
        # st.write("Temperature:", temperature, "°C")
        # st.write("Relative Humidity:", humidity, "%")
        # ... write other values

    else:
        # pass
        # # Iterate through DataFrame columns
        for column in weather_data.columns:
            if column not in ['Datetime', 'Thunderstorm Occurrence']:
                param_cols.append(column)
         
    with col1:
        
        # Create line plot for wind speed
        fig1 = px.line(data_frame=weather_data, x="Datetime", y="Relative Humidity (%)", title="Relative Humiddity")
        st.plotly_chart(fig1, use_container_width=True)

        # Line Plot for precipetation
        fig2 = px.line(data_frame=weather_data, x="Datetime", y="Temperature (°C)", title="Temperature")
        st.plotly_chart(fig2, use_container_width=True)

        # Line Plot for precipetation
        fig3 = px.line(data_frame=weather_data, x="Datetime", y="Precipitation (mm)", title="Rainfall")
        st.plotly_chart(fig3, use_container_width=True)
        
        st.divider()
        st.write(param_cols[0], weather_data[param_cols[0]])
        st.write(param_cols[1], weather_data[param_cols[1]])
        st.write(param_cols[2], weather_data[param_cols[2]])

    with col2:
        # Line Plot for precipetation
        fig4 = px.line(data_frame=weather_data, x="Datetime", y="Wind Speed (m/s)", title="Wind Speed")
        st.plotly_chart(fig4, use_container_width=True)

        # Line Plot for precipetation
        fig5 = px.line(data_frame=weather_data, x="Datetime", y="Cloud Coverage (%)", title="Cloud Coverage")
        st.plotly_chart(fig5, use_container_width=True)

        # Line plot for Temperature
        fig6 = px.line(data_frame=weather_data, x="Datetime", y="Thunderstorm Occurrence", title="Thunderstorm Probability")
        st.plotly_chart(fig6, use_container_width=True)

        st.divider()
        st.write(param_cols[3], weather_data[param_cols[3]])
        st.write(param_cols[4], weather_data[param_cols[4]])
        st.write(param_cols[5], weather_data[param_cols[5]])
        
    gig = px.line(data_frame=weather_data, x="Datetime", y=weather_data["Temperature (°C)"], title="Temperature")
    
    gig.add_scatter(x=bhopal_data_df['Datetime'], y=weather_data['Temperature (°C)'], name="Bhopal")
    gig.add_scatter(x=bangalore_data_df['Datetime'], y=weather_data['Temperature (°C)'], name="Banglore")
    gig.add_scatter(x=srinagar_data_df['Datetime'], y=weather_data['Temperature (°C)'], name="Srinagar")
    gig.add_scatter(x=gandhi_nagar_data_df['Datetime'], y=weather_data['Temperature (°C)'], name="Gandhi Nagar")

    # Set the size of the plot
    gig.update_layout(width=716, height=350)
    st.plotly_chart(gig, use_container_width=True)


# Aerial representation page (example using a placeholder image)
elif page_to_show == "Aerial Representation":
    st.subheader("Aerial Representation")
    # image = Image.open("placeholder_image.jpg")  # Replace with your actual aerial image
    st.title("Coming Soon..")
    # st.image(image)
    
