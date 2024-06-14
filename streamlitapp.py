# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from PIL import Image

# # Set page configuration
# st.set_page_config(layout="wide")

# # Load data from CSV
# weather_data = pd.read_csv(r"C:\Projs\COde\Meteo\MetP\data\bhopal_weather_hourly_10_days.csv")
# weather_data['Longitude'] = 77.4126
# weather_data['Latitude'] = 23.2599
# weather_data['Datetime'] = pd.to_datetime(weather_data['Datetime'])
# # print(weather_data)

# # Create the main container
# container = st.container()

# # Add the heading
# with container:
#     st.markdown("<h1 style='text-align: center;'>Extreme Weather Forecasting</h1>", unsafe_allow_html=True)
#     st.markdown("---")

# # Create columns for the menu toggle, current, and next sections
# col1, col2, col3 = st.columns([1, 2, 1])

# # # Menu toggle section
# # with col1:
# #     st.markdown("<h3 style='text-align: center;'>MENU TOGGLE</h3>", unsafe_allow_html=True)


# # Current section
# with col2:
#     st.markdown("<h3 style='text-align: center;'>CURRENT</h3>", unsafe_allow_html=True)
#     st.markdown("---")

#     # City selector
#     # city_selector = st.selectbox("City Selector")#, weather_data["city"].unique())

#     # Replace city selector with default text (choose your desired city)
#     default_city = "Bhopal"  # Modify this to your preferred default city
#     st.write(f"Current City: {default_city}")


#     # Graph representation
#     st.markdown("<h4 style='text-align: center;'>GRAPHS REPRESENTATION</h4>", unsafe_allow_html=True)
#     col4, col5 = st.columns(2)

#     with col4:
#         # # Filter data based on selected city
#         # filtered_data = weather_data[weather_data["city"] == city_selector]
#         filtered_data = weather_data

#         ts = weather_data.index

#         # # Create line plot for temperature
#         # fig1 = px.line(filtered_data, x=ts, y="temp", title="Temperature")
#         # st.plotly_chart(fig1, use_container_width=True)
#         # Filter data based on selected location
#         filtered_data = weather_data
        

#         # Create temperature heatmap
#         fig1 = px.imshow(filtered_data.pivot_table(index='Datetime', columns='Latitude', values='Temperature (°C)'),
#                          labels=dict(x="Latitude", y="Longitude", color="Temperature (°C)"),
#                          title="Temperature Heatmap",
#                          color_continuous_scale='Plasma')
#         st.plotly_chart(fig1, use_container_width=True)


#         # Create line plot for wind speed
#         fig3 = px.line(filtered_data, x=ts, y="wind_spd", title="Wind Speed")
#         st.plotly_chart(fig3, use_container_width=True)

#     with col5:
#         # Create line plot for humidity
#         fig2 = px.line(filtered_data, x=ts, y="rh", title="Humidity")
#         st.plotly_chart(fig2, use_container_width=True)

#         # Create line plot for precipitation
#         fig4 = px.line(filtered_data, x=ts, y="precip", title="Precipitation")
#         st.plotly_chart(fig4, use_container_width=True)


# # Next section
# with col3:
#     st.markdown("<h3 style='text-align: center;'>NEXT</h3>", unsafe_allow_html=True)

# # Weather parameters representation
# st.markdown("<h3 style='text-align: left;'>WEATHER PARAMETERS REPRESENTATION</h3>", unsafe_allow_html=True)


# # Load the image
# image = Image.open("WRF2kmRAIN.png")

# # Set desired width and height (or use aspect ratio)
# width = 800  # Adjust these values as needed
# height = 800

# # Resize the image (maintains aspect ratio by default)
# resized_image = image.resize((width, height))

# # # Display the resized image
# # st.image(resized_image, caption="Aerial Image (resized)", use_column_width=True)

# # Aerial representation
# st.markdown("<h3 style='text-align: left;'>AERIAL REPRESENTATION</h3>", unsafe_allow_html=True)
# st.image(resized_image, caption="MP Weather", use_column_width=False)



# import streamlit as st
# import pandas as pd
# import plotly.express as px

# # Set page configuration
# st.set_page_config(layout="wide")

# # Load data from CSV
# weather_data = pd.read_csv("weather_viz.csv")

# # Create the main container
# container = st.container()

# # Add the heading
# with container:
#   st.markdown("<h1 style='text-align: center;'>HEADING</h1>", unsafe_allow_html=True)
#   st.markdown("---")

# # Create columns for menu toggle (optional) and buttons
# col1, col2 = st.columns([1, 2])  # Adjust column sizes if needed

# # Menu toggle section (optional)
# # with col1:
# #   st.markdown("<h3 style='text-align: center;'>MENU TOGGLE</h3>", unsafe_allow_html=True)

# # Current and Next buttons (toggleable sections)
# current_state = st.button("Current")  # Create a button for toggling current section
# next_state = st.button("Next")  # Create a button for toggling next section

# # Empty container for content (will be filled based on button clicks)
# content_container = st.empty()

# # Function to display graphs based on selected section and graph type
# def display_graphs(section, graph_type):
#   filtered_data = weather_data  # Default: use all data (modify for filtering if needed)
#   # You can add logic here to filter data based on selected city or other criteria
#   ts = weather_data.index

#   if section == "Current":
#     if graph_type == "Temperature":
#       fig = px.line(filtered_data, x=ts, y="temp", title="Temperature")
#     elif graph_type == "Wind Speed":
#       fig = px.line(filtered_data, x=ts, y="wind_spd", title="Wind Speed")
#     # ... Add logic for other graph types for current section
#   elif section == "Next":
#     # Add logic for graphs in the next section (modify placeholders)
#     if graph_type == "Precipitation Chance":
#       fig = px.bar(filtered_data, x=ts, y="precip", title="Precipitation Chance")
#     elif graph_type == "Humidity":
#       fig = px.line(filtered_data, x=ts, y="rh", title="Humidity")
#     # ... Add logic for other graph types for next section
#   else:
#     # Handle case of unexpected section
#     fig = None

#   # Display the graph if a valid selection is made
#   if fig is not None:
#     content_container.plotly_chart(fig, use_container_width=True)

# # Handle button clicks to display relevant graphs
# if current_state:
#   # Display options for current section graphs
#   graph_type = st.selectbox("Select Current Weather Graph", ["Temperature", "Wind Speed", "Humidity"])  # Add more options
#   display_graphs("Current", graph_type)

# if next_state:
#   # Display options for next section graphs
#   graph_type = st.selectbox("Select Next Weather Graph", ["Precipitation Chance", "Humidity"])  # Add more options
#   display_graphs("Next", graph_type)

# # Aerial representation
# st.markdown("<h3 style='text-align: left;'>AERIAL REPRESENTATION</h3>", unsafe_allow_html=True)
# st.image("aerial_image.png", caption="Images", use_column_width=True)

# # First, import the elements you need
# from streamlit_elements import elements, mui, html

import streamlit as st
import pandas as pd
import json

# Sample Weather Data (replace with your actual data)
data = {
    "date": pd.date_range(start="2024-06-10", end="2024-06-16", freq="H"),
    "temperature": [28, 29, 31, 33, 35, 34, 32, 30, 29, 28, 27, 26, 25, 26, 28, 30, 32, 34, 36, 37, 35, 33, 32, 31] * 7,
    "precipitation": [0, 0, 0.2, 0.5, 1.2, 0.8, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * 7,
    "wind_speed": [5, 6, 8, 10, 12, 11, 9, 7, 6, 5, 4, 3, 2, 3, 4, 5, 6, 7, 8, 9, 10, 8, 7, 6] * 7,
}
df = pd.DataFrame(data)


st.title("Weather Meteogram - Bhopal, India")

# Generate data for Nivo chart
nivo_data = df.to_dict(orient='records')

# Nivo chart configuration in JSON format
nivo_props = {
    "data": nivo_data,
    "xScale": {"type": "time", "format": "%Y-%m-%d"},
    "xFormat": "time:%Y-%m-%d",
    "yScale": {"type": "linear", "min": "auto", "max": "auto"},
    "axisBottom": {"format": "%b %d"},
    "axisLeft": {"orient": "left", "tickSize": 5, "tickPadding": 5, "tickRotation": 0},
    "useMesh": True,
}

# HTML code for embedding Nivo chart (adjust width and height as needed)
html_code = f"""
<div id="chart"></div>
<script type="text/javascript">
  // Load Nivo library (you'll need to include the actual script tags here)
  
  const data = {json.dumps(nivo_data)};
  const props = {json.dumps(nivo_props)};

  // Create the Nivo chart
  nivo.render(document.getElementById('chart'), nivo.LineChart, data, props);
</script>
"""

# Display the chart in Streamlit
st.components.v1.html(html_code, width=800, height=400)