# import xarray as xr
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import numpy as np 

# # Step 1: Loading Data 
# variable = 'tmp2m'
# data = ds_surf[variable].squeeze()
# temperature_celsius = data - 273.15

import streamlit as st

# Style a heading
st.markdown("""
<style>
h1 {
    color: blue;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Your content
st.title("My Styled Heading")

# Style a paragraph
st.markdown("""
<style>
p {
    font-size: 18px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

st.write("This is a styled paragraph.")
