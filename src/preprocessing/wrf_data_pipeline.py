import os
import ecmwflibs
import numpy as np
import pandas as pd
import xarray as xr
from cfgrib.xarray_store import open_dataset

from utils.helpers import distance_calc


grib_file = r'C:\Projs\COde\Meteo\MetP\data\gfs.0p25.2015011500.f000.grib2'

KEYS = ['surface', 'atmosphereSingleLayer', 'atmosphere']

# Loading the Dataset
# class GRIB2_Dataset():
def load_grib2() -> pd.DataFrame:
    """
    This function runs a loop for the parameter arguments from the KEYS list and iterates thought each key feeding into the xarray dataset argumnets and fetches all the varialbes from the whole dataset.
    """
    count = 0
    new_grib_dataframe = pd.DataFrame()

    for i in KEYS:
        try:
            grib_dataset = open_dataset(grib_file,
                                    engine='cfgrib',
                                    backend_kwargs={'filter_by_keys': {'typeOfLevel': f'{i}'}})
            
            if len(grib_dataset.sizes) != 0:
                grib_dataframe = grib_dataset.to_dataframe()
                print(f"\nFetched and converted\n {grib_dataframe} Dataframe of the key {i}.\n")
  
                if count == 0:
                    new_grib_dataframe= pd.concat([new_grib_dataframe, grib_dataframe], axis=1)
                    print(f"First dataframe: {new_grib_dataframe.shape}.\n")

                elif count > 0:
                    try: 
                        filtered_grib_dataframe = grib_dataframe.iloc[:, 4:]
                        
                        new_grib_dataframe = pd.concat([new_grib_dataframe, filtered_grib_dataframe], axis=1)
                    except pd.errors.DataError as e:
                        print(f"{e}: Dataset is either empty or not correctly converted. Please Check the Dataset")
            else:
                print(f"Dataset with '{i}'\nKey Is Empty\n")
                pass

            count += 1
        except KeyError as e:
            print(f"{e}: The given parameter key is not present in the dataset.")
            pass
    return new_grib_dataframe

# def load_grib2():
#     """
#     This function loads a grib2 file and combines DataArrays based on specified keys.
#     """
#     try:
#         grib_ds = open_dataset(grib_file,
#                                 engine='cfgrib',
#                                 backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}}
#                                 )
#         return grib_ds
#     except KeyError as e:
#         print(f"{e}: The given parameter key is not present in the dataset.")
#         pass
# print(load_grib2())

def dataF() -> pd.DataFrame:
    """
    Takes the loaded Gridded dataset in xarray format, converts it to pandas dataframe, resets its index and sets the timestamp as its index.

    Returns:
        wrf_df: A Pandas Dataframe with the converted gridded dataset.
    """
    try:
        wrf_dataframe = load_grib2()
        wrf_dataframe = wrf_dataframe.reset_index()
        wrf_dataframe = wrf_dataframe.set_index('time')
        return wrf_dataframe
    except AttributeError as e:
        print(f"{e}: The Gridded Data was not correctly loaded. Please check the Dataset file and its format.")
        pass

def wrf_data_BPL() -> pd.DataFrame:
    # Creating a new column for writing the calculating the distance
    data_wrf = dataF()
    data_wrf['distance'] = data_wrf.apply(lambda row: distance_calc(row['latitude'], 23.25, row['longitude'], 77.41), axis=1)
    
    # data_wrf = data_wrf.drop(columns=['siconc', 'lsm', 'unknown', '4lftx', 'SUNSD', 'fldcp', 'wilt', 'sde', 'sdwe', 'hindex'])

    return data_wrf.loc[data_wrf['distance'] < 50]

print(wrf_data_BPL())
