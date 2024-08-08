# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 09:03:04 2021

@author: benja
"""

import pandas as pd
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

ERA5_mask = nc.Dataset('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/ERA5/Mask_segmentation_200807.nc')
ERA5_2000 = pd.read_hdf('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/ERA5/Tracks_2000.h5', 'table')

ERA5_lat = ERA5_mask['latitude'][:]
ERA5_lon = ERA5_mask['longitude'][:]
ERA5_time = ERA5_mask['time'][:]
ERA5_SM = ERA5_mask['segmentation_mask'][350,:,:] # needs to be in 2-d, so now it is time(1) of all lats and lons

HAR_ds = nc.Dataset('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/HAR v2_d10km_h_2d_prcp_2008.nc')
HAR_lon = HAR_ds['lon'][:]
HAR_lat = HAR_ds['lat'][:]
HAR_prcp = HAR_ds['prcp'][350,:,:]
HAR_2000 = pd.read_hdf('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/HAR_tracks/Tracks_2000.h5', 'table')

WRF_ds = nc.Dataset('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/WRFOut_TP9km_HourlyP_2008_07.nc')
WRF_lon = WRF_ds['lon'][:]
WRF_lat = WRF_ds['lat'][:]
WRF_prcp = WRF_ds['Prep'][350,:,:]
WRF_2000 = pd.read_hdf('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/WRF_tracks/Tracks_2000.h5', 'table')
WRF_2001 = pd.read_hdf('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/WRF_tracks/Tracks_2001.h5', 'table')

# %% Plotting
# Meshgridding the lon and lats to 2-D
ERA5_loni, ERA5_lati = np.meshgrid(ERA5_lon, ERA5_lat)

plt.pcolor(HAR_lat, HAR_lon, HAR_prcp)
# Basic plotting of seg_mask in 2-D
plt.contour(ERA5_loni,ERA5_lati,ERA5_SM)
#plt.colorbar()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xticks(np.arange(60, 140, 20))
plt.yticks(np.arange(10, 50, 10))
plt.xlim(60,140)
plt.ylim(10, 50)

# %%
plt.pcolor(WRF_lon, WRF_lat, WRF_prcp)
plt.colorbar()
plt.contour(ERA5_loni,ERA5_lati,ERA5_SM)

# %%
plt.pcolor(HAR_lon, HAR_lat, HAR_prcp)
plt.colorbar()
plt.contour(ERA5_loni,ERA5_lati,ERA5_SM)

# %%
"""
Some indexing:
    Picking out cell nr 1
    ERA5_2000_cell1 = ERA5_2000[ERA5_2000.cell == 1]
    HAR_2000_cell1 = HAR_2000[HAR_2000.cell ==1]
    WRF_2000_cell1 = WRF_2000[WRF_2000.cell ==1]
    
    Above threshold 5
    TV5 = ERA5_2000[ERA5_2000.threshold_value > 5]
    
    Index certain dates/months
    june = WRF_2000[WRF_2000.time.dt.month == 6]
    first = WRF_2000[WRF_2000.time.dt.day == 1]
    FirstofJan = WRF_2000[(WRF_2000.cell == 1 ) & (WRF_2000.time.dt.month == 1)]
    SeventhJJA = WRF_2000[(WRF_2000.time.dt.day == 7 ) & (WRF_2000.time.dt.month == 6) | (WRF_2000.time.dt.day == 7 ) & (WRF_2000.time.dt.month == 7) | (WRF_2000.time.dt.day == 7 ) & (WRF_2000.time.dt.month == 8)]
    
    """
#%%

"""
    Concatenating H5 files ( Creates an enourmous 2D table)
    WRF_tracks = pd.concat([WRF_2000, WRF_2001])
    
    Daily average of tv, 1st of January in entire dataframe (should be accurate):
        Daily_Values = WRF_tracks.threshold_value[(WRF_tracks.time.dt.month == 1) & (WRF_tracks.time.dt.day == 1)].mean()
"""
#%% 
""" Maybe this is the way to go?? It returns the daily mean of each day, each year. (Does it ignore NaN though?)

Step by step tfrom hourly to annual

1. WRF_tracks.set_index('time', inplace=True) (To reset the time as column again WRF_tracks.reset_index('time', inplace=True))
2. daily = WRF_tracks.threshold_value.resample('d').mean()
    To make it a DataFrame with time as column:
    daily = daily.to_frame().reset_index()
3. monthly = daily.resample('m').mean()
4. annual = monthly.resample('y').mean()

Next step would be to calculate daily mean over years, to find out the diurnal cycles over time
E.g. Jan1 = daily.threshold_value[(daily.time.dt.month == 1) & (daily.time.dt.day == 1)].mean()
Using Groupby:
    Annual_dm = WRF_dm.groupby([WRF_dm.index.month, WRF_dm.index.day]).mean() (Make sure you have time as index column)

This might be useful for group by cells=
https://pbpython.com/pandas-grouper-agg.html