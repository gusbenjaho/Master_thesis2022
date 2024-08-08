# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 10:53:20 2021

@author: benja
"""
import glob
import pandas as pd
import numpy as np
import math
import xarray as xr

#%%
# WRF
# get data file names
path =r'C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/WRF/'
all_files = glob.glob(path + "/*.h5")

li = []
for filename in all_files:
    df = pd.read_hdf(filename, 'table')
    li.append(df)
WRF_tracks = pd.concat(li, axis=0, ignore_index=True)

del li; del df; del path; del filename; del all_files


#%%
# Loading in HAR
# get data file names
path =r'C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/HAR/'
all_files = glob.glob(path + "/*.h5")

li = []
for filename in all_files:
    df = pd.read_hdf(filename, 'table')
    li.append(df)
HAR_tracks = pd.concat(li, axis=0, ignore_index=True)

del li; del df; del path; del filename; del all_files

#%%

# ERA5

path =r'C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/ERA5/'
all_files = glob.glob(path + "/*.h5")

li = []
for filename in all_files:
    df = pd.read_hdf(filename, 'table')
    li.append(df)
ERA5_tracks = pd.concat(li, axis=0, ignore_index=True)

del li; del df; del path; del filename; del all_files

#%%
# Loading in GPM
# get data file names
path =r'C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/GPM_tracks/'
all_files = glob.glob(path + "/*.h5")

li = []
for filename in all_files:
    df = pd.read_hdf(filename, 'table')
    li.append(df)
GPM_tracks = pd.concat(li, axis=0, ignore_index=True)

del li; del df; del path; del filename; del all_files

#%%

# Remove duplicates 
WRF_tracks = WRF_tracks.drop_duplicates(subset='hdim_2'); HAR_tracks = HAR_tracks.drop_duplicates(subset='hdim_2')
GPM_tracks = GPM_tracks.drop_duplicates(subset='hdim_2'); ERA5_tracks = ERA5_tracks.drop_duplicates(subset='hdim_2')

#%% 

# Solving the missing longitudes in WRF
WRF_nc = xr.open_dataset('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/WRFOut_TP9km_HourlyP_2008_07.nc')
lons = WRF_nc.lon.values
lats = WRF_nc.lat.values
WE = WRF_tracks.longitude.values; SN = WRF_tracks.latitude.values
loni, lati = np.meshgrid(WE,SN)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# The larger loop but divided in three steps for faster execution (took 30 hours)
xlon = []
for lo in WRF_tracks.west_east.values:
    xlon_fn = int(find_nearest(WRF_nc.west_east.values, lo))
    xlon.append(xlon_fn)

xlat = []
for la in WRF_tracks.south_north.values:
    xlat_fn = int(find_nearest(WRF_nc.south_north.values, la))
    xlat.append(xlat_fn)

del xlon_fn; del xlat_fn

longitude = []
for lo in xlon:
    for la in xlat:
        lon = lons[la, lo]
    longitude.append(lon)
    

latitude = []
for lo in xlon:
    for la in xlat:
        lat = lats[la, lo]
    latitude.append(lat)
del lo; del la; del lon; del xlon; del xlat; del lons; del lats; del WRF_nc;


longitude = pd.DataFrame(longitude)
latitude = pd.DataFrame(latitude)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/longitude', longitude)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/latitude', latitude)
WRF_tracks['longitude'] = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/longitude.npy')

"""
From Julia:
    
we = 311.1235
sn = 296.13456

xlon = int(find_nearest(WRF_nc.west_east.values, we))
xlat = int(find_nearest(WRF_nc.south_north.values, sn))

lat2 = lats[xlat, xlon]
lon2 = lons[xlat, xlon]# Above loop but all at once, 800 values in 14h (!)
# Using int()
longitude2 = np.array(())
for lo in WRF_tracks.west_east.values:
    xlon = int(find_nearest(WRF_nc.west_east.values, lo))
    for la in WRF_tracks.south_north.values:
        xlat = int(find_nearest(WRF_nc.south_north.values, la))
        lon = lons[xlat, xlon]
    longitude2 = np.append(longitude2, lon)
"""

#%%

# Set the datasets to same extent (that of HAR)
minlon = 67; maxlon = 104; minlat = 26; maxlat = 44
GPM_tracks = GPM_tracks[(GPM_tracks.longitude >= minlon) & (GPM_tracks.longitude <= maxlon) & (GPM_tracks.latitude >= minlat) & (GPM_tracks.latitude <= maxlat)]
WRF_tracks = WRF_tracks[(WRF_tracks.longitude >= minlon) & (WRF_tracks.longitude <= maxlon) & (WRF_tracks.latitude >= minlat) & (WRF_tracks.latitude <= maxlat)]
HAR_tracks = HAR_tracks[(HAR_tracks.longitude >= minlon) & (HAR_tracks.longitude <= maxlon) & (HAR_tracks.latitude >= minlat) & (HAR_tracks.latitude <= maxlat)]
ERA5_tracks = ERA5_tracks[(ERA5_tracks.longitude >= minlon) & (ERA5_tracks.longitude <= maxlon) & (ERA5_tracks.latitude >= minlat) & (ERA5_tracks.latitude <= maxlat)]
del minlon; del maxlon; del minlat; del maxlat


#%% Convert ncells to km2
## HAR and WRF just use the km resolution 
WRF_tracks['km2'] = WRF_tracks.ncells*(9*9)
HAR_tracks['km2'] = HAR_tracks.ncells*(10*10)

#%%
## GPM and ERA5 data changes with latitude
## https://towardsdatascience.com/the-correct-way-to-average-the-globe-92ceecd172b7 & https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
def weighted_lat(tracks):
    lat_weight = [] ## weighting latitude
    for lat in tracks.latitude:
        weights = math.cos(np.deg2rad(lat))
        lat_weight.append(weights)
    return lat_weight

# Running for both datasets
ERA5_wl = weighted_lat(ERA5_tracks)
GPM_wl = weighted_lat(GPM_tracks)

## Converting ERA5/GPM grid_size at the equator to latitude dependent by multiplying grid_size at equator by latitude radian (see above)
ERA5_gs = 111 * 0.28125 # ERA5 (equator longitudinal distance  * resolution of dataset)
GPM_gs = 111 * 0.1
def weighted_grid(tracks_wl,tracks_gs):
    weighted_gridsize = []
    for wl in tracks_wl:
        weighted_grid = wl * tracks_gs
        weighted_gridsize.append(weighted_grid)
    return weighted_gridsize

# Running for both datasets (remember to change grid_size)
ERA5_wg = weighted_grid(ERA5_wl,ERA5_gs); del ERA5_wl; del ERA5_gs
GPM_wg = weighted_grid(GPM_wl, GPM_gs); del GPM_wl; del GPM_gs

# Converting to km2
def km2_conv(tracks_wg,tracks):
    km2 = []
    for idx,wg in enumerate(tracks_wg):
        cell_km2 = wg**2 * tracks.ncells.values[idx]
        km2.append(cell_km2)
    return km2
ERA5_tracks['km2'] = km2_conv(ERA5_wg, ERA5_tracks)
GPM_tracks['km2'] = km2_conv(GPM_wg, GPM_tracks)
del ERA5_wg; del GPM_wg
     
#%% Save the tracks

WRF_tracks.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_tracks.pkl')
HAR_tracks.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_tracks.pkl')
GPM_tracks.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_tracks.pkl')
ERA5_tracks.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_tracks.pkl')

#%% Wet season (WS) only, saved as model_ws.pkl
GPM_tracks['time'] = pd.to_datetime(GPM_tracks.timestr)
WRF_ws = WRF_tracks[(WRF_tracks.time.dt.month == 5) | (WRF_tracks.time.dt.month == 6) | (WRF_tracks.time.dt.month == 7) | (WRF_tracks.time.dt.month == 8) | (WRF_tracks.time.dt.month == 9)]
HAR_ws = HAR_tracks[(HAR_tracks.time.dt.month == 5) | (HAR_tracks.time.dt.month == 6) | (HAR_tracks.time.dt.month == 7) | (HAR_tracks.time.dt.month == 8) | (HAR_tracks.time.dt.month == 9)]
ERA5_ws = ERA5_tracks[(ERA5_tracks.time.dt.month == 5) | (ERA5_tracks.time.dt.month == 6) | (ERA5_tracks.time.dt.month == 7) | (ERA5_tracks.time.dt.month == 8) | (ERA5_tracks.time.dt.month == 9)]
GPM_ws = GPM_tracks[(GPM_tracks.time.dt.month == 5) | (GPM_tracks.time.dt.month == 6) | (GPM_tracks.time.dt.month == 7) | (GPM_tracks.time.dt.month == 8) | (GPM_tracks.time.dt.month == 9)]

#%%
WRF_ws.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_ws.pkl')
HAR_ws.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_ws.pkl')
GPM_ws.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_ws.pkl')
ERA5_ws.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_ws.pkl')

#%% 
"""
Data over the TP only
Functions and data needed

"""
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

## REMINDER: Lo/la in tracks is what you want to convert to that of lon/lat
## First, convert tracks lon/lat to their nearest points in elevation dataset:
def get_xlon(tracks,lon):
    xlon = []
    for lo in tracks.longitude.values:
        xlon_temp = find_nearest(lon, lo)
        xlon.append(xlon_temp)
    return xlon

def get_xlat(tracks,lat):
    xlat = []
    for la in tracks.latitude.values:
        xlat_temp = find_nearest(lat, la)
        xlat.append(xlat_temp)
    return xlat

## Function to get the elevation for each point (only above 3000m in this case)
## Using the above converted lon/lats to identify the elevation for each feature in tracks
def get_elevation(tracks, lon, lat, DEM):
    ele = []
    for lo, la in zip(tracks.ele_lon.values, tracks.ele_lat.values):
        idx_col = np.where(lon == lo)[0]
        idx_row = np.nonzero(lat == la)[0]
        ele_cell = DEM[idx_col, idx_row][0]
        ele.append(ele_cell)
    return ele

DEM = xr.open_dataset('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/SRTM_elevation_tp_domain.nc')
TP_DEM = DEM.__xarray_dataarray_variable__.values
TP_DEM[TP_DEM < 3000] = 0
TP_lon = DEM.lon.values
TP_lat = DEM.lat.values; del DEM

## These are the filtered tracks from 'Remove_below_3hours', this to see how many cells are lost when subsetting the TP only region
## The filtered over TP only tracks will have the x2 extension in their savings
WRF_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_filtered.pkl')
HAR_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_filtered.pkl')
GPM_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_filtered.pkl')
ERA5_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_filtered.pkl')

#%% Creating files with cells over 3000m only
WRF_tracks['ele_lon'] = get_xlon(WRF_tracks,TP_lon)
WRF_tracks['ele_lat'] = get_xlat(WRF_tracks, TP_lat)
WRF_tracks['ele'] = get_elevation(WRF_tracks, TP_lon, TP_lat, TP_DEM)
WRF_TP = WRF_tracks[WRF_tracks.ele > 0]

HAR_tracks['ele_lon'] = get_xlon(HAR_tracks,TP_lon)
HAR_tracks['ele_lat'] = get_xlat(HAR_tracks, TP_lat)
HAR_tracks['ele'] = get_elevation(HAR_tracks, TP_lon, TP_lat, TP_DEM)
HAR_TP = HAR_tracks[HAR_tracks.ele > 0]

ERA5_tracks['ele_lon'] = get_xlon(ERA5_tracks,TP_lon)
ERA5_tracks['ele_lat'] = get_xlat(ERA5_tracks, TP_lat)
ERA5_tracks['ele'] = get_elevation(ERA5_tracks, TP_lon, TP_lat, TP_DEM)
ERA5_TP = ERA5_tracks[ERA5_tracks.ele > 0]

GPM_tracks['ele_lon'] = get_xlon(GPM_tracks,TP_lon)
GPM_tracks['ele_lat'] = get_xlat(GPM_tracks, TP_lat)
GPM_tracks['ele'] = get_elevation(GPM_tracks, TP_lon, TP_lat, TP_DEM)
GPM_TP = GPM_tracks[GPM_tracks.ele > 0]
del TP_DEM, TP_lat, TP_lon, WRF_tracks, HAR_tracks, ERA5_tracks, GPM_tracks

#%% Save the tracks
WRF_TP.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_TP.pkl')
HAR_TP.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_TP.pkl')
GPM_TP.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_TP.pkl')
ERA5_TP.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_TP.pkl')

#%% WARM SEASON only
WRF_TP = WRF_TP[(WRF_TP.time.dt.month == 5) | (WRF_TP.time.dt.month == 6) | (WRF_TP.time.dt.month == 7) | (WRF_TP.time.dt.month == 8) | (WRF_TP.time.dt.month == 9)]
HAR_TP = HAR_TP[(HAR_TP.time.dt.month == 5) | (HAR_TP.time.dt.month == 6) | (HAR_TP.time.dt.month == 7) | (HAR_TP.time.dt.month == 8) | (HAR_TP.time.dt.month == 9)]
ERA5_TP = ERA5_TP[(ERA5_TP.time.dt.month == 5) | (ERA5_TP.time.dt.month == 6) | (ERA5_TP.time.dt.month == 7) | (ERA5_TP.time.dt.month == 8) | (ERA5_TP.time.dt.month == 9)]
GPM_TP = GPM_TP[(GPM_TP.time.dt.month == 5) | (GPM_TP.time.dt.month == 6) | (GPM_TP.time.dt.month == 7) | (GPM_TP.time.dt.month == 8) | (GPM_TP.time.dt.month == 9)]

#%% Save the tracks
WRF_TP.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/TP_WRF_filtered_x2.pkl')
HAR_TP.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/TP_HAR_filtered_x2.pkl')
GPM_TP.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/TP_GPM_filtered_x2.pkl')
ERA5_TP.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/TP_ERA5_filtered_x2.pkl')