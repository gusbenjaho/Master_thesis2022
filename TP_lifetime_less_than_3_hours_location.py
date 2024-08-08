# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib

#%%
WRF_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_ws.pkl')
HAR_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_ws.pkl')
GPM_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_ws.pkl')
ERA5_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_ws.pkl')
GPM_tracks['time'] = pd.to_datetime(GPM_tracks.timestr)

#%%
WRF_TP = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/TP_WRF_ws.pkl')
HAR_TP = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/TP_HAR_ws.pkl')
ERA5_TP = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/TP_ERA5_ws.pkl')
GPM_TP = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/TP_GPM_ws.pkl')

#%%
def get_lifetime_location(preciptracks, start, end):
    FMT = "%Y-%m-%d %H:%M:%S"
    tracks_lt = pd.DataFrame()
    lifetime = []; init_lon = []; init_lat = [];
    for y in np.arange(start, end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            y_initlon = ytracks[ytracks.cell == cell].longitude.values[0]
            y_initlat = ytracks[ytracks.cell == cell].latitude.values[0]
            tdelta = datetime.strptime(ytracks[ytracks.cell == cell].timestr.iloc[-1], FMT) - datetime.strptime(ytracks[ytracks.cell == cell].timestr.iloc[0], FMT)
            days, seconds = tdelta.days, tdelta.seconds
            hours = (days*24) + (seconds/3600)
            lifetime.append(hours)
            init_lon.append(y_initlon)
            init_lat.append(y_initlat)
    tracks_lt['init_lon'] = pd.DataFrame(init_lon)
    tracks_lt['init_lat'] = pd.DataFrame(init_lat)
    tracks_lt['lifetime'] = pd.DataFrame(lifetime)
    return tracks_lt


#%% Study area
WRF = get_lifetime_location(WRF_tracks, 2001, 2017); WRF['break_point'] = (WRF['lifetime'] >= 3).astype(int)
HAR = get_lifetime_location(HAR_tracks, 2001, 2017); HAR['break_point'] = (HAR['lifetime'] >= 3).astype(int)
ERA5 = get_lifetime_location(ERA5_tracks, 2001, 2017); ERA5['break_point'] = (ERA5['lifetime'] >= 3).astype(int)
GPM = get_lifetime_location(GPM_tracks, 2001, 2017); GPM['break_point'] = (GPM['lifetime'] >= 3).astype(int)
tracks_list = [WRF, HAR, ERA5, GPM]

del WRF_tracks, HAR_tracks, ERA5_tracks, GPM_tracks
#%% TP only
WRF = get_lifetime_location(WRF_TP, 2001, 2017); WRF['break_point'] = (WRF['lifetime'] >= 3).astype(int)
HAR = get_lifetime_location(HAR_TP, 2001, 2017); HAR['break_point'] = (HAR['lifetime'] >= 3).astype(int)
ERA5 = get_lifetime_location(ERA5_TP, 2001, 2017); ERA5['break_point'] = (ERA5['lifetime'] >= 3).astype(int)
GPM = get_lifetime_location(GPM_TP, 2001, 2017); GPM['break_point'] = (GPM['lifetime'] >= 3).astype(int)
tp_list = [WRF, HAR, ERA5, GPM]

del WRF_TP, HAR_TP, ERA5_TP, GPM_TP

#%% Study area plot
TP_DEM = xr.open_dataset('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/GMTED2010_15n030_0125deg.nc')
rows = 2; cols = 2; proj = ccrs.PlateCarree()
labels = ['Lifetime < 3 hours','Lifetime >= 3 hours']
cmap = matplotlib.cm.get_cmap("RdYlGn")
titles = ['a)','b)','c)','d)']

fig = plt.figure(figsize=(15,8.5))
for i, n in enumerate(tracks_list):
    ax = fig.add_subplot(rows, cols, i+1, projection=proj)
    ax.set_extent([65, 105, 25, 46], proj)
    plot = plt.scatter(x=n['init_lon'], y=n['init_lat'], c=n['break_point'], s=4, cmap=cmap, transform=proj)
    plt.legend(plot.legend_elements()[0],labels,loc='lower left')
    ax.gridlines(crs=proj, draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
    ax.contour(TP_DEM.longitude, TP_DEM.latitude, TP_DEM.elevation, [3000], cmap='Greys_r', linewidths = [1.0])
    ax.set_title(titles[i])
    
txt = 'WRF [a], HAR [b], ERA5 [c] & GPM [d] location of precipitation cells with lifetime above or below three hours between 2001 - 2016'
plt.figtext(0.125, 0.07, txt, horizontalalignment='left')
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/studyarea_lifetime_location.png', dpi=100)
del rows, cols, proj, labels, cmap, titles, fig, i, n, ax, plot, txt


#%% TP only plot
rows = 2; cols = 2; proj = ccrs.PlateCarree()
labels = ['Lifetime < 3 hours','Lifetime >= 3 hours']
cmap = matplotlib.cm.get_cmap("RdYlGn")
titles = ['a)','b)','c)','d)']

fig = plt.figure(figsize=(15,8.5))
for i, n in enumerate(tp_list):
    ax = fig.add_subplot(rows, cols, i+1, projection=proj)
    ax.set_extent([65, 105, 25, 46], proj)
    plot = plt.scatter(x=n['init_lon'], y=n['init_lat'], c=n['break_point'], s=4, cmap=cmap, transform=proj)
    plt.legend(plot.legend_elements()[0],labels,loc='lower left')
    ax.gridlines(crs=proj, draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
    ax.contour(TP_DEM.longitude, TP_DEM.latitude, TP_DEM.elevation, [3000], cmap='Greys_r', linewidths = [1.0])
    ax.set_title(titles[i])
    
txt = 'WRF [a], HAR [b], ERA5 [c] & GPM [d] location of precipitation cells over the TP with lifetime above or below three hours between 2001 - 2016'
plt.figtext(0.125, 0.07, txt, horizontalalignment='left')
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/all_lifetime_location.png', dpi=100)
del rows, cols, proj, labels, cmap, titles, fig, i, n, ax, plot, txt