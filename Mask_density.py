# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 10:59:01 2021

@author: benja
"""
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import matplotlib
from matplotlib.colors import BoundaryNorm
import dask
dask.config.set({"array.slicing.split_large_chunks": True})

#%% Skip to data for plotting after running all datasets 
## REMEMBER WS ONLY
WRF_masks = xr.open_mfdataset('D:/WRF_masks/*.nc', combine='nested', concat_dim='time')
HAR_masks = xr.open_mfdataset('D:/HAR_masks/*.nc', combine='nested', concat_dim='time')
ERA5_masks = xr.open_mfdataset('D:/ERA5_masks/*.nc', combine='nested', concat_dim='time')
GPM_masks = xr.open_mfdataset('D:/GPM_masks/*.nc', combine='nested', concat_dim='time')

#%%
WRF_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_filtered.pkl')
HAR_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_filtered.pkl')
ERA5_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_filtered.pkl')
GPM_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_filtered.pkl')
GPM_tracks['time_dt'] = pd.to_datetime(GPM_tracks.timestr)

#%% Density
"""
For all cells

"""
## Data
WRF_sm = WRF_masks['segmentation_mask'].sel(time=WRF_masks.time.isin(WRF_tracks.time))
HAR_sm = HAR_masks['segmentation_mask'].sel(time=HAR_masks.time.isin(HAR_tracks.time))
ERA5_sm = ERA5_masks['segmentation_mask'].sel(time=ERA5_masks.time.isin(ERA5_tracks.time))

## Need to convert the GPM_masks time dimension to datetime first
GPM_masks['time'] = GPM_masks.indexes['time'].to_datetimeindex()
GPM_sm = GPM_masks['segmentation_mask'].sel(time=GPM_masks.time.isin(GPM_tracks.time))

#%% Function
def get_density_all(mask, tracks):
    density = np.zeros_like(mask[0,:,:])
    for t in mask.time.values:
        tvalues = np.squeeze(mask[mask.time == t])
        for f in tracks[tracks.time == t].feature.values:
            idx = tvalues.isin(f).astype(int)
            density += idx.values
    return density

#%% Running and saving
WRF_dense = get_density_all(WRF_sm, WRF_tracks)
HAR_dense = get_density_all(HAR_sm, HAR_tracks)
ERA5_dense = get_density_all(ERA5_sm, ERA5_tracks)
GPM_dense = get_density_all(GPM_sm, GPM_tracks); GPM_dense = np.transpose(GPM_dense)

np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_dense_filtered_all', WRF_dense)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_dense_filtered_all', HAR_dense)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_dense_filtered_all', ERA5_dense)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_dense_filtered_all', GPM_dense)

#%% Data for plotting (files from get_density)
WRF_dense = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/WRF_dense_filtered_all.npy')
HAR_dense = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/HAR_dense_filtered_all.npy')
ERA5_dense = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/ERA5_dense_filtered_all.npy')
GPM_dense = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/GPM_dense_filtered_all.npy')

WRF_dense = np.ma.masked_where(WRF_dense == 0, WRF_dense); HAR_dense = np.ma.masked_where(HAR_dense == 0, HAR_dense)
ERA5_dense = np.ma.masked_where(ERA5_dense == 0, ERA5_dense); GPM_dense = np.ma.masked_where(GPM_dense == 0, GPM_dense)


dense_list = [WRF_dense, HAR_dense, ERA5_dense, GPM_dense]
del WRF_dense, HAR_dense, ERA5_dense, GPM_dense

## Lons and lats

ERA5_lon = ERA5_masks['longitude']; ERA5_lat = ERA5_masks['latitude']; ERA5_lon, ERA5_lat = np.meshgrid(ERA5_lon, ERA5_lat)
GPM_lon = GPM_masks['lon']; GPM_lat = GPM_masks['lat']; GPM_lon, GPM_lat = np.meshgrid(GPM_lon, GPM_lat)
lon_list = [WRF_masks['lon'][0], HAR_masks['lon'][0], ERA5_lon, GPM_lon]; lat_list = [WRF_masks['lat'][0], HAR_masks['lat'][0], ERA5_lat, GPM_lat]
TP_DEM = xr.open_dataset('D:/Benjamin/MasterThesis2021-2022/data/GMTED2010_15n030_0125deg.nc')
del WRF_masks, HAR_masks, ERA5_masks, GPM_masks, ERA5_lon, ERA5_lat, GPM_lon, GPM_lat

#%% Plotting
plt.style.use('classic')
ticks = [1, 100, 200,300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
bounds = np.linspace(0, 2000, 21)
ncolors = len(bounds) - 1
cmap = matplotlib.cm.get_cmap("viridis_r", ncolors)
norm = BoundaryNorm(boundaries=bounds, ncolors=ncolors)
vmax=2000
proj = ccrs.PlateCarree()
titles = ['a)','b)','c)','d)']

fig = plt.figure(figsize=(15,8.5))
for i,x in enumerate(dense_list):
    ax = fig.add_subplot(2,2,i+1, projection=proj)
    ax.set_extent([65, 105, 25, 46], proj)
    plot = ax.pcolormesh(lon_list[i], lat_list[i], x, cmap=cmap, vmax=vmax, norm=norm)
    grids = ax.gridlines(crs=proj, draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
    grids.top_labels = False; grids.right_labels = False;
    ax.contour(TP_DEM.longitude, TP_DEM.latitude, TP_DEM.elevation, [3000], cmap='Greys_r', linewidths = [1.0])
    ax.set_title(titles[i])
    
axins = fig.add_axes([0.915, 0.15, 0.015, 0.68])
cbar = fig.colorbar(plot, cax=axins, ticks=ticks, extend='max')
cbar.set_label('Number of MPSs')
cmap.set_over("cyan")
cmap.set_bad(color='white')
fig.subplots_adjust(hspace=0.06);
fig.subplots_adjust(wspace=0.09);
txt = 'WRF [a], HAR [b], ERA5 [c] & GPM [d] density of preciptation cell features between 2001 - 2016'
plt.figtext(0.12, 0.08, txt, horizontalalignment='left')
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/mask_density_filtered_all.png', dpi=100)

del bounds, ncolors, cmap, norm, vmax, proj, titles, fig, ax, plot, axins, cbar, txt, i, x, grids, ticks
#%% 
"""
For cells at intitation

"""
## trying to select cell number and time at initiation stage
def get_features(tracks,start,end): #Remember: +1 at end time
    feature = []
    time = []
    tracks_feature = pd.DataFrame()
    for y in np.arange(start, end):
        ytracks = tracks[tracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell):
            featcell = ytracks[ytracks.cell == cell].feature.values[0]
            feattime = ytracks[ytracks.cell == cell].time.values[0]
            feature.append(featcell)
            time.append(feattime)
    tracks_feature['feature'] = pd.DataFrame(feature)
    tracks_feature['time'] = pd.DataFrame(time)
    return tracks_feature

#%% Extracting initiation features
WRF_feats = get_features(WRF_tracks, 2001, 2017)
HAR_feats = get_features(HAR_tracks, 2001, 2017)
ERA5_feats = get_features(ERA5_tracks, 2001, 2017)
GPM_feats = get_features(GPM_tracks, 2001, 2017)

del WRF_tracks, HAR_tracks, ERA5_tracks, GPM_tracks

#%% Select data correseponding to tracks_feats (to get masks at initiation and during MJJAS only)
WRF_sm = WRF_masks['segmentation_mask'].sel(time=WRF_masks.time.isin(WRF_feats.time))
HAR_sm = HAR_masks['segmentation_mask'].sel(time=HAR_masks.time.isin(HAR_feats.time))
ERA5_sm = ERA5_masks['segmentation_mask'].sel(time=ERA5_masks.time.isin(ERA5_feats.time))

## Need to convert the GPM_masks time dimension to datetime first
GPM_masks['time'] = GPM_masks.indexes['time'].to_datetimeindex()
GPM_sm = GPM_masks['segmentation_mask'].sel(time=GPM_masks.time.isin(GPM_feats.time))


#%% Calculate density function for initiation stage

## Which is faster?? (This will one make an np array directly, so xr.load will be unnecessary)
def get_density(mask, track_feats):
    density = np.zeros_like(mask[0,:,:])
    for t in mask.time.values:
        tvalues = np.squeeze(mask[mask.time == t])
        for f in track_feats[track_feats.time == t].feature.values:
            idx = tvalues.isin(f).astype(int)
            density += idx.values
    return density

"""
## I think this might be it!!! (This will make an xr file so xr.load will be necessary)
def get_density(mask, track_feats):
    density = []
    for t in mask.time.values:
        tvalues = np.squeeze(mask[mask.time == t])
        for f in track_feats[track_feats.time == t].feature.values:
            idx = tvalues.isin(f).astype(int)
            density.append(idx)
    concat_dense = xr.concat(density, dim='time')
    density_sum = concat_dense.sum(dim='time')
    return density_sum
"""

#%% Running and saving
WRF_dense = get_density(WRF_sm, WRF_feats)
HAR_dense = get_density(HAR_sm, HAR_feats)
ERA5_dense = get_density(ERA5_sm, ERA5_feats)
GPM_dense = get_density(GPM_sm, GPM_feats); GPM_dense = np.transpose(GPM_dense)

## Saving takes a long time!!! (remember xarray.load() first)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_dense_filtered', WRF_dense)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_dense_filtered', HAR_dense)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_dense_filtered', ERA5_dense)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_dense_filtered', GPM_dense)

#%%
del WRF_sm, WRF_feats, HAR_sm, HAR_feats, ERA5_sm, ERA5_feats, GPM_sm, GPM_feats

#%% Data for plotting (files from get_density)
WRF_dense = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/WRF_dense_filtered.npy')
HAR_dense = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/HAR_dense_filtered.npy')
ERA5_dense = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/ERA5_dense_filtered.npy')
GPM_dense = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/GPM_dense_filtered.npy')

WRF_dense = np.ma.masked_where(WRF_dense == 0, WRF_dense); HAR_dense = np.ma.masked_where(HAR_dense == 0, HAR_dense)
ERA5_dense = np.ma.masked_where(ERA5_dense == 0, ERA5_dense); GPM_dense = np.ma.masked_where(GPM_dense == 0, GPM_dense)


dense_list = [WRF_dense, HAR_dense, ERA5_dense, GPM_dense]
del WRF_dense, HAR_dense, ERA5_dense, GPM_dense

## Lons and lats

ERA5_lon = ERA5_masks['longitude']; ERA5_lat = ERA5_masks['latitude']; ERA5_lon, ERA5_lat = np.meshgrid(ERA5_lon, ERA5_lat)
GPM_lon = GPM_masks['lon']; GPM_lat = GPM_masks['lat']; GPM_lon, GPM_lat = np.meshgrid(GPM_lon, GPM_lat)
lon_list = [WRF_masks['lon'][0], HAR_masks['lon'][0], ERA5_lon, GPM_lon]; lat_list = [WRF_masks['lat'][0], HAR_masks['lat'][0], ERA5_lat, GPM_lat]
TP_DEM = xr.open_dataset('D:/Benjamin/MasterThesis2021-2022/data/GMTED2010_15n030_0125deg.nc')
del WRF_masks, HAR_masks, ERA5_masks, GPM_masks, ERA5_lon, ERA5_lat, GPM_lon, GPM_lat

#%% Plotting
plt.style.use('classic')
ticks = [1, 10, 20,30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
bounds = np.linspace(0, 200, 21)
ncolors = len(bounds) - 1
cmap = matplotlib.cm.get_cmap("viridis_r", ncolors)
norm = BoundaryNorm(boundaries=bounds, ncolors=ncolors)
vmax=200
proj = ccrs.PlateCarree()
titles = ['a)','b)','c)','d)']

fig = plt.figure(figsize=(15,8.5))
for i,x in enumerate(dense_list):
    ax = fig.add_subplot(2,2,i+1, projection=proj)
    ax.set_extent([65, 105, 25, 46], proj)
    plot = ax.pcolormesh(lon_list[i], lat_list[i], x, cmap=cmap, vmax=vmax, norm=norm)
    grids = ax.gridlines(crs=proj, draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
    grids.top_labels = False; grids.right_labels = False;
    ax.contour(TP_DEM.longitude, TP_DEM.latitude, TP_DEM.elevation, [3000], cmap='Greys_r', linewidths = [1.0])
    ax.set_title(titles[i])
    
axins = fig.add_axes([0.915, 0.15, 0.015, 0.68])
cbar = fig.colorbar(plot, cax=axins, ticks=ticks, extend='max')
cbar.set_label('Number of MPSs')
cmap.set_over("cyan")
cmap.set_bad(color='white')
fig.subplots_adjust(hspace=0.06);
fig.subplots_adjust(wspace=0.09);
txt = 'WRF [a], HAR [b], ERA5 [c] & GPM [d] density of preciptation cells at initiation stage between 2001 - 2016'
plt.figtext(0.12, 0.08, txt, horizontalalignment='left')
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/mask_density_filtered_init.png', dpi=100)

del bounds, ncolors, cmap, norm, vmax, proj, titles, fig, ax, plot, axins, cbar, txt, i, x, grids, ticks

#%%
"""
Some plotting tools for making a discrete colorbar with centered labels at each color change
from matplotlib.colors import BoundaryNorm
## To use WRF max values
bounds = np.linspace(0, 200, 21)
ncolors = len(bounds) - 1
cmap = matplotlib.cm.get_cmap("ocean_r", ncolors)
norm = BoundaryNorm(boundaries=bounds, ncolors=ncolors)
vmax=200

#%% Above but to use ERA5/HAR max values
bounds = np.linspace(0, 600, 21)
ncolors = len(bounds) - 1
cmap = matplotlib.cm.get_cmap("ocean_r", ncolors)
norm = BoundaryNorm(boundaries=bounds, ncolors=ncolors)
vmax=600

#%%
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([65, 105, 25, 46], ccrs.PlateCarree())
fig = ax.pcolormesh(WRF_lon, WRF_lat, WRF_dense, cmap=cmap, vmax=vmax, norm=norm)
axins = inset_axes(ax,width="100%",  height="5%",loc='lower center',borderpad=-4)
cbar = plt.colorbar(fig, cax=axins, orientation='horizontal', ticks=bounds)
cbar.set_label('Precipitation cell frequency')
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
ax.contour(TP_DEM.longitude, TP_DEM.latitude, TP_DEM.elevation, [3000], cmap='Greys_r', linewidths = [1.0])


txt = 'Density of preciptation cells at initiation stage based on WRF dataset'
plt.figtext(0.12, 0.08, txt, horizontalalignment='left')
fig = plt.gcf(); fig.set_size_inches(15, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/WRF_dense.png', dpi=100)

#%%
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([65, 105, 25, 46], ccrs.PlateCarree())
fig = ax.pcolormesh(HAR_lon, HAR_lat, HAR_dense, cmap=cmap, vmax=vmax, norm=norm)
axins = inset_axes(ax,width="100%",  height="5%",loc='lower center',borderpad=-4)
cbar = plt.colorbar(fig, cax=axins, orientation='horizontal', ticks=bounds, extend='max')
cbar.set_label('Precipitation cell frequency')
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
ax.contour(TP_loni,TP_lati,TP_ele.data, [3000], cmap='Greys_r', linewidths = [2.0])
cmap.set_over("limegreen")

txt = 'Density of preciptation cells at initiation stage based on HAR dataset'
plt.figtext(0.12, 0.08, txt, horizontalalignment='left')
fig = plt.gcf(); fig.set_size_inches(15, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/HAR_dense.png', dpi=100)

#%%
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([65, 105, 25, 46], ccrs.PlateCarree())
fig = ax.pcolormesh(ERA5_lon, ERA5_lat, ERA5_dense, cmap=cmap, vmax=vmax, norm=norm)
axins = inset_axes(ax,width="100%",  height="5%",loc='lower center',borderpad=-4)
cbar = plt.colorbar(fig, cax=axins, orientation='horizontal', ticks=bounds, extend='max')
cbar.set_label('Precipitation cell frequency')
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
ax.contour(TP_loni,TP_lati,TP_ele.data, [3000], cmap='Greys_r', linewidths = [2.0])
cmap.set_over("limegreen")

txt = 'Density of preciptation cells at initiation stage based on ERA5 dataset'
plt.figtext(0.12, 0.08, txt, horizontalalignment='left')
fig = plt.gcf(); fig.set_size_inches(15, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/ERA5_dense.png', dpi=100)

#%% ALl in one plot!

fig = plt.figure(figsize=(15,8.5))
ax1 = fig.add_subplot(2,2,1, projection=ccrs.PlateCarree())
ax1.set_extent([65, 105, 25, 46], ccrs.PlateCarree())
ax1.pcolormesh(WRF_lon, WRF_lat, WRF_dense, cmap=cmap, vmax=vmax, norm=norm)
grids = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
grids.bottom_labels = False; grids.right_labels = False
ax1.contour(TP_loni,TP_lati,TP_ele.data, [3000], cmap='Greys_r', linewidths = [2.0])
ax1.set_title('a)')

ax2 = fig.add_subplot(2,2,2, projection=ccrs.PlateCarree())
ax2.set_extent([65, 105, 25, 46], ccrs.PlateCarree())
ax2.pcolormesh(HAR_lon, HAR_lat, HAR_dense, cmap=cmap, vmax=vmax, norm=norm)
grids = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
grids.bottom_labels = False; grids.left_labels = False 
ax2.contour(TP_loni,TP_lati,TP_ele.data, [3000], cmap='Greys_r', linewidths = [2.0])
ax2.set_title('b)')

ax3 = fig.add_subplot(2,2,3, projection=ccrs.PlateCarree())
ax3.set_extent([65, 105, 25, 46], ccrs.PlateCarree())
fig1 = ax3.pcolormesh(ERA5_lon, ERA5_lat, ERA5_dense, cmap=cmap, vmax=vmax, norm=norm)
grids = ax3.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
grids.top_labels = False; grids.right_labels = False
ax3.contour(TP_loni,TP_lati,TP_ele.data, [3000], cmap='Greys_r', linewidths = [2.0])
ax3.set_title('c)')

ax4 = fig.add_subplot(2,2,4, projection=ccrs.PlateCarree())
ax4.set_extent([65, 105, 25, 46], ccrs.PlateCarree())
ax4.pcolormesh(GPM_lon, GPM_lat, GPM_dense, cmap=cmap, vmax=vmax, norm=norm)
grids = ax4.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
grids.top_labels = False; grids.left_labels = False
ax4.contour(TP_loni,TP_lati,TP_ele.data, [3000], cmap='Greys_r', linewidths = [2.0])
ax4.set_title('d)')

fig.subplots_adjust(hspace=0);
fig.subplots_adjust(wspace=0.025);

axins = fig.add_axes([0.935, 0.15, 0.015, 0.68])
cbar = fig.colorbar(fig1, cax=axins, ticks=bounds, extend='max')
cbar.set_label('Precipitation cell frequency')
cmap.set_over("limegreen")

txt = 'WRF [a], HAR [b], ERA5 [c] & GPM [d] density of preciptation cells at initiation stage between 2001 - 2016'
plt.figtext(0.12, 0.08, txt, horizontalalignment='left')
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/all_mask_density.png', dpi=100)
"""

#%% TESTING

""" UNNECESSARY?? USE CARTOPY TO SET EXTEND INSTEAD??
minlon = 65; maxlon = 110; minlat = 22; maxlat = 45
mask = np.asarray((WRF_lon >= minlon) & (WRF_lon <= maxlon) & (WRF_lat >= minlat) & (WRF_lat <= maxlat)).astype(int)
del minlon, maxlon, minlat, maxlat


idx = np.where(mask == 1)
row_idx = np.arange(idx[0].min(), idx[0].max())
col_idx = np.arange(idx[1].min(), idx[1].max())
WRF_dense = WRF_dense[row_idx[0]:row_idx[-1], col_idx[0]:col_idx[-1]]#.astype(float)
WRF_lon = WRF_lon[row_idx[0]:row_idx[-1], col_idx[0]:col_idx[-1]]
WRF_lat = WRF_lat[row_idx[0]:row_idx[-1], col_idx[0]:col_idx[-1]]
#WRF_dense[WRF_dense == 0] = np.nan
del idx, row_idx, col_idx, mask


 TAKES FOREVER
WRF_dense = np.zeros([len(WRF_lat[:,1]), len(WRF_lon[0])])
for t in WRF_sm.time.values:
    tvalues = np.squeeze(WRF_sm[WRF_sm.time == t])
    for f in WRF_feats[WRF_feats.time == t].feature.values:
        idx = tvalues.isin(f).astype(int)
        WRF_dense[idx] += idx


WRF_dense = np.zeros([len(WRF_lat[:,1]), len(WRF_lon[0])])
for y in np.arange(2001,2017):
    for m in WRF_sm.time.dt.month.values:
        ym_values = np.squeeze(WRF_sm[(WRF_sm.time.dt.year == y) & (WRF_sm.time.dt.month == m)])
        f = WRF_feats[(WRF_feats.time.dt.year ==y) & (WRF_feats.time.dt.month == m)].feature
        idx = ym_values.isin(f)
        WRF_dense[idx] += 1


WRF_dense = np.zeros([len(WRF_lat[:,1]), len(WRF_lon[0])])
for f,t in zip(WRF_feats.feature, WRF_feats.time):
    tvalues = np.squeeze(WRF_sm[WRF_sm.time == t])
    idx = tvalues.isin(f)
    WRF_dense[idx] += 1


## Takes all features, even those not assigned to a cell (So wrong:))
WRF_dense = np.zeros([len(WRF_lat[:,1]), len(WRF_lon[0])])
for t in WRF_feats.time:
    idx = np.squeeze(np.asarray(WRF_sm[WRF_sm.time == t], dtype=bool))
    WRF_dense[idx] += 1
"""    

# for i in tqdm (range (24), desc="Loading..."): # Not sure if its working, try with a smaller loop



