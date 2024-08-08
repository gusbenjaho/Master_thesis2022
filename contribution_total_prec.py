# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 11:25:08 2021

@author: benja
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd

import dask
dask.config.set({"array.slicing.split_large_chunks": True})

#%% 
"""
Contribution to total precipitation
WRF is in ??
HAR is in mm/h
ERA5 is in m/h

"""
#%% WRF - data prep
WRF_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_filtered.pkl')
WRF_prec = xr.open_mfdataset('D:/WRF_prec/*.nc'); WRF_prec = WRF_prec.Prep.sel(time=WRF_prec.time.isin(WRF_tracks.time))
WRF_masks = xr.open_mfdataset('D:/WRF_masks/*.nc', combine='nested', concat_dim='time')
WRF_masks = WRF_masks.sel(time=WRF_masks.time.isin(WRF_tracks.time)); WRF_masks = WRF_masks['segmentation_mask']

#%% HAR - data prep
HAR_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_filtered.pkl')
HAR_prec = xr.open_mfdataset('D:/HAR_prec/*.nc'); HAR_prec = HAR_prec.prcp.sel(time=HAR_prec.time.isin(HAR_tracks.time))
HAR_masks = xr.open_mfdataset('D:/HAR_masks/*.nc', combine='nested', concat_dim='time')
HAR_masks = HAR_masks.sel(time=HAR_masks.time.isin(HAR_tracks.time)); HAR_masks = HAR_masks['segmentation_mask']

#%% ERA5  - data prep
ERA5_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_filtered.pkl')
ERA5_prec = xr.open_mfdataset('D:/ERA5_prec/*.nc'); ERA5_prec = ERA5_prec.tp *1000; ERA5_prec = ERA5_prec.sel(time=ERA5_prec.time.isin(ERA5_tracks.time))
ERA5_masks = xr.open_mfdataset('D:/ERA5_masks/*.nc', combine='nested', concat_dim='time')
ERA5_masks = ERA5_masks.sel(time=ERA5_masks.time.isin(ERA5_tracks.time)); ERA5_masks = ERA5_masks['segmentation_mask']
ERA5_prec = ERA5_prec.sel(longitude=slice(ERA5_masks.longitude.min(), ERA5_masks.longitude.max()), latitude=slice(ERA5_masks.latitude.max(), ERA5_masks.latitude.min()))

#%% Summing up precipitation where there have been precipitation cells (using mask file)
## For GPM
GPM_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_filtered.pkl')
mask_prec = []
for y in np.arange(2001,2017):
    yprec = xr.open_dataset('D:/GPM_prec/GPM_'+str(y)+'.nc4_prec.nc4'); 
    yprec['time'] = yprec.indexes['time'].to_datetimeindex()
    ymasks = xr.open_mfdataset('D:/GPM_masks/Mask_segmentation_'+str(y)+'*.nc'); 
    ymasks['time'] = ymasks.indexes['time'].to_datetimeindex()
    ymasks = ymasks.sel(lon=slice(yprec.lon.min(), yprec.lon.max()), lat=slice(yprec.lat.min(), yprec.lat.max()))
    ymasks = ymasks.segmentation_mask.sel(time=ymasks.time.isin(GPM_tracks.time)); 
    yprec = yprec.precipitationCal.sel(time=yprec.time.isin(ymasks.time))
    for t in ymasks.time.values:
        tfeat = GPM_tracks[GPM_tracks.time == t].feature
        tmask = np.squeeze(ymasks[ymasks.time == t])
        idx = tmask.isin(tfeat)
        tprec = np.squeeze(yprec[yprec.time == t])
        prec = tprec.where(idx,0)
        mask_prec.append(prec)
concat_prec = xr.concat(mask_prec, dim='time')
GPM_mask_prec = concat_prec.sum(dim='time')
del t, tmask, idx, tprec, prec, mask_prec, concat_prec, tfeat, ymasks, yprec, y, GPM_tracks
GPM_mask_prec.load()
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_mask_prec_filtered', GPM_mask_prec)

## FOR WRF, HAR, ERA5
mask_prec = []
for t in ERA5_masks.time.values:
    tfeat = ERA5_tracks[ERA5_tracks.time == t].feature
    tmask = np.squeeze(ERA5_masks[ERA5_masks.time == t])
    idx = tmask.isin(tfeat)
    tprec = np.squeeze(ERA5_prec[ERA5_prec.time == t])
    prec = tprec.where(idx,0)
    mask_prec.append(prec)
concat_prec = xr.concat(mask_prec, dim='time')
ERA5_mask_prec = concat_prec.sum(dim='time')
del t, tmask, idx, tprec, prec, mask_prec, concat_prec, tfeat, ERA5_prec, ERA5_masks, ERA5_tracks


"""
## For GPM (ONLY RETURNS NANS)
mask_prec = np.zeros([600,350])
for t in GPM_masks.time.values:
    tfeat = GPM_tracks[GPM_tracks.time == t].feature
    tmask = GPM_masks[GPM_masks.time == t].squeeze()
    idx = tmask.isin(tfeat)
    tprec = GPM_prec[GPM_prec.time == t].squeeze()
    prec = tprec.where(idx)
    mask_prec += np.array(prec)
GPM_mask_prec = np.transpose(mask_prec)
del t, tmask, idx, tprec, prec, mask_prec, tfeat

## SUPER DUPER MEGA SLOW
HAR_mask_prec = np.zeros_like(HAR[0,:,:])
for tmask,t in zip(WRF,WRF.time.values):
    idx = ~tmask.isin(tmask >= 1)
    tprec = np.squeeze(HAR_prec[HAR_prec.time == t])
    prec = tprec.where(idx,0)
    HAR_mask_prec = HAR_mask_prec + prec.values
del t, tmask, idx, tprec, prec

HAR_mask_prec = []
for tmask,t in zip(HAR,HAR.time.values):
    idx = ~tmask.isin(tmask >= 1)
    tprec = np.squeeze(HAR_prec[HAR_prec.time == t])
    prec = tprec.where(idx,0)
    HAR_mask_prec.append(prec)
concat_prec = xr.concat(HAR_mask_prec, dim='time')
HAR_mask_prec = concat_prec.sum(dim='time')
del t, tmask, idx, tprec, prec, concat_prec


## Something wrong when using the function :( 
## ValueError: 0d-boolean array is used for indexing along dimension 'latitude', but only 1d boolean arrays are supported.
def get_masked_prec(mask, prec):
    mask_prec = np.zeros_like(mask[0,:,:])
    for t in mask.time.values:
        tmask = np.squeeze(mask[mask.time == t])
        idx = ~tmask.isin(tmask >= 1)
        tprec = np.squeeze(prec[prec.time == t])
        prec = tprec.where(idx,0)
        mask_prec = mask_prec + prec.values
    return mask_prec

"""

#%% Using precipitation from ALL timesteps to divide mask_prec with
WRF_prec = xr.open_mfdataset('D:/WRF_prec/*.nc')
HAR_prec = xr.open_mfdataset('D:/HAR_prec/*.nc')
HAR_prec = HAR_prec.sel(time=HAR_prec.time.dt.month.isin([5,6,7,8,9]))
ERA5_prec = xr.open_mfdataset('D:/ERA5_prec/*.nc').tp.sel(longitude=slice(ERA5_mask_prec.longitude.min(), ERA5_mask_prec.longitude.max()), latitude=slice(ERA5_mask_prec.latitude.max(), ERA5_mask_prec.latitude.min()))

WRF_sum_prec = WRF_prec.Prep.sum(dim='time'); WRF_contr_prec = WRF_mask_prec/WRF_sum_prec * 100
HAR_sum_prec = HAR_prec.prcp.sum(dim='time'); HAR_contr_prec = HAR_mask_prec/HAR_sum_prec * 100
ERA5_sum_prec = ERA5_prec.sum(dim='time')*1000 ; ERA5_contr_prec = ERA5_mask_prec/ERA5_sum_prec * 100
del ERA5_prec, ERA5_sum_prec, ERA5_mask_prec

np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_contr_prec_filtered', WRF_contr_prec)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_contr_prec_filtered', HAR_contr_prec)
ERA5_contr_prec.load()
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_contr_prec_filtered', ERA5_contr_prec)

#%% For GPM
GPM_sum_prec = np.zeros([600,350])
for y in np.arange(2001,2017): 
    yprec = xr.open_dataset('D:/GPM_prec/GPM_'+str(y)+'.nc4_prec.nc4'); 
    yprec['time'] = yprec.indexes['time'].to_datetimeindex()
    ysum = np.squeeze(yprec.precipitationCal.sum(dim='time'))
    GPM_sum_prec += np.array(ysum)
del y, yprec, ysum
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_sum_prec', GPM_sum_prec)

GPM_mask_prec = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_mask_prec_filtered.npy')
GPM_contr_prec = GPM_mask_prec/GPM_sum_prec * 100; GPM_contr_prec = np.transpose(GPM_contr_prec)
del GPM_sum_prec, GPM_mask_prec
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_contr_prec_filtered', GPM_contr_prec)


#%% Loading the above + lon/lats
WRF_contr_prec = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/WRF_contr_prec_filtered.npy'); WRF_contr_prec = np.ma.masked_where(WRF_contr_prec == 0, WRF_contr_prec)
HAR_contr_prec = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/HAR_contr_prec_filtered.npy'); HAR_contr_prec = np.ma.masked_where(HAR_contr_prec == 0, HAR_contr_prec)
ERA5_contr_prec = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/ERA5_contr_prec_filtered.npy'); ERA5_contr_prec = np.ma.masked_where(ERA5_contr_prec == 0, ERA5_contr_prec)
GPM_contr_prec = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/GPM_contr_prec_filtered.npy'); GPM_contr_prec = np.ma.masked_where(GPM_contr_prec == 0, GPM_contr_prec)

WRF_lon = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/WRF_lon.npy')
WRF_lat = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/WRF_lat.npy')
HAR_lon = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/HAR_lon.npy')
HAR_lat = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/HAR_lat.npy')
ERA5_lon = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/ERA5_lon_contr_prec.npy')
ERA5_lat = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/ERA5_lat_contr_prec.npy')
GPM_lon = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/GPM_lon.npy')
GPM_lat = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/GPM_lat.npy')

prec_list = [WRF_contr_prec, HAR_contr_prec, ERA5_contr_prec, GPM_contr_prec]
lon_list = [WRF_lon, HAR_lon, ERA5_lon, GPM_lon]; lat_list = [WRF_lat, HAR_lat, ERA5_lat, GPM_lat]

del WRF_contr_prec, HAR_contr_prec, ERA5_contr_prec, GPM_contr_prec,WRF_lon, HAR_lon, ERA5_lon, GPM_lon,WRF_lat, HAR_lat, ERA5_lat, GPM_lat
TP_DEM = xr.open_dataset('D:/Benjamin/MasterThesis2021-2022/data/GMTED2010_15n030_0125deg.nc')

#%% 
## Plotting necessities
bounds = np.linspace(0, 60, 13); ncolors = len(bounds) - 1
cmap = matplotlib.cm.get_cmap('viridis_r', ncolors)
norm = BoundaryNorm(boundaries=bounds, ncolors=ncolors)
vmax=100; titles = ['a)','b)','c)','d)']; proj = ccrs.PlateCarree()

## Plotting
fig = plt.figure(figsize=(15,8.5))
for i,x in enumerate(prec_list):
    ax = fig.add_subplot(2,2,i+1, projection=proj)
    ax.set_extent([67, 104, 26, 44], proj)
    plot = ax.pcolormesh(lon_list[i], lat_list[i], x, cmap=cmap, vmax=vmax, norm=norm)
    grids = ax.gridlines(crs=proj, draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
    grids.top_labels = False; grids.right_labels = False;
    ax.contour(TP_DEM.longitude, TP_DEM.latitude, TP_DEM.elevation, [3000], cmap='Greys_r', linewidths = [1.0])
    ax.set_title(titles[i])

axins = fig.add_axes([0.935, 0.15, 0.015, 0.68])
cbar = fig.colorbar(plot, cax=axins, ticks=bounds, extend='max')
cmap.set_over("cyan")
cmap.set_bad(color='white')
cbar.set_label('MPSs contribution to total MJJAS precipitation [%]')
fig.subplots_adjust(hspace=0.06);
fig.subplots_adjust(wspace=0.11);
txt = 'WRF [a], HAR [b], ERA5 [c] & GPM [d] precipitation cell contribution to total precipitation between 2001 - 2016'
plt.figtext(0.12, 0.08, txt, horizontalalignment='left')
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/all_contr_prec_filtered.png', dpi=100)

del bounds, ncolors, cmap, norm, vmax, proj, titles, fig, ax, plot, axins, cbar, txt, x, i, grids
    

#%% WRF - Plotting
"""
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([65, 105, 25, 46], ccrs.PlateCarree())
fig = ax.pcolormesh(WRF_lon, WRF_lat, WRF_contr_prec, cmap=cmap, vmax=vmax, norm=norm)
axins = inset_axes(ax,width="100%",  height="5%",loc='lower center',borderpad=-4)
cbar = plt.colorbar(fig, cax=axins, orientation='horizontal', ticks=bounds)
cbar.set_label('[%]')
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
ax.contour(TP_loni,TP_lati,TP_ele.data, [3000], cmap='Greys_r', linewidths = [2.0])

txt = 'WRF cell contribution to total precipitation between 2001 - 2016'
plt.figtext(0.12, 0.08, txt, horizontalalignment='left')
fig = plt.gcf(); fig.set_size_inches(15, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/WRF_contr_prec.png', dpi=100)

#%% HAR - Plotting
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([65, 105, 25, 46], ccrs.PlateCarree())
fig = ax.pcolormesh(HAR_lon, HAR_lat, HAR_contr_prec, cmap=cmap, vmax=vmax, norm=norm)
axins = inset_axes(ax,width="100%",  height="5%",loc='lower center',borderpad=-4)
cbar = plt.colorbar(fig, cax=axins, orientation='horizontal', ticks=bounds)
cbar.set_label('[%]')
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
ax.contour(TP_loni,TP_lati,TP_ele.data, [3000], cmap='Greys_r', linewidths = [2.0])

txt = 'HAR cell contribution to total precipitation between 2001 - 2016'
plt.figtext(0.12, 0.08, txt, horizontalalignment='left')
fig = plt.gcf(); fig.set_size_inches(15, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/HAR_contr_prec.png', dpi=100)

#%% ERA5 - Plotting
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([65, 105, 25, 46], ccrs.PlateCarree())
fig = ax.pcolormesh(ERA5_lon, ERA5_lat, ERA5_contr_prec, cmap=cmap, vmax=vmax, norm=norm)
axins = inset_axes(ax,width="100%",  height="5%",loc='lower center',borderpad=-4)
cbar = plt.colorbar(fig, cax=axins, orientation='horizontal', ticks=bounds)
cbar.set_label('[%]')
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
ax.contour(TP_loni,TP_lati,TP_ele.data, [3000], cmap='Greys_r', linewidths = [2.0])

txt = 'ERA5 cell contribution to total precipitation between 2001 - 2016'
plt.figtext(0.12, 0.08, txt, horizontalalignment='left')
fig = plt.gcf(); fig.set_size_inches(15, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/ERA5_contr_prec.png', dpi=100)

#%% ALl in one plot!

fig = plt.figure(figsize=(15,8.5))
ax1 = fig.add_subplot(2,2,1, projection=ccrs.PlateCarree())
ax1.set_extent([65, 105, 25, 46], ccrs.PlateCarree())
ax1.pcolormesh(WRF_lon, WRF_lat, WRF_contr_prec, cmap=cmap, vmax=vmax, norm=norm)
grids = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
grids.bottom_labels = False; grids.right_labels = False
ax1.contour(TP_loni,TP_lati,TP_ele.data, [3000], cmap='Greys_r', linewidths = [2.0])
ax1.set_title('a)')

ax2 = fig.add_subplot(2,2,2, projection=ccrs.PlateCarree())
ax2.set_extent([65, 105, 25, 46], ccrs.PlateCarree())
ax2.pcolormesh(HAR_lon, HAR_lat, HAR_contr_prec, cmap=cmap, vmax=vmax, norm=norm)
grids = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
grids.bottom_labels = False; grids.left_labels = False 
ax2.contour(TP_loni,TP_lati,TP_ele.data, [3000], cmap='Greys_r', linewidths = [2.0])
ax2.set_title('b)')

ax3 = fig.add_subplot(2,2,3, projection=ccrs.PlateCarree())
ax3.set_extent([65, 105, 25, 46], ccrs.PlateCarree())
fig1 = ax3.pcolormesh(ERA5_lon, ERA5_lat, ERA5_contr_prec, cmap=cmap, vmax=vmax, norm=norm)
grids = ax3.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
grids.top_labels = False; grids.right_labels = False
ax3.contour(TP_loni,TP_lati,TP_ele.data, [3000], cmap='Greys_r', linewidths = [2.0])
ax3.set_title('c)')

fig.subplots_adjust(hspace=0);
fig.subplots_adjust(wspace=0.025);

axins = fig.add_axes([0.935, 0.15, 0.015, 0.68])
cbar = fig.colorbar(fig1, cax=axins, ticks=bounds)
cbar.set_label('Cell fraction of total precipitation [%]')
cmap.set_over("limegreen")

txt = 'WRF [a], HAR [b] & ERA5 [c] cell contribution to total precipitation between 2001 - 2016'
plt.figtext(0.12, 0.08, txt, horizontalalignment='left')
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/all_contr_prec.png', dpi=100)