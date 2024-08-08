# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 08:56:27 2022

@author: benja
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import dask
dask.config.set({"array.slicing.split_large_chunks": True})

#%%

"""
Mean annual total precipitation

HAR is in mm/h
ERA5 is in m/h

"""
## DATA
WRF_prec = xr.open_mfdataset('D:/WRF_prec/*.nc')

HAR_prec = xr.open_mfdataset('D:/HAR_prec/*.nc')
HAR_prec = HAR_prec.sel(time=HAR_prec.time.dt.month.isin([5,6,7,8,9]))

ERA5_prec = xr.open_mfdataset('D:/ERA5_prec/*.nc').sel(longitude=slice(65, 105), latitude=slice(46, 25))
ERA5_prec= ERA5_prec.sel(time=ERA5_prec.time.dt.month.isin([5,6,7,8,9])) ## Only wet season (ws) may - sept


#%% Calculations and data WRF (jump load calculations)
WRF_asum = WRF_prec.Prep.resample(time='Y').sum(dim='time') ## Annual ws sum (asum) of precipitation
WRF_amean = WRF_asum.mean(dim='time') ## WS annual mean (amean) of 2001 - 2016 (in mm?) 

del WRF_asum, WRF_prec

## Saving to same time in the future (takes time)
WRF_amean.to_netcdf('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_amean_prec.nc')
WRF_amean.load()
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_mean_prec', WRF_amean)

#%% Calculations and data HAR (jump load calculations)
HAR_asum = HAR_prec.prcp.resample(time='Y').sum(dim='time')## Annual ws sum (asum) of precipitation
HAR_amean = HAR_asum.mean(dim='time') ## WS annual mean (amean) of 2001 - 2016 in mm

del HAR_asum, HAR_prec

## Saving to same time in the future (takes time)
HAR_amean.to_netcdf('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_amean_prec')
HAR_amean.load()
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_mean_prec', HAR_amean)

#%% Calculations and data ERA5 (jump load calculations)
## Try groupby
ERA5_asum = ERA5_prec.tp.resample(time='Y').sum(dim='time') ## tp = total prec dimension in m
ERA5_amean = ERA5_asum.mean(dim='time')*1000 ## WS annual mean (amean) of 2001 - 2016 in mm (*1000)

ERA5_lon = ERA5_prec.longitude; ERA5_lat = ERA5_prec.latitude
ERA5_loni, ERA5_lati = np.meshgrid(ERA5_lon, ERA5_lat); del ERA5_lon, ERA5_lat
del ERA5_asum, ERA5_prec

## Saving to same time in the future (takes time)
ERA5_amean.to_netcdf('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_amean_prec.nc')
#ERA5_amean.load()
#np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_mean_prec', ERA5_amean)


#%% Calculations and data GPM (jump load calculations)
## This way much faster than loading all nc files in as one!!
yearly_sum = np.zeros([600,350])
for y in np.arange(2001,2017): 
    yprec = xr.open_dataset('D:/GPM_prec/GPM_'+str(y)+'.nc4_prec.nc4'); 
    yprec['time'] = yprec.indexes['time'].to_datetimeindex()
    yhourly = yprec.precipitationCal.resample(time='H').mean(dim='time')
    ysum = np.squeeze(yhourly.resample(time='Y').sum(dim='time'))
    yearly_sum += np.array(ysum)

GPM_amean = yearly_sum / 16
GPM_amean = np.transpose(GPM_amean)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_mean_prec', GPM_amean)

"""
GPM_hourly = GPM_prec.precipitationCal.resample(time='H').mean(dim='time')
GPM_asum = GPM_hourly.resample(time='Y').sum(dim='time') ## Annual ws sum (asum) of precipitation
GPM_amean = GPM_asum.mean(dim='time') #.astype(int) not helping
#GPM_lon = GPM_prec.lon; GPM_lat = GPM_prec.lat; GPM_lon, GPM_lat = np.meshgrid(GPM_lon, GPM_lat)
del GPM_asum, GPM_prec, GPM_hourly

## Try to save to netcdf file instead
## If nothing works, loop through each file instead of loading them all in at once
## Saving to same time in the future (takes time)
GPM_amean.to_netcdf('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_amean_prec')
#GPM_amean.load()
#np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_mean_prec', GPM_amean)
"""
#%% Load calculations + lon/lats
WRF = xr.open_dataset('D:/Benjamin/MasterThesis2021-2022/data/Python_files/WRF_amean_prec')
HAR = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/HAR_mean_prec.npy')
ERA5 = xr.open_dataset('D:/Benjamin/MasterThesis2021-2022/data/Python_files/ERA5_amean_prec')
GPM = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/GPM_mean_prec.npy')

WRF_lon = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/WRF_lon.npy')
WRF_lat = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/WRF_lat.npy')

HAR_lon = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/HAR_lon.npy')
HAR_lat = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/HAR_lat.npy')

ERA5_lon = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/ERA5_lon.npy')
ERA5_lat = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/ERA5_lat.npy')

GPM_lon = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/GPM_lon.npy')
GPM_lat = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/GPM_lat.npy')

prec_list = [WRF.Prep, HAR, ERA5.tp, GPM]
lon_list = [WRF_lon, HAR_lon, ERA5_lon, GPM_lon]; lat_list = [WRF_lat, HAR_lat, ERA5_lat, GPM_lat]

del WRF, HAR, ERA5, GPM, WRF_lon, WRF_lat, HAR_lon, HAR_lat, ERA5_lon, ERA5_lat, GPM_lon, GPM_lat
TP_DEM = xr.open_dataset('D:/Benjamin/MasterThesis2021-2022/data/GMTED2010_15n030_0125deg.nc')


#%% Plotting necessities
vmax=5000
bounds = np.linspace(0, 5000, 21)
ncolors = len(bounds) - 1
cmap = matplotlib.cm.get_cmap("gnuplot2_r", ncolors)
norm = BoundaryNorm(boundaries=bounds, ncolors=ncolors)
titles = ['a)','b)','c)','d)']
proj = ccrs.PlateCarree()

fig = plt.figure(figsize=(15,8.5))
for i,x in enumerate(prec_list):
    ax = fig.add_subplot(2,2,i+1, projection=proj)
    ax.set_extent([65, 105, 25, 46], proj)
    plot = ax.pcolormesh(lon_list[i], lat_list[i], x, cmap=cmap, vmax=vmax, norm=norm)
    grids = ax.gridlines(crs=proj, draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
    grids.top_labels = False; grids.right_labels = False;
    ax.contour(TP_DEM.longitude, TP_DEM.latitude, TP_DEM.elevation, [3000], cmap='Greys_r', linewidths = [1.0])
    ax.set_title(titles[i])

axins = fig.add_axes([0.935, 0.15, 0.015, 0.68])
cbar = fig.colorbar(plot, cax=axins, ticks=bounds, extend='max')
cbar.set_label('Mean annual sum of MJJAS precipitation [mm]')
cmap.set_over("cyan")
fig.subplots_adjust(hspace=0.06);
fig.subplots_adjust(wspace=0.09);
txt = 'WRF [a], HAR [b], ERA5 [c] & GPM [d] mean annual sum of precipitation between 2001 - 2016'
plt.figtext(0.12, 0.08, txt, horizontalalignment='left')
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/all_mean_prec.png', dpi=100)

del bounds, ncolors, cmap, norm, vmax, proj, titles, fig, ax, plot, axins, cbar, txt, x, i, grids


#%% Plotting WRF
"""
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([65, 105, 25, 46], ccrs.PlateCarree())
fig = ax.pcolormesh(WRF_lon, WRF_lat, WRF.Prep, cmap=cmap, vmax=vmax, norm=norm)
axins = inset_axes(ax,width="100%",  height="5%",loc='lower center', borderpad=-4)
cbar = plt.colorbar(fig, ticks=bounds, extend='max', cax=axins, orientation='horizontal')
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
ax.contour(TP_loni,TP_lati,TP_ele.data, [3000], cmap='Greys_r', linewidths = [2.0])
cmap.set_over("limegreen")
cbar.set_label('Precipitation [mm $y^{-1}$]')

txt = 'WRF mean annual sum of precipitation between 2001 - 2016'
plt.figtext(0.12, 0.08, txt, horizontalalignment='left')
fig = plt.gcf(); fig.set_size_inches(15, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/WRF_mean_prec.png', dpi=100)

#%% Plotting HAR
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([65, 105, 25, 46], ccrs.PlateCarree())
fig = ax.pcolormesh(HAR_lon, HAR_lat, HAR, cmap=cmap, vmax=vmax, norm=norm)
axins = inset_axes(ax,width="100%",  height="5%",loc='lower center', borderpad=-4)
cbar = plt.colorbar(fig, ticks=bounds, extend='max', cax=axins, orientation='horizontal')
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
ax.contour(TP_loni,TP_lati,TP_ele.data, [3000], cmap='Greys_r', linewidths = [2.0])
cmap.set_over("limegreen")
cbar.set_label('Precipitation [mm $y^{-1}$]')

txt = 'HAR mean annual sum of precipitation between 2001 - 2016'
plt.figtext(0.12, 0.08, txt, horizontalalignment='left')
fig = plt.gcf(); fig.set_size_inches(15, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/HAR_mean_prec.png', dpi=100)

#%% Plotting ERA5
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([65, 105, 25, 46], ccrs.PlateCarree())
fig = ax.pcolormesh(ERA5_lon, ERA5_lat, ERA5['tp'], cmap=cmap, vmax=vmax, norm=norm)
axins = inset_axes(ax,width="100%",  height="5%",loc='lower center', borderpad=-4)
cbar = plt.colorbar(fig, ticks=bounds, extend='max', cax=axins, orientation='horizontal')
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
ax.contour(TP_loni,TP_lati,TP_ele.data, [3000], cmap='Greys_r', linewidths = [2.0])
cmap.set_over("limegreen")
cbar.set_label('Precipitation [mm $y^{-1}$]')

txt = 'ERA5 mean annual sum of precipitation between 2001 - 2016'
plt.figtext(0.12, 0.08, txt, horizontalalignment='left')
fig = plt.gcf(); fig.set_size_inches(15, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/ERA5_mean_prec.png', dpi=100)

#%% Plotting necessities
vmax=5000
bounds = np.linspace(0, 5000, 21)
ncolors = len(bounds) - 1
cmap = matplotlib.cm.get_cmap("ocean_r", ncolors)
norm = BoundaryNorm(boundaries=bounds, ncolors=ncolors)

#%% ALl in one plot!

fig = plt.figure(figsize=(15,8.5))
ax1 = fig.add_subplot(2,2,1, projection=ccrs.PlateCarree())
ax1.set_extent([65, 105, 25, 46], ccrs.PlateCarree())
ax1.pcolormesh(WRF_lon, WRF_lat, WRF, cmap=cmap, vmax=vmax, norm=norm)
grids = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
grids.bottom_labels = False; grids.right_labels = False
ax1.contour(TP_loni,TP_lati,TP_ele.data, [3000], cmap='Greys_r', linewidths = [2.0])
ax1.set_title('a)')

ax2 = fig.add_subplot(2,2,2, projection=ccrs.PlateCarree())
ax2.set_extent([65, 105, 25, 46], ccrs.PlateCarree())
ax2.pcolormesh(HAR_lon, HAR_lat, HAR, cmap=cmap, vmax=vmax, norm=norm)
grids = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
grids.bottom_labels = False; grids.left_labels = False 
ax2.contour(TP_loni,TP_lati,TP_ele.data, [3000], cmap='Greys_r', linewidths = [2.0])
ax2.set_title('b)')

ax3 = fig.add_subplot(2,2,3, projection=ccrs.PlateCarree())
ax3.set_extent([65, 105, 25, 46], ccrs.PlateCarree())
fig1 = ax3.pcolormesh(ERA5_lon, ERA5_lat, ERA5, cmap=cmap, vmax=vmax, norm=norm)
grids = ax3.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
grids.top_labels = False; grids.right_labels = False
ax3.contour(TP_loni,TP_lati,TP_ele.data, [3000], cmap='Greys_r', linewidths = [2.0])
ax3.set_title('c)')

fig.subplots_adjust(hspace=0);
fig.subplots_adjust(wspace=0.025);

axins = fig.add_axes([0.935, 0.15, 0.015, 0.68])
cbar = fig.colorbar(fig1, cax=axins, ticks=bounds, extend='max')
cbar.set_label('Precipitation [mm $y^{-1}$]')
cmap.set_over("limegreen")

txt = 'WRF [a], HAR [b] & ERA5 [c] mean annual sum of precipitation between 2001 - 2016'
plt.figtext(0.12, 0.08, txt, horizontalalignment='left')
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/all_mean_prec.png', dpi=100)
"""