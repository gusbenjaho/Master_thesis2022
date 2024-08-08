# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#%%
"""
Elevation map
Dont set <0 to NaN, use vmin instead??

"""
# Elevation map data
DEM = xr.open_dataset('D:/Benjamin/MasterThesis2021-2022/data/GMTED2010_15n030_0125deg.nc')
Elevation = np.ma.masked_where(DEM.elevation < 0.1, DEM.elevation)

#%% Creating polygons for WRF and HAR extents
WRF_data = xr.open_dataset('D:/WRF_masks/Mask_segmentation_200001.nc')
HAR_data = xr.open_dataset('D:/HAR_masks/Mask_segmentation_2000_1.nc')
WRF_lon = WRF_data.lon; WRF_lat = WRF_data.lat; WRF_lon[1:-1, 1:-1] = np.nan; WRF_lat[1:-1, 1:-1] = np.nan
HAR_lon = HAR_data.lon; HAR_lat = HAR_data.lat; HAR_lon[1:-1, 1:-1] = np.nan; HAR_lat[1:-1, 1:-1] = np.nan

del HAR_data, WRF_data
#%% Plotting elevation map
# Removing blue from terrain cmap
orig_cmap = plt.cm.terrain
colors = orig_cmap(np.linspace(0.3, 1, 10))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mycmap", colors)

# Plotting
ax = plt.axes(projection=ccrs.PlateCarree())
axins = inset_axes(ax,width="2%",  height="100%", loc='center right', borderpad=-4.6) # Location of colorbar
ax.set_extent([50, 140, 2, 55], ccrs.PlateCarree())
fig = ax.pcolormesh(DEM.longitude, DEM.latitude, Elevation, cmap=cmap)
cbar = plt.colorbar(fig, cax=axins)
cbar.set_label('Altitude [m]')
ax.coastlines(); 
grids = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
ax.add_patch(mpatches.Rectangle(xy=[67, 26], width=37, height=18, ec='k', lw=2, ls='--', fill=False))
ax.scatter(WRF_lon, WRF_lat, s=0.2, c='k')
ax.scatter(HAR_lon, HAR_lat, s=0.1, c='k')
ax.add_feature(cfeature.OCEAN, zorder=0); grids.right_labels = False
ax.contour(DEM.longitude, DEM.latitude, DEM.elevation, [3000], cmap='Greys_r', linewidths = [2.0])
ax.text(55, 44, 'WRF_GU', transform=ccrs.PlateCarree(), size=14)
ax.text(63, 44, 'HAR', transform=ccrs.PlateCarree(), size=16)
ax.text(68, 27, 'TP region domain', transform=ccrs.PlateCarree(), size=12)
ax.text(80, 33, 'TP domain', transform=ccrs.PlateCarree(), size=12)
txt = 'Elevation in the TP region, including extent of study area (black dashed line), the TP defined by the 3000 m level (black solid line), \nthe HAR dataset extent (black dotted line) and the WRF dataset extent (red dotted line)'
plt.figtext(0.125, 0.05, txt, horizontalalignment='left')
fig = plt.gcf(); fig.set_size_inches(15, 10)
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/elevation.png', dpi=100)
del axins, cbar, colors, fig, grids, txt, ax