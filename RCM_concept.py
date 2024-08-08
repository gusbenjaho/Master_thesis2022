# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import xarray as xr

DEM = xr.open_dataset('D:/Benjamin/MasterThesis2021-2022/data/GMTED2010_15n030_0125deg.nc')
Elevation = np.ma.masked_where(DEM.elevation < 0.1, DEM.elevation)

#%% Global Cartopy map
projPC = ccrs.PlateCarree()
proj = ccrs.Robinson()
cLon = 80; cLat = 10
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1, projection=projPC)

ax.stock_img()
ax.coastlines()
ax.add_patch(mpatches.Rectangle(xy=[50, 2], width=90, height=53, ec='k', lw=2, ls='--', fill=False))
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/Global_cartopy_map.png', dpi=100)
#%% Global ele map
orig_cmap = plt.cm.terrain
colors = orig_cmap(np.linspace(0.3, 1, 10))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mycmap", colors)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
fig = ax.pcolormesh(DEM.longitude, DEM.latitude, Elevation, cmap=cmap)
ax.add_feature(cfeature.OCEAN, zorder=0); ax.add_feature(cfeature.COASTLINE,linewidth=0.3)
ax.add_patch(mpatches.Rectangle(xy=[50, 2], width=90, height=53, ec='k', lw=2, ls='--', fill=False))
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/global_ele_map.png', dpi=100)

#%% Region with elevation data
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([50, 140, 2, 55], ccrs.PlateCarree())
fig = ax.pcolormesh(DEM.longitude, DEM.latitude, Elevation, cmap=cmap)
ax.add_feature(cfeature.OCEAN, zorder=0); ax.add_feature(cfeature.COASTLINE,linewidth=0.3)
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/regional_ele_map.png', dpi=100)

#%% Map_inset
fig = plt.figure(figsize=(10, 5))
