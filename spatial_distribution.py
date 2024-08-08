# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 15:34:50 2021

@author: benja

Get the data

"""
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
#import cmasher as cmr
from progressbar import ProgressBar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
pbar = ProgressBar()

WRF_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_tracks.pkl')
HAR_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_tracks.pkl')
GPM_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_tracks.pkl')
ERA5_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_tracks.pkl')

#%%
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

#%% Lon and lat from nc file

## Irregular grid (I think)
HAR_nc = xr.open_dataset('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/HAR v2_d10km_m_2d_prcp_2020.nc')

xlon = []
for lo in HAR_tracks.longitude.values:
    xlon_temp = find_nearest((np.unique(HAR_nc.lon.values)), lo)
    xlon.append(xlon_temp)
del xlon_temp

xlat = []
for la in HAR_tracks.latitude.values:
    xlat_temp = find_nearest((np.unique(HAR_nc.lat.values)), la)
    xlat.append(xlat_temp)
del xlat_temp

np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_nclon', xlon)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_nclat', xlat)

HAR_tracks['nclat'] = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_nclat.npy')
HAR_tracks['nclon'] = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_nclon.npy')
#%% Extracting initiation cells function

def init_cells(tracks, start, end):
    init_cells = pd.DataFrame()
    for y in np.arange(2001,2016):
        ytracks = tracks[tracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            init_cell = ytracks[ytracks.cell == cell][['cell','nclon','nclat']].iloc[0]
            init_cells = init_cells.append(init_cell)
    init_cells = init_cells.cell.groupby([init_cells.nclon, init_cells.nclat]).count()
    return init_cells

#%% Run and reset index (need to create xlon and xlat for each data set first)
WRF_init = init_cells(WRF_tracks,2001,2016); WRF_init = WRF_init.reset_index()
HAR_init = init_cells(HAR_tracks,2001,2016); HAR_init = HAR_init.reset_index()
ERA5_init = init_cells(ERA5_tracks,2001,2016); ERA5_init = ERA5_init.reset_index()
GPM_tracks['time'] = pd.to_datetime(GPM_tracks.timestr)
GPM_init = init_cells(GPM_tracks,2001,2016); GPM_init = GPM_init.reset_index()

#%% Calculate density over the nc grid
## NC lon/lat data
HAR_nc = xr.open_dataset('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/HAR v2_d10km_m_2d_prcp_2020.nc')
loni = HAR_nc.lon.values; lati = HAR_nc.lat.values; del HAR_nc

dense_test = np.zeros([len(lati[:,1]), len(loni[0])])
for row_idx, lo in enumerate(loni):
    la = lati[row_idx]
    for lat in np.array(HAR_init.nclat):
        col_idx = np.asarray(la == lat)
        dense_test[row_idx, col_idx] += 1


for lon_idx, lo in enumerate(loni):
    bla = loni[lon_idx]


# IS THIS IT?? Really fast but yields 7641 instead 7571 :( NOOOOO!!!!
dense_test = np.zeros([len(lati[:,1]), len(loni[0])])
for lon, lat in zip(HAR_init.nclon, HAR_init.nclat):
        idx_col = np.nonzero(loni == lon)
        idx_row = np.nonzero(lati == lat)
        dense_test[idx_row[0], idx_col[1]] += 1
dense_test[dense_test==0]=['NaN']
del lon; del lat; del idx_row; del idx_col

#%% TP elevation for plotting

## Elevation data
TP_DEM = xr.open_dataset('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/GMTED2010_15n030_0125deg.nc')
ele_lon = TP_DEM.longitude; ele_lat = TP_DEM.latitude; TP_ele = TP_DEM.elevation
TP_loni, TP_lati = np.meshgrid(ele_lon,ele_lat); del ele_lon; del ele_lat; del TP_DEM
#ele_lon = DEM.lon[(DEM.lon >= 67) & (DEM.lon <= 103)].values; ele_lat = DEM.lat[(DEM.lat >= 26) & (DEM.lat <= 44)].values
#ele_loni, ele_lati = np.meshgrid(ele_lon,ele_lat); del ele_lon; del ele_lat

#%% Plotting
cmap = plt.cm.get_cmap('Reds', 12); #cmap = cmap.reversed()
ax = plt.axes(projection=ccrs.PlateCarree())
axins = inset_axes(ax,width="100%",  height="5%",loc='lower center',borderpad=-5) # Location of colorbar
ax.set_extent([67, 105, 26, 44], ccrs.PlateCarree())
fig = ax.pcolor(loni,lati,dense_test, cmap=cmap)
cbar = plt.colorbar(fig, cax=axins, orientation='horizontal')
cbar.set_label('Precipitation cell frequency')
ax.coastlines()
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
ax.add_feature(cfeature.LAND, facecolor=(0.66,0.66,0.66))
ax.contour(TP_lati,TP_loni,TP_ele, cmap='Greys', levels=list(range(3000,5000,1000))) # Add contour of 3000m elevation
txt = 'Density map of precipitation cells at initiation stage between 2001 and 2016 based on the HAR model data. Black line denotes the TP.'
plt.figtext(0.12, 0.00001, txt, horizontalalignment='left')
fig = plt.gcf(); fig.set_size_inches(18, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/HAR_density.png', dpi=100)

#%%
"""
Elevation map
Dont set <0 to NaN, use vmin instead??

"""
# Elevation map data
DEM = xr.open_dataset('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/GMTED2010_15n030_0125deg.nc')
glob_ele = DEM.elevation; glob_lon = DEM.longitude; glob_lat = DEM.latitude; del DEM
idx_lon = np.asarray((glob_lon >= 60) & (glob_lon <= 120)); idx_lat = np.asarray((glob_lat >= 10) & (glob_lat <= 50))
region_lon = np.array(glob_lon[idx_lon]); region_lat = np.array(glob_lat[idx_lat]); del glob_lon; del glob_lat
region_ele = glob_ele[idx_lat, idx_lon].values; del glob_ele; del idx_lon; del idx_lat
region_ele = region_ele.astype(float); region_ele[region_ele <= 0] = np.nan;
region_loni, region_lati = np.meshgrid(region_lon, region_lat); del region_lat; del region_lon;

#%% Plotting elevation map
# Removing blue from terrain cmap
orig_cmap = plt.cm.terrain
colors = orig_cmap(np.linspace(0.3, 1, 10))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mycmap", colors)

# Plotting
ax = plt.axes(projection=ccrs.PlateCarree())
axins = inset_axes(ax,width="5%",  height="100%", loc='center right', borderpad=-4.6) # Location of colorbar
ax.set_extent([60, 120, 10, 50], ccrs.PlateCarree())
fig = ax.pcolor(region_loni, region_lati, region_ele, cmap=cmap)
cbar = plt.colorbar(fig, cax=axins)
cbar.set_label('Altitude [m]')
ax.coastlines()
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='black', alpha=0.5, linestyle='--')
ax.add_patch(mpatches.Rectangle(xy=[67, 26], width=37, height=18, ec='k', lw=2, fill=False)) # Adding study area to map
ax.add_feature(cfeature.OCEAN, zorder=0)
ax.contour(TP_loni,TP_lati,TP_ele.data, [3000], cmap='Greys_r', linewidths = [2.0])
txt = 'Elevation in the TP region, including extent of study area (box) and the TP defined by the 3000 m level (black line)'
plt.figtext(0.125, 0.05, txt, horizontalalignment='left')
fig = plt.gcf(); fig.set_size_inches(15, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/elevation.png', dpi=100)

#%%  Failed stuff
## This results in 7571, same as for DC
"""
def density(tracks, start, end):
    density = pd.DataFrame()
    for y in np.arange(2001,2016):
        ytracks = tracks[tracks.time.dt.year == y]
        ytracks = ytracks.drop_duplicates(subset='cell')
        count = ytracks.cell.groupby([ytracks.nclon, ytracks.nclat]).count().to_frame()
        density = density.append(count)
    return density
"""


""" Almost working, still wrong though

# asarray went much faster than nonzero, and yielded the same output 850 in total)
dense_test = np.zeros([len(lati[:,1]), len(loni[0])])
for lon in HAR_dense.nclon:
    for lat in HAR_dense.nclat:
        idx = np.nonzero((loni == lon) & (lati == lat))
        dense_test[idx] += 1


for lat, long in zip(lati,loni):
    print(lat,long)

lonlat = np.zeros([len(lati[:,1]), len(loni[0])])
coords = []
for xrow_index, xrow in enumerate(loni):
    lat = lats[idx]

    
    for xcol_index, xitem in enumerate(xrow):
        for yrow_idx, yrow in enumerate(lati):
            for ycol_idx, yitem in enumerate(yrow):
                coords.append([xitem,yitem])
                lonlat[xrow_index][xcol_index].append(coords)

for xrow_index, xrow in enumerate(bla):
    for xcol_index, xitem in enumerate(xrow):
        print(xcol_index)


dense_test = np.zeros([len(lati[:,1]), len(loni[0])])
for xrow_idx, xrow in enumerate(loni):
    for xcol_idx, xitem in enumerate(xrow):
        idx = np.where((xitem == HAR_dense.nclon) & (lati[xrow_idx][xcol_idx] == HAR_dense.nclat))
        dense_test[xrow_idx][xcol_idx] += 1


## Files from density calculation above
HAR_lon = []; HAR_lat = []
density_lon = np.zeros([len(lati[:,1]), len(loni[0])])
for row_index, row in enumerate(loni):
    for col_index, item in enumerate(row):
        for cell in HAR_lon.nclon.values:
            if cell == item:
                density_lon[row_index][col_index] += HAR_lon.cell[HAR_lon.nclon.values == cell]
            else:
                continue

density_lat = np.zeros([len(lati[:,1]), len(loni[0])])
for row_index, row in enumerate(lati):
    for col_index, item in enumerate(row):
        for cell in HAR_lat.nclat.values:
            if cell == item:
                density_lat[row_index][col_index] += HAR_lat.cell[HAR_lat.nclat.values == cell]
            else:
                continue

density_init = density_lon + density_lat

plt.pcolor(loni,lati,density_lat)
plt.colorbar()
#%% And finally
for index, row in HAR_dense.iterrows():
    bla = row['nclon'], row['nclat']
# Or do a meshgrid of the values after? And then add them together maybe?
""" 

"""
density = np.zeros([len(lati[:,1]), len(loni[0])])
for y in np.arange(2001,2016):
    ytracks = WRF_tracks[WRF_tracks.time.dt.year == y]
    for i in range(len(lons)):
        for j in range(len(lati[i])):
            for cell in np.unique(ytracks.cell.values):
                if (WRF_tracks.cell[(WRF_tracks.longitude == lons[i]) & (WRF_tracks.latitude == lats[j])] == cell).all():
                    count = 1
density[i][j] = density[i][j].append(count)
#%%

def density(tracks, start, end):
    density = np.zeros([len(lati[:,1]), len(loni[0])])
    for y in np.arange(start,end):

        
ytracks = HAR_tracks[HAR_tracks.time.dt.year == 2001]
ytracks = ytracks.drop_duplicates(subset='cell')
for lo in ytracks.longitude.values:
    xlon = int(find_nearest((np.unique(HAR_nc.lon.values)), lo)
    for la in ytracks.latitude.values:
        xlat = int(find_nearest((np.unique(HAR_nc.lat.values)), la)
        count = ytracks.cell[(ytracks.longitude == lo) & (ytracks.latitude == la)].count()
density[xlon,xlat] = density[xlon,xlat].append(count)



density = np.zeros([len(lati[:,1]), len(loni[0])])
for lo in np.arange(67,103.1,0.1):
    for la in np.arange(26,44.1,0.1):
        poslon = loni[loni == 67]
        poslat = lati[lati == la]
        count = WRF_lon.cell[WRF_lon.lon == lo] + WRF_lat.cell[WRF_lat.lat == la]
        density[poslat,poslon] = count

bla = np.stack((lati.flatten(),loni.flatten()), axis=-1)
lons = bla[:,1]; lats = bla[:,0]
test = zip(lats,lons); test = list(test)
#%%




WRF_tracks.cell.plot(x='longitude', y='latitude', kind='area')


plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([60, 120, 10, 50], ccrs.PlateCarree())
ax.stock_img()
ax.coastlines()
plt.show()


## Regular grid
HAR_nc = xr.open_dataset('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/HAR v2_d10km_m_2d_prcp_2020.nc')
WRF_nc = xr.open_dataset('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/WRFOut_TP9km_HourlyP_2008_07.nc')
minlon = 67; maxlon = 103; minlat = 26; maxlat=44
lats = HAR_nc.lat.values[:,1]; lats = lats[(lats >= minlat) & (lats <= maxlat)]
lons = HAR_nc.lon.values[1,:]; lons = lons[(lons >= minlon) & (lons <= maxlon)]
del minlat; del maxlat; del minlon; del maxlon;



# Step 1
xlon = []
for lo in HAR_tracks.longitude.values:
    xlon_temp = find_nearest(lons, lo)
    xlon.append(xlon_temp)
del xlon_temp


# Step 2
xlat = []
for la in pbar(HAR_tracks.latitude.values):
    xlat_temp = find_nearest(lats, la)
    xlat.append(xlat_temp)
del xlat_temp

HAR_tracks['nclon'] = xlon
HAR_tracks['nclat'] = xlat

HAR_lon_mesh, HAR_lat_mesh = np.meshgrid(HAR_lon.cell, HAR_lat.cell)

plt.plot(x=lons, y=lats, data=test)
plt.pcolormesh(lons, lats, density, shading='flat',vmin=1)
plt.colorbar()
plt.scatter(loni,lati,c=test[loni,lati])

#%% Loop for assigning new cell ID
#ID_IDX = np.arange(0,WRF_DC.sum())


def get_unique_cell(preciptracks, start, end): 
    preciptracks['year']= preciptracks.time.dt.year
    YCell=0
    NewCell = []
    for y in np.arange(start,end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            for i in np.arange(0,WRF_DC.sum()):
                if cell == ytracks.cell:
                    YCell = i
            NewCell.append(YCell)
            return NewCell
#NewCell = get_unique_cell(WRF_tracks, 2001, 2016)
#test = pd.DataFrame(test)


This returns a list of 7251 objects, which is equal to the sum of count
def get_unique_cell(preciptracks, start, end): 
    preciptracks['year']= preciptracks.time.dt.year
    NewCell=[] 
    for y in np.arange(start,end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            YCell = ytracks.groupby([ytracks.cell == cell])
            NewCell.append(YCell)
            
    return NewCell
tracks[condition] = newvalue
generate a big array of value
tracks[condition].cellid = str(cellid) + str(year)
str(cellid) + '_' + str(year)

"""
