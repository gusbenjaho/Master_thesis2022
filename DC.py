# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 13:53:26 2021

@author: benja
"""
""" 

Count / hour and month

Read in tracks file

"""

# %% Get the data

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

WRF_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_tracks.pkl')
HAR_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_tracks.pkl')
GPM_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_tracks.pkl')
ERA5_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_tracks.pkl')

# %%
"""
Diurnal count for init hour

"""

# DEF creates a function based on coming loop, these args necessary to call function later:
# diurnal_precip, bins = get_diurnal_init(tracks, 2000, 2015)
def get_diurnal_init(preciptracks, start, end):
    preciptracks['hour'] = preciptracks.time.dt.hour
    diurnal = []
    for y in np.arange(start, end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            init_hour = ytracks[ytracks.cell == cell].hour.values[0]
            diurnal.append(init_hour)

    diurnal_precip, bins = np.histogram(diurnal, bins=np.arange(0, 25))
    return diurnal_precip, bins

# %% Called it with:
# WRF
WRF_DC, bins = get_diurnal_init(WRF_tracks, 2001, 2016)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_DC.npy', WRF_DC)

# %% HAR
HAR_DC, bins = get_diurnal_init(HAR_tracks, 2001, 2016)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_DC.npy', HAR_DC)

# %% ERA5
ERA5_DC, bins = get_diurnal_init(ERA5_tracks, 2001, 2016)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_DC.npy', ERA5_DC)

# %% GPM
GPM_tracks['time'] = pd.to_datetime(GPM_tracks.timestr)
GPM_DC, bins = get_diurnal_init(GPM_tracks, 2001, 2016)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_DC.npy', GPM_DC)
del bins

# %% Daily init counts from above
WRF_DC = np.load(
    'C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_DC.npy')
HAR_DC = np.load(
    'C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_DC.npy')
ERA5_DC = np.load(
    'C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_DC.npy')
GPM_DC = np.load(
    'C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_DC.npy')
#DC_bins = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/DC_bins.npy')

# %% Total count
DC_tot = pd.DataFrame()
DC_tot.insert(0, 'Hour', range(0, 24))
DC_tot['WRF'], DC_tot['HAR'], DC_tot['ERA5'], DC_tot['GPM'] = [WRF_DC, HAR_DC, ERA5_DC, GPM_DC]
del WRF_DC; del HAR_DC; del ERA5_DC; del GPM_DC
DC_tot.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/DC_tot.pkl')

# %% Normalize
DC = pd.DataFrame()
DC.insert(0, 'Hour', range(0, 24))
DC['WRF'] = DC_tot.WRF/DC_tot.WRF.sum() * 100
DC['HAR'] = DC_tot.HAR/DC_tot.HAR.sum() * 100
DC['ERA5'] = DC_tot.ERA5/DC_tot.ERA5.sum() * 100
DC['GPM'] = DC_tot.GPM/DC_tot.GPM.sum() * 100
DC.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/DC.pkl')

# %% Plotting
txt = 'The diurnal cycle of precipitation cell frequency (top) and fraction (bottom) at initiation stage between 2001 and 2016 based on the three\nmodel datasets (WRF, HAR & ERA5) and the satellite dataset (GPM)'
fig, axes = plt.subplots(nrows=2, ncols=1)
DC_tot.plot(x='Hour', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', grid=True, ylabel='Frequency', ax=axes[0], linestyle='--', xlim=[-0.5, 23.5], xticks=np.arange(0, 24, step=1), xlabel='Hour (UTC)')
DC.plot(x='Hour', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', grid=True, ylabel='Fraction (%)',ax=axes[1], linestyle='--', xlim=[-0.5, 23.5], xticks=np.arange(0, 24, step=1), xlabel='Hour (UTC)')
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
fig = plt.gcf()
fig.set_size_inches(12, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/DC.png', dpi=100)

# %%
""" 
Diurnal count of dissipation hour

"""

def get_diurnal_diss(preciptracks, start, end):
    preciptracks['hour'] = preciptracks.time.dt.hour
    diurnal = []
    for y in np.arange(start, end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            init_hour = ytracks[ytracks.cell == cell].hour.values[-1]
            diurnal.append(init_hour)

    diurnal_precip, bins = np.histogram(diurnal, bins=np.arange(0, 25))
    return diurnal_precip, bins

# %% Called it with:
# WRF
WRF_DC_diss, bins = get_diurnal_diss(WRF_tracks, 2001, 2016)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_DC_diss.npy', WRF_DC_diss)

# %% HAR
HAR_DC_diss, bins = get_diurnal_diss(HAR_tracks, 2001, 2016)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_DC_diss.npy', HAR_DC_diss)

# %% ERA5
ERA5_DC_diss, bins = get_diurnal_diss(ERA5_tracks, 2001, 2016)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_DC_diss.npy', ERA5_DC_diss)

# %% GPM
#GPM_tracks['time'] = pd.to_datetime(GPM_tracks.timestr)
GPM_DC_diss, bins = get_diurnal_diss(GPM_tracks, 2001, 2016)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_DC_diss.npy', GPM_DC_diss)
del bins

# %% Daily diss counts from above
WRF_DC_diss = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_DC_diss.npy')
HAR_DC_diss = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_DC_diss.npy')
ERA5_DC_diss = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_DC_diss.npy')
GPM_DC_diss = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_DC_diss.npy')

# %% Total count
DC_diss_tot = pd.DataFrame()
DC_diss_tot.insert(0, 'Hour', range(0, 24))
DC_diss_tot['WRF'], DC_diss_tot['HAR'], DC_diss_tot['ERA5'], DC_diss_tot['GPM'] = [WRF_DC_diss, HAR_DC_diss, ERA5_DC_diss, GPM_DC_diss]
del WRF_DC_diss; del HAR_DC_diss; del ERA5_DC_diss; del GPM_DC_diss

# %% Normalize
DC_diss = pd.DataFrame()
DC_diss.insert(0, 'Hour', range(0, 24))
DC_diss['WRF'] = DC_diss_tot.WRF/DC_diss_tot.WRF.sum() * 100
DC_diss['HAR'] = DC_diss_tot.HAR/DC_diss_tot.HAR.sum() * 100
DC_diss['ERA5'] = DC_diss_tot.ERA5/DC_diss_tot.ERA5.sum() * 100
DC_diss['GPM'] = DC_diss_tot.GPM/DC_diss_tot.GPM.sum() * 100

# %% Plotting
txt = 'The diurnal cycle of precipitation cell frequency (top) and fraction (bottom) at dissipation stage between 2001 and 2016 based on the three\nmodel datasets (WRF, HAR & ERA5) and the satellite dataset (GPM)'
fig, axes = plt.subplots(nrows=2, ncols=1)
DC_diss_tot.plot(x='Hour', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', grid=True, ylabel='Frequency',ax=axes[0], linestyle='--', xlim=[-0.5, 23.5], xticks=np.arange(0, 24, step=1), xlabel='Hour (UTC)')
DC_diss.plot(x='Hour', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', grid=True, ylabel='Fraction (%)', ax=axes[1], linestyle='--', xlim=[-0.5, 23.5], xticks=np.arange(0, 24, step=1), xlabel='Hour (UTC)')
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
fig = plt.gcf()
fig.set_size_inches(12, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/DC_diss.png', dpi=100)

# %%

"""
Diurnal count at maximum size

"""

def get_diurnal_max(preciptracks, start, end):
    preciptracks['hour'] = preciptracks.time.dt.hour
    diurnal = []
    for y in np.arange(start, end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            idx = np.argmax(ytracks.ncells[ytracks.cell == cell])
            max_hour = ytracks[ytracks.cell == cell].hour.values[idx]
            diurnal.append(max_hour)
    diurnal_precip, bins = np.histogram(diurnal, bins=np.arange(0, 25))
    return diurnal_precip, bins

# %% Running and saving
WRF_DC_max, bins = get_diurnal_max(WRF_tracks, 2001, 2016)
HAR_DC_max, bins = get_diurnal_max(HAR_tracks, 2001, 2016)
ERA5_DC_max, bins = get_diurnal_max(ERA5_tracks, 2001, 2016)
GPM_tracks['time'] = pd.to_datetime(GPM_tracks.timestr)
GPM_DC_max, bins = get_diurnal_max(GPM_tracks, 2001, 2016)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_DC_max.npy', WRF_DC_max)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_DC_max.npy', HAR_DC_max)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_DC_max.npy', ERA5_DC_max)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_DC_max.npy', GPM_DC_max)

# %% Loading above files
WRF_DC_max = np.load(
    'C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_DC_max.npy')
HAR_DC_max = np.load(
    'C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_DC_max.npy')
ERA5_DC_max = np.load(
    'C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_DC_max.npy')
GPM_DC_max = np.load(
    'C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_DC_max.npy')

# %% Total count
DC_max_tot = pd.DataFrame()
DC_max_tot.insert(0, 'Hour', range(0, 24))
DC_max_tot['WRF'], DC_max_tot['HAR'], DC_max_tot['ERA5'], DC_max_tot['GPM'] = [WRF_DC_max, HAR_DC_max, ERA5_DC_max, GPM_DC_max]
del WRF_DC_max; del HAR_DC_max; del ERA5_DC_max; del GPM_DC_max

# %% Normalize
DC_max = pd.DataFrame()
DC_max.insert(0, 'Hour', range(0, 24))
DC_max['WRF'] = DC_max_tot.WRF/DC_max_tot.WRF.sum() * 100
DC_max['HAR'] = DC_max_tot.HAR/DC_max_tot.HAR.sum() * 100
DC_max['ERA5'] = DC_max_tot.ERA5/DC_max_tot.ERA5.sum() * 100
DC_max['GPM'] = DC_max_tot.GPM/DC_max_tot.GPM.sum() * 100

# %% Plotting
txt = 'The diurnal cycle of precipitation cell frequency (top) and fraction (bottom) at maximum extension stage between 2001 and 2016\nbased on the three model datasets (WRF, HAR & ERA5) and the satellite dataset (GPM)'
fig, axes = plt.subplots(nrows=2, ncols=1)
DC_max_tot.plot(x='Hour', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', grid=True, ylabel='Frequency', ax=axes[0], linestyle='--', xlim=[-0.5, 23.5], xticks=np.arange(0, 24, step=1), xlabel='Hour (UTC)')
DC_max.plot(x='Hour', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', grid=True, ylabel='Fraction (%)', ax=axes[1], linestyle='--', xlim=[-0.5, 23.5], xticks=np.arange(0, 24, step=1), xlabel='Hour (UTC)')
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
fig = plt.gcf()
fig.set_size_inches(12, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/DC_max.png', dpi=100)

# %%
"""
DC of largest cells (>95th percentile)

"""

# %% Extracting initiation/dissipation cells functions

def init_cells(tracks, start, end):
    init_cells = pd.DataFrame()
    for y in np.arange(2001, 2016):
        ytracks = tracks[tracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            ycells = ytracks[ytracks.cell == cell].iloc[0]
            init_cells = init_cells.append(ycells)
    return init_cells


def diss_cells(tracks, start, end):
    diss_cells = pd.DataFrame()
    for y in np.arange(2001, 2016):
        ytracks = tracks[tracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            ycells = ytracks[ytracks.cell == cell].iloc[-1]
            diss_cells = diss_cells.append(ycells)
    return diss_cells

# %% DC of HAR initiation cells > 95th percentile
HAR_init = init_cells(HAR_tracks, 2001, 2016)
# HAR_init['km2'] = HAR_init.ncells*(10*10)
HAR_ncell_95 = np.percentile(HAR_init.ncells, 95)
HAR_95_mask = np.asarray(HAR_init.ncells >= HAR_ncell_95)
HAR_init_95 = HAR_init[HAR_95_mask]
DC_HAR_init_95 = HAR_init_95.cell.groupby(
    HAR_init_95.time.dt.hour).count().to_frame()
del HAR_ncell_95
del HAR_95_mask
del HAR_init
del HAR_init_95

# %% DC of HAR dissipation cells > 95th percentile
HAR_diss = diss_cells(HAR_tracks, 2001, 2016)
HAR_ncell_95 = np.percentile(HAR_diss.ncells, 95)
HAR_95_mask = np.asarray(HAR_diss.ncells >= HAR_ncell_95)
HAR_diss_95 = HAR_diss[HAR_95_mask]
DC_HAR_diss_95 = HAR_diss_95.cell.groupby(
    HAR_diss_95.time.dt.hour).count().to_frame()
del HAR_ncell_95
del HAR_95_mask
del HAR_diss
del HAR_diss_95

DC_HAR_95 = pd.DataFrame()
DC_HAR_95.insert(0, 'Hour', range(0, 24))
DC_HAR_95['Initiation'], DC_HAR_95['Dissipation'] = [
    DC_HAR_init_95, DC_HAR_diss_95]
del DC_HAR_diss_95
del DC_HAR_init_95

# %% Plotting
DC_HAR_95.plot(x='Hour', y=['Initiation', 'Dissipation'], grid=True,
               xlim=[-0.5, 23.5], xticks=np.arange(0, 24, step=1), xlabel='Hour (UTC)')
fig = plt.gcf()
fig.set_size_inches(12, 5)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/DC_HAR_95.png', dpi=100)
# %%
"""
DC of feature

"""
## Without the loops I need to remove some years to have the same timeline in each file 
GPM_tracks['time'] = pd.to_datetime(GPM_tracks.timestr)
WRF_tracks = WRF_tracks[(WRF_tracks.time.dt.year >= 2001) & (WRF_tracks.time.dt.year <= 2016)]
HAR_tracks = HAR_tracks[(HAR_tracks.time.dt.year >= 2001) & (HAR_tracks.time.dt.year <= 2016)]
ERA5_tracks = ERA5_tracks[(ERA5_tracks.time.dt.year >= 2001) & (ERA5_tracks.time.dt.year <= 2016)]
GPM_tracks = GPM_tracks[(GPM_tracks.time.dt.year >= 2001) & (GPM_tracks.time.dt.year <= 2016)]

DC_all_tot = WRF_tracks.cell.groupby(WRF_tracks.time.dt.hour).count().to_frame()
DC_all_tot.insert(0, 'Hour', range(0, 24))
DC_all_tot.rename(columns={'cell': 'WRF'}, inplace=True)
DC_all_tot['HAR'] = HAR_tracks.cell.groupby(HAR_tracks.time.dt.hour).count()
DC_all_tot['ERA5'] = ERA5_tracks.cell.groupby(ERA5_tracks.time.dt.hour).count()
DC_all_tot['GPM'] = GPM_tracks.cell.groupby(GPM_tracks.time.dt.hour).count()

## Change to local time (+6)
DC_all_tot.Hour = DC_all_tot.Hour + 6
DC_all_tot.replace(to_replace={'Hour': {24:0, 25:1, 26:2, 27:3, 28:4, 29:5}}, inplace=True)
DC_all_tot = DC_all_tot.sort_values(by=['Hour'])

DC_all = pd.DataFrame()
DC_all.insert(0, 'Hour', range(0, 24))
DC_all['WRF'] = DC_all_tot.WRF/DC_all_tot.WRF.sum() * 100
DC_all['HAR'] = DC_all_tot.HAR/DC_all_tot.HAR.sum() * 100
DC_all['ERA5'] = DC_all_tot.ERA5/DC_all_tot.ERA5.sum() * 100
DC_all['GPM'] = DC_all_tot.GPM/DC_all_tot.GPM.sum() * 100

DC_all.Hour = DC_all.Hour + 6
DC_all.replace(to_replace={'Hour': {24:0, 25:1, 26:2, 27:3, 28:4, 29:5}}, inplace=True)
DC_all = DC_all.sort_values(by=['Hour'])
# %% Plotting
txt = 'The diurnal cycle of precipitation cell frequency (top) and fraction (bottom) at every recorded stage between 2001 and 2016\nbased on the three model datasets (WRF, HAR & ERA5) and the satellite dataset (GPM)'
fig, axes = plt.subplots(nrows=2, ncols=1)
DC_all_tot.plot(x='Hour', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', grid=True, ylabel='Frequency', ax=axes[0], linestyle='--', xlim=[-0.5, 23.5], xticks=np.arange(0, 24, step=1), xlabel='Hour (UTC+6)')
DC_all.plot(x='Hour', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', grid=True, ylabel='Fraction (%)', ax=axes[1], linestyle='--', xlim=[-0.5, 23.5], xticks=np.arange(0, 24, step=1), xlabel='Hour (UTC+6)')
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
fig = plt.gcf()
fig.set_size_inches(12, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/DC_all.png', dpi=100)

# %%
""" 
DC of features of TP only (>3000)


"""
WRF_TP = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_TP.pkl')
HAR_TP = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_TP.pkl')
ERA5_TP = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_TP.pkl')
GPM_TP = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_TP.pkl')

#%% DC of TP only
DC_TP_tot = WRF_TP.cell.groupby(WRF_TP.time.dt.hour).count().to_frame()
DC_TP_tot.insert(0, 'Hour', range(0, 24))
DC_TP_tot.rename(columns={'cell': 'WRF'}, inplace=True)
DC_TP_tot['HAR'] = HAR_TP.cell.groupby(HAR_TP.time.dt.hour).count()
DC_TP_tot['ERA5'] = ERA5_TP.cell.groupby(ERA5_TP.time.dt.hour).count()
DC_TP_tot['GPM'] = GPM_TP.cell.groupby(GPM_TP.time.dt.hour).count()

DC_TP = pd.DataFrame()
DC_TP.insert(0, 'Hour', range(0, 24))
DC_TP['WRF'] = DC_TP_tot.WRF/DC_TP_tot.WRF.sum() * 100
DC_TP['HAR'] = DC_TP_tot.HAR/DC_TP_tot.HAR.sum() * 100
DC_TP['ERA5'] = DC_TP_tot.ERA5/DC_TP_tot.ERA5.sum() * 100
DC_TP['GPM'] = DC_TP_tot.GPM/DC_TP_tot.GPM.sum() * 100
del WRF_TP; del HAR_TP; del ERA5_TP; del GPM_TP

## Change to local time (+6)
DC_TP_tot.Hour = DC_TP_tot.Hour + 6
DC_TP_tot.replace(to_replace={'Hour': {24:0, 25:1, 26:2, 27:3, 28:4, 29:5}}, inplace=True)
DC_TP_tot = DC_TP_tot.sort_values(by=['Hour'])

DC_TP.Hour = DC_TP.Hour + 6
DC_TP.replace(to_replace={'Hour': {24:0, 25:1, 26:2, 27:3, 28:4, 29:5}}, inplace=True)
DC_TP = DC_TP.sort_values(by=['Hour'])

#%% Plotting
txt = 'The diurnal cycle of precipitation cell frequency (top) and fraction (bottom) at every recorded stage on the TP only\nbetween 2001 and 2016 based on the three model datasets (WRF, HAR & ERA5) and the satellite dataset (GPM)'
fig, axes = plt.subplots(nrows=2, ncols=1)
DC_TP_tot.plot(x='Hour', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', grid=True, ylabel='Frequency', ax=axes[0], linestyle='--', xlim=[-0.5, 23.5], xticks=np.arange(0, 24, step=1), xlabel='Hour (UTC+6)')
DC_TP.plot(x='Hour', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', grid=True, ylabel='Fraction (%)', ax=axes[1], linestyle='--', xlim=[-0.5, 23.5], xticks=np.arange(0, 24, step=1), xlabel='Hour (UTC+6)')
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
fig = plt.gcf()
fig.set_size_inches(12, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/DC_TP.png', dpi=100)