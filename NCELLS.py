# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 12:42:16 2021

@author: benja
"""

# Read in tracks file

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%% Get the data

WRF_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_tracks.pkl')
HAR_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_tracks.pkl')
GPM_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_tracks.pkl')
ERA5_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_tracks.pkl')
GPM_tracks['time'] = pd.to_datetime(GPM_tracks.timestr)

WRF_TP = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_TP.pkl')
HAR_TP = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_TP.pkl')
ERA5_TP = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_TP.pkl')
GPM_TP = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_TP.pkl')

#%% 
"""
Count of NCELLS (km2) at initiation stage

Alternative 1

"""

def get_ncells_init(preciptracks, start, end):      
    ncells = []
    for y in np.arange(start,end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            init_ncells = ytracks[ytracks.cell == cell].km2.values[0]
            ncells.append(init_ncells)
    ncells_size, bins = np.histogram(ncells, bins = np.arange(0,525000,25000))
    return ncells_size, bins

#%% Calling function for each dataset
WRF_NC_init,bins = get_ncells_init(WRF_tracks,2001,2016)
HAR_NC_init,bins = get_ncells_init(HAR_tracks,2001,2016)
ERA5_NC_init,bins = get_ncells_init(ERA5_tracks,2001,2016)
GPM_NC_init,bins = get_ncells_init(GPM_tracks,2001,2016)

#%% Concatenating
NC_tot_init = pd.DataFrame(); NC_tot_init.insert(0, 'Size', bins[1:21])
NC_tot_init['WRF'],NC_tot_init['HAR'],NC_tot_init['ERA5'],NC_tot_init['GPM'] = [WRF_NC_init, HAR_NC_init, ERA5_NC_init, GPM_NC_init]
del WRF_NC_init; del HAR_NC_init; del ERA5_NC_init; del GPM_NC_init

#%% Plotting
NC_tot_init.plot(x='Size', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='bar', grid=True, ylabel='Count', xlabel='$km^2$')
txt = 'Histogram of precipitation cell size at initiation stage between 2001 and 2016 for the three model datasets (WRF, HAR & ERA5) and the satellite dataset (GPM)'
plt.figtext(0.12, 0.007, txt, horizontalalignment='left')
fig = plt.gcf(); fig.set_size_inches(20, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/NC_init.png', dpi=100)

#%%
"""
Alternative 2

"""
def get_ncells_init2(preciptracks, start, end):
    temp_ncells = []      
    ncells = pd.DataFrame()
    for y in np.arange(start,end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            init_ncells = ytracks[ytracks.cell == cell].km2.values[0]
            temp_ncells.append(init_ncells)
    ncells['Size'] = pd.DataFrame(temp_ncells)
    return ncells

#%% Study area
WRFinit = get_ncells_init2(WRF_tracks,2001,2016)
HARinit = get_ncells_init2(HAR_tracks,2001,2016)
ERA5init = get_ncells_init2(ERA5_tracks,2001,2016)
GPMinit = get_ncells_init2(GPM_tracks,2001,2016)

#%%
fig, axes = plt.subplots(2,2)
axes[0,0].hist(WRFinit, bins=100, density=True, rwidth = 0.8); axes[0,0].set_xticks(np.arange(10000,510000,50000)); axes[0,0].set_xlim(0,510000);
axes[0,0].set_title('a)'); axes[0,0].set_ylabel('Probability'); axes[0,0].set_ylim(0,0.000022)
axes[0,1].hist(HARinit, bins=100, density=True, rwidth = 0.8); axes[0,1].set_xticks(np.arange(10000,510000,50000)); axes[0,1].set_xlim(0,510000);
axes[0,1].set_title('b)'); axes[0,1].set_ylim(0,0.000022)
axes[1,0].hist(ERA5init, bins=100, density=True, rwidth = 0.8); axes[1,0].set_xticks(np.arange(10000,510000,50000)); axes[1,0].set_xlim(0,510000);
axes[1,0].set_title('c)'); axes[1,0].set_ylabel('Probability'); axes[1,0].set_xlabel('Size (km2)'); axes[1,0].set_ylim(0,0.000022)
axes[1,1].hist(GPMinit, bins=100, density=True, rwidth = 0.8); axes[1,1].set_xticks(np.arange(10000,510000,50000)); axes[1,1].set_xlim(0,510000);
axes[1,1].set_title('d)'); axes[1,1].set_xlabel('Size (km2)'); axes[1,1].set_ylim(0,0.000022)
plt.style.use('seaborn')
txt = 'Probability density function of precipitation cell size at initiation stage for WRF (a), HAR (b), ERA5 (c) and GPM (d)'
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
fig.set_size_inches(16, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/PDF_km2_init.png', dpi=100)

#%% TP only
TP_WRFinit = get_ncells_init2(WRF_TP,2001,2016)
TP_HARinit = get_ncells_init2(HAR_TP,2001,2016)
TP_ERA5init = get_ncells_init2(ERA5_TP,2001,2016)
TP_GPMinit = get_ncells_init2(GPM_TP,2001,2016)

#%%
fig, axes = plt.subplots(2,2)
axes[0,0].hist(TP_WRFinit, bins=100, density=True, rwidth = 0.8); axes[0,0].set_xticks(np.arange(10000,510000,50000)); axes[0,0].set_xlim(0,510000);
axes[0,0].set_title('a)'); axes[0,0].set_ylabel('Probability'); axes[0,0].set_ylim(0,0.000022)
axes[0,1].hist(TP_HARinit, bins=100, density=True, rwidth = 0.8); axes[0,1].set_xticks(np.arange(10000,510000,50000)); axes[0,1].set_xlim(0,510000);
axes[0,1].set_title('b)'); axes[0,1].set_ylim(0,0.000022)
axes[1,0].hist(TP_ERA5init, bins=100, density=True, rwidth = 0.8); axes[1,0].set_xticks(np.arange(10000,510000,50000)); axes[1,0].set_xlim(0,510000);
axes[1,0].set_title('c)'); axes[1,0].set_ylabel('Probability'); axes[1,0].set_xlabel('Size (km2)'); axes[1,0].set_ylim(0,0.000022)
axes[1,1].hist(TP_GPMinit, bins=100, density=True, rwidth = 0.8); axes[1,1].set_xticks(np.arange(10000,510000,50000)); axes[1,1].set_xlim(0,510000);
axes[1,1].set_title('d)'); axes[1,1].set_xlabel('Size (km2)'); axes[1,1].set_ylim(0,0.000022)
plt.style.use('seaborn')
txt = 'Probability density function of precipitation cell size at initiation stage for WRF (a), HAR (b), ERA5 (c) and GPM (d)'
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
fig.set_size_inches(16, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/PDF_km2_init_TP.png', dpi=100)

#%%
del WRFinit, HARinit, ERA5init, GPMinit, fig, axes, txt, bins, TP_WRFinit, TP_HARinit, TP_ERA5init, TP_GPMinit, NC_tot_init

#%%
"""
Count of NCELLS at dissolution stage

"""
def get_ncells_diss(preciptracks, start, end):      
    ncells = []
    for y in np.arange(start,end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            diss_ncells = ytracks[ytracks.cell == cell].km2.values[-1]
            ncells.append(diss_ncells)
    ncells_size, bins = np.histogram(ncells, bins = np.arange(0,525000,25000))
    return ncells_size, bins

#%%
WRF_NC_diss,bins = get_ncells_diss(WRF_tracks,2001,2016)
HAR_NC_diss,bins = get_ncells_diss(HAR_tracks,2001,2016)
ERA5_NC_diss,bins = get_ncells_diss(ERA5_tracks,2001,2016)
GPM_NC_diss,bins = get_ncells_diss(GPM_tracks,2001,2016)

#%%
NC_tot_diss = pd.DataFrame(); NC_tot_diss.insert(0, 'Size', bins[1:21])
NC_tot_diss['WRF'],NC_tot_diss['HAR'],NC_tot_diss['ERA5'],NC_tot_diss['GPM'] = [WRF_NC_diss, HAR_NC_diss, ERA5_NC_diss, GPM_NC_diss]
del WRF_NC_diss; del HAR_NC_diss; del ERA5_NC_diss; del GPM_NC_diss

#%% Plotting
NC_tot_diss.plot(x='Size', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='bar', grid=True, ylabel='Count', xlabel='$km^2$')
txt = 'Histogram of precipitation cell size at dissolution stage between 2001 and 2016 for the three model datasets (WRF, HAR & ERA5) and the satellite dataset (GPM)'
plt.figtext(0.12, 0.007, txt, horizontalalignment='left')
fig = plt.gcf(); fig.set_size_inches(20, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/NC_diss.png', dpi=100)

#%%
"""
Alternative 2

"""
def get_ncells_diss2(preciptracks, start, end):
    temp_ncells = []      
    ncells = pd.DataFrame()
    for y in np.arange(start,end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            diss_ncells = ytracks[ytracks.cell == cell].km2.values[-1]
            temp_ncells.append(diss_ncells)
    ncells['Size'] = pd.DataFrame(temp_ncells)
    return ncells

#%% Study area
WRFdiss = get_ncells_diss2(WRF_tracks,2001,2016)
HARdiss = get_ncells_diss2(HAR_tracks,2001,2016)
ERA5diss = get_ncells_diss2(ERA5_tracks,2001,2016)
GPMdiss = get_ncells_diss2(GPM_tracks,2001,2016)

#%%
fig, axes = plt.subplots(2,2)
axes[0,0].hist(WRFdiss, bins=100, density=True, rwidth = 0.8); axes[0,0].set_xticks(np.arange(10000,510000,50000)); axes[0,0].set_xlim(0,510000);
axes[0,0].set_title('a)'); axes[0,0].set_ylabel('Probability'); axes[0,0].set_ylim(0,0.000022)
axes[0,1].hist(HARdiss, bins=100, density=True, rwidth = 0.8); axes[0,1].set_xticks(np.arange(10000,510000,50000)); axes[0,1].set_xlim(0,510000);
axes[0,1].set_title('b)'); axes[0,1].set_ylim(0,0.000022)
axes[1,0].hist(ERA5diss, bins=100, density=True, rwidth = 0.8); axes[1,0].set_xticks(np.arange(10000,510000,50000)); axes[1,0].set_xlim(0,510000);
axes[1,0].set_title('c)'); axes[1,0].set_ylabel('Probability'); axes[1,0].set_xlabel('Size (km2)'); axes[1,0].set_ylim(0,0.000022)
axes[1,1].hist(GPMdiss, bins=100, density=True, rwidth = 0.8); axes[1,1].set_xticks(np.arange(10000,510000,50000)); axes[1,1].set_xlim(0,510000);
axes[1,1].set_title('d)'); axes[1,1].set_xlabel('Size (km2)'); axes[1,1].set_ylim(0,0.000022)
plt.style.use('seaborn')
txt = 'Probability density function of precipitation cell size at dissolution stage for WRF (a), HAR (b), ERA5 (c) and GPM (d)'
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
fig.set_size_inches(16, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/PDF_km2_diss.png', dpi=100)

#%% TP only
TP_WRFdiss = get_ncells_diss2(WRF_TP,2001,2016)
TP_HARdiss = get_ncells_diss2(HAR_TP,2001,2016)
TP_ERA5diss = get_ncells_diss2(ERA5_TP,2001,2016)
TP_GPMdiss = get_ncells_diss2(GPM_TP,2001,2016)

#%%
fig, axes = plt.subplots(2,2)
axes[0,0].hist(TP_WRFdiss, bins=100, density=True, rwidth = 0.8); axes[0,0].set_xticks(np.arange(10000,510000,50000)); axes[0,0].set_xlim(0,510000);
axes[0,0].set_title('a)'); axes[0,0].set_ylabel('Probability'); axes[0,0].set_ylim(0,0.000025)
axes[0,1].hist(TP_HARdiss, bins=100, density=True, rwidth = 0.8); axes[0,1].set_xticks(np.arange(10000,510000,50000)); axes[0,1].set_xlim(0,510000);
axes[0,1].set_title('b)'); axes[0,1].set_ylim(0,0.000025)
axes[1,0].hist(TP_ERA5diss, bins=100, density=True, rwidth = 0.8); axes[1,0].set_xticks(np.arange(10000,510000,50000)); axes[1,0].set_xlim(0,510000);
axes[1,0].set_title('c)'); axes[1,0].set_ylabel('Probability'); axes[1,0].set_xlabel('Size (km2)'); axes[1,0].set_ylim(0,0.000025)
axes[1,1].hist(TP_GPMdiss, bins=100, density=True, rwidth = 0.8); axes[1,1].set_xticks(np.arange(10000,510000,50000)); axes[1,1].set_xlim(0,510000);
axes[1,1].set_title('d)'); axes[1,1].set_xlabel('Size (km2)'); axes[1,1].set_ylim(0,0.000025)
plt.style.use('seaborn')
txt = 'Probability density function of precipitation cell size at dissolution stage for WRF (a), HAR (b), ERA5 (c) and GPM (d)'
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
fig.set_size_inches(16, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/PDF_km2_diss_TP.png', dpi=100)

#%% 
del WRFdiss, HARdiss, ERA5diss, GPMdiss, fig, axes, txt, TP_WRFdiss, TP_HARdiss, TP_ERA5diss, TP_GPMdiss, NC_tot_diss, bins

#%%
""" 
Count of NCELLS at maximum size

"""

def get_ncells_max(preciptracks, start, end):         
    ncells=[]
    for y in np.arange(start,end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            idx = np.argmax(ytracks.km2[ytracks.cell == cell])
            max_ncells = ytracks[ytracks.cell == cell].km2.values[idx]
            ncells.append(max_ncells)
    ncells_size, bins = np.histogram(ncells, bins = np.arange(0,525000,25000))
    return ncells_size, bins

#%%
WRF_NC_max,bins = get_ncells_max(WRF_tracks,2001,2016)
HAR_NC_max,bins = get_ncells_max(HAR_tracks,2001,2016)
ERA5_NC_max,bins = get_ncells_max(ERA5_tracks,2001,2016)
GPM_NC_max,bins = get_ncells_max(GPM_tracks,2001,2016)

#%%
NC_tot_max = pd.DataFrame(); NC_tot_max.insert(0, 'Size', bins[1:21])
NC_tot_max['WRF'],NC_tot_max['HAR'],NC_tot_max['ERA5'],NC_tot_max['GPM'] = [WRF_NC_max, HAR_NC_max, ERA5_NC_max, GPM_NC_max]
del WRF_NC_max; del HAR_NC_max; del ERA5_NC_max; del GPM_NC_max

#%% Plotting
NC_tot_max.plot(x='Size', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='bar', grid=True, ylabel='Count', xlabel='$km^2$')
txt = 'Histogram of precipitation cell size at maximum extension stage between 2001 and 2016 for the three model datasets (WRF, HAR & ERA5) and the satellite dataset (GPM)'
plt.figtext(0.12, 0.007, txt, horizontalalignment='left')
fig = plt.gcf(); fig.set_size_inches(20, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/NC_max.png', dpi=100)

#%%
"""
Alternative 2

"""
def get_ncells_max2(preciptracks, start, end):
    temp_ncells = []      
    ncells = pd.DataFrame()
    for y in np.arange(start,end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            idx = np.argmax(ytracks.km2[ytracks.cell == cell])
            max_ncells = ytracks[ytracks.cell == cell].km2.values[idx]
            temp_ncells.append(max_ncells)
    ncells['Size'] = pd.DataFrame(temp_ncells)
    return ncells

#%% Study area
WRFmax = get_ncells_max2(WRF_tracks,2001,2016)
HARmax = get_ncells_max2(HAR_tracks,2001,2016)
ERA5max = get_ncells_max2(ERA5_tracks,2001,2016)
GPMmax = get_ncells_max2(GPM_tracks,2001,2016)

#%%
fig, axes = plt.subplots(2,2)
axes[0,0].hist(WRFmax, bins=100, density=True, rwidth = 0.8); axes[0,0].set_xticks(np.arange(10000,510000,50000)); axes[0,0].set_xlim(0,510000);
axes[0,0].set_title('a)'); axes[0,0].set_ylabel('Probability'); axes[0,0].set_ylim(0,0.000015)
axes[0,1].hist(HARmax, bins=100, density=True, rwidth = 0.8); axes[0,1].set_xticks(np.arange(10000,510000,50000)); axes[0,1].set_xlim(0,510000);
axes[0,1].set_title('b)'); axes[0,1].set_ylim(0,0.000015)
axes[1,0].hist(ERA5max, bins=100, density=True, rwidth = 0.8); axes[1,0].set_xticks(np.arange(10000,510000,50000)); axes[1,0].set_xlim(0,510000);
axes[1,0].set_title('c)'); axes[1,0].set_ylabel('Probability'); axes[1,0].set_xlabel('Size (km2)'); axes[1,0].set_ylim(0,0.000015)
axes[1,1].hist(GPMmax, bins=100, density=True, rwidth = 0.8); axes[1,1].set_xticks(np.arange(10000,510000,50000)); axes[1,1].set_xlim(0,510000);
axes[1,1].set_title('d)'); axes[1,1].set_xlabel('Size (km2)'); axes[1,1].set_ylim(0,0.000015)
plt.style.use('seaborn')
txt = 'Probability density function of precipitation cell size at maximum extension stage for WRF (a), HAR (b), ERA5 (c) and GPM (d)'
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
fig.set_size_inches(16, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/PDF_km2_max.png', dpi=100)

#%% TP only 
TP_WRFmax = get_ncells_max2(WRF_TP,2001,2016)
TP_HARmax = get_ncells_max2(HAR_TP,2001,2016)
TP_ERA5max = get_ncells_max2(ERA5_TP,2001,2016)
TP_GPMmax = get_ncells_max2(GPM_TP,2001,2016)

#%%
fig, axes = plt.subplots(2,2)
axes[0,0].hist(TP_WRFmax, bins=100, density=True, rwidth = 0.8); axes[0,0].set_xticks(np.arange(10000,510000,50000)); axes[0,0].set_xlim(0,510000);
axes[0,0].set_title('a)'); axes[0,0].set_ylabel('Probability'); axes[0,0].set_ylim(0,0.000015)
axes[0,1].hist(TP_HARmax, bins=100, density=True, rwidth = 0.8); axes[0,1].set_xticks(np.arange(10000,510000,50000)); axes[0,1].set_xlim(0,510000);
axes[0,1].set_title('b)'); axes[0,1].set_ylim(0,0.000015)
axes[1,0].hist(TP_ERA5max, bins=100, density=True, rwidth = 0.8); axes[1,0].set_xticks(np.arange(10000,510000,50000)); axes[1,0].set_xlim(0,510000);
axes[1,0].set_title('c)'); axes[1,0].set_ylabel('Probability'); axes[1,0].set_xlabel('Size (km2)'); axes[1,0].set_ylim(0,0.000015)
axes[1,1].hist(TP_GPMmax, bins=100, density=True, rwidth = 0.8); axes[1,1].set_xticks(np.arange(10000,510000,50000)); axes[1,1].set_xlim(0,510000);
axes[1,1].set_title('d)'); axes[1,1].set_xlabel('Size (km2)'); axes[1,1].set_ylim(0,0.000015)
plt.style.use('seaborn')
txt = 'Probability density function of precipitation cell size at maximum extension stage for WRF (a), HAR (b), ERA5 (c) and GPM (d)'
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
fig.set_size_inches(16, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/PDF_km2_max_TP.png', dpi=100)

#%% 
del WRFmax, HARmax, ERA5max, GPMmax, fig, axes, txt, TP_WRFmax, TP_HARmax, TP_ERA5max, TP_GPMmax, NC_tot_max, bins