# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:37:19 2021

@author: benja


Skip this section if original track files has not been modified

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Read in tracks file

WRF_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_tracks.pkl')
HAR_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_tracks.pkl')
GPM_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_tracks.pkl')
ERA5_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_tracks.pkl')
GPM_tracks['time'] = pd.to_datetime(GPM_tracks.timestr)
#%% 
"""
Creating an annual cycle of the monthly means of init and max th value

"""
def get_monthly_th_initmean(preciptracks, start, end): 
    preciptracks['month']= preciptracks.time.dt.month
    month = []
    th = []
    monthly_mean = pd.DataFrame() 
    for y in np.arange(start,end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            init_month = ytracks[ytracks.cell == cell].month.values[0]
            init_th = ytracks[ytracks.cell == cell].threshold_value.iloc[0]
            month.append(init_month)
            th.append(init_th)
    monthly_mean['month'] = pd.DataFrame(month)
    monthly_mean['th'] = pd.DataFrame(th)
    monthly_mean = monthly_mean.th.groupby(monthly_mean.month).mean().to_frame()
    return monthly_mean

def get_monthly_th_maxmean(preciptracks, start, end): 
    preciptracks['month']= preciptracks.time.dt.month
    month = []
    th = []
    monthly_mean = pd.DataFrame() 
    for y in np.arange(start,end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            idx = np.argmax(ytracks.ncells[ytracks.cell == cell])
            max_month = ytracks[ytracks.cell == cell].month.values[idx]
            max_th = ytracks[ytracks.cell == cell].threshold_value.iloc[idx]
            month.append(max_month)
            th.append(max_th)
    monthly_mean['month'] = pd.DataFrame(month)
    monthly_mean['th'] = pd.DataFrame(th)
    monthly_mean = monthly_mean.th.groupby(monthly_mean.month).mean().to_frame()
    return monthly_mean

#%% Running and saving monthly mean init
WRF_initmean = get_monthly_th_initmean(WRF_tracks, 2001, 2016)
HAR_initmean = get_monthly_th_initmean(HAR_tracks, 2001, 2016)
ERA5_initmean = get_monthly_th_initmean(ERA5_tracks, 2001, 2016)
GPM_initmean = get_monthly_th_initmean(GPM_tracks, 2001, 2016)

Monthly_initmean = pd.DataFrame(); Monthly_initmean.insert(0, 'Month', range(1, 13)); Monthly_initmean.set_index('Month', inplace=True)
Monthly_initmean['WRF'], Monthly_initmean['HAR'], Monthly_initmean['ERA5'], Monthly_initmean['GPM'] = [WRF_initmean, HAR_initmean, ERA5_initmean, GPM_initmean]
Monthly_initmean.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/Monthly_initmean.pkl')
del WRF_initmean, HAR_initmean, ERA5_initmean, GPM_initmean;

#%% Running and saving monthly mean max
WRF_maxmean = get_monthly_th_maxmean(WRF_tracks, 2001, 2016)
HAR_maxmean = get_monthly_th_maxmean(HAR_tracks, 2001, 2016)
ERA5_maxmean = get_monthly_th_maxmean(ERA5_tracks, 2001, 2016)
GPM_maxmean = get_monthly_th_maxmean(GPM_tracks, 2001, 2016)

Monthly_maxmean = pd.DataFrame(); Monthly_maxmean.insert(0, 'Month', range(1, 13)); Monthly_maxmean.set_index('Month', inplace=True)
Monthly_maxmean['WRF'], Monthly_maxmean['HAR'], Monthly_maxmean['ERA5'], Monthly_maxmean['GPM'] = [WRF_maxmean, HAR_maxmean, ERA5_maxmean, GPM_maxmean]
Monthly_maxmean.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/Monthly_maxmean.pkl')
del WRF_maxmean, HAR_maxmean, ERA5_maxmean, GPM_maxmean;

#%% Plotting initmean
Monthly_initmean = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/Monthly_initmean.pkl')
Monthly_initmean.plot(grid=True)
plt.ylabel('mm/hr > 10 000km2')
plt.xlim(1,12)
plt.xticks(np.arange(1,13, step=1))
#plt.tight_layout()
txt = 'Mean monthly threshold value of precipitation cell at initiation stage between 2001 and 2016 based on the three model datasets\n(WRF, HAR & ERA5) and the satellite dataset (GPM)'
plt.figtext(0.12, 0.01, txt, horizontalalignment='left')
fig = plt.gcf(); fig.set_size_inches(12, 8)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/TH_initmean.png', dpi=100)

#%% Plotting maxmean
Monthly_maxmean = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/Monthly_maxmean.pkl')
Monthly_maxmean.plot(grid=True)
plt.ylabel('mm/hr > 10 000km2')
plt.xlim(1,12)
plt.xticks(np.arange(1,13, step=1))
#plt.tight_layout()
txt = 'Mean monthly threshold value of precipitation cell at maximum threshold value between 2001 and 2016 based on the three model datasets\n(WRF, HAR & ERA5) and the satellite dataset (GPM)'
plt.figtext(0.12, 0.01, txt, horizontalalignment='left')
fig = plt.gcf(); fig.set_size_inches(12, 8)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/TH_maxmean.png', dpi=100)

#%% 
del fig, Monthly_maxmean, Monthly_initmean, txt
#%%
""" 
Creating a histogram of the distribution of each TV (matches with DC amount)

"""

def get_th_initcount(preciptracks,start,end):
    th = []
    th_count = pd.DataFrame()
    for y in np.arange(start,end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            init_th = ytracks[ytracks.cell == cell].threshold_value.values[0]
            th.append(init_th)
    th_count['th'] = pd.DataFrame(th)
    th_count = th_count.th.groupby(th_count.th).count().to_frame()
    return th_count

def get_th_maxcount(preciptracks,start,end):
    th = []
    th_count = pd.DataFrame()
    for y in np.arange(start,end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            idx = np.argmax(ytracks.ncells[ytracks.cell == cell])
            max_th = ytracks[ytracks.cell == cell].threshold_value.values[idx]
            th.append(max_th)
    th_count['th'] = pd.DataFrame(th)
    th_count = th_count.th.groupby(th_count.th).count().to_frame()
    return th_count


#%% Running and saving
## Init
WRF_th_initcount = get_th_initcount(WRF_tracks,2001,2016)
HAR_th_initcount = get_th_initcount(HAR_tracks,2001,2016)
ERA5_th_initcount = get_th_initcount(ERA5_tracks,2001,2016)
GPM_th_initcount = get_th_initcount(GPM_tracks,2001,2016)

TH_initcount = pd.DataFrame(); TH_initcount.insert(0, 'th', range(3,11)); TH_initcount.set_index('th', inplace=True)
TH_initcount['WRF'], TH_initcount['HAR'], TH_initcount['ERA5'], TH_initcount['GPM'] = [WRF_th_initcount, HAR_th_initcount, ERA5_th_initcount, GPM_th_initcount]
TH_initcount.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/TH_initcount.pkl')
del WRF_th_initcount, HAR_th_initcount, ERA5_th_initcount, GPM_th_initcount;

#%% Running and saving
## Max TH
WRF_th_maxcount = get_th_maxcount(WRF_tracks,2001,2016)
HAR_th_maxcount = get_th_maxcount(HAR_tracks,2001,2016)
ERA5_th_maxcount = get_th_maxcount(ERA5_tracks,2001,2016)
GPM_th_maxcount = get_th_maxcount(GPM_tracks,2001,2016)

TH_maxcount = pd.DataFrame(); TH_maxcount.insert(0, 'th', range(3,11)); TH_maxcount.set_index('th', inplace=True)
TH_maxcount['WRF'], TH_maxcount['HAR'], TH_maxcount['ERA5'], TH_maxcount['GPM'] = [WRF_th_maxcount, HAR_th_maxcount, ERA5_th_maxcount, GPM_th_maxcount]
TH_maxcount.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/TH_maxcount.pkl')
del WRF_th_maxcount; del HAR_th_maxcount; del ERA5_th_maxcount; del GPM_th_maxcount;

#%% Creating fraction file
## Init
TH_initcount = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/TH_initcount.pkl')
TH_initfrac = pd.DataFrame(); TH_initfrac.insert(0, 'th', range(3,11)); TH_initfrac.set_index('th', inplace=True)
TH_initfrac['WRF'] = TH_initcount.WRF/TH_initcount.WRF.sum() * 100; TH_initfrac['HAR'] = TH_initcount.HAR/TH_initcount.HAR.sum() * 100
TH_initfrac['ERA5'] = TH_initcount.ERA5/TH_initcount.ERA5.sum() * 100; TH_initfrac['GPM'] = TH_initcount.GPM/TH_initcount.GPM.sum() * 100

#%% Max
TH_maxcount = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/TH_maxcount.pkl')
TH_maxfrac = pd.DataFrame(); TH_maxfrac.insert(0, 'th', range(3,11)); TH_maxfrac.set_index('th', inplace=True)
TH_maxfrac['WRF'] = TH_maxcount.WRF/TH_maxcount.WRF.sum() * 100; TH_maxfrac['HAR'] = TH_maxcount.HAR/TH_maxcount.HAR.sum() * 100
TH_maxfrac['ERA5'] = TH_maxcount.ERA5/TH_maxcount.ERA5.sum() * 100; TH_maxfrac['GPM'] = TH_maxcount.GPM/TH_maxcount.GPM.sum() * 100

#%% Plotting
## Init
TH_initfrac.plot(use_index=True, y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='bar', grid=True, ylabel='Fraction (%)', xlabel='Threshold value')
txt = 'Fraction of precipitation cells for each threshold of precipitation at initation stage based on the three model datasets (WRF, HAR & ERA5)\nand the satellite dataset (GPM)'
plt.figtext(0.12, 0.01, txt, horizontalalignment='left')
fig = plt.gcf(); fig.set_size_inches(12, 8)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/TH_initcount.png', dpi=100)

#%% Max
TH_maxfrac.plot(use_index=True, y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='bar', grid=True, ylabel='Fraction (%)', xlabel='Threshold value')
txt = 'Fraction of precipitation cells for each threshold of precipitation at maximum threshold values based on the three model datasets (WRF, HAR & ERA5)\nand the satellite dataset (GPM)'
plt.figtext(0.12, 0.01, txt, horizontalalignment='left')
fig = plt.gcf(); fig.set_size_inches(12, 8)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/TH_maxcount.png', dpi=100)