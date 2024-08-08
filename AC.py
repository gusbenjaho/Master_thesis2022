# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 11:18:35 2021

@author: benja
"""

"""
# Annual cycle (AC)

"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

WRF_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/WRF_tracks_filtered.pkl')
HAR_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/HAR_tracks_filtered.pkl')
GPM_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/GPM_tracks_filtered.pkl')
ERA5_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/ERA5_tracks_filtered.pkl')

#%% Functions for AC
def get_monthly_init(preciptracks, start, end): 
    preciptracks['month']= preciptracks.time.dt.month 
    monthly=[] 
    for y in np.arange(start,end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            init_month = ytracks[ytracks.cell == cell].month.values[0]
            monthly.append(init_month)
            
    monthly_precip, bins = np.histogram(monthly, bins = np.arange(1,14))
    return monthly_precip, bins

def get_monthly_diss(preciptracks, start, end): 
    preciptracks['month']= preciptracks.time.dt.month 
    monthly=[] 
    for y in np.arange(start,end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            diss_month = ytracks[ytracks.cell == cell].month.values[-1]
            monthly.append(diss_month)
            
    monthly_precip, bins = np.histogram(monthly, bins = np.arange(1,14))
    return monthly_precip, bins

def get_monthly_max(preciptracks, start, end): 
    preciptracks['month']= preciptracks.time.dt.month 
    monthly=[] 
    for y in np.arange(start,end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            idx = np.argmax(ytracks.ncells[ytracks.cell == cell])
            max_month = ytracks[ytracks.cell == cell].month.values[idx]
            monthly.append(max_month)
            
    monthly_precip, bins = np.histogram(monthly, bins = np.arange(1,14))
    return monthly_precip, bins

#%% Running and saving
# WRF
WRF_init, bins = get_monthly_init(WRF_tracks, 2001, 2017)
#WRF_diss, bins = get_monthly_diss(WRF_tracks, 2001, 2017)
#WRF_max, bins = get_monthly_max(WRF_tracks, 2001, 2017)
np.save('D:/Benjamin/MasterThesis2021-2022/data/Python_files/WRF_AC_init.npy', WRF_init)
#np.save('D:/Benjamin/MasterThesis2021-2022/data/Python_files/WRF_AC_diss.npy', WRF_diss)
#np.save('D:/Benjamin/MasterThesis2021-2022/data/Python_files/WRF_AC_max.npy', WRF_max)
del WRF_init, bins;# del WRF_diss; del WRF_max; del bins

#%% HAR
HAR_init, bins = get_monthly_init(HAR_tracks, 2001, 2017)
#HAR_diss, bins = get_monthly_diss(HAR_tracks, 2001, 2017)
#HAR_max, bins = get_monthly_max(HAR_tracks, 2001, 2017)
np.save('D:/Benjamin/MasterThesis2021-2022/data/Python_files/HAR_AC_init.npy', HAR_init)
#np.save('D:/Benjamin/MasterThesis2021-2022/data/Python_files/HAR_AC_diss.npy', HAR_diss)
#np.save('D:/Benjamin/MasterThesis2021-2022/data/Python_files/HAR_AC_max.npy', HAR_max)
del HAR_init, HAR_tracks, bins;# del HAR_diss; del HAR_max; del bins

#%% ERA5
ERA5_init, bins = get_monthly_init(ERA5_tracks, 2001, 2017)
#ERA5_diss, bins = get_monthly_diss(ERA5_tracks, 2001, 2017)
#ERA5_max, bins = get_monthly_max(ERA5_tracks, 2001, 2017)
np.save('D:/Benjamin/MasterThesis2021-2022/data/Python_files/ERA5_AC_init.npy', ERA5_init)
#np.save('D:/Benjamin/MasterThesis2021-2022/data/Python_files/ERA5_AC_diss.npy', ERA5_diss)
#np.save('D:/Benjamin/MasterThesis2021-2022/data/Python_files/ERA5_AC_max.npy', ERA5_max)
del ERA5_init, ERA5_tracks, bins;# del ERA5_diss; del ERA5_max; del bins

#%% GPM
GPM_tracks['time'] = pd.to_datetime(GPM_tracks.timestr)
GPM_init, bins = get_monthly_init(GPM_tracks, 2001, 2017)
#GPM_diss, bins = get_monthly_diss(GPM_tracks, 2001, 2017)
#GPM_max, bins = get_monthly_max(GPM_tracks, 2001, 2017)
np.save('D:/Benjamin/MasterThesis2021-2022/data/Python_files/GPM_AC_init.npy', GPM_init)
#np.save('D:/Benjamin/MasterThesis2021-2022/data/Python_files/GPM_AC_diss.npy', GPM_diss)
#np.save('D:/Benjamin/MasterThesis2021-2022/data/Python_files/GPM_AC_max.npy', GPM_max)
del GPM_init, GPM_tracks, bins;# del GPM_diss; del GPM_max; del bins

#%%
# AC_init files from above

WRF_init = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/WRF_AC_init.npy')
HAR_init = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/HAR_AC_init.npy')
ERA5_init = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/ERA5_AC_init.npy')
GPM_init = np.load('D:/Benjamin/MasterThesis2021-2022/data/Python_files/GPM_AC_init.npy')


#%% Total count
AC_tot = pd.DataFrame(); AC_tot.insert(0, 'Month', range(1, 13))
AC_tot['WRF_GU'],AC_tot['HAR'],AC_tot['ERA5'],AC_tot['GPM'] = [WRF_init, HAR_init, ERA5_init, GPM_init]
del WRF_init; del HAR_init; del ERA5_init; del GPM_init

#%% Normalize 
AC = pd.DataFrame(); AC.insert(0, 'Month', range(1, 13))
AC['WRF_GU'] = AC_tot.WRF_GU/AC_tot.WRF_GU.sum() * 100; AC['HAR'] = AC_tot.HAR/AC_tot.HAR.sum() * 100
AC['ERA5'] = AC_tot.ERA5/AC_tot.ERA5.sum() * 100 ; AC['GPM'] = AC_tot.GPM/AC_tot.GPM.sum() * 100

#%% Plotting
plt.style.use('seaborn')
txt = 'Annual cycle of precipitation cell frequency (top) and fraction (bottom) at initiation stage between 2001 and 2016 based on the three\nmodel datasets (WRF, HAR & ERA5) and the satellite dataset (GPM)'
fig, axes = plt.subplots(nrows=2, ncols=1)
colors = ['#1E90FF', '#104E8B', '#FF6103', '#FF9912']; y_data=['WRF_GU', 'HAR', 'ERA5', 'GPM']
AC_tot.plot(x='Month', y=y_data, kind='line', grid=True, ylabel='Frequency', xlabel='', ax=axes[0], linestyle='-', xlim=[0.5,12.5], xticks=np.arange(0,24, step=1), color=colors, title='a)')
AC.plot(x='Month', y=y_data, kind='line', grid=True, ylabel='Fraction [%]', ax=axes[1], linestyle='-', xlim=[0.5,12.5], xticks=np.arange(0,24, step=1), color=colors, title='b)')
axes[0].legend(loc='upper left',bbox_to_anchor=(1.0, 1))
axes[1].legend(loc='upper left',bbox_to_anchor=(1.0, 1))
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
fig = plt.gcf(); fig.set_size_inches(12, 10)
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/AC_init_color.png', dpi=100)

#%%
"""
SEASONS

"""

DJF = AC.loc[[11,0,1],:]; MAM = AC.loc[[2,3,4],:]; JJA = AC.loc[[5,6,7],:]; SON = AC.loc[[8,9,10],:]
DJF_tot = AC_tot.loc[[11,0,1],:]; MAM_tot = AC_tot.loc[[2,3,4],:]; JJA_tot = AC_tot.loc[[5,6,7],:]; SON_tot = AC_tot.loc[[8,9,10],:]

#%% Plotting seasons (%)
fig, axes = plt.subplots(nrows=2, ncols=2)
DJF.plot(x='Month', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='bar', grid=True, ylabel='Count(%)', title='DJF', ax=axes[0,0])
MAM.plot(x='Month', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='bar', grid=True, ylabel='Count(%)', title='MAM', ax=axes[0,1])
JJA.plot(x='Month', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='bar', grid=True, ylabel='Count(%)', title='JJA', ax=axes[1,0])
SON.plot(x='Month', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='bar', grid=True, ylabel='Count(%)', title='SON', ax=axes[1,1])

fig = plt.gcf(); fig.set_size_inches(12, 10)
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/AC_seasons(%).png')

#%% Plotting seasons (total)
fig, axes = plt.subplots(nrows=2, ncols=2)
DJF_tot.plot(x='Month', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='bar', grid=True, ylabel='Count', title='DJF', ylim=(0,2600), ax=axes[0,0])
MAM_tot.plot(x='Month', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='bar', grid=True, ylabel='Count', title='MAM', ylim=(0,2600), ax=axes[0,1])
JJA_tot.plot(x='Month', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='bar', grid=True, ylabel='Count', title='JJA', ylim=(0,2600), ax=axes[1,0])
SON_tot.plot(x='Month', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='bar', grid=True, ylabel='Count', title='SON', ylim=(0,2600), ax=axes[1,1])

fig = plt.gcf(); fig.set_size_inches(12, 10)
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/AC_seasons(total).png', dpi=100)