# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 09:26:40 2021

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

#%% 
"""
Summer months

"""

WRF_JJA = WRF_tracks[(WRF_tracks.time.dt.month == 6) | (WRF_tracks.time.dt.month == 7) | (WRF_tracks.time.dt.month == 8)]
HAR_JJA = HAR_tracks[(HAR_tracks.time.dt.month == 6) | (HAR_tracks.time.dt.month == 7) | (HAR_tracks.time.dt.month == 8)]
ERA5_JJA = ERA5_tracks[(ERA5_tracks.time.dt.month == 6) | (ERA5_tracks.time.dt.month == 7) | (ERA5_tracks.time.dt.month == 8)]
GPM_JJA = GPM_tracks[(GPM_tracks.time.dt.month == 6) | (GPM_tracks.time.dt.month == 7) | (GPM_tracks.time.dt.month == 8)]

#%% 
"""
Diurnal count for init hour

"""

# DEF creates a function based on coming loop, these args necessary to call function later:
# diurnal_precip, bins = get_diurnal_init(tracks, 2000, 2015)
def get_diurnal_init(preciptracks, start, end): 
    preciptracks['hour']= preciptracks.time.dt.hour 
    diurnal=[] 
    for y in np.arange(start,end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            init_hour = ytracks[ytracks.cell == cell].hour.values[0]
            diurnal.append(init_hour)
            
    diurnal_precip, bins = np.histogram(diurnal, bins = np.arange(0,25))
    return diurnal_precip, bins

#%% WRF_JJA
WRFJJA_DC, bins = get_diurnal_init(WRF_JJA, 2001, 2016)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRFJJA_DC.npy', WRFJJA_DC)

#%% HAR_JJA
HARJJA_DC, bins = get_diurnal_init(HAR_JJA, 2001, 2016)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HARJJA_DC.npy', HARJJA_DC)

#%% ERA5_JJA
ERA5JJA_DC, bins = get_diurnal_init(ERA5_JJA, 2001, 2016)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5JJA_DC.npy', ERA5JJA_DC)

#%% GPM_JJA
GPMJJA_DC, bins = get_diurnal_init(GPM_JJA, 2001, 2016)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPMJJA_DC.npy', GPMJJA_DC)
del bins

#%% Summer months from above
WRFJJA_DC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRFJJA_DC.npy')
HARJJA_DC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HARJJA_DC.npy')
ERA5JJA_DC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5JJA_DC.npy')
GPMJJA_DC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPMJJA_DC.npy')

#%% Total count JJA
DC_JJA_tot = pd.DataFrame(); DC_JJA_tot.insert(0, 'Hour', range(0, 24))
DC_JJA_tot['WRF'],DC_JJA_tot['HAR'],DC_JJA_tot['ERA5'],DC_JJA_tot['GPM'] = [WRFJJA_DC, HARJJA_DC, ERA5JJA_DC, GPMJJA_DC]
del WRFJJA_DC; del HARJJA_DC; del ERA5JJA_DC; del GPMJJA_DC

#%% Normalize JJA
DC_JJA = pd.DataFrame(); DC_JJA.insert(0, 'Hour', range(0, 24))
DC_JJA['WRF'] = DC_JJA_tot.WRF/DC_JJA_tot.WRF.sum() * 100; DC_JJA['HAR'] = DC_JJA_tot.HAR/DC_JJA_tot.HAR.sum() * 100
DC_JJA['ERA5'] = DC_JJA_tot.ERA5/DC_JJA_tot.ERA5.sum() * 100 ; DC_JJA['GPM'] = DC_JJA_tot.GPM/DC_JJA_tot.GPM.sum() * 100

#%% Plotting JJA
txt = 'The diurnal cycle of precipitation cell frequency (top) and fraction (bottom) at initiation stage during summer months (JJA) between\n2001 and 2016 based on the three model datasets (WRF, HAR & ERA5) and the satellite dataset (GPM)'
fig, axes = plt.subplots(nrows=2, ncols=1)
DC_JJA_tot.plot(x='Hour', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', grid=True, ylabel='Frequency', ax=axes[0], linestyle='--', xlim=[-0.5,23.5], xticks=np.arange(0,24, step=1), xlabel='Hour (UTC)')
DC_JJA.plot(x='Hour', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', grid=True, ylabel='Fraction (%)', ax=axes[1], linestyle='--', xlim=[-0.5,23.5], xticks=np.arange(0,24, step=1), xlabel='Hour (UTC)')
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
fig = plt.gcf(); fig.set_size_inches(12, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/DCJJA.png', dpi=100)

#%% JJA + All year (%)
txt = 'The diurnal cycle of precipitation cell fraction at initiation stage for every month (top) and summer months (JJA) (bottom) between\n2001 and 2016 based on the three model datasets (WRF, HAR & ERA5) and the satellite dataset (GPM)'
DC = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/DC.pkl')
fig, axes = plt.subplots(nrows=2, ncols=1)
DC.plot(x='Hour', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', grid=True, ylabel='Count(%)', title='All year', ax=axes[0], linestyle='--', xlim=[-0.5,23.5], xticks=np.arange(0,24, step=1), xlabel='Hour (UTC)')
DC_JJA.plot(x='Hour', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', grid=True, ylabel='Count(%)', title='JJA', ax=axes[1], linestyle='--', xlim=[-0.5,23.5], xticks=np.arange(0,24, step=1), xlabel='Hour (UTC)')
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
fig = plt.gcf(); fig.set_size_inches(12, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/DC_DCJJA(%).png', dpi=100)

#%% JJA + All year (total)
txt = 'The diurnal cycle of precipitation cell frequency at initiation stage for every month (top) and summer months (JJA) (bottom) between\n2001 and 2016 based on the three model datasets (WRF, HAR & ERA5) and the satellite dataset (GPM)'
DC_tot = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/DC_tot.pkl')
fig, axes = plt.subplots(nrows=2, ncols=1)
DC_tot.plot(x='Hour', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', grid=True, ylabel='Frequency', title='All year', ax=axes[0], linestyle='--', xlim=[-0.5,23.5], xticks=np.arange(0,24, step=1), xlabel='Hour (UTC)')
DC_JJA_tot.plot(x='Hour', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', grid=True, ylabel='Frequency', title='JJA', ax=axes[1], linestyle='--', xlim=[-0.5,23.5], xticks=np.arange(0,24, step=1), xlabel='Hour (UTC)')
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
fig = plt.gcf(); fig.set_size_inches(12, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/DC_DCJJA(total).png', dpi=100)
#%% 
"""
Monthly HAR

"""
# Does not work yet!!
def get_monthly_diurnal_init(preciptracks, start, end): 
    preciptracks['hour']= preciptracks.time.dt.hour 
    diurnal_monthly=[]
    diurnal_monthly_count = pd.DataFrame()
    for y in np.arange(start,end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for m in np.arange(1,13):
            mtracks = ytracks[ytracks.time.dt.month == m]
            for cell in np.unique(mtracks.cell.values):
                init_hour = mtracks[mtracks.cell == cell].hour.values[0]
                diurnal_monthly.append(init_hour)
                diurnal_monthly_count[m], bins = np.histogram(diurnal_monthly, bins = np.arange(0,25))
            
    return diurnal_monthly_count

#%% Testing above code
diurnal_monthly_count = get_monthly_diurnal_init(WRF_tracks,2001,2016)

#%% Do this for now
HAR_JAN = HAR_tracks[HAR_tracks.time.dt.month == 1]; HAR_JAN, bins = get_diurnal_init(HAR_JAN,2001,2016)
HAR_JAN = pd.DataFrame(HAR_JAN)
HAR_JAN.insert(0, 'Hour', range(1, 25))
HAR_JAN.rename(columns={ 0:'Jan'}, inplace=True)
HAR_FEB = HAR_tracks[HAR_tracks.time.dt.month == 2]; HAR_FEB, bins = get_diurnal_init(HAR_FEB,2001,2016); HAR_FEB = pd.DataFrame(HAR_FEB); HAR_FEB.rename(columns={ 0:'Feb'}, inplace=True)
HAR_MAR = HAR_tracks[HAR_tracks.time.dt.month == 3]; HAR_MAR, bins = get_diurnal_init(HAR_MAR,2001,2016); HAR_MAR = pd.DataFrame(HAR_MAR); HAR_MAR.rename(columns={ 0:'Mar'}, inplace=True)
HAR_APR = HAR_tracks[HAR_tracks.time.dt.month == 4]; HAR_APR, bins = get_diurnal_init(HAR_APR,2001,2016); HAR_APR = pd.DataFrame(HAR_APR); HAR_APR.rename(columns={ 0:'Apr'}, inplace=True)
HAR_MAY = HAR_tracks[HAR_tracks.time.dt.month == 5]; HAR_MAY, bins = get_diurnal_init(HAR_MAY,2001,2016); HAR_MAY = pd.DataFrame(HAR_MAY); HAR_MAY.rename(columns={ 0:'May'}, inplace=True)
HAR_JUN = HAR_tracks[HAR_tracks.time.dt.month == 6]; HAR_JUN, bins = get_diurnal_init(HAR_JUN,2001,2016); HAR_JUN = pd.DataFrame(HAR_JUN); HAR_JUN.rename(columns={ 0:'Jun'}, inplace=True)
HAR_JUL = HAR_tracks[HAR_tracks.time.dt.month == 7]; HAR_JUL, bins = get_diurnal_init(HAR_JUL,2001,2016); HAR_JUL = pd.DataFrame(HAR_JUL); HAR_JUL.rename(columns={ 0:'Jul'}, inplace=True)
HAR_AUG = HAR_tracks[HAR_tracks.time.dt.month == 8]; HAR_AUG, bins = get_diurnal_init(HAR_AUG,2001,2016); HAR_AUG = pd.DataFrame(HAR_AUG); HAR_AUG.rename(columns={ 0:'Aug'}, inplace=True)
HAR_SEP = HAR_tracks[HAR_tracks.time.dt.month == 9]; HAR_SEP, bins = get_diurnal_init(HAR_SEP,2001,2016); HAR_SEP = pd.DataFrame(HAR_SEP); HAR_SEP.rename(columns={ 0:'Sep'}, inplace=True)
HAR_OCT = HAR_tracks[HAR_tracks.time.dt.month == 10]; HAR_OCT, bins = get_diurnal_init(HAR_OCT,2001,2016); HAR_OCT = pd.DataFrame(HAR_OCT); HAR_OCT.rename(columns={ 0:'Oct'}, inplace=True)
HAR_NOV = HAR_tracks[HAR_tracks.time.dt.month == 11]; HAR_NOV, bins = get_diurnal_init(HAR_NOV,2001,2016); HAR_NOV = pd.DataFrame(HAR_NOV); HAR_NOV.rename(columns={ 0:'Nov'}, inplace=True)
HAR_DEC = HAR_tracks[HAR_tracks.time.dt.month == 12]; HAR_DEC, bins = get_diurnal_init(HAR_DEC,2001,2016); HAR_DEC = pd.DataFrame(HAR_DEC); HAR_DEC.rename(columns={ 0:'Dec'}, inplace=True)

DC_monthly = pd.concat([HAR_JAN, HAR_FEB, HAR_MAR, HAR_APR, HAR_MAY, HAR_JUN, HAR_JUL, HAR_AUG, HAR_SEP, HAR_OCT, HAR_NOV, HAR_DEC],axis=1)
del HAR_JAN; del HAR_FEB; del HAR_MAR; del HAR_APR; del HAR_MAY; del HAR_JUN; del HAR_JUL; del HAR_AUG; del HAR_SEP; del HAR_OCT; del HAR_NOV; del HAR_DEC

#%% Plotting monthly 
plt.subplot(3,4,1)
plt.bar(DC_monthly.Hour, DC_monthly.Jan); plt.ylabel('Count'); plt.xticks(np.arange(1,25,step=2)); plt.title('Jan'); plt.grid()

plt.subplot(3,4,2)
plt.bar(DC_monthly.Hour, DC_monthly.Feb); plt.ylabel('Count'); plt.xticks(np.arange(1,25,step=2)); plt.title('Feb'); plt.grid()

plt.subplot(3,4,3)
plt.bar(DC_monthly.Hour, DC_monthly.Mar); plt.ylabel('Count'); plt.xticks(np.arange(1,25,step=2)); plt.title('Mar'); plt.grid()

plt.subplot(3,4,4)
plt.bar(DC_monthly.Hour, DC_monthly.Apr); plt.ylabel('Count'); plt.xticks(np.arange(1,25,step=2)); plt.title('Apr'); plt.grid()

plt.subplot(3,4,5)
plt.bar(DC_monthly.Hour, DC_monthly.May); plt.ylabel('Count'); plt.xticks(np.arange(1,25,step=2)); plt.title('May'); plt.grid()

plt.subplot(3,4,6)
plt.bar(DC_monthly.Hour, DC_monthly.Jun); plt.ylabel('Count'); plt.xticks(np.arange(1,25,step=2)); plt.title('Jun'); plt.grid()

plt.subplot(3,4,7)
plt.bar(DC_monthly.Hour, DC_monthly.Jul); plt.ylabel('Count'); plt.xticks(np.arange(1,25,step=2)); plt.title('Jul'); plt.grid()

plt.subplot(3,4,8)
plt.bar(DC_monthly.Hour, DC_monthly.Aug); plt.ylabel('Count'); plt.xticks(np.arange(1,25,step=2)); plt.title('Aug'); plt.grid()

plt.subplot(3,4,9)
plt.bar(DC_monthly.Hour, DC_monthly.Sep); plt.ylabel('Count'); plt.xlabel('Hour'); plt.xticks(np.arange(1,25,step=2)); plt.title('Sep'); plt.grid()

plt.subplot(3,4,10)
plt.bar(DC_monthly.Hour, DC_monthly.Oct); plt.ylabel('Count'); plt.xlabel('Hour'); plt.xticks(np.arange(1,25,step=2)); plt.title('Oct'); plt.grid()

plt.subplot(3,4,11)
plt.bar(DC_monthly.Hour, DC_monthly.Nov); plt.ylabel('Count'); plt.xlabel('Hour'); plt.xticks(np.arange(1,25,step=2)); plt.title('Nov'); plt.grid()

plt.subplot(3,4,12)
plt.bar(DC_monthly.Hour, DC_monthly.Dec); plt.ylabel('Count'); plt.xlabel('Hour'); plt.xticks(np.arange(1,25,step=2)); plt.title('Dec'); plt.grid()
fig = plt.gcf(); fig.set_size_inches(20, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/HAR_monthly.png', dpi=100)