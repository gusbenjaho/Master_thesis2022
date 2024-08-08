# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 09:28:52 2021

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
## Diurnal count based on inititation threshold (matches with amount of DC_init)

Maybe restructure if to run again by threshold value instrad of model

"""

def get_diurnal_init_th(preciptracks, start, end):
    preciptracks['hour']= preciptracks.time.dt.hour           
    diurnal=[]
    for y in np.arange(start,end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            if ytracks[ytracks.cell == cell].threshold_value.iloc[0] == 10:
                init_hour = ytracks[ytracks.cell == cell].hour.values[0]
                diurnal.append(init_hour)
            else:
                continue
    diurnal_precip, bins = np.histogram(diurnal, bins = np.arange(0,25))
    return diurnal_precip, bins

#%% WRF DC for each TH, remember to set th value in the loop
WRF_DC_TH03, bins = get_diurnal_init_th(WRF_tracks, 2001, 2016)
WRF_DC_TH03 = pd.DataFrame(WRF_DC_TH03); WRF_DC_TH03.insert(0, 'Hour', range(1, 25)); WRF_DC_TH03.rename(columns={ 0:3}, inplace=True)
WRF_DC_TH04, bins = get_diurnal_init_th(WRF_tracks, 2001, 2016)
WRF_DC_TH04 = pd.DataFrame(WRF_DC_TH04); WRF_DC_TH04.rename(columns={ 0:4}, inplace=True)
WRF_DC_TH05, bins = get_diurnal_init_th(WRF_tracks, 2001, 2016)
WRF_DC_TH05 = pd.DataFrame(WRF_DC_TH05); WRF_DC_TH05.rename(columns={ 0:5}, inplace=True)
WRF_DC_TH06, bins = get_diurnal_init_th(WRF_tracks, 2001, 2016)
WRF_DC_TH06 = pd.DataFrame(WRF_DC_TH06); WRF_DC_TH06.rename(columns={ 0:6}, inplace=True)
WRF_DC_TH07, bins = get_diurnal_init_th(WRF_tracks, 2001, 2016)
WRF_DC_TH07 = pd.DataFrame(WRF_DC_TH07); WRF_DC_TH07.rename(columns={ 0:7}, inplace=True)
WRF_DC_TH08, bins = get_diurnal_init_th(WRF_tracks, 2001, 2016)
WRF_DC_TH08 = pd.DataFrame(WRF_DC_TH08); WRF_DC_TH08.rename(columns={ 0:8}, inplace=True)
WRF_DC_TH09, bins = get_diurnal_init_th(WRF_tracks, 2001, 2016)
WRF_DC_TH09 = pd.DataFrame(WRF_DC_TH09); WRF_DC_TH09.rename(columns={ 0:9}, inplace=True)
WRF_DC_TH10, bins = get_diurnal_init_th(WRF_tracks, 2001, 2016)
WRF_DC_TH10 = pd.DataFrame(WRF_DC_TH10); WRF_DC_TH10.rename(columns={ 0:10}, inplace=True)
WRF_DC_TH = pd.concat([WRF_DC_TH03, WRF_DC_TH04, WRF_DC_TH05, WRF_DC_TH06, WRF_DC_TH07, WRF_DC_TH08, WRF_DC_TH09, WRF_DC_TH10], axis=1)
WRF_DC_TH.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_DC_th_init.pkl')
del WRF_DC_TH03; del WRF_DC_TH04; del WRF_DC_TH05; del WRF_DC_TH06; del WRF_DC_TH07; del WRF_DC_TH08; del WRF_DC_TH09; del WRF_DC_TH10; del bins

#%% HAR
HAR_DC_TH03, bins = get_diurnal_init_th(HAR_tracks, 2001, 2016)
HAR_DC_TH03 = pd.DataFrame(HAR_DC_TH03); HAR_DC_TH03.insert(0, 'Hour', range(1, 25)); HAR_DC_TH03.rename(columns={ 0:3}, inplace=True)
HAR_DC_TH04, bins = get_diurnal_init_th(HAR_tracks, 2001, 2016)
HAR_DC_TH04 = pd.DataFrame(HAR_DC_TH04); HAR_DC_TH04.rename(columns={ 0:4}, inplace=True)
HAR_DC_TH05, bins = get_diurnal_init_th(HAR_tracks, 2001, 2016)
HAR_DC_TH05 = pd.DataFrame(HAR_DC_TH05); HAR_DC_TH05.rename(columns={ 0:5}, inplace=True)
HAR_DC_TH06, bins = get_diurnal_init_th(HAR_tracks, 2001, 2016)
HAR_DC_TH06 = pd.DataFrame(HAR_DC_TH06); HAR_DC_TH06.rename(columns={ 0:6}, inplace=True)
HAR_DC_TH07, bins = get_diurnal_init_th(HAR_tracks, 2001, 2016)
HAR_DC_TH07 = pd.DataFrame(HAR_DC_TH07); HAR_DC_TH07.rename(columns={ 0:7}, inplace=True)
HAR_DC_TH08, bins = get_diurnal_init_th(HAR_tracks, 2001, 2016)
HAR_DC_TH08 = pd.DataFrame(HAR_DC_TH08); HAR_DC_TH08.rename(columns={ 0:8}, inplace=True)
HAR_DC_TH09, bins = get_diurnal_init_th(HAR_tracks, 2001, 2016)
HAR_DC_TH09 = pd.DataFrame(HAR_DC_TH09); HAR_DC_TH09.rename(columns={ 0:9}, inplace=True)
HAR_DC_TH10, bins = get_diurnal_init_th(HAR_tracks, 2001, 2016)
HAR_DC_TH10 = pd.DataFrame(HAR_DC_TH10); HAR_DC_TH10.rename(columns={ 0:10}, inplace=True)
HAR_DC_TH = pd.concat([HAR_DC_TH03, HAR_DC_TH04, HAR_DC_TH05, HAR_DC_TH06, HAR_DC_TH07, HAR_DC_TH08, HAR_DC_TH09, HAR_DC_TH10], axis=1)
HAR_DC_TH.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_DC_th_init.pkl')
del HAR_DC_TH03; del HAR_DC_TH04; del HAR_DC_TH05; del HAR_DC_TH06; del HAR_DC_TH07; del HAR_DC_TH08; del HAR_DC_TH09; del HAR_DC_TH10; del bins

#%% ERA5
ERA5_DC_TH03, bins = get_diurnal_init_th(ERA5_tracks, 2001, 2016)
ERA5_DC_TH03 = pd.DataFrame(ERA5_DC_TH03); ERA5_DC_TH03.insert(0, 'Hour', range(1, 25)); ERA5_DC_TH03.rename(columns={ 0:3}, inplace=True)
ERA5_DC_TH04, bins = get_diurnal_init_th(ERA5_tracks, 2001, 2016)
ERA5_DC_TH04 = pd.DataFrame(ERA5_DC_TH04); ERA5_DC_TH04.rename(columns={ 0:4}, inplace=True)
ERA5_DC_TH05, bins = get_diurnal_init_th(ERA5_tracks, 2001, 2016)
ERA5_DC_TH05 = pd.DataFrame(ERA5_DC_TH05); ERA5_DC_TH05.rename(columns={ 0:5}, inplace=True)
ERA5_DC_TH06, bins = get_diurnal_init_th(ERA5_tracks, 2001, 2016)
ERA5_DC_TH06 = pd.DataFrame(ERA5_DC_TH06); ERA5_DC_TH06.rename(columns={ 0:6}, inplace=True)
ERA5_DC_TH07, bins = get_diurnal_init_th(ERA5_tracks, 2001, 2016)
ERA5_DC_TH07 = pd.DataFrame(ERA5_DC_TH07); ERA5_DC_TH07.rename(columns={ 0:7}, inplace=True)
ERA5_DC_TH08, bins = get_diurnal_init_th(ERA5_tracks, 2001, 2016)
ERA5_DC_TH08 = pd.DataFrame(ERA5_DC_TH08); ERA5_DC_TH08.rename(columns={ 0:8}, inplace=True)
ERA5_DC_TH09, bins = get_diurnal_init_th(ERA5_tracks, 2001, 2016)
ERA5_DC_TH09 = pd.DataFrame(ERA5_DC_TH09); ERA5_DC_TH09.rename(columns={ 0:9}, inplace=True)
ERA5_DC_TH10, bins = get_diurnal_init_th(ERA5_tracks, 2001, 2016)
ERA5_DC_TH10 = pd.DataFrame(ERA5_DC_TH10); ERA5_DC_TH10.rename(columns={ 0:10}, inplace=True)
ERA5_DC_TH = pd.concat([ERA5_DC_TH03, ERA5_DC_TH04, ERA5_DC_TH05, ERA5_DC_TH06, ERA5_DC_TH07, ERA5_DC_TH08, ERA5_DC_TH09, ERA5_DC_TH10], axis=1)
ERA5_DC_TH.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_DC_th_init.pkl')
del ERA5_DC_TH03; del ERA5_DC_TH04; del ERA5_DC_TH05; del ERA5_DC_TH06; del ERA5_DC_TH07; del ERA5_DC_TH08; del ERA5_DC_TH09; del ERA5_DC_TH10; del bins

#%% GPM
GPM_tracks['time'] = pd.to_datetime(GPM_tracks.timestr)

GPM_DC_TH03, bins = get_diurnal_init_th(GPM_tracks, 2001, 2016)
GPM_DC_TH03 = pd.DataFrame(GPM_DC_TH03); GPM_DC_TH03.insert(0, 'Hour', range(1, 25)); GPM_DC_TH03.rename(columns={ 0:3}, inplace=True)
GPM_DC_TH04, bins = get_diurnal_init_th(GPM_tracks, 2001, 2016)
GPM_DC_TH04 = pd.DataFrame(GPM_DC_TH04); GPM_DC_TH04.rename(columns={ 0:4}, inplace=True)
GPM_DC_TH05, bins = get_diurnal_init_th(GPM_tracks, 2001, 2016)
GPM_DC_TH05 = pd.DataFrame(GPM_DC_TH05); GPM_DC_TH05.rename(columns={ 0:5}, inplace=True)
GPM_DC_TH06, bins = get_diurnal_init_th(GPM_tracks, 2001, 2016)
GPM_DC_TH06 = pd.DataFrame(GPM_DC_TH06); GPM_DC_TH06.rename(columns={ 0:6}, inplace=True)
GPM_DC_TH07, bins = get_diurnal_init_th(GPM_tracks, 2001, 2016)
GPM_DC_TH07 = pd.DataFrame(GPM_DC_TH07); GPM_DC_TH07.rename(columns={ 0:7}, inplace=True)
GPM_DC_TH08, bins = get_diurnal_init_th(GPM_tracks, 2001, 2016)
GPM_DC_TH08 = pd.DataFrame(GPM_DC_TH08); GPM_DC_TH08.rename(columns={ 0:8}, inplace=True)
GPM_DC_TH09, bins = get_diurnal_init_th(GPM_tracks, 2001, 2016)
GPM_DC_TH09 = pd.DataFrame(GPM_DC_TH09); GPM_DC_TH09.rename(columns={ 0:9}, inplace=True)
GPM_DC_TH10, bins = get_diurnal_init_th(GPM_tracks, 2001, 2016)
GPM_DC_TH10 = pd.DataFrame(GPM_DC_TH10); GPM_DC_TH10.rename(columns={ 0:10}, inplace=True)
GPM_DC_TH = pd.concat([GPM_DC_TH03, GPM_DC_TH04, GPM_DC_TH05, GPM_DC_TH06, GPM_DC_TH07, GPM_DC_TH08, GPM_DC_TH09, GPM_DC_TH10], axis=1)
GPM_DC_TH.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_DC_th_init.pkl')
del GPM_DC_TH03; del GPM_DC_TH04; del GPM_DC_TH05; del GPM_DC_TH06; del GPM_DC_TH07; del GPM_DC_TH08; del GPM_DC_TH09; del GPM_DC_TH10; del bins

#%% Files from above
WRF_DC_TH = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_DC_th_init.pkl')
HAR_DC_TH= pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_DC_th_init.pkl')
ERA5_DC_TH = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_DC_th_init.pkl')
GPM_DC_TH = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_DC_th_init.pkl')

#%% Frequency
DC_TH3_tot = pd.DataFrame(); DC_TH3_tot['WRF'] = WRF_DC_TH.iloc[:,1]; DC_TH3_tot.insert(0, 'Hour', range(0, 24))
DC_TH3_tot['HAR'] = HAR_DC_TH.iloc[:,1]; DC_TH3_tot['ERA5'] = ERA5_DC_TH.iloc[:,1]; DC_TH3_tot['GPM'] = GPM_DC_TH.iloc[:,1]

DC_TH4_tot = pd.DataFrame(); DC_TH4_tot['WRF'] = WRF_DC_TH.iloc[:,2]; DC_TH4_tot.insert(0, 'Hour', range(0, 24))
DC_TH4_tot['HAR'] = HAR_DC_TH.iloc[:,2]; DC_TH4_tot['ERA5'] = ERA5_DC_TH.iloc[:,2]; DC_TH4_tot['GPM'] = GPM_DC_TH.iloc[:,2]

DC_TH5_tot = pd.DataFrame(); DC_TH5_tot['WRF'] = WRF_DC_TH.iloc[:,3]; DC_TH5_tot.insert(0, 'Hour', range(0, 24))
DC_TH5_tot['HAR'] = HAR_DC_TH.iloc[:,3]; DC_TH5_tot['ERA5'] = ERA5_DC_TH.iloc[:,3]; DC_TH5_tot['GPM'] = GPM_DC_TH.iloc[:,3]

DC_TH6_tot = pd.DataFrame(); DC_TH6_tot['WRF'] = WRF_DC_TH.iloc[:,4]; DC_TH6_tot.insert(0, 'Hour', range(0, 24))
DC_TH6_tot['HAR'] = HAR_DC_TH.iloc[:,4]; DC_TH6_tot['ERA5'] = ERA5_DC_TH.iloc[:,4]; DC_TH6_tot['GPM'] = GPM_DC_TH.iloc[:,4]

DC_TH7_tot = pd.DataFrame(); DC_TH7_tot['WRF'] = WRF_DC_TH.iloc[:,5]; DC_TH7_tot.insert(0, 'Hour', range(0, 24))
DC_TH7_tot['HAR'] = HAR_DC_TH.iloc[:,5]; DC_TH7_tot['ERA5'] = ERA5_DC_TH.iloc[:,5]; DC_TH7_tot['GPM'] = GPM_DC_TH.iloc[:,5]

DC_TH8_tot = pd.DataFrame(); DC_TH8_tot['WRF'] = WRF_DC_TH.iloc[:,6]; DC_TH8_tot.insert(0, 'Hour', range(0, 24))
DC_TH8_tot['HAR'] = HAR_DC_TH.iloc[:,6]; DC_TH8_tot['ERA5'] = ERA5_DC_TH.iloc[:,6]; DC_TH8_tot['GPM'] = GPM_DC_TH.iloc[:,6]

DC_TH9_tot = pd.DataFrame(); DC_TH9_tot['WRF'] = WRF_DC_TH.iloc[:,7]; DC_TH9_tot.insert(0, 'Hour', range(0, 24))
DC_TH9_tot['HAR'] = HAR_DC_TH.iloc[:,7]; DC_TH9_tot['ERA5'] = ERA5_DC_TH.iloc[:,7]; DC_TH9_tot['GPM'] = GPM_DC_TH.iloc[:,7]

DC_TH10_tot = pd.DataFrame(); DC_TH10_tot['WRF'] = WRF_DC_TH.iloc[:,8]; DC_TH10_tot.insert(0, 'Hour', range(0, 24))
DC_TH10_tot['HAR'] = HAR_DC_TH.iloc[:,8]; DC_TH10_tot['ERA5'] = ERA5_DC_TH.iloc[:,8]; DC_TH10_tot['GPM'] = GPM_DC_TH.iloc[:,8]

del WRF_DC_TH; del HAR_DC_TH; del ERA5_DC_TH; del GPM_DC_TH;
#%% Fraction
DC_TH3 = pd.DataFrame(); DC_TH3.insert(0, 'Hour', range(0, 24));
DC_TH3['WRF'] = DC_TH3_tot.WRF/DC_TH3_tot.WRF.sum() * 100; DC_TH3['HAR'] = DC_TH3_tot.HAR/DC_TH3_tot.HAR.sum() * 100; 
DC_TH3['ERA5'] = DC_TH3_tot.ERA5/DC_TH3_tot.ERA5.sum() * 100; DC_TH3['GPM'] = DC_TH3_tot.GPM/DC_TH3_tot.GPM.sum() * 100;

DC_TH4 = pd.DataFrame(); DC_TH4.insert(0, 'Hour', range(0, 24));
DC_TH4['WRF'] = DC_TH4_tot.WRF/DC_TH4_tot.WRF.sum() * 100; DC_TH4['HAR'] = DC_TH4_tot.HAR/DC_TH4_tot.HAR.sum() * 100; 
DC_TH4['ERA5'] = DC_TH4_tot.ERA5/DC_TH4_tot.ERA5.sum() * 100; DC_TH4['GPM'] = DC_TH4_tot.GPM/DC_TH4_tot.GPM.sum() * 100;

DC_TH5 = pd.DataFrame(); DC_TH5.insert(0, 'Hour', range(0, 24));
DC_TH5['WRF'] = DC_TH5_tot.WRF/DC_TH5_tot.WRF.sum() * 100; DC_TH5['HAR'] = DC_TH5_tot.HAR/DC_TH5_tot.HAR.sum() * 100; 
DC_TH5['ERA5'] = DC_TH5_tot.ERA5/DC_TH5_tot.ERA5.sum() * 100; DC_TH5['GPM'] = DC_TH5_tot.GPM/DC_TH5_tot.GPM.sum() * 100;

DC_TH6 = pd.DataFrame(); DC_TH6.insert(0, 'Hour', range(0, 24));
DC_TH6['WRF'] = DC_TH6_tot.WRF/DC_TH6_tot.WRF.sum() * 100; DC_TH6['HAR'] = DC_TH6_tot.HAR/DC_TH6_tot.HAR.sum() * 100; 
DC_TH6['ERA5'] = DC_TH6_tot.ERA5/DC_TH6_tot.ERA5.sum() * 100; DC_TH6['GPM'] = DC_TH6_tot.GPM/DC_TH6_tot.GPM.sum() * 100;

DC_TH7 = pd.DataFrame(); DC_TH7.insert(0, 'Hour', range(0, 24));
DC_TH7['WRF'] = DC_TH7_tot.WRF/DC_TH7_tot.WRF.sum() * 100; DC_TH7['HAR'] = DC_TH7_tot.HAR/DC_TH7_tot.HAR.sum() * 100; 
DC_TH7['ERA5'] = DC_TH7_tot.ERA5/DC_TH7_tot.ERA5.sum() * 100; DC_TH7['GPM'] = DC_TH7_tot.GPM/DC_TH7_tot.GPM.sum() * 100;

DC_TH8 = pd.DataFrame(); DC_TH8.insert(0, 'Hour', range(0, 24));
DC_TH8['WRF'] = DC_TH8_tot.WRF/DC_TH8_tot.WRF.sum() * 100; DC_TH8['HAR'] = DC_TH8_tot.HAR/DC_TH8_tot.HAR.sum() * 100; 
DC_TH8['ERA5'] = DC_TH8_tot.ERA5/DC_TH8_tot.ERA5.sum() * 100; DC_TH8['GPM'] = DC_TH8_tot.GPM/DC_TH8_tot.GPM.sum() * 100;

DC_TH9 = pd.DataFrame(); DC_TH9.insert(0, 'Hour', range(0, 24));
DC_TH9['WRF'] = DC_TH9_tot.WRF/DC_TH9_tot.WRF.sum() * 100; DC_TH9['HAR'] = DC_TH9_tot.HAR/DC_TH9_tot.HAR.sum() * 100; 
DC_TH9['ERA5'] = DC_TH9_tot.ERA5/DC_TH9_tot.ERA5.sum() * 100; DC_TH9['GPM'] = DC_TH9_tot.GPM/DC_TH9_tot.GPM.sum() * 100;

DC_TH10 = pd.DataFrame(); DC_TH10.insert(0, 'Hour', range(0, 24));
DC_TH10['WRF'] = DC_TH10_tot.WRF/DC_TH10_tot.WRF.sum() * 100; DC_TH10['HAR'] = DC_TH10_tot.HAR/DC_TH10_tot.HAR.sum() * 100; 
DC_TH10['ERA5'] = DC_TH10_tot.ERA5/DC_TH10_tot.ERA5.sum() * 100; DC_TH10['GPM'] = DC_TH10_tot.GPM/DC_TH10_tot.GPM.sum() * 100;
#%% Plotting frequency

fig, axes = plt.subplots(nrows=2, ncols=4)
plt.subplot(2,4,1)
DC_TH3_tot.plot(x='Hour', kind='line', title='TH3', ax=axes[0,0], grid=True, linestyle='--', xlim=[-0.5,23.5], ylabel='Frequency', xlabel='')
plt.subplot(2,4,2)
DC_TH4_tot.plot(x='Hour', kind='line', title='TH4', ax=axes[0,1], grid=True, linestyle='--', xlim=[-0.5,23.5], xlabel='')
plt.subplot(2,4,3)
DC_TH5_tot.plot(x='Hour', kind='line', title='TH5', ax=axes[0,2], grid=True, linestyle='--', xlim=[-0.5,23.5], xlabel='')
plt.subplot(2,4,4)
DC_TH6_tot.plot(x='Hour', kind='line', title='TH6', ax=axes[0,3], grid=True, linestyle='--', xlim=[-0.5,23.5], xlabel='')
plt.subplot(2,4,5)
DC_TH7_tot.plot(x='Hour', kind='line', title='TH7', ax=axes[1,0], grid=True, linestyle='--', xlim=[-0.5,23.5], ylabel='Frequency')
plt.subplot(2,4,6)
DC_TH8_tot.plot(x='Hour', kind='line', title='TH8', ax=axes[1,1], grid=True, linestyle='--', xlim=[-0.5,23.5])
plt.subplot(2,4,7)
DC_TH9_tot.plot(x='Hour', kind='line', title='TH9', ax=axes[1,2], grid=True, linestyle='--', xlim=[-0.5,23.5])
plt.subplot(2,4,8)
DC_TH10_tot.plot(x='Hour', kind='line', title='TH10', ax=axes[1,3], grid=True, linestyle='--', xlim=[-0.5,23.5])

fig = plt.gcf(); fig.set_size_inches(23, 7)
txt = 'The diurnal cycle of precipitation cell frequency at initiation stage for each precipitation threshold (THmm) between 2001 and 2016 based on the three model datasets (WRF, HAR & ERA5) and the satellite dataset (GPM)'
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/DC_init_th_tot.png', dpi=100)

#%% Plotting fraction

fig, axes = plt.subplots(nrows=2, ncols=4)
plt.subplot(2,4,1)
DC_TH3.plot(x='Hour', kind='line', title='TH3', ax=axes[0,0], grid=True, ylabel='Fraction (%)', xlabel='', linestyle='--', xlim=[-0.5,23.5])
plt.subplot(2,4,2)
DC_TH4.plot(x='Hour', kind='line', title='TH4', ax=axes[0,1], grid=True, xlabel='', linestyle='--', xlim=[-0.5,23.5])
plt.subplot(2,4,3)
DC_TH5.plot(x='Hour', kind='line', title='TH5', ax=axes[0,2], grid=True, xlabel='', linestyle='--', xlim=[-0.5,23.5])
plt.subplot(2,4,4)
DC_TH6.plot(x='Hour', kind='line', title='TH6', ax=axes[0,3], grid=True, xlabel='', linestyle='--', xlim=[-0.5,23.5])
plt.subplot(2,4,5)
DC_TH7.plot(x='Hour', kind='line', title='TH7', ax=axes[1,0], grid=True, ylabel='Fraction (%)', linestyle='--', xlim=[-0.5,23.5], xlabel='Hour (UTC)')
plt.subplot(2,4,6)
DC_TH8.plot(x='Hour', kind='line', title='TH8', ax=axes[1,1], grid=True, linestyle='--', xlim=[-0.5,23.5], xlabel='Hour (UTC)')
plt.subplot(2,4,7)
DC_TH9.plot(x='Hour', kind='line', title='TH9', ax=axes[1,2], grid=True, linestyle='--', xlim=[-0.5,23.5], xlabel='Hour (UTC)')
plt.subplot(2,4,8)
DC_TH10.plot(x='Hour', kind='line', title='TH10', ax=axes[1,3], grid=True, linestyle='--', xlim=[-0.5,23.5], xlabel='Hour (UTC)')

fig = plt.gcf(); fig.set_size_inches(23, 7)
txt = 'The diurnal cycle of precipitation cell fraction at initiation stage for each precipitation threshold (THmm) between 2001 and 2016 based on the three model datasets (WRF, HAR & ERA5) and the satellite dataset (GPM)'
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/DC_init_th.png', dpi=100)
#%%
"""
Diurnal count based on maximum threshold

"""
def get_diurnal_th_max(preciptracks, start, end):
    preciptracks['hour']= preciptracks.time.dt.hour           
    diurnal=[]
    for y in np.arange(start,end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            idx = np.argmax(ytracks.threshold_value[ytracks.cell == cell])
            max_hour = ytracks[ytracks.cell == cell].hour.values[idx]
            diurnal.append(max_hour)
    diurnal_precip, bins = np.histogram(diurnal, bins = np.arange(0,25))
    return diurnal_precip, bins

#%% Called it with:
# WRF
WRF_DC_max_th, bins = get_diurnal_th_max(WRF_tracks, 2001, 2016)
HAR_DC_max_th, bins = get_diurnal_th_max(HAR_tracks, 2001, 2016)
ERA5_DC_max_th, bins = get_diurnal_th_max(ERA5_tracks, 2001, 2016)
GPM_tracks['time'] = pd.to_datetime(GPM_tracks.timestr)
GPM_DC_max_th, bins = get_diurnal_th_max(GPM_tracks, 2001, 2016)
del bins
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_DC_max_th.npy', WRF_DC_max_th)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_DC_max_th.npy', HAR_DC_max_th)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_DC_max_th.npy', ERA5_DC_max_th)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_DC_max_th.npy', GPM_DC_max_th)

#%% Loading above files
WRF_DC_max_th = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_DC_max_th.npy')
HAR_DC_max_th = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_DC_max_th.npy')
ERA5_DC_max_th = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_DC_max_th.npy')
GPM_DC_max_th = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_DC_max_th.npy')

#%% Total count
DC_max_th_tot = pd.DataFrame(); DC_max_th_tot.insert(0, 'Hour', range(0, 24))
DC_max_th_tot['WRF'],DC_max_th_tot['HAR'],DC_max_th_tot['ERA5'],DC_max_th_tot['GPM'] = [WRF_DC_max_th, HAR_DC_max_th, ERA5_DC_max_th, GPM_DC_max_th]
del WRF_DC_max_th; del HAR_DC_max_th; del ERA5_DC_max_th; del GPM_DC_max_th

#%% Normalize 
DC_max_th = pd.DataFrame(); DC_max_th.insert(0, 'Hour', range(0, 24))
DC_max_th['WRF'] = DC_max_th_tot.WRF/DC_max_th_tot.WRF.sum() * 100; DC_max_th['HAR'] = DC_max_th_tot.HAR/DC_max_th_tot.HAR.sum() * 100
DC_max_th['ERA5'] = DC_max_th_tot.ERA5/DC_max_th_tot.ERA5.sum() * 100 ; DC_max_th['GPM'] = DC_max_th_tot.GPM/DC_max_th_tot.GPM.sum() * 100

#%% Plotting

fig, axes = plt.subplots(nrows=2, ncols=1)
DC_max_th_tot.plot(x='Hour', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', grid=True, ylabel='Frequency', ax=axes[0], linestyle='--', xlim=[-0.5,23.5], xticks=np.arange(0,24, step=1), xlabel='Hour (UTC)')
DC_max_th.plot(x='Hour', y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', grid=True, ylabel='Fraction (%)', ax=axes[1], linestyle='--', xlim=[-0.5,23.5], xticks=np.arange(0,24, step=1), xlabel='Hour (UTC)')
fig = plt.gcf(); fig.set_size_inches(12, 10)
txt = 'The diurnal cycle of precipitation cell frequency (top) and fraction (bottom) at maximum precipitation threshold between 2001 and 2016\nbased on the three model datasets (WRF, HAR & ERA5) and the satellite dataset (GPM)'
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/DC_max_th.png', dpi=100)