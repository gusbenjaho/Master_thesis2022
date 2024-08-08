# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 11:31:33 2021

@author: benja
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
#%%
WRF_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/WRF_filtered.pkl')
HAR_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/HAR_filtered.pkl')
GPM_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/GPM_filtered.pkl')
ERA5_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/ERA5_filtered.pkl')
GPM_tracks['time'] = pd.to_datetime(GPM_tracks.timestr)

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

def get_th_disscount(preciptracks,start,end):
    th = []
    th_count = pd.DataFrame()
    for y in np.arange(start,end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            diss_th = ytracks[ytracks.cell == cell].threshold_value.values[-1]
            th.append(diss_th)
    th_count['th'] = pd.DataFrame(th)
    th_count = th_count.th.groupby(th_count.th).count().to_frame()
    return th_count

def get_th_maxcount(preciptracks,start,end):
    th = []
    th_count = pd.DataFrame()
    for y in np.arange(start,end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            idx = np.argmax(ytracks.threshold_value[ytracks.cell == cell])
            max_th = ytracks[ytracks.cell == cell].threshold_value.values[idx]
            th.append(max_th)
    th_count['th'] = pd.DataFrame(th)
    th_count = th_count.th.groupby(th_count.th).count().to_frame()
    return th_count

def get_th_extcount(preciptracks,start,end):
    th = []
    th_count = pd.DataFrame()
    for y in np.arange(start,end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            idx = np.argmax(ytracks.ncells[ytracks.cell == cell])
            max_ext = ytracks[ytracks.cell == cell].threshold_value.values[idx]
            th.append(max_ext)
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
TH_initcount.to_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TH_initcount_filtered_TP.pkl')
del WRF_th_initcount, HAR_th_initcount, ERA5_th_initcount, GPM_th_initcount;

#%% Dissipation
WRF_th_disscount = get_th_disscount(WRF_tracks,2001,2016)
HAR_th_disscount = get_th_disscount(HAR_tracks,2001,2016)
ERA5_th_disscount = get_th_disscount(ERA5_tracks,2001,2016)
GPM_th_disscount = get_th_disscount(GPM_tracks,2001,2016)

TH_disscount = pd.DataFrame(); TH_disscount.insert(0, 'th', range(3,11)); TH_disscount.set_index('th', inplace=True)
TH_disscount['WRF'], TH_disscount['HAR'], TH_disscount['ERA5'], TH_disscount['GPM'] = [WRF_th_disscount, HAR_th_disscount, ERA5_th_disscount, GPM_th_disscount]
TH_disscount.to_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TH_disscount_filtered_TP.pkl')
del WRF_th_disscount, HAR_th_disscount, ERA5_th_disscount, GPM_th_disscount;

#%% Max extension
WRF_th_extcount = get_th_extcount(WRF_tracks,2001,2016)
HAR_th_extcount = get_th_extcount(HAR_tracks,2001,2016)
ERA5_th_extcount = get_th_extcount(ERA5_tracks,2001,2016)
GPM_th_extcount = get_th_extcount(GPM_tracks,2001,2016)

TH_extcount = pd.DataFrame(); TH_extcount.insert(0, 'th', range(3,11)); TH_extcount.set_index('th', inplace=True)
TH_extcount['WRF'], TH_extcount['HAR'], TH_extcount['ERA5'], TH_extcount['GPM'] = [WRF_th_extcount, HAR_th_extcount, ERA5_th_extcount, GPM_th_extcount]
TH_extcount.to_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TH_extcount_filtered_TP.pkl')
del WRF_th_extcount, HAR_th_extcount, ERA5_th_extcount, GPM_th_extcount;

#%% Max TH
WRF_th_maxcount = get_th_maxcount(WRF_tracks,2001,2016)
HAR_th_maxcount = get_th_maxcount(HAR_tracks,2001,2016)
ERA5_th_maxcount = get_th_maxcount(ERA5_tracks,2001,2016)
GPM_th_maxcount = get_th_maxcount(GPM_tracks,2001,2016)

TH_maxcount = pd.DataFrame(); TH_maxcount.insert(0, 'th', range(3,11)); TH_maxcount.set_index('th', inplace=True)
TH_maxcount['WRF'], TH_maxcount['HAR'], TH_maxcount['ERA5'], TH_maxcount['GPM'] = [WRF_th_maxcount, HAR_th_maxcount, ERA5_th_maxcount, GPM_th_maxcount]
TH_maxcount.to_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TH_maxcount_filtered_TP.pkl')
del WRF_th_maxcount, HAR_th_maxcount, ERA5_th_maxcount, GPM_th_maxcount, WRF_tracks, HAR_tracks, ERA5_tracks, GPM_tracks

#%% Creating fraction file
## Init
TH_initcount = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TH_initcount_filtered.pkl')
TH_initfrac = pd.DataFrame(); TH_initfrac.insert(0, 'th', range(3,11)); TH_initfrac.set_index('th', inplace=True)
TH_initfrac['WRF_GU'] = TH_initcount.WRF/TH_initcount.WRF.sum() * 100; TH_initfrac['HAR'] = TH_initcount.HAR/TH_initcount.HAR.sum() * 100
TH_initfrac['ERA5'] = TH_initcount.ERA5/TH_initcount.ERA5.sum() * 100; TH_initfrac['GPM'] = TH_initcount.GPM/TH_initcount.GPM.sum() * 100
TH_initcount= TH_initcount.rename(columns={'WRF':'WRF_GU'})
#%% Dissipation
TH_disscount = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TH_disscount_filtered.pkl')
TH_dissfrac = pd.DataFrame(); TH_dissfrac.insert(0, 'th', range(3,11)); TH_dissfrac.set_index('th', inplace=True)
TH_dissfrac['WRF_GU'] = TH_disscount.WRF/TH_disscount.WRF.sum() * 100; TH_dissfrac['HAR'] = TH_disscount.HAR/TH_disscount.HAR.sum() * 100
TH_dissfrac['ERA5'] = TH_disscount.ERA5/TH_disscount.ERA5.sum() * 100; TH_dissfrac['GPM'] = TH_disscount.GPM/TH_disscount.GPM.sum() * 100
TH_disscount= TH_initcount.rename(columns={'WRF':'WRF_GU'})
#%% Max TH
TH_maxcount = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TH_maxcount_filtered.pkl')
TH_maxfrac = pd.DataFrame(); TH_maxfrac.insert(0, 'th', range(3,11)); TH_maxfrac.set_index('th', inplace=True)
TH_maxfrac['WRF_GU'] = TH_maxcount.WRF/TH_maxcount.WRF.sum() * 100; TH_maxfrac['HAR'] = TH_maxcount.HAR/TH_maxcount.HAR.sum() * 100
TH_maxfrac['ERA5'] = TH_maxcount.ERA5/TH_maxcount.ERA5.sum() * 100; TH_maxfrac['GPM'] = TH_maxcount.GPM/TH_maxcount.GPM.sum() * 100
TH_maxcount= TH_initcount.rename(columns={'WRF':'WRF_GU'})
#%% Max extension
TH_extcount = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TH_extcount_filtered.pkl')
TH_extfrac = pd.DataFrame(); TH_extfrac.insert(0, 'th', range(3,11)); TH_extfrac.set_index('th', inplace=True)
TH_extfrac['WRF_GU'] = TH_extcount.WRF/TH_extcount.WRF.sum() * 100; TH_extfrac['HAR'] = TH_extcount.HAR/TH_extcount.HAR.sum() * 100
TH_extfrac['ERA5'] = TH_extcount.ERA5/TH_extcount.ERA5.sum() * 100; TH_extfrac['GPM'] = TH_extcount.GPM/TH_extcount.GPM.sum() * 100
TH_extcount= TH_initcount.rename(columns={'WRF':'WRF_GU'})
#%% Plotting
plt.style.use('seaborn'); y_data = ['WRF_GU', 'HAR', 'ERA5', 'GPM'];colors = ['#1E90FF', '#104E8B', '#FF6103', '#FF9912']
fig, axes = plt.subplots(2,2)
TH_initfrac.plot(use_index=True, y=y_data, ax=axes[0,0], kind='bar', grid=True, ylabel='Fraction [%]', xlabel='', ylim=(0,90), title='a)', color=colors)
TH_dissfrac.plot(use_index=True, y=y_data, ax=axes[0,1], kind='bar', grid=True, ylabel='', xlabel='', ylim=(0,90), title='b)', color=colors)
TH_maxfrac.plot(use_index=True, y=y_data, ax=axes[1,0], kind='bar', grid=True, ylabel='Fraction [%]', xlabel='Intensity level [mm $h^{-1}$]', ylim=(0,90), title='c)', color=colors)
TH_extfrac.plot(use_index=True, y=y_data, ax=axes[1,1], kind='bar', grid=True, ylabel='', xlabel='Intensity level [mm $h^{-1}$]', ylim=(0,90), title='d)', color=colors)
txt = 'Fraction of precipitation cells for each intensity level at initation (a), dissipation (b), max threshold value (c) and\nmax extension stage (d) based on the three model datasets (WRF, HAR & ERA5) and the satellite dataset (GPM)'
plt.figtext(0.12, 0.01, txt, horizontalalignment='left')
fig.set_size_inches(12, 8)
fig.show()
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/PTV_filtered_color.png', dpi=100)
