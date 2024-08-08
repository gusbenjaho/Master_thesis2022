# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 09:07:04 2021

@author: benja
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

WRF_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/WRF_filtered.pkl')
HAR_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/HAR_filtered.pkl')
GPM_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/GPM_filtered.pkl')
ERA5_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/ERA5_filtered.pkl')

#%%
GPM_tracks['time'] = pd.to_datetime(GPM_tracks.timestr)
WRF_tracks = WRF_tracks[(WRF_tracks.time.dt.year >= 2001) & (WRF_tracks.time.dt.year <= 2016)]
HAR_tracks = HAR_tracks[(HAR_tracks.time.dt.year >= 2001) & (HAR_tracks.time.dt.year <= 2016)]
ERA5_tracks = ERA5_tracks[(ERA5_tracks.time.dt.year >= 2001) & (ERA5_tracks.time.dt.year <= 2016)]
GPM_tracks = GPM_tracks[(GPM_tracks.time.dt.year >= 2001) & (GPM_tracks.time.dt.year <= 2016)]

DC_all_tot = WRF_tracks.cell.groupby(WRF_tracks.time.dt.hour).count().to_frame()
DC_all_tot.insert(0, 'Hour', range(0, 24))
DC_all_tot.rename(columns={'cell': 'WRF_GU'}, inplace=True)
DC_all_tot['HAR'] = HAR_tracks.cell.groupby(HAR_tracks.time.dt.hour).count()
DC_all_tot['ERA5'] = ERA5_tracks.cell.groupby(ERA5_tracks.time.dt.hour).count()
DC_all_tot['GPM'] = GPM_tracks.cell.groupby(GPM_tracks.time.dt.hour).count()

## Change to local time (+6)
DC_all_tot.Hour = DC_all_tot.Hour + 6
DC_all_tot.replace(to_replace={'Hour': {24:0, 25:1, 26:2, 27:3, 28:4, 29:5}}, inplace=True)
DC_all_tot = DC_all_tot.sort_values(by=['Hour'])

DC_all = pd.DataFrame()
DC_all.insert(0, 'Hour', range(0, 24))
DC_all['WRF_GU'] = DC_all_tot.WRF_GU/DC_all_tot.WRF_GU.sum() * 100
DC_all['HAR'] = DC_all_tot.HAR/DC_all_tot.HAR.sum() * 100
DC_all['ERA5'] = DC_all_tot.ERA5/DC_all_tot.ERA5.sum() * 100
DC_all['GPM'] = DC_all_tot.GPM/DC_all_tot.GPM.sum() * 100

DC_all.Hour = DC_all.Hour + 6
DC_all.replace(to_replace={'Hour': {24:0, 25:1, 26:2, 27:3, 28:4, 29:5}}, inplace=True)
DC_all = DC_all.sort_values(by=['Hour'])

# %% Plotting
#styles = ['-','--',':','-.'];
colors = ['#1E90FF', '#104E8B', '#FF6103', '#FF9912']
plt.style.use('seaborn')
txt = 'The diurnal cycle of precipitation cell frequency [a] and fraction [b] at every recorded stage between 2001 and 2016\nbased on the three model datasets (WRF, HAR & ERA5) and the satellite dataset (GPM)'
fig, axes = plt.subplots(nrows=2, ncols=1)
for col, color in zip(DC_all_tot.columns[1:], colors):
    DC_all_tot.plot(x='Hour', y=col, kind='line', grid=True, ylabel='Frequency', ax=axes[0], linestyle='-', xlim=[-0.5, 23.5], xticks=np.arange(0, 24, step=1), xlabel='Hour (UTC+6)', color=color, title='a)')
    DC_all.plot(x='Hour', y=col, kind='line', grid=True, ylabel='Fraction [%]', ax=axes[1], linestyle='-', xlim=[-0.5, 23.5], xticks=np.arange(0, 24, step=1), xlabel='Hour (UTC+6)', color=color, title='b)')
axes[0].legend(loc='upper left',bbox_to_anchor=(1.0, 1))
axes[1].legend(loc='upper left',bbox_to_anchor=(1.0, 1))
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
fig = plt.gcf()
fig.set_size_inches(12, 10)
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/DC_all_filtered_color.png', dpi=100)

#%%
del WRF_tracks, HAR_tracks, ERA5_tracks, GPM_tracks, fig, axes, txt, DC_all, DC_all_tot, col, color, colors #, style, styles
# %%
""" 
DC of features of TP only (>3000)


"""
WRF_TP = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_WRF_filtered.pkl')
HAR_TP = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_HAR_filtered.pkl')
ERA5_TP = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_ERA5_filtered.pkl')
GPM_TP = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_GPM_filtered.pkl')

#%%
WRF_TP = WRF_TP[(WRF_TP.time.dt.year >= 2001) & (WRF_TP.time.dt.year <= 2016)]
HAR_TP = HAR_TP[(HAR_TP.time.dt.year >= 2001) & (HAR_TP.time.dt.year <= 2016)]
ERA5_TP = ERA5_TP[(ERA5_TP.time.dt.year >= 2001) & (ERA5_TP.time.dt.year <= 2016)]
GPM_TP = GPM_TP[(GPM_TP.time.dt.year >= 2001) & (GPM_TP.time.dt.year <= 2016)]

#%% DC of TP only
DC_TP_tot = WRF_TP.cell.groupby(WRF_TP.time.dt.hour).count().to_frame()
DC_TP_tot.insert(0, 'Hour', range(0, 24))
DC_TP_tot.rename(columns={'cell': 'WRF_GU'}, inplace=True)
DC_TP_tot['HAR'] = HAR_TP.cell.groupby(HAR_TP.time.dt.hour).count()
DC_TP_tot['ERA5'] = ERA5_TP.cell.groupby(ERA5_TP.time.dt.hour).count()
DC_TP_tot['GPM'] = GPM_TP.cell.groupby(GPM_TP.time.dt.hour).count()

DC_TP = pd.DataFrame()
DC_TP.insert(0, 'Hour', range(0, 24))
DC_TP['WRF_GU'] = DC_TP_tot.WRF_GU/DC_TP_tot.WRF_GU.sum() * 100
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
colors = ['#1E90FF', '#104E8B', '#FF6103', '#FF9912']
#styles = ['-','--',':','-.'];
plt.style.use('seaborn')
txt = 'The diurnal cycle of precipitation cell frequency [a] and fraction [b] at every recorded stage on the TP only\nbetween 2001 and 2016 based on the three model datasets (WRF, HAR & ERA5) and the satellite dataset (GPM)'
fig, axes = plt.subplots(nrows=2, ncols=1)
for col, color in zip(DC_TP_tot.columns[1:], colors):
    DC_TP_tot.plot(x='Hour', y=col, kind='line', grid=True, ylabel='Frequency', ax=axes[0], linestyle='-', xlim=[-0.5, 23.5], xticks=np.arange(0, 24, step=1), xlabel='Hour (UTC+6)', color=color, title='a)')
    DC_TP.plot(x='Hour', y=col, kind='line', grid=True, ylabel='Fraction [%]', ax=axes[1], linestyle='-', xlim=[-0.5, 23.5], xticks=np.arange(0, 24, step=1), xlabel='Hour (UTC+6)', color=color, title='b)')
axes[0].legend(loc='upper left',bbox_to_anchor=(1.0, 1))
axes[1].legend(loc='upper left',bbox_to_anchor=(1.0, 1))
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
fig = plt.gcf()
#plt.style.use('seaborn')
fig.set_size_inches(12, 10)
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/DC_TP_filtered_color.png', dpi=100)