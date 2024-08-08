# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 14:00:07 2021

@author: benja
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%
WRF_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/WRF_filtered.pkl')
HAR_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/HAR_filtered.pkl')
GPM_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/GPM_filtered.pkl')
ERA5_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/ERA5_filtered.pkl')
GPM_tracks['time'] = pd.to_datetime(GPM_tracks.timestr)

## To same annual extent
WRF_tracks = WRF_tracks[(WRF_tracks.time.dt.year >= 2001) & (WRF_tracks.time.dt.year <= 2016)]
HAR_tracks = HAR_tracks[(HAR_tracks.time.dt.year >= 2001) & (HAR_tracks.time.dt.year <= 2016)]
ERA5_tracks = ERA5_tracks[(ERA5_tracks.time.dt.year >= 2001) & (ERA5_tracks.time.dt.year <= 2016)]
GPM_tracks = GPM_tracks[(GPM_tracks.time.dt.year >= 2001) & (GPM_tracks.time.dt.year <= 2016)]

#%% Entire study area

keys = 'PTV3', 'PTV4', 'PTV5', 'PTV6', 'PTV7', 'PTV8', 'PTV9', 'PTV10'
WRF = WRF_tracks.cell.groupby([WRF_tracks.time.dt.hour, WRF_tracks.threshold_value]).count().to_frame(); WRF.reset_index(inplace=True)
HAR = HAR_tracks.cell.groupby([HAR_tracks.time.dt.hour, HAR_tracks.threshold_value]).count().to_frame(); HAR.reset_index(inplace=True)
ERA5 = ERA5_tracks.cell.groupby([ERA5_tracks.time.dt.hour, ERA5_tracks.threshold_value]).count().to_frame(); ERA5.reset_index(inplace=True)
GPM = GPM_tracks.cell.groupby([GPM_tracks.time.dt.hour, GPM_tracks.threshold_value]).count().to_frame(); GPM.reset_index(inplace=True)

WRF.time = WRF.time + 6; HAR.time = HAR.time + 6; ERA5.time = ERA5.time + 6; GPM.time = GPM.time + 6;
WRF.replace(to_replace={'time': {24:0, 25:1, 26:2, 27:3, 28:4, 29:5}}, inplace=True)
HAR.replace(to_replace={'time': {24:0, 25:1, 26:2, 27:3, 28:4, 29:5}}, inplace=True)
ERA5.replace(to_replace={'time': {24:0, 25:1, 26:2, 27:3, 28:4, 29:5}}, inplace=True)
GPM.replace(to_replace={'time': {24:0, 25:1, 26:2, 27:3, 28:4, 29:5}}, inplace=True)
WRF.sort_values(by='time', inplace=True); HAR.sort_values(by='time', inplace=True)
ERA5.sort_values(by='time', inplace=True); GPM.sort_values(by='time', inplace=True)

PTV3 = pd.DataFrame();
PTV3['WRF'] = pd.DataFrame(WRF[WRF.threshold_value == 3].cell.values); PTV3['HAR'] = pd.DataFrame(HAR[HAR.threshold_value == 3].cell.values)
PTV3['ERA5'] = pd.DataFrame(ERA5[ERA5.threshold_value == 3].cell.values); PTV3['GPM'] = pd.DataFrame(GPM[GPM.threshold_value == 3].cell.values)
PTV3.WRF = PTV3.WRF/PTV3.WRF.sum() * 100; PTV3.HAR = PTV3.HAR/PTV3.HAR.sum() * 100
PTV3.ERA5 = PTV3.ERA5/PTV3.ERA5.sum() * 100; PTV3.GPM = PTV3.GPM/PTV3.GPM.sum() * 100


PTV4 = pd.DataFrame(); 
PTV4['WRF'] = pd.DataFrame(WRF[WRF.threshold_value == 4].cell.values); PTV4['HAR'] = pd.DataFrame(HAR[HAR.threshold_value == 4].cell.values)
PTV4['ERA5'] = pd.DataFrame(ERA5[ERA5.threshold_value == 4].cell.values); PTV4['GPM'] = pd.DataFrame(GPM[GPM.threshold_value == 4].cell.values)
PTV4.WRF = PTV4.WRF/PTV4.WRF.sum() * 100; PTV4.HAR = PTV4.HAR/PTV4.HAR.sum() * 100
PTV4.ERA5 = PTV4.ERA5/PTV4.ERA5.sum() * 100; PTV4.GPM = PTV4.GPM/PTV4.GPM.sum() * 100

PTV5 = pd.DataFrame();
PTV5['WRF'] = pd.DataFrame(WRF[WRF.threshold_value == 5].cell.values); PTV5['HAR'] = pd.DataFrame(HAR[HAR.threshold_value == 5].cell.values)
PTV5['ERA5'] = pd.DataFrame(ERA5[ERA5.threshold_value == 5].cell.values); PTV5['GPM'] = pd.DataFrame(GPM[GPM.threshold_value == 5].cell.values)
PTV5.WRF = PTV5.WRF/PTV5.WRF.sum() * 100; PTV5.HAR = PTV5.HAR/PTV5.HAR.sum() * 100
PTV5.ERA5 = PTV5.ERA5/PTV5.ERA5.sum() * 100; PTV5.GPM = PTV5.GPM/PTV5.GPM.sum() * 100

PTV6 = pd.DataFrame();
PTV6['WRF'] = pd.DataFrame(WRF[WRF.threshold_value == 6].cell.values); PTV6['HAR'] = pd.DataFrame(HAR[HAR.threshold_value == 6].cell.values)
PTV6['ERA5'] = pd.DataFrame(ERA5[ERA5.threshold_value == 6].cell.values); PTV6['GPM'] = pd.DataFrame(GPM[GPM.threshold_value == 6].cell.values)
PTV6.WRF = PTV6.WRF/PTV6.WRF.sum() * 100; PTV6.HAR = PTV6.HAR/PTV6.HAR.sum() * 100
PTV6.ERA5 = PTV6.ERA5/PTV6.ERA5.sum() * 100; PTV6.GPM = PTV6.GPM/PTV6.GPM.sum() * 100

PTV7 = pd.DataFrame(); 
PTV7['WRF'] = pd.DataFrame(WRF[WRF.threshold_value == 7].cell.values); PTV7['HAR'] = pd.DataFrame(HAR[HAR.threshold_value == 7].cell.values)
PTV7['ERA5'] = pd.DataFrame(ERA5[ERA5.threshold_value == 7].cell.values); PTV7['GPM'] = pd.DataFrame(GPM[GPM.threshold_value == 7].cell.values)
PTV7.WRF = PTV7.WRF/PTV7.WRF.sum() * 100; PTV7.HAR = PTV7.HAR/PTV7.HAR.sum() * 100
PTV7.ERA5 = PTV7.ERA5/PTV7.ERA5.sum() * 100; PTV7.GPM = PTV7.GPM/PTV7.GPM.sum() * 100

PTV8 = pd.DataFrame();
PTV8['WRF'] = pd.DataFrame(WRF[WRF.threshold_value == 8].cell.values); PTV8['HAR'] = pd.DataFrame(HAR[HAR.threshold_value == 8].cell.values)
PTV8['ERA5'] = pd.DataFrame(ERA5[ERA5.threshold_value == 8].cell.values); PTV8['GPM'] = pd.DataFrame(GPM[GPM.threshold_value == 8].cell.values)
PTV8.WRF = PTV8.WRF/PTV8.WRF.sum() * 100; PTV8.HAR = PTV8.HAR/PTV8.HAR.sum() * 100
PTV8.ERA5 = PTV8.ERA5/PTV8.ERA5.sum() * 100; PTV8.GPM = PTV8.GPM/PTV8.GPM.sum() * 100

PTV9 = pd.DataFrame();
PTV9['WRF'] = pd.DataFrame(WRF[WRF.threshold_value == 9].cell.values); PTV9['HAR'] = pd.DataFrame(HAR[HAR.threshold_value == 9].cell.values)
PTV9['ERA5'] = pd.DataFrame(ERA5[ERA5.threshold_value == 9].cell.values); PTV9['GPM'] = pd.DataFrame(GPM[GPM.threshold_value == 9].cell.values)
PTV9.WRF = PTV9.WRF/PTV9.WRF.sum() * 100; PTV9.HAR = PTV9.HAR/PTV9.HAR.sum() * 100
PTV9.ERA5 = PTV9.ERA5/PTV9.ERA5.sum() * 100; PTV9.GPM = PTV9.GPM/PTV9.GPM.sum() * 100

PTV10 = pd.DataFrame();
PTV10['WRF'] = pd.DataFrame(WRF[WRF.threshold_value == 10].cell.values); PTV10['HAR'] = pd.DataFrame(HAR[HAR.threshold_value == 10].cell.values)
PTV10['ERA5'] = pd.DataFrame(ERA5[ERA5.threshold_value == 10].cell.values); PTV10['GPM'] = pd.DataFrame(GPM[GPM.threshold_value == 10].cell.values)
PTV10.WRF = PTV10.WRF/PTV10.WRF.sum() * 100; PTV10.HAR = PTV10.HAR/PTV10.HAR.sum() * 100
PTV10.ERA5 = PTV10.ERA5/PTV10.ERA5.sum() * 100; PTV10.GPM = PTV10.GPM/PTV10.GPM.sum() * 100

TH_values = pd.concat([PTV3, PTV4, PTV5, PTV6, PTV7, PTV8, PTV9, PTV10], keys=keys).reset_index().rename(columns={'level_0':'threshold_value', 'level_1':'Hour','WRF':'WRF_GU'})
del WRF, HAR, ERA5, GPM, WRF_tracks, HAR_tracks, ERA5_tracks, GPM_tracks, PTV3, PTV4, PTV5, PTV6, PTV7, PTV8, PTV9, PTV10, keys
    
#%% Plotting
colors = ['#1E90FF', '#104E8B', '#FF6103', '#FF9912']; rows = [0,0,1,1,2,2,3,3]; cols = [0,1,0,1,0,1,0,1]; TV = ['PTV3', 'PTV4', 'PTV5', 'PTV6', 'PTV7', 'PTV8', 'PTV9', 'PTV10']
plt.style.use('seaborn')
labels = ['Intensity level 3','Intensity level 4','Intensity level 5','Intensity level 6','Intensity level 7','Intensity level 8','Intensity level 9','Intensity level 10']
xticks = np.arange(0,24,2)
fig, axes = plt.subplots(4,2)
for r,c,tv,lbl in zip(rows, cols, TV, labels):
    for col, color in zip(TH_values.columns[2:], colors):
        TH_values[TH_values.threshold_value == tv].plot(x='Hour',y=col, kind='line', ax=axes[r,c], grid=True, linestyle='-', xlim=[-0.5,23.5], xlabel='', ylim=(0,11), xticks=xticks, color=color).legend(fontsize=16)
        axes[r,c].set_title(lbl, fontsize=16)
    axes[r,c].set_ylabel('Fraction [%]', fontsize=16)
axes[3,0].set_xlabel('Hour (UTC+6)', fontsize=16); axes[3,1].set_xlabel('Hour (UTC+6)', fontsize=16)
fig = plt.gcf(); fig.set_size_inches(30, 30)
#txt = 'The diurnal cycle of precipitation cell fraction in the entire study area at every recorded stage for each precipitation threshold value (TV [mm $h^{-1}$] between 2001 and 2016\nbased on the three model datasets (WRF, HAR & ERA5) and the satellite dataset (GPM)'
#plt.figtext(0.12, 0.001, txt, horizontalalignment='left')
plt.show()
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/DC_th_filtered_color.png', dpi=200)

#%%
del axes, fig, xticks, TH_values, c, col, cols, r, rows, style, styles, TV, tv
#%% 
""" 
DC of features of TP only (>3000)

Using same name (tracks extension) but have pd.read.pickle the TP file

Returns a lot of NaNs, WHY???

"""
WRF_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_WRF_filtered.pkl')
HAR_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_HAR_filtered.pkl')
ERA5_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_ERA5_filtered.pkl')
GPM_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_GPM_filtered.pkl')

## To same annual extent
WRF_tracks = WRF_tracks[(WRF_tracks.time.dt.year >= 2001) & (WRF_tracks.time.dt.year <= 2016)]
HAR_tracks = HAR_tracks[(HAR_tracks.time.dt.year >= 2001) & (HAR_tracks.time.dt.year <= 2016)]
ERA5_tracks = ERA5_tracks[(ERA5_tracks.time.dt.year >= 2001) & (ERA5_tracks.time.dt.year <= 2016)]
GPM_tracks = GPM_tracks[(GPM_tracks.time.dt.year >= 2001) & (GPM_tracks.time.dt.year <= 2016)]

#%% 

keys = 'TH3', 'TH4', 'TH5', 'TH6', 'TH7', 'TH8', 'TH9', 'TH10'
WRF = WRF_tracks.cell.groupby([WRF_tracks.time.dt.hour, WRF_tracks.threshold_value]).count().to_frame(); WRF.reset_index(inplace=True)
HAR = HAR_tracks.cell.groupby([HAR_tracks.time.dt.hour, HAR_tracks.threshold_value]).count().to_frame(); HAR.reset_index(inplace=True)
ERA5 = ERA5_tracks.cell.groupby([ERA5_tracks.time.dt.hour, ERA5_tracks.threshold_value]).count().to_frame(); ERA5.reset_index(inplace=True)
GPM = GPM_tracks.cell.groupby([GPM_tracks.time.dt.hour, GPM_tracks.threshold_value]).count().to_frame(); GPM.reset_index(inplace=True)

WRF.time = WRF.time + 6; HAR.time = HAR.time + 6; ERA5.time = ERA5.time + 6; GPM.time = GPM.time + 6;
WRF.replace(to_replace={'time': {24:0, 25:1, 26:2, 27:3, 28:4, 29:5}}, inplace=True)
HAR.replace(to_replace={'time': {24:0, 25:1, 26:2, 27:3, 28:4, 29:5}}, inplace=True)
ERA5.replace(to_replace={'time': {24:0, 25:1, 26:2, 27:3, 28:4, 29:5}}, inplace=True)
GPM.replace(to_replace={'time': {24:0, 25:1, 26:2, 27:3, 28:4, 29:5}}, inplace=True)
WRF.sort_values(by='time', inplace=True); HAR.sort_values(by='time', inplace=True)
ERA5.sort_values(by='time', inplace=True); GPM.sort_values(by='time', inplace=True)

TH3 = pd.DataFrame();
TH3['WRF'] = pd.DataFrame(WRF[WRF.threshold_value == 3].cell.values); TH3['HAR'] = pd.DataFrame(HAR[HAR.threshold_value == 3].cell.values)
TH3['ERA5'] = pd.DataFrame(ERA5[ERA5.threshold_value == 3].cell.values); TH3['GPM'] = pd.DataFrame(GPM[GPM.threshold_value == 3].cell.values)
TH3.WRF = TH3.WRF/TH3.WRF.sum() * 100; TH3.HAR = TH3.HAR/TH3.HAR.sum() * 100
TH3.ERA5 = TH3.ERA5/TH3.ERA5.sum() * 100; TH3.GPM = TH3.GPM/TH3.GPM.sum() * 100


TH4 = pd.DataFrame(); 
TH4['WRF'] = pd.DataFrame(WRF[WRF.threshold_value == 4].cell.values); TH4['HAR'] = pd.DataFrame(HAR[HAR.threshold_value == 4].cell.values)
TH4['ERA5'] = pd.DataFrame(ERA5[ERA5.threshold_value == 4].cell.values); TH4['GPM'] = pd.DataFrame(GPM[GPM.threshold_value == 4].cell.values)
TH4.WRF = TH4.WRF/TH4.WRF.sum() * 100; TH4.HAR = TH4.HAR/TH4.HAR.sum() * 100
TH4.ERA5 = TH4.ERA5/TH4.ERA5.sum() * 100; TH4.GPM = TH4.GPM/TH4.GPM.sum() * 100

TH5 = pd.DataFrame();
TH5['WRF'] = pd.DataFrame(WRF[WRF.threshold_value == 5].cell.values); TH5['HAR'] = pd.DataFrame(HAR[HAR.threshold_value == 5].cell.values)
TH5['ERA5'] = pd.DataFrame(ERA5[ERA5.threshold_value == 5].cell.values); TH5['GPM'] = pd.DataFrame(GPM[GPM.threshold_value == 5].cell.values)
TH5.WRF = TH5.WRF/TH5.WRF.sum() * 100; TH5.HAR = TH5.HAR/TH5.HAR.sum() * 100
TH5.ERA5 = TH5.ERA5/TH5.ERA5.sum() * 100; TH5.GPM = TH5.GPM/TH5.GPM.sum() * 100

TH6 = pd.DataFrame();
TH6['WRF'] = pd.DataFrame(WRF[WRF.threshold_value == 6].cell.values); TH6['HAR'] = pd.DataFrame(HAR[HAR.threshold_value == 6].cell.values)
TH6['ERA5'] = pd.DataFrame(ERA5[ERA5.threshold_value == 6].cell.values); TH6['GPM'] = pd.DataFrame(GPM[GPM.threshold_value == 6].cell.values)
TH6.WRF = TH6.WRF/TH6.WRF.sum() * 100; TH6.HAR = TH6.HAR/TH6.HAR.sum() * 100
TH6.ERA5 = TH6.ERA5/TH6.ERA5.sum() * 100; TH6.GPM = TH6.GPM/TH6.GPM.sum() * 100

TH7 = pd.DataFrame(); 
TH7['WRF'] = pd.DataFrame(WRF[WRF.threshold_value == 7].cell.values); TH7['HAR'] = pd.DataFrame(HAR[HAR.threshold_value == 7].cell.values)
TH7['ERA5'] = pd.DataFrame(ERA5[ERA5.threshold_value == 7].cell.values); TH7['GPM'] = pd.DataFrame(GPM[GPM.threshold_value == 7].cell.values)
TH7.WRF = TH7.WRF/TH7.WRF.sum() * 100; TH7.HAR = TH7.HAR/TH7.HAR.sum() * 100
TH7.ERA5 = TH7.ERA5/TH7.ERA5.sum() * 100; TH7.GPM = TH7.GPM/TH7.GPM.sum() * 100

TH8 = pd.DataFrame();
TH8['WRF'] = pd.DataFrame(WRF[WRF.threshold_value == 8].cell.values); TH8['HAR'] = pd.DataFrame(HAR[HAR.threshold_value == 8].cell.values)
TH8['ERA5'] = pd.DataFrame(ERA5[ERA5.threshold_value == 8].cell.values); TH8['GPM'] = pd.DataFrame(GPM[GPM.threshold_value == 8].cell.values)
TH8.WRF = TH8.WRF/TH8.WRF.sum() * 100; TH8.HAR = TH8.HAR/TH8.HAR.sum() * 100
TH8.ERA5 = TH8.ERA5/TH8.ERA5.sum() * 100; TH8.GPM = TH8.GPM/TH8.GPM.sum() * 100

TH9 = pd.DataFrame();
TH9['WRF'] = pd.DataFrame(WRF[WRF.threshold_value == 9].cell.values); TH9['HAR'] = pd.DataFrame(HAR[HAR.threshold_value == 9].cell.values)
TH9['ERA5'] = pd.DataFrame(ERA5[ERA5.threshold_value == 9].cell.values); TH9['GPM'] = pd.DataFrame(GPM[GPM.threshold_value == 9].cell.values)
TH9.WRF = TH9.WRF/TH9.WRF.sum() * 100; TH9.HAR = TH9.HAR/TH9.HAR.sum() * 100
TH9.ERA5 = TH9.ERA5/TH9.ERA5.sum() * 100; TH9.GPM = TH9.GPM/TH9.GPM.sum() * 100

TH10 = pd.DataFrame();
TH10['WRF'] = pd.DataFrame(WRF[WRF.threshold_value == 10].cell.values); TH10['HAR'] = pd.DataFrame(HAR[HAR.threshold_value == 10].cell.values)
TH10['ERA5'] = pd.DataFrame(ERA5[ERA5.threshold_value == 10].cell.values); TH10['GPM'] = pd.DataFrame(GPM[GPM.threshold_value == 10].cell.values)
TH10.WRF = TH10.WRF/TH10.WRF.sum() * 100; TH10.HAR = TH10.HAR/TH10.HAR.sum() * 100
TH10.ERA5 = TH10.ERA5/TH10.ERA5.sum() * 100; TH10.GPM = TH10.GPM/TH10.GPM.sum() * 100

TH_values = pd.concat([TH3, TH4, TH5, TH6, TH7, TH8, TH9, TH10], keys=keys).reset_index().rename(columns={'level_0':'threshold_value', 'level_1':'Hour'})
TH_values = TH_values.fillna(0)
del WRF, HAR, ERA5, GPM, WRF_tracks, HAR_tracks, ERA5_tracks, GPM_tracks, TH3, TH4, TH5, TH6, TH7, TH8, TH9, TH10, keys
    
#%% Plotting
xticks = np.arange(0,24,2)
fig, axes = plt.subplots(2,4)
TH_values[TH_values.threshold_value == 'TH3'].plot(x='Hour',y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', title='TV3', ax=axes[0,0], grid=True, linestyle='--', xlim=[-0.5,23.5], ylabel='Fraction [%]', xlabel='', ylim=(0,25), xticks=xticks)
TH_values[TH_values.threshold_value == 'TH4'].plot(x='Hour',y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', title='TV4', ax=axes[0,1], grid=True, linestyle='--', xlim=[-0.5,23.5], ylabel='', xlabel='', ylim=(0,25), xticks=xticks)
TH_values[TH_values.threshold_value == 'TH5'].plot(x='Hour',y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', title='TV5', ax=axes[0,2], grid=True, linestyle='--', xlim=[-0.5,23.5], ylabel='', xlabel='', ylim=(0,25), xticks=xticks)
TH_values[TH_values.threshold_value == 'TH6'].plot(x='Hour',y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', title='TV6', ax=axes[0,3], grid=True, linestyle='--', xlim=[-0.5,23.5], ylabel='', xlabel='', ylim=(0,25), xticks=xticks)
TH_values[TH_values.threshold_value == 'TH7'].plot(x='Hour',y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', title='TV7', ax=axes[1,0], grid=True, linestyle='--', xlim=[-0.5,23.5], ylabel='Fraction [%]', xlabel='Hour (UTC+6)', ylim=(0,25), xticks=xticks)
TH_values[TH_values.threshold_value == 'TH8'].plot(x='Hour',y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', title='TV8', ax=axes[1,1], grid=True, linestyle='--', xlim=[-0.5,23.5], ylabel='', xlabel='Hour (UTC+6)', ylim=(0,25), xticks=xticks)
TH_values[TH_values.threshold_value == 'TH9'].plot(x='Hour',y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', title='TV9', ax=axes[1,2], grid=True, linestyle='--', xlim=[-0.5,23.5], ylabel='', xlabel='Hour (UTC+6)', ylim=(0,25), xticks=xticks)
TH_values[TH_values.threshold_value == 'TH10'].plot(x='Hour',y=['WRF', 'HAR', 'ERA5', 'GPM'], kind='line', title='TV10', ax=axes[1,3], grid=True, linestyle='--', xlim=[-0.5,23.5], ylabel='', xlabel='Hour (UTC+6)', ylim=(0,25), xticks=xticks)

fig = plt.gcf(); fig.set_size_inches(25, 10)
txt = 'The diurnal cycle of precipitation cell fraction on the TP only at every recorded stage for each precipitation threshold value (TV [mm $h^{-1}$] between 2001 and 2016\nbased on the three model datasets (WRF, HAR & ERA5) and the satellite dataset (GPM)'
plt.figtext(0.12, 0.001, txt, horizontalalignment='left')
plt.show()
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/DC_th_TP_filtered.png', dpi=100)

#%%
del axes, fig, txt, xticks, TH_values