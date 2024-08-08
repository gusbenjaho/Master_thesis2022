# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 09:23:48 2021

@author: benja
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, norm
from datetime import datetime
import seaborn as sns
import matplotlib as mpl

#%%
WRF_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/WRF_filtered.pkl')
HAR_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/HAR_filtered.pkl')
GPM_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/GPM_filtered.pkl')
ERA5_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/ERA5_filtered.pkl')
GPM_tracks['time'] = pd.to_datetime(GPM_tracks.timestr)

#%% 
""" 
Data of features of TP only (>3000)

"""
WRF_TP = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_WRF_filtered.pkl')
HAR_TP = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_HAR_filtered.pkl')
ERA5_TP = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_ERA5_filtered.pkl')
GPM_TP = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_GPM_filtered.pkl')

#%% VERY long function but it yield a DF with the duration for each cell plus its max size and max th value
def get_lifetime(preciptracks, start, end):
    FMT = "%Y-%m-%d %H:%M:%S"
    tracks_lt = pd.DataFrame()
    lifetime = []
    km2 = []
    th = []
    for y in np.arange(start, end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            cell_km2 = ytracks[ytracks.cell == cell].km2.values.max()
            cell_th = ytracks[ytracks.cell == cell].threshold_value.values.max()
            tdelta = datetime.strptime(ytracks[ytracks.cell == cell].timestr.iloc[-1], FMT) - datetime.strptime(ytracks[ytracks.cell == cell].timestr.iloc[0], FMT)
            days, seconds = tdelta.days, tdelta.seconds
            hours = (days*24) + (seconds/3600)
            lifetime.append(hours)
            km2.append(cell_km2)
            th.append(cell_th)
    tracks_lt['km2'] = pd.DataFrame(km2)
    tracks_lt['threshold_value'] = pd.DataFrame(th)
    tracks_lt['lifetime'] = pd.DataFrame(lifetime)
    return tracks_lt

#%% Life time for entire region
WRF_lt = get_lifetime(WRF_tracks, 2001, 2017)
HAR_lt = get_lifetime(HAR_tracks,2001,2017)
ERA5_lt = get_lifetime(ERA5_tracks, 2001, 2017)
GPM_lt = get_lifetime(GPM_tracks, 2001, 2017)
del WRF_tracks, HAR_tracks, ERA5_tracks, GPM_tracks

WRF_lt.to_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/WRF_lt_filtered.pkl')
HAR_lt.to_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/HAR_lt_filtered.pkl')
GPM_lt.to_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/GPM_lt_filtered.pkl')
ERA5_lt.to_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/ERA5_lt_filtered.pkl')

#%% Lifetime for TP only
TP_WRF_lt = get_lifetime(WRF_TP, 2001, 2017)
TP_HAR_lt = get_lifetime(HAR_TP,2001,2017)
TP_ERA5_lt = get_lifetime(ERA5_TP, 2001, 2017)
TP_GPM_lt = get_lifetime(GPM_TP, 2001, 2017)
del WRF_TP, HAR_TP, ERA5_TP, GPM_TP

TP_WRF_lt.to_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_WRF_lt_filtered.pkl')
TP_HAR_lt.to_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_HAR_lt_filtered.pkl')
TP_GPM_lt.to_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_GPM_lt_filtered.pkl')
TP_ERA5_lt.to_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_ERA5_lt_filtered.pkl')

#%% Load files from above
WRF_lt = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/WRF_lt_filtered.pkl')
HAR_lt = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/HAR_lt_filtered.pkl')
GPM_lt = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/GPM_lt_filtered.pkl')
ERA5_lt = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/ERA5_lt_filtered.pkl')

TP_WRF_lt = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_WRF_lt_filtered.pkl')
TP_HAR_lt = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_HAR_lt_filtered.pkl')
TP_GPM_lt = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_GPM_lt_filtered.pkl')
TP_ERA5_lt = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_ERA5_lt_filtered.pkl')

#%% 
"""
Lifetime distribution

"""
## Boxlpot of lifetime dataprep
WRF_lt['Model'] = 'WRF_GU'; HAR_lt['Model'] = 'HAR'; ERA5_lt['Model'] = 'ERA5'; GPM_lt['Model'] = 'GPM'
tracks_lt = WRF_lt.melt(id_vars=['Model'], value_vars=['lifetime']); tracks_lt = tracks_lt.append(HAR_lt.melt(id_vars=['Model'], value_vars=['lifetime']));
tracks_lt = tracks_lt.append(ERA5_lt.melt(id_vars=['Model'], value_vars=['lifetime'])); tracks_lt = tracks_lt.append(GPM_lt.melt(id_vars=['Model'], value_vars=['lifetime']));

TP_WRF_lt['Model'] = 'WRF_GU'; TP_HAR_lt['Model'] = 'HAR'; TP_ERA5_lt['Model'] = 'ERA5'; TP_GPM_lt['Model'] = 'GPM'
TP_tracks_lt = TP_WRF_lt.melt(id_vars=['Model'], value_vars=['lifetime']); TP_tracks_lt = TP_tracks_lt.append(TP_HAR_lt.melt(id_vars=['Model'], value_vars=['lifetime']));
TP_tracks_lt = TP_tracks_lt.append(TP_ERA5_lt.melt(id_vars=['Model'], value_vars=['lifetime'])); TP_tracks_lt = TP_tracks_lt.append(TP_GPM_lt.melt(id_vars=['Model'], value_vars=['lifetime']));

#%% Plotting
yticklabels = '0', '2', '4', '6', '8', '10', '12','14','16','18','>20'
colors = ['#1E90FF', '#104E8B', '#FF6103', '#FF9912']
color_dict = dict(zip(tracks_lt.Model.unique(),colors)); del colors
fig, (ax1,ax2) = plt.subplots(2,1)
sns.set(rc = {'figure.figsize':(15,8)})
bplot = sns.boxplot(x='Model', y='value', data=tracks_lt, ax=ax1, dodge=False, width=0.4)
for i in range(0,4):
    mybox = bplot.artists[i]
    mybox.set_facecolor(color_dict[tracks_lt.Model.unique()[i]])
ax1.set(ylim=(-1,20), yticks=np.arange(0,22,2), ylabel='Hours', xlabel='', title='a)')
ax1.set_yticklabels(yticklabels)

sns.set(rc = {'figure.figsize':(15,8)})
bplot = sns.boxplot(x='Model', y='value', data=TP_tracks_lt, ax=ax2, dodge=False, width=0.4, palette='Blues')
for i in range(0,4):
    mybox = bplot.artists[i]
    mybox.set_facecolor(color_dict[tracks_lt.Model.unique()[i]])
ax2.set(ylim=(-1,20), yticks=np.arange(0,22,2), ylabel='Hours', xlabel='', title='b)')
ax2.set_yticklabels(yticklabels)
fig.set_size_inches(14, 10)
txt = 'Boxplot of precipitation cell lifetime in the entire study area (a) and TP only (b), based on the three model datasets (WRF, HAR & ERA5)\nand the satellite dataset (GPM).'
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/lifetime_filtered.png', dpi=100)
del ax1, ax2, fig, txt, yticklabels, i, mybox, bplot, color_dict
#%% 
del tracks_lt, TP_tracks_lt

#%% Boxplot of lifetime for each TH dataprep
WRF_lt_th = WRF_lt.melt(id_vars=['threshold_value'], value_vars=['lifetime'])
HAR_lt_th = HAR_lt.melt(id_vars=['threshold_value'], value_vars=['lifetime'])
ERA5_lt_th = ERA5_lt.melt(id_vars=['threshold_value'], value_vars=['lifetime'])
GPM_lt_th = GPM_lt.melt(id_vars=['threshold_value'], value_vars=['lifetime'])

TP_WRF_lt_th = TP_WRF_lt.melt(id_vars=['threshold_value'], value_vars=['lifetime'])
TP_HAR_lt_th = TP_HAR_lt.melt(id_vars=['threshold_value'], value_vars=['lifetime'])
TP_ERA5_lt_th = TP_ERA5_lt.melt(id_vars=['threshold_value'], value_vars=['lifetime'])
TP_GPM_lt_th = TP_GPM_lt.melt(id_vars=['threshold_value'], value_vars=['lifetime'])

#%% Plotting
## Entire study area
yticklabels = '0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60', '>65'
fig, axes = plt.subplots(2,2)
sns.boxplot(x='threshold_value', y='value', hue='threshold_value', data=WRF_lt_th, ax=axes[0,0], dodge=False, color='cyan').set(xlabel='', ylabel='Hours', title='a)')
sns.boxplot(x='threshold_value', y='value', hue='threshold_value', data=HAR_lt_th, ax=axes[0,1], dodge=False, color='cyan').set(xlabel='', ylabel='', title='b)')
sns.boxplot(x='threshold_value', y='value', hue='threshold_value', data=ERA5_lt_th, ax=axes[1,0], dodge=False, color='cyan').set(xlabel='Threshold value [mm $h^{-1}$]', ylabel='Hours', title='c)')
sns.boxplot(x='threshold_value', y='value', hue='threshold_value', data=GPM_lt_th, ax=axes[1,1], dodge=False, color='cyan').set(xlabel='Threshold value [mm $h^{-1}$]', ylabel='', title='d)')
axes[0,0].get_legend().remove();axes[0,1].get_legend().remove();axes[1,0].get_legend().remove();axes[1,1].get_legend().remove()
axes[0,0].set(ylim=(-1,65), yticks=np.arange(0,70,5)); axes[0,1].set(ylim=(-1,65), yticks=np.arange(0,70,5), yticklabels=yticklabels)
axes[1,0].set(ylim=(-1,65), yticks=np.arange(0,70,5), yticklabels=yticklabels); axes[1,1].set(ylim=(-1,65), yticks=np.arange(0,70,5))

fig.set_size_inches(14, 10)
txt = 'Boxplot of precipitation cell lifetime in entire study area based on each threshold value of precipitation for WRF (a), HAR (b), ERA5 (c) and\nGPM (d). (Note: b and c have three values each above 65)'
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/Lifetime_TH_filtered.png', dpi=100)

#%% TP only
fig, axes = plt.subplots(2,2)
sns.boxplot(x='threshold_value', y='value', hue='threshold_value', data=TP_WRF_lt_th, ax=axes[0,0], dodge=False, color='cyan').set(xlabel='', ylabel='Hours', title='a)')
sns.boxplot(x='threshold_value', y='value', hue='threshold_value', data=TP_HAR_lt_th, ax=axes[0,1], dodge=False, color='cyan').set(xlabel='', ylabel='', title='b)')
sns.boxplot(x='threshold_value', y='value', hue='threshold_value', data=TP_ERA5_lt_th, ax=axes[1,0], dodge=False, color='cyan').set(xlabel='Threshold value [mm $h^{-1}$]', ylabel='Hours', title='c)')
sns.boxplot(x='threshold_value', y='value', hue='threshold_value', data=TP_GPM_lt_th, ax=axes[1,1], dodge=False, color='cyan').set(xlabel='Threshold value [mm $h^{-1}$]', ylabel='', title='d)')
axes[0,0].get_legend().remove();axes[0,1].get_legend().remove();axes[1,0].get_legend().remove();axes[1,1].get_legend().remove()
axes[0,0].set(ylim=(-1,65), yticks=np.arange(0,70,5)); axes[0,1].set(ylim=(-1,65), yticks=np.arange(0,70,5), yticklabels=yticklabels)
axes[1,0].set(ylim=(-1,65), yticks=np.arange(0,70,5), yticklabels=yticklabels); axes[1,1].set(ylim=(-1,65), yticks=np.arange(0,70,5))

fig.set_size_inches(14, 10)
txt = 'Boxplot of precipitation cell lifetime over the TP only based on each threshold value of precipitation for WRF (a), HAR (b), ERA5 (c) and GPM (d)'
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/Lifetime_TH_filtered_TP.png', dpi=100)

#%%
del WRF_lt_th, TP_WRF_lt_th, HAR_lt_th, TP_HAR_lt_th, ERA5_lt_th, TP_ERA5_lt_th, GPM_lt_th, TP_GPM_lt_th, txt, yticklabels, fig, axes

#%% 
"""
Maximum extension in each TH

"""
## Boxplot of km2 for each TH dataprep
WRF_km2_th = WRF_lt.melt(id_vars=['threshold_value'], value_vars=['km2'])
HAR_km2_th = HAR_lt.melt(id_vars=['threshold_value'], value_vars=['km2'])
ERA5_km2_th = ERA5_lt.melt(id_vars=['threshold_value'], value_vars=['km2'])
GPM_km2_th = GPM_lt.melt(id_vars=['threshold_value'], value_vars=['km2'])

TP_WRF_km2_th = TP_WRF_lt.melt(id_vars=['threshold_value'], value_vars=['km2'])
TP_HAR_km2_th = TP_HAR_lt.melt(id_vars=['threshold_value'], value_vars=['km2'])
TP_ERA5_km2_th = TP_ERA5_lt.melt(id_vars=['threshold_value'], value_vars=['km2'])
TP_GPM_km2_th = TP_GPM_lt.melt(id_vars=['threshold_value'], value_vars=['km2'])
del WRF_lt, TP_WRF_lt, HAR_lt, TP_HAR_lt, ERA5_lt, TP_ERA5_lt, GPM_lt, TP_GPM_lt

#%% Plotting entire area
#yticklabels = '0', '100000', '200000', '300000', '400000', '500000', '600000', '700000', '800000', '900000', '>1000000'
fig, axes = plt.subplots(2,2)
yticks = np.arange(0,1200000,100000)
sns.boxplot(x='threshold_value', y='value', hue='threshold_value', data=WRF_km2_th, ax=axes[0,0], dodge=False, color='cyan').set(xlabel='', ylabel='Cell size [$km^2$]', title='a)')
sns.boxplot(x='threshold_value', y='value', hue='threshold_value', data=HAR_km2_th, ax=axes[0,1], dodge=False, color='cyan').set(xlabel='', ylabel='', title='b)')
sns.boxplot(x='threshold_value', y='value', hue='threshold_value', data=ERA5_km2_th, ax=axes[1,0], dodge=False, color='cyan').set(xlabel='Threshold value [mm $h^{-1}$]', ylabel='Cell size [$km^2$]', title='c)')
sns.boxplot(x='threshold_value', y='value', hue='threshold_value', data=GPM_km2_th, ax=axes[1,1], dodge=False, color='cyan').set(xlabel='Threshold value [mm $h^{-1}$]', ylabel='', title='d)')
axes[0,0].get_legend().remove();axes[0,1].get_legend().remove();axes[1,0].get_legend().remove();axes[1,1].get_legend().remove()
axes[0,0].set(ylim=(0,1000000), yticks=yticks); axes[0,1].set(ylim=(0,1000000), yticks=yticks)
axes[1,0].set(ylim=(0,1000000), yticks=yticks); axes[1,1].set(ylim=(0,1000000), yticks=yticks)

plt.style.use('seaborn')
fig.show()
fig.set_size_inches(14, 10)
txt = 'Boxplot of precipitation cells at max extension in the entire study area based on each threshold value of precipitation for WRF (a), HAR (b), ERA5 (c) and GPM (d)'
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/KM2_TH_filtered.png', dpi=100)

#%% TP only
fig, axes = plt.subplots(2,2)
yticks = np.arange(0,650000,50000)
sns.boxplot(x='threshold_value', y='value', hue='threshold_value', data=TP_WRF_km2_th, ax=axes[0,0], dodge=False, color='cyan').set(xlabel='', ylabel='Cell size [$km^2$]', title='a)')
sns.boxplot(x='threshold_value', y='value', hue='threshold_value', data=TP_HAR_km2_th, ax=axes[0,1], dodge=False, color='cyan').set(xlabel='', ylabel='', title='b)')
sns.boxplot(x='threshold_value', y='value', hue='threshold_value', data=TP_ERA5_km2_th, ax=axes[1,0], dodge=False, color='cyan').set(xlabel='Threshold value [mm $h^{-1}$]', ylabel='Cell size [$km^2$]', title='c)')
sns.boxplot(x='threshold_value', y='value', hue='threshold_value', data=TP_GPM_km2_th, ax=axes[1,1], dodge=False, color='cyan').set(xlabel='Threshold value [mm $h^{-1}$]', ylabel='', title='d)')
axes[0,0].get_legend().remove();axes[0,1].get_legend().remove();axes[1,0].get_legend().remove();axes[1,1].get_legend().remove()
axes[0,0].set(ylim=(0,600000), yticks=yticks); axes[0,1].set(ylim=(0,600000), yticks=yticks)
axes[1,0].set(ylim=(0,600000), yticks=yticks); axes[1,1].set(ylim=(0,600000), yticks=yticks)

#plt.style.use('seaborn')
fig.show()
fig.set_size_inches(14, 10)
txt = 'Boxplot of precipitation cells at max extension on the TP only based on each threshold value of precipitation for WRF (a), HAR (b), ERA5 (c) and GPM (d)'
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/KM2_TH_TP_filtered.png', dpi=100)

#%% 
"""
Correlations

"""
## Load files from above
WRF_lt = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/WRF_lt_filtered.pkl'); WRF_lt = WRF_lt[['km2', 'threshold_value', 'lifetime']]
HAR_lt = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/HAR_lt_filtered.pkl'); HAR_lt = HAR_lt[['km2', 'threshold_value', 'lifetime']]
GPM_lt = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/GPM_lt_filtered.pkl'); GPM_lt = GPM_lt[['km2', 'threshold_value', 'lifetime']]
ERA5_lt = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/ERA5_lt_filtered.pkl'); ERA5_lt = ERA5_lt[['km2', 'threshold_value', 'lifetime']]

TP_WRF_lt = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_WRF_lt_filtered.pkl'); TP_WRF_lt = WRF_lt[['km2', 'threshold_value', 'lifetime']]
TP_HAR_lt = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_HAR_lt_filtered.pkl'); TP_HAR_lt = TP_HAR_lt[['km2', 'threshold_value', 'lifetime']]
TP_GPM_lt = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_GPM_lt_filtered.pkl'); TP_GPM_lt = TP_GPM_lt[['km2', 'threshold_value', 'lifetime']]
TP_ERA5_lt = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_ERA5_lt_filtered.pkl'); TP_ERA5_lt = TP_ERA5_lt[['km2', 'threshold_value', 'lifetime']]

#%% Correlation Spearman, since TH (and lifetime?) is ordinal?
WRF_scorr, WRF_spval = spearmanr(WRF_lt); WRF_scorr[WRF_scorr > 0.99] = np.nan
HAR_scorr, HAR_spval = spearmanr(HAR_lt); HAR_scorr[HAR_scorr > 0.99] = np.nan
ERA5_scorr, ERA5_spval = spearmanr(ERA5_lt); ERA5_scorr[ERA5_scorr > 0.99] = np.nan
GPM_scorr, GPM_spval = spearmanr(GPM_lt); GPM_scorr[GPM_scorr > 0.99] = np.nan

TP_WRF_scorr, TP_WRF_spval = spearmanr(TP_WRF_lt); TP_WRF_scorr[TP_WRF_scorr > 0.99] = np.nan
TP_HAR_scorr, TP_HAR_spval = spearmanr(TP_HAR_lt); TP_HAR_scorr[TP_HAR_scorr > 0.99] = np.nan
TP_ERA5_scorr, TP_ERA5_spval = spearmanr(TP_ERA5_lt); TP_ERA5_scorr[TP_ERA5_scorr > 0.99] = np.nan
TP_GPM_scorr, TP_GPM_spval = spearmanr(TP_GPM_lt); TP_GPM_scorr[TP_GPM_scorr > 0.99] = np.nan

del WRF_lt, HAR_lt, ERA5_lt, GPM_lt, TP_WRF_lt, TP_HAR_lt, TP_ERA5_lt, TP_GPM_lt

#%% Plots
## Entire study area
plt.style.use('seaborn')
yticklabels=['Size', 'Intensity level', 'Lifetime']; xticklabels=yticklabels
vmin = 0; vmax = 0.6
cmap = sns.cm.rocket_r
fig, axes = plt.subplots(2,2)
sns.heatmap(WRF_scorr, annot=True, ax=axes[0,0], xticklabels=xticklabels, yticklabels=yticklabels, vmin=vmin, vmax=vmax, linewidth=1, linecolor='k', square=True, cbar=False, cmap=cmap)
axes[0,0].set_title('a)')

sns.heatmap(HAR_scorr, annot=True, ax=axes[0,1], xticklabels=xticklabels, yticklabels=yticklabels, vmin=vmin, vmax=vmax, linewidth=1, linecolor='k', square=True, cbar=False, cmap=cmap)
axes[0,1].set_title('b)')

sns.heatmap(ERA5_scorr, annot=True, xticklabels=xticklabels, ax=axes[1,0], yticklabels=yticklabels, vmin=vmin, vmax=vmax, linewidth=1, linecolor='k', square=True, cbar=False, cmap=cmap)
axes[1,0].set_title('c)')

sns.heatmap(GPM_scorr, annot=True, xticklabels=xticklabels, ax=axes[1,1], yticklabels=yticklabels, vmin=vmin, vmax=vmax, linewidth=1, linecolor='k', square=True, cbar=False, cmap=cmap)
axes[1,1].set_title('d)')
fig.subplots_adjust(wspace=-0.4)

cbar_ax = fig.add_axes([0.8, 0.112, 0.02, 0.764])
norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, cax=cbar_ax).set_label('Spearman´s rank correlation coefficient (rho)')
txt = 'Spearman correlation of cell size  [$km^2$], precipitation threshold value (TV [mm $h^{-1}$]) and cell lifetime (LT [Hours])\nin the entire study area based on WRF (a), HAR (b), ERA5 (c) and GPM (d)'
plt.figtext(0.24, 0.03, txt, horizontalalignment='left')
fig.set_size_inches(14, 10)
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/SpearCorr_Lifet_TH_km2_filtered.png', dpi=100)

#%%
## TP only
plt.style.use('seaborn')
cmap = sns.cm.flare
fig, axes = plt.subplots(2,2)
sns.heatmap(TP_WRF_scorr, annot=True, ax=axes[0,0], xticklabels=xticklabels, yticklabels=yticklabels, vmin=vmin, vmax=vmax, linewidth=1, linecolor='k', square=True, cbar=False, cmap=cmap)
axes[0,0].set_title('a)')

sns.heatmap(TP_HAR_scorr, annot=True, ax=axes[0,1], xticklabels=xticklabels, yticklabels=yticklabels, vmin=vmin, vmax=vmax, linewidth=1, linecolor='k', square=True, cbar=False, cmap=cmap)
axes[0,1].set_title('b)')

sns.heatmap(TP_ERA5_scorr, annot=True, xticklabels=xticklabels, ax=axes[1,0], yticklabels=yticklabels, vmin=vmin, vmax=vmax, linewidth=1, linecolor='k', square=True, cbar=False, cmap=cmap)
axes[1,0].set_title('c)')

sns.heatmap(TP_GPM_scorr, annot=True, xticklabels=xticklabels, ax=axes[1,1], yticklabels=yticklabels, vmin=vmin, vmax=vmax, linewidth=1, linecolor='k', square=True, cbar=False, cmap=cmap)
axes[1,1].set_title('d)')
fig.subplots_adjust(wspace=-0.4)

cbar_ax = fig.add_axes([0.8, 0.112, 0.02, 0.764])
norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, cax=cbar_ax).set_label('Spearman´s rank correlation coefficient (rho)')
txt = 'Spearman correlation of cell size [$km^2$], precipitation threshold value (TV [mm $h^{-1}$]) and cell lifetime (LT [Hours])\nover the TP only based on WRF (a), HAR (b), ERA5 (c) and GPM (d)'
plt.figtext(0.24, 0.03, txt, horizontalalignment='left')
fig.set_size_inches(14, 10)
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/SpearCorr_Lifet_TH_km2_TP_filtered.png', dpi=100)