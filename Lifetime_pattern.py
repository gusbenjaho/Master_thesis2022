# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 16:09:25 2021

@author: benja
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr, norm
from datetime import datetime
import seaborn as sns
import matplotlib as mpl
import xarray as xr

#%%
WRF_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_tracks.pkl')
HAR_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_tracks.pkl')
GPM_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_tracks.pkl')
ERA5_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_tracks.pkl')
GPM_tracks['time'] = pd.to_datetime(GPM_tracks.timestr)

#%% 
"""
Data over the TP only
Functions and data needed

"""
WRF_TP = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_TP.pkl')
HAR_TP = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_TP.pkl')
ERA5_TP = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_TP.pkl')
GPM_TP = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_TP.pkl')

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
WRF_lt = get_lifetime(WRF_tracks, 2001, 2016)
HAR_lt = get_lifetime(HAR_tracks,2001,2016)
ERA5_lt = get_lifetime(ERA5_tracks, 2001, 2016)
GPM_lt = get_lifetime(GPM_tracks, 2001, 2016)
del WRF_tracks, HAR_tracks, ERA5_tracks, GPM_tracks

WRF_lt.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_lt.pkl')
HAR_lt.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_lt.pkl')
GPM_lt.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_lt.pkl')
ERA5_lt.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_lt.pkl')

#%% Lifetime for TP only
TP_WRF_lt = get_lifetime(WRF_TP, 2001, 2016)
TP_HAR_lt = get_lifetime(HAR_TP,2001,2016)
TP_ERA5_lt = get_lifetime(ERA5_TP, 2001, 2016)
TP_GPM_lt = get_lifetime(GPM_TP, 2001, 2016)
del WRF_TP, HAR_TP, ERA5_TP, GPM_TP

TP_WRF_lt.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/TP_WRF_lt.pkl')
TP_HAR_lt.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/TP_HAR_lt.pkl')
TP_GPM_lt.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/TP_GPM_lt.pkl')
TP_ERA5_lt.to_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/TP_ERA5_lt.pkl')

#%% Load files from above
WRF_lt = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_lt.pkl')
HAR_lt = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_lt.pkl')
GPM_lt = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_lt.pkl')
ERA5_lt = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_lt.pkl')

TP_WRF_lt = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/TP_WRF_lt.pkl')
TP_HAR_lt = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/TP_HAR_lt.pkl')
TP_GPM_lt = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/TP_GPM_lt.pkl')
TP_ERA5_lt = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/TP_ERA5_lt.pkl')

#%% 
"""
Lifetime distribution

"""
## Boxlpot of lifetime dataprep
WRF_lt['Model'] = 'WRF'; HAR_lt['Model'] = 'HAR'; ERA5_lt['Model'] = 'ERA5'; GPM_lt['Model'] = 'GPM'
tracks_lt = WRF_lt.melt(id_vars=['Model'], value_vars=['lifetime']); tracks_lt = tracks_lt.append(HAR_lt.melt(id_vars=['Model'], value_vars=['lifetime']));
tracks_lt = tracks_lt.append(ERA5_lt.melt(id_vars=['Model'], value_vars=['lifetime'])); tracks_lt = tracks_lt.append(GPM_lt.melt(id_vars=['Model'], value_vars=['lifetime']));

TP_WRF_lt['Model'] = 'WRF'; TP_HAR_lt['Model'] = 'HAR'; TP_ERA5_lt['Model'] = 'ERA5'; TP_GPM_lt['Model'] = 'GPM'
TP_tracks_lt = TP_WRF_lt.melt(id_vars=['Model'], value_vars=['lifetime']); TP_tracks_lt = TP_tracks_lt.append(TP_HAR_lt.melt(id_vars=['Model'], value_vars=['lifetime']));
TP_tracks_lt = TP_tracks_lt.append(TP_ERA5_lt.melt(id_vars=['Model'], value_vars=['lifetime'])); TP_tracks_lt = TP_tracks_lt.append(TP_GPM_lt.melt(id_vars=['Model'], value_vars=['lifetime']));

#%% Plotting
yticklabels = '0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '>60'
fig, (ax1,ax2) = plt.subplots(2,1)
sns.set(rc = {'figure.figsize':(15,8)})
sns.boxplot(x='Model', y='value', hue='Model', data=tracks_lt, ax=ax1, dodge=False, width=0.4)
ax1.set(ylim=(-1,60), yticks=np.arange(0,65,5), ylabel='Hours', xlabel='', title='a)')
ax1.set_yticklabels(yticklabels)

sns.set(rc = {'figure.figsize':(15,8)})
sns.boxplot(x='Model', y='value', hue='Model', data=TP_tracks_lt, ax=ax2, dodge=False, width=0.4)
ax2.set(ylim=(-1,60), yticks=np.arange(0,65,5), ylabel='Hours', xlabel='', title='b)')
ax2.set_yticklabels(yticklabels)


txt = 'Boxplot of precipitation cell lifetime in entire study area (a) and TP only (b), based on the three model datasets (WRF, HAR & ERA5) and the satellite dataset (GPM)'
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/lifetime_box.png', dpi=100)

#%% 
del tracks_lt, TP_tracks_lt, ax1, ax2, fig, txt, yticklabels

#%% Boxplot of lifetime for each TH dataprep
WRF_lt_th = WRF_lt.melt(id_vars=['threshold_value'], value_vars=['lifetime'])
HAR_lt_th = HAR_lt.melt(id_vars=['threshold_value'], value_vars=['lifetime'])
ERA5_lt_th = ERA5_lt.melt(id_vars=['threshold_value'], value_vars=['lifetime'])
GPM_lt_th = GPM_lt.melt(id_vars=['threshold_value'], value_vars=['lifetime'])

TP_WRF_lt_th = TP_WRF_lt.melt(id_vars=['threshold_value'], value_vars=['lifetime'])
TP_HAR_lt_th = TP_HAR_lt.melt(id_vars=['threshold_value'], value_vars=['lifetime'])
TP_ERA5_lt_th = TP_ERA5_lt.melt(id_vars=['threshold_value'], value_vars=['lifetime'])
TP_GPM_lt_th = TP_GPM_lt.melt(id_vars=['threshold_value'], value_vars=['lifetime'])

del WRF_lt, TP_WRF_lt, HAR_lt, TP_HAR_lt, ERA5_lt, TP_ERA5_lt, GPM_lt, TP_GPM_lt

#%% Plotting
## Entire study area
yticklabels = '0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '>60'
fig, axes = plt.subplots(2,2)
sns.boxplot(x='threshold_value', y='value', hue='threshold_value', data=WRF_lt_th, ax=axes[0,0], dodge=False, color='cyan').set(xlabel='', ylabel='Hours', title='a)')
sns.boxplot(x='threshold_value', y='value', hue='threshold_value', data=HAR_lt_th, ax=axes[0,1], dodge=False, color='cyan').set(xlabel='', ylabel='', title='b)')
sns.boxplot(x='threshold_value', y='value', hue='threshold_value', data=ERA5_lt_th, ax=axes[1,0], dodge=False, color='cyan').set(xlabel='Threshold value', ylabel='Hours', title='c)')
sns.boxplot(x='threshold_value', y='value', hue='threshold_value', data=GPM_lt_th, ax=axes[1,1], dodge=False, color='cyan').set(xlabel='Threshold value', ylabel='', title='d)')
axes[0,0].get_legend().remove();axes[0,1].get_legend().remove();axes[1,0].get_legend().remove();axes[1,1].get_legend().remove()
axes[0,0].set(ylim=(0,60), yticks=np.arange(0,65,5)); axes[0,1].set(ylim=(0,60), yticks=np.arange(0,65,5), yticklabels=yticklabels)
axes[1,0].set(ylim=(0,60), yticks=np.arange(0,65,5), yticklabels=yticklabels); axes[1,1].set(ylim=(0,60), yticks=np.arange(0,65,5))

txt = 'Boxplot of precipitation cell lifetime in entire study area based on each threshold value of precipitation for WRF (a), HAR (b), ERA5 (c) and GPM (d)'
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/Lifetime_TH_box.png', dpi=100)

#%% TP only
fig, axes = plt.subplots(2,2)
sns.boxplot(x='threshold_value', y='value', hue='threshold_value', data=TP_WRF_lt_th, ax=axes[0,0], dodge=False, color='cyan').set(xlabel='', ylabel='Hours', title='a)')
sns.boxplot(x='threshold_value', y='value', hue='threshold_value', data=TP_HAR_lt_th, ax=axes[0,1], dodge=False, color='cyan').set(xlabel='', ylabel='', title='b)')
sns.boxplot(x='threshold_value', y='value', hue='threshold_value', data=TP_ERA5_lt_th, ax=axes[1,0], dodge=False, color='cyan').set(xlabel='Threshold value', ylabel='Hours', title='c)')
sns.boxplot(x='threshold_value', y='value', hue='threshold_value', data=TP_GPM_lt_th, ax=axes[1,1], dodge=False, color='cyan').set(xlabel='Threshold value', ylabel='', title='d)')
axes[0,0].get_legend().remove();axes[0,1].get_legend().remove();axes[1,0].get_legend().remove();axes[1,1].get_legend().remove()
axes[0,0].set(ylim=(-1,60), yticks=np.arange(0,65,5)); axes[0,1].set(ylim=(-1,60), yticks=np.arange(0,65,5), yticklabels=yticklabels)
axes[1,0].set(ylim=(-1,60), yticks=np.arange(0,65,5), yticklabels=yticklabels); axes[1,1].set(ylim=(-1,60), yticks=np.arange(0,65,5))

txt = 'Boxplot of precipitation cell lifetime over the TP only based on each threshold value of precipitation for WRF (a), HAR (b), ERA5 (c) and GPM (d)'
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/Lifetime_TH_box_TP.png', dpi=100)

#%%
del WRF_lt_th, TP_WRF_lt_th, HAR_lt_th, TP_HAR_lt_th, ERA5_lt_th, TP_ERA5_lt_th, GPM_lt_th, TP_GPM_lt_th, txt, yticklabels, fig, axes

#%% Load files from above
WRF_lt = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_lt.pkl')
HAR_lt = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_lt.pkl')
GPM_lt = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_lt.pkl')
ERA5_lt = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_lt.pkl')

TP_WRF_lt = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/TP_WRF_lt.pkl')
TP_HAR_lt = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/TP_HAR_lt.pkl')
TP_GPM_lt = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/TP_GPM_lt.pkl')
TP_ERA5_lt = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/TP_ERA5_lt.pkl')

#%%
"""
Maybe create a PDF of lifespan? As in Hu2016 and Yaodong 2008 (also a cdf)

"""
## Entire study area
## from https://machinelearningmastery.com/probability-density-estimation/
WRF_mean = WRF_lt.lifetime.mean(); WRF_std = WRF_lt.lifetime.std(); WRF_dist = norm(WRF_mean, WRF_std)
WRF_val = [value for value in np.unique(WRF_lt.lifetime)]; WRF_prob = [WRF_dist.pdf(value) for value in WRF_val]

HAR_mean = HAR_lt.lifetime.mean(); HAR_std = HAR_lt.lifetime.std(); HAR_dist = norm(HAR_mean, HAR_std)
HAR_val = [value for value in np.unique(HAR_lt.lifetime)]; HAR_prob = [HAR_dist.pdf(value) for value in HAR_val]

ERA5_mean = ERA5_lt.lifetime.mean(); ERA5_std = ERA5_lt.lifetime.std(); ERA5_dist = norm(ERA5_mean, ERA5_std)
ERA5_val = [value for value in np.unique(ERA5_lt.lifetime)]; ERA5_prob = [ERA5_dist.pdf(value) for value in ERA5_val]

GPM_mean = GPM_lt.lifetime.mean(); GPM_std = GPM_lt.lifetime.std(); GPM_dist = norm(GPM_mean, GPM_std)
GPM_val = [value for value in np.unique(GPM_lt.lifetime)]; GPM_prob = [GPM_dist.pdf(value) for value in GPM_val]

fig, axes = plt.subplots(2,2)
axes[0,0].hist(WRF_lt.lifetime, bins=np.unique(WRF_lt.lifetime)-0.5, density=True); axes[0,0].plot(WRF_val, WRF_prob); axes[0,0].set_title('a)'); 
axes[0,0].set_xlim(0,30); axes[0,0].set_ylabel('Probability'); axes[0,0].set_xticks(np.arange(0,30,1)); axes[0,0].set_xlim([-1, 30])

axes[0,1].hist(HAR_lt.lifetime, bins=np.unique(HAR_lt.lifetime)-0.5, density=True); axes[0,1].plot(HAR_val, HAR_prob); axes[0,1].set_title('b)'); 
axes[0,1].set_xlim(0,30); axes[0,1].set_xticks(np.arange(0,30,1)); axes[0,1].set_xlim([-1, 30])

axes[1,0].hist(ERA5_lt.lifetime, bins=np.unique(ERA5_lt.lifetime)-0.5, density=True); axes[1,0].plot(ERA5_val, ERA5_prob); axes[1,0].set_title('c)'); 
axes[1,0].set_xlim(0,30); axes[1,0].set_xlabel('Hours'); axes[1,0].set_ylabel('Probability'); axes[1,0].set_xticks(np.arange(0,30,1)); axes[1,0].set_xlim([-1, 30])

axes[1,1].hist(GPM_lt.lifetime, bins=np.unique(GPM_lt.lifetime)-0.5, density=True); axes[1,1].plot(GPM_val, GPM_prob); axes[1,1].set_title('d)'); 
axes[1,1].set_xlim(0,30); axes[1,1].set_xlabel('Hours'); axes[1,1].set_xticks(np.arange(0,30,1)); axes[1,1].set_xlim([-1, 30])

txt = 'Probability density function of precipitation cell lifetime in entire study area for WRF (a), HAR (b), ERA5 (c) and GPM (d)'
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/PDF_lifetime.png', dpi=100)

#%% 
del fig, axes, WRF_dist, WRF_mean, WRF_prob, WRF_std, WRF_val, HAR_dist, HAR_mean, HAR_prob, HAR_std, HAR_val
del ERA5_dist, ERA5_mean, ERA5_prob, ERA5_std, ERA5_val, GPM_dist, GPM_mean, GPM_prob, GPM_std, GPM_val, txt

#%% TP only

TP_WRF_mean = TP_WRF_lt.lifetime.mean(); TP_WRF_std = TP_WRF_lt.lifetime.std(); TP_WRF_dist = norm(TP_WRF_mean, TP_WRF_std)
TP_WRF_val = [value for value in np.unique(TP_WRF_lt.lifetime)]; TP_WRF_prob = [TP_WRF_dist.pdf(value) for value in TP_WRF_val]

TP_HAR_mean = TP_HAR_lt.lifetime.mean(); TP_HAR_std = TP_HAR_lt.lifetime.std(); TP_HAR_dist = norm(TP_HAR_mean, TP_HAR_std)
TP_HAR_val = [value for value in np.unique(TP_HAR_lt.lifetime)]; TP_HAR_prob = [TP_HAR_dist.pdf(value) for value in TP_HAR_val]

TP_ERA5_mean = TP_ERA5_lt.lifetime.mean(); TP_ERA5_std = TP_ERA5_lt.lifetime.std(); TP_ERA5_dist = norm(TP_ERA5_mean, TP_ERA5_std)
TP_ERA5_val = [value for value in np.unique(TP_ERA5_lt.lifetime)]; TP_ERA5_prob = [TP_ERA5_dist.pdf(value) for value in TP_ERA5_val]

TP_GPM_mean = TP_GPM_lt.lifetime.mean(); TP_GPM_std = TP_GPM_lt.lifetime.std(); TP_GPM_dist = norm(TP_GPM_mean, TP_GPM_std)
TP_GPM_val = [value for value in np.unique(TP_GPM_lt.lifetime)]; TP_GPM_prob = [TP_GPM_dist.pdf(value) for value in TP_GPM_val]

fig, axes = plt.subplots(2,2)
axes[0,0].hist(TP_WRF_lt.lifetime, bins=np.unique(TP_WRF_lt.lifetime)-0.5, density=True); axes[0,0].plot(TP_WRF_val, TP_WRF_prob); axes[0,0].set_title('a)'); 
axes[0,0].set_xlim(0,30); axes[0,0].set_ylabel('Probability'); axes[0,0].set_xticks(np.arange(0,30,1)); axes[0,0].set_xlim([-1, 30])

axes[0,1].hist(TP_HAR_lt.lifetime, bins=np.unique(TP_HAR_lt.lifetime)-0.5, density=True); axes[0,1].plot(TP_HAR_val, TP_HAR_prob); axes[0,1].set_title('b)'); 
axes[0,1].set_xlim(0,30); axes[0,1].set_xticks(np.arange(0,30,1)); axes[0,1].set_xlim([-1, 30])

axes[1,0].hist(TP_ERA5_lt.lifetime, bins=np.unique(TP_ERA5_lt.lifetime)-0.5, density=True); axes[1,0].plot(TP_ERA5_val, TP_ERA5_prob); axes[1,0].set_title('c)'); 
axes[1,0].set_xlim(0,30); axes[1,0].set_xlabel('Hours'); axes[1,0].set_ylabel('Probability'); axes[1,0].set_xticks(np.arange(0,30,1)); axes[1,0].set_xlim([-1, 30])

axes[1,1].hist(TP_GPM_lt.lifetime, bins=np.unique(TP_GPM_lt.lifetime)-0.5, density=True); axes[1,1].plot(TP_GPM_val, TP_GPM_prob); axes[1,1].set_title('d)'); 
axes[1,1].set_xlim(0,30); axes[1,1].set_xlabel('Hours'); axes[1,1].set_xticks(np.arange(0,30,1)); axes[1,1].set_xlim([-1, 30])

txt = 'Probability density function of precipitation cell lifetime over the TP only for WRF (a), HAR (b), ERA5 (c) and GPM (d)'
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/PDF_lifetime_TP.png', dpi=100)

#%% 
del fig, axes,TP_WRF_dist,TP_WRF_mean,TP_WRF_prob,TP_WRF_std,TP_WRF_val, TP_HAR_dist, TP_HAR_mean, TP_HAR_prob, TP_HAR_std, TP_HAR_val
del TP_ERA5_dist, TP_ERA5_mean, TP_ERA5_prob, TP_ERA5_std, TP_ERA5_val, TP_GPM_dist, TP_GPM_mean, TP_GPM_prob, TP_GPM_std, TP_GPM_val, txt

#%% 
"""
Correlations

"""
## Pearson
WRF_pcorr = WRF_lt.corr()
HAR_pcorr = HAR_lt.corr()
ERA5_pcorr = ERA5_lt.corr()
GPM_pcorr = GPM_lt.corr()

#%% Check Pvalue
def pearsonr_pval(x,y):
        return pearsonr(x,y)[1]

WRF_ppval = WRF_lt.corr(method=pearsonr_pval)
HAR_ppval = HAR_lt.corr(method=pearsonr_pval)
ERA5_ppval = ERA5_lt.corr(method=pearsonr_pval)
GPM_ppval = GPM_lt.corr(method=pearsonr_pval)

#%% Correlation Spearman, since TH (and lifetime?) is ordinal?
WRF_scorr, WRF_spval = spearmanr(WRF_lt)
HAR_scorr, HAR_spval = spearmanr(HAR_lt)
ERA5_scorr, ERA5_spval = spearmanr(ERA5_lt)
GPM_scorr, GPM_spval = spearmanr(GPM_lt)

TP_WRF_scorr, TP_WRF_spval = spearmanr(TP_WRF_lt)
TP_HAR_scorr, TP_HAR_spval = spearmanr(TP_HAR_lt)
TP_ERA5_scorr, TP_ERA5_spval = spearmanr(TP_ERA5_lt)
TP_GPM_scorr, TP_GPM_spval = spearmanr(TP_GPM_lt)

del WRF_lt, HAR_lt, ERA5_lt, GPM_lt, TP_WRF_lt, TP_HAR_lt, TP_ERA5_lt, TP_GPM_lt

#%% Plots
## Entire study area

fig, axes = plt.subplots(2,2)
sns.heatmap(WRF_scorr, annot=True, ax=axes[0,0], xticklabels='', yticklabels=['km2', 'TH', 'LT'], vmin=0, vmax=1, linewidth=1, linecolor='w', square=True, cbar=False)
axes[0,0].set_title('a)')

sns.heatmap(HAR_scorr, annot=True, ax=axes[0,1], xticklabels='', yticklabels=['km2', 'TH', 'LT'], vmin=0, vmax=1, linewidth=1, linecolor='w', square=True, cbar=False)
axes[0,1].set_title('b)')

sns.heatmap(ERA5_scorr, annot=True, xticklabels=['km2', 'TH', 'LT'], ax=axes[1,0], yticklabels=['km2', 'TH', 'LT'], vmin=0, vmax=1, linewidth=1, linecolor='w', square=True, cbar=False)
axes[1,0].set_title('c)')

sns.heatmap(GPM_scorr, annot=True, xticklabels=['km2', 'TH', 'LT'], ax=axes[1,1], yticklabels=['km2', 'TH', 'LT'], vmin=0, vmax=1, linewidth=1, linecolor='w', square=True, cbar=False)
axes[1,1].set_title('d)')
fig.subplots_adjust(wspace=-0.6)

cbar_ax = fig.add_axes([0.72, 0.112, 0.03, 0.764])
cmap = sns.cm.rocket
norm = mpl.colors.Normalize(vmin=0,vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, cax=cbar_ax).set_label('R')
txt = 'Spearman correlation of cell size (km2), precipitation threshold value (TH) and cell lifetime (LT)\nin the entire study area based on WRF (a), HAR (b), ERA5 (c) and GPM (d)'
plt.figtext(0.315, 0.03, txt, horizontalalignment='left')
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/SpearCorr_Lifet_TH_km2.png', dpi=100)

#%%
## TP only
fig, axes = plt.subplots(2,2)
sns.heatmap(TP_WRF_scorr, annot=True, ax=axes[0,0], xticklabels='', yticklabels=['km2', 'TH', 'LT'], vmin=0, vmax=1, linewidth=1, linecolor='w', square=True, cbar=False)
axes[0,0].set_title('a)')

sns.heatmap(TP_HAR_scorr, annot=True, ax=axes[0,1], xticklabels='', yticklabels=['km2', 'TH', 'LT'], vmin=0, vmax=1, linewidth=1, linecolor='w', square=True, cbar=False)
axes[0,1].set_title('b)')

sns.heatmap(TP_ERA5_scorr, annot=True, xticklabels=['km2', 'TH', 'LT'], ax=axes[1,0], yticklabels=['km2', 'TH', 'LT'], vmin=0, vmax=1, linewidth=1, linecolor='w', square=True, cbar=False)
axes[1,0].set_title('c)')

sns.heatmap(TP_GPM_scorr, annot=True, xticklabels=['km2', 'TH', 'LT'], ax=axes[1,1], yticklabels=['km2', 'TH', 'LT'], vmin=0, vmax=1, linewidth=1, linecolor='w', square=True, cbar=False)
axes[1,1].set_title('d)')
fig.subplots_adjust(wspace=-0.6)

cbar_ax = fig.add_axes([0.72, 0.112, 0.03, 0.764])
cmap = sns.cm.rocket
norm = mpl.colors.Normalize(vmin=0,vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, cax=cbar_ax).set_label('R')
plt.colorbar(sm, cax=cbar_ax).set_label('R')
txt = 'Spearman correlation of cell size (km2), precipitation threshold value (TH) and cell lifetime (LT)\nover the TP only based on WRF (a), HAR (b), ERA5 (c) and GPM (d)'
plt.figtext(0.315, 0.03, txt, horizontalalignment='left')
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/SpearCorr_Lifet_TH_km2_TP.png', dpi=100)