# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 10:35:49 2021

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

#%% 
""" 
DC of features of TP only (>3000)

"""
WRF_TP = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_WRF_filtered.pkl')
HAR_TP = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_HAR_filtered.pkl')
ERA5_TP = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_ERA5_filtered.pkl')
GPM_TP = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_GPM_filtered.pkl')

#%%

#%%
"""
Inititation stage
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
WRFinit = get_ncells_init2(WRF_tracks,2001,2017)
HARinit = get_ncells_init2(HAR_tracks,2001,2017)
ERA5init = get_ncells_init2(ERA5_tracks,2001,2017)
GPMinit = get_ncells_init2(GPM_tracks,2001,2017)

#%%
bins = np.arange(10000,510000,10000)
xticks = np.arange(0,510000,50000)
xlim = 0,500000
ylim = 0,0.000025
from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
plt.style.use('seaborn')
fig, axes = plt.subplots(2,2)
axes[0,0].hist(WRFinit, bins=bins, density=True, align='left', rwidth = 0.8, color='#1E90FF'); axes[0,0].set_xticks(xticks); axes[0,0].set_xlim(xlim)
axes[0,0].set_title('a)'); axes[0,0].set_ylabel('Density'); axes[0,0].set_ylim(ylim); axes[0,0].grid(True); axes[0,0].xaxis.set_major_formatter(formatter)
axes[0,1].hist(HARinit, bins=bins, density=True, align='left', rwidth = 0.8, color='#104E8B'); axes[0,1].set_xticks(xticks); axes[0,1].set_xlim(xlim);
axes[0,1].set_title('b)'); axes[0,1].set_ylim(ylim); axes[0,1].grid(True); axes[0,1].xaxis.set_major_formatter(formatter)
axes[1,0].hist(ERA5init, bins=bins, density=True, align='left', rwidth = 0.8, color='#FF6103'); axes[1,0].set_xticks(xticks); axes[1,0].set_xlim(xlim); axes[1,0].xaxis.set_major_formatter(formatter)
axes[1,0].set_title('c)'); axes[1,0].set_ylabel('Density'); axes[1,0].set_xlabel('Size [$km^2$]'); axes[1,0].set_ylim(ylim); axes[1,0].grid(True)
axes[1,1].hist(GPMinit, bins=bins, density=True, align='left', rwidth = 0.8, color='#FF9912'); axes[1,1].set_xticks(xticks); axes[1,1].set_xlim(xlim);
axes[1,1].set_title('d)'); axes[1,1].set_xlabel('Size [$km^2$]'); axes[1,1].set_ylim(ylim); axes[1,1].grid(True); axes[1,1].xaxis.set_major_formatter(formatter)

txt = 'Density of precipitation cell size at initiation stage in entire study area for WRF (a), HAR (b), ERA5 (c) and GPM (d)'
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
fig.set_size_inches(16, 10)
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/PDF_km2_init_filtered_color.png', dpi=100)
del axes, bins, fig, xlim, xticks, ylim, txt
#%% TP only
TP_WRFinit = get_ncells_init2(WRF_TP,2001,2017)
TP_HARinit = get_ncells_init2(HAR_TP,2001,2017)
TP_ERA5init = get_ncells_init2(ERA5_TP,2001,2017)
TP_GPMinit = get_ncells_init2(GPM_TP,2001,2017)

#%%
bins = np.arange(10000,500000,50000)
xticks = np.arange(10000,510000,50000)
xlim = 10000,510000
ylim = 0,0.000025
fig, axes = plt.subplots(2,2)
axes[0,0].hist(TP_WRFinit, bins=bins, density=True, rwidth = 0.8); axes[0,0].set_xticks(xticks); axes[0,0].set_xlim(xlim);
axes[0,0].set_title('a)'); axes[0,0].set_ylabel('Fraction'); axes[0,0].set_ylim(ylim)
axes[0,1].hist(TP_HARinit, bins=bins, density=True, rwidth = 0.8); axes[0,1].set_xticks(xticks); axes[0,1].set_xlim(xlim);
axes[0,1].set_title('b)'); axes[0,1].set_ylim(ylim)
axes[1,0].hist(TP_ERA5init, bins=bins, density=True, rwidth = 0.8); axes[1,0].set_xticks(xticks); axes[1,0].set_xlim(xlim);
axes[1,0].set_title('c)'); axes[1,0].set_ylabel('Fraction'); axes[1,0].set_xlabel('Size [$km^2$]'); axes[1,0].set_ylim(ylim)
axes[1,1].hist(TP_GPMinit, bins=bins, density=True, rwidth = 0.8); axes[1,1].set_xticks(xticks); axes[1,1].set_xlim(xlim);
axes[1,1].set_title('d)'); axes[1,1].set_xlabel('Size [$km^2$]'); axes[1,1].set_ylim(ylim)
plt.style.use('seaborn')
txt = 'Fraction of precipitation cell size at initiation stage on the TP only for WRF (a), HAR (b), ERA5 (c) and GPM (d)'
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
fig.set_size_inches(16, 10)
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/PDF_km2_init_TP_filtered.png', dpi=100)
del axes, bins, fig, xlim, xticks, ylim, txt
#%%
del WRFinit, HARinit, ERA5init, GPMinit, TP_WRFinit, TP_HARinit, TP_ERA5init, TP_GPMinit

#%%
"""
NCELLS at dissipation stage
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
WRFdiss = get_ncells_diss2(WRF_tracks,2001,2017)
HARdiss = get_ncells_diss2(HAR_tracks,2001,2017)
ERA5diss = get_ncells_diss2(ERA5_tracks,2001,2017)
GPMdiss = get_ncells_diss2(GPM_tracks,2001,2017)

#%%
fig, axes = plt.subplots(2,2)
axes[0,0].hist(WRFdiss, bins=bins, density=True, rwidth = 0.8); axes[0,0].set_xticks(xticks); axes[0,0].set_xlim(xlim);
axes[0,0].set_title('a)'); axes[0,0].set_ylabel('Fraction'); axes[0,0].set_ylim(ylim)
axes[0,1].hist(HARdiss, bins=bins, density=True, rwidth = 0.8); axes[0,1].set_xticks(xticks); axes[0,1].set_xlim(xlim);
axes[0,1].set_title('b)'); axes[0,1].set_ylim(ylim)
axes[1,0].hist(ERA5diss, bins=bins, density=True, rwidth = 0.8); axes[1,0].set_xticks(xticks); axes[1,0].set_xlim(xlim);
axes[1,0].set_title('c)'); axes[1,0].set_ylabel('Fraction'); axes[1,0].set_xlabel('Size [$km^2$]'); axes[1,0].set_ylim(ylim)
axes[1,1].hist(GPMdiss, bins=bins, density=True, rwidth = 0.8); axes[1,1].set_xticks(xticks); axes[1,1].set_xlim(xlim);
axes[1,1].set_title('d)'); axes[1,1].set_xlabel('Size [$km^2$]'); axes[1,1].set_ylim(ylim)
plt.style.use('seaborn')
txt = 'Fraction of precipitation cell size at dissipation stage in the entire study area for WRF (a), HAR (b), ERA5 (c) and GPM (d)'
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
fig.set_size_inches(16, 10)
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/PDF_km2_diss_filtered.png', dpi=100)

#%% TP only
TP_WRFdiss = get_ncells_diss2(WRF_TP,2001,2017)
TP_HARdiss = get_ncells_diss2(HAR_TP,2001,2017)
TP_ERA5diss = get_ncells_diss2(ERA5_TP,2001,2017)
TP_GPMdiss = get_ncells_diss2(GPM_TP,2001,2017)

#%%
fig, axes = plt.subplots(2,2)
axes[0,0].hist(TP_WRFdiss, bins=bins, density=True, rwidth = 0.8); axes[0,0].set_xticks(xticks); axes[0,0].set_xlim(xlim);
axes[0,0].set_title('a)'); axes[0,0].set_ylabel('Fraction'); axes[0,0].set_ylim(0,0.000027)
axes[0,1].hist(TP_HARdiss, bins=bins, density=True, rwidth = 0.8); axes[0,1].set_xticks(xticks); axes[0,1].set_xlim(xlim);
axes[0,1].set_title('b)'); axes[0,1].set_ylim(0,0.000027)
axes[1,0].hist(TP_ERA5diss, bins=bins, density=True, rwidth = 0.8); axes[1,0].set_xticks(xticks); axes[1,0].set_xlim(xlim);
axes[1,0].set_title('c)'); axes[1,0].set_ylabel('Fraction'); axes[1,0].set_xlabel('Size [$km^2$]'); axes[1,0].set_ylim(0,0.000027)
axes[1,1].hist(TP_GPMdiss, bins=bins, density=True, rwidth = 0.8); axes[1,1].set_xticks(xticks); axes[1,1].set_xlim(xlim);
axes[1,1].set_title('d)'); axes[1,1].set_xlabel('Size [$km^2$]'); axes[1,1].set_ylim(0,0.000027)
plt.style.use('seaborn')
txt = 'Fraction of precipitation cell size at dissipation stage on the TP only for WRF (a), HAR (b), ERA5 (c) and GPM (d)'
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
fig.set_size_inches(16, 10)
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/PDF_km2_diss_TP_filtered.png', dpi=100)

#%% 
del WRFdiss, HARdiss, ERA5diss, GPMdiss, fig, axes, txt, TP_WRFdiss, TP_HARdiss, TP_ERA5diss, TP_GPMdiss

#%%
"""
Maximum extension stage
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
WRFmax = get_ncells_max2(WRF_tracks,2001,2017)
HARmax = get_ncells_max2(HAR_tracks,2001,2017)
ERA5max = get_ncells_max2(ERA5_tracks,2001,2017)
GPMmax = get_ncells_max2(GPM_tracks,2001,2017)

#%%
bins = np.arange(10000,510000,10000)
xticks = np.arange(0,510000,50000)
xlim = 0,500000
ylim = 0,0.000025
plt.style.use('default')
fig, axes = plt.subplots(2,2)
axes[0,0].hist(WRFmax, bins=bins, density=True, align='left', rwidth = 0.8); axes[0,0].set_xticks(xticks); axes[0,0].set_xlim(xlim); axes[0,0].grid()
axes[0,0].set_title('a)'); axes[0,0].set_ylabel('Density'); axes[0,0].set_ylim(0,0.000020); axes[0,0].xaxis.set_major_formatter(formatter)
axes[0,1].hist(HARmax, bins=bins, density=True, align='left', rwidth = 0.8); axes[0,1].set_xticks(xticks); axes[0,1].set_xlim(xlim);
axes[0,1].set_title('b)'); axes[0,1].set_ylim(0,0.000020); axes[0,1].xaxis.set_major_formatter(formatter); axes[0,1].grid()
axes[1,0].hist(ERA5max, bins=bins, density=True, align='left', rwidth = 0.8); axes[1,0].set_xticks(xticks); axes[1,0].set_xlim(xlim); axes[1,0].xaxis.set_major_formatter(formatter)
axes[1,0].set_title('c)'); axes[1,0].set_ylabel('Density'); axes[1,0].set_xlabel('Size [$km^2$]'); axes[1,0].set_ylim(0,0.000020); axes[1,0].grid()
axes[1,1].hist(GPMmax, bins=bins, density=True, align='left', rwidth = 0.8); axes[1,1].set_xticks(xticks); axes[1,1].set_xlim(xlim); axes[1,1].xaxis.set_major_formatter(formatter)
axes[1,1].set_title('d)'); axes[1,1].set_xlabel('Size [$km^2$]'); axes[1,1].set_ylim(0,0.000020); axes[1,1].grid()
txt = 'Density of precipitation cell size at maximum extension stage in the entire study area for WRF (a), HAR (b), ERA5 (c) and GPM (d)'
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
fig.set_size_inches(16, 10)
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/PDF_km2_max_filtered.png', dpi=100)
del axes, bins, fig, xlim, xticks, ylim, txt

#%% TP only 
TP_WRFmax = get_ncells_max2(WRF_TP,2001,2017)
TP_HARmax = get_ncells_max2(HAR_TP,2001,2017)
TP_ERA5max = get_ncells_max2(ERA5_TP,2001,2017)
TP_GPMmax = get_ncells_max2(GPM_TP,2001,2017)

#%%
fig, axes = plt.subplots(2,2)
axes[0,0].hist(TP_WRFmax, bins=bins, density=True, rwidth = 0.8); axes[0,0].set_xticks(xticks); axes[0,0].set_xlim(xlim);
axes[0,0].set_title('a)'); axes[0,0].set_ylabel('Fraction'); axes[0,0].set_ylim(0,0.000020)
axes[0,1].hist(TP_HARmax, bins=bins, density=True, rwidth = 0.8); axes[0,1].set_xticks(xticks); axes[0,1].set_xlim(xlim);
axes[0,1].set_title('b)'); axes[0,1].set_ylim(0,0.000015)
axes[1,0].hist(TP_ERA5max, bins=bins, density=True, rwidth = 0.8); axes[1,0].set_xticks(xticks); axes[1,0].set_xlim(xlim);
axes[1,0].set_title('c)'); axes[1,0].set_ylabel('Fraction'); axes[1,0].set_xlabel('Size [$km^2$]'); axes[1,0].set_ylim(0,0.000015)
axes[1,1].hist(TP_GPMmax, bins=bins, density=True, rwidth = 0.8); axes[1,1].set_xticks(xticks); axes[1,1].set_xlim(xlim);
axes[1,1].set_title('d)'); axes[1,1].set_xlabel('Size [$km^2$]'); axes[1,1].set_ylim(0,0.000015)
plt.style.use('seaborn')
txt = 'Fraction of precipitation cell size at maximum extension stage on the TP only for WRF (a), HAR (b), ERA5 (c) and GPM (d)'
plt.figtext(0.12, 0.03, txt, horizontalalignment='left')
fig.set_size_inches(16, 10)
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/PDF_km2_max_TP_filtered.png', dpi=100)

#%% 
del WRFmax, HARmax, ERA5max, GPMmax, fig, axes, txt, TP_WRFmax, TP_HARmax, TP_ERA5max, TP_GPMmax