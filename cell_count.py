# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 13:53:26 2021

@author: benja
"""
""" 

Count / hour and month

Read in tracks file

"""
import seaborn as sns
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
# Daily count


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


#%% Called it with:
# WRF
diurnal_precip, bins = get_diurnal_init(WRF_tracks, 2001, 2016)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_DC.npy', diurnal_precip)
WRF_DC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_DC.npy')
del diurnal_precip; del bins

#%% HAR
diurnal_precip, bins = get_diurnal_init(HAR_tracks, 2001, 2016)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_DC.npy', diurnal_precip)
HAR_DC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_DC.npy')
del diurnal_precip; del bins

#%% ERA5
diurnal_precip, bins = get_diurnal_init(ERA5_tracks, 2001, 2016)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_DC.npy', diurnal_precip)
ERA5_DC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_DC.npy')
del diurnal_precip; del bins

#%% GPM
GPM_tracks['time'] = pd.to_datetime(GPM_tracks.timestr)
diurnal_precip, bins = get_diurnal_init(GPM_tracks, 2001, 2016)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_DC.npy', diurnal_precip)
GPM_DC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_DC.npy')
del diurnal_precip; del bins

#%% 
"""
Summer months

"""

WRF_JJA = WRF_tracks[(WRF_tracks.time.dt.month == 6) | (WRF_tracks.time.dt.month == 7) | (WRF_tracks.time.dt.month == 8)]
HAR_JJA = HAR_tracks[(HAR_tracks.time.dt.month == 6) | (HAR_tracks.time.dt.month == 7) | (HAR_tracks.time.dt.month == 8)]
ERA5_JJA = ERA5_tracks[(ERA5_tracks.time.dt.month == 6) | (ERA5_tracks.time.dt.month == 7) | (ERA5_tracks.time.dt.month == 8)]
GPM_JJA = GPM_tracks[(GPM_tracks.time.dt.month == 6) | (GPM_tracks.time.dt.month == 7) | (GPM_tracks.time.dt.month == 8)]

#%% WRF_JJA
diurnal_precip, bins = get_diurnal_init(WRF_JJA, 2001, 2016)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRFJJA_DC.npy', diurnal_precip)
WRFJJA_DC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRFJJA_DC.npy')
del diurnal_precip; del bins

#%% HAR_JJA
diurnal_precip, bins = get_diurnal_init(HAR_JJA, 2001, 2016)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HARJJA_DC.npy', diurnal_precip)
HARJJA_DC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HARJJA_DC.npy')
del diurnal_precip; del bins

#%% ERA5_JJA
diurnal_precip, bins = get_diurnal_init(ERA5_JJA, 2001, 2016)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5JJA_DC.npy', diurnal_precip)
ERA5JJA_DC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5JJA_DC.npy')
del diurnal_precip; del bins

#%% GPM_JJA
diurnal_precip, bins = get_diurnal_init(GPM_JJA, 2001, 2016)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPMJJA_DC.npy', diurnal_precip)
GPMJJA_DC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPMJJA_DC.npy')
del diurnal_precip; del bins

#%%
"""
#%%
# Daily count files from above

"""
WRF_DC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_DC.npy')
HAR_DC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_DC.npy')
ERA5_DC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_DC.npy')
GPM_DC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_DC.npy')
#DC_bins = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/DC_bins.npy')

#%% Total count
WRF_DC = pd.DataFrame(WRF_DC)
WRF_DC.insert(0, 'Hour', range(1, 25))
WRF_DC.rename(columns={ 0:'WRF'}, inplace=True)

HAR_DC = pd.DataFrame(HAR_DC)
HAR_DC.rename(columns={0:'HAR'}, inplace=True)

ERA5_DC = pd.DataFrame(ERA5_DC)
ERA5_DC.rename(columns={0:'ERA5'}, inplace=True)

GPM_DC = pd.DataFrame(GPM_DC)
GPM_DC.rename(columns={0:'GPM'}, inplace=True)

DC_tot_col = pd.concat([WRF_DC, HAR_DC, ERA5_DC, GPM_DC],axis=1)
DC_tot = DC_tot_col.melt(id_vars=['Hour'], value_vars=['WRF', 'HAR', 'ERA5', 'GPM'])
DC_tot.rename(columns={'variable':'Model', 'value':'Count'}, inplace=True)

#%% Normalize 
WRF_DC.WRF = WRF_DC.WRF/WRF_DC.WRF.sum() * 100
HAR_DC.HAR = HAR_DC.HAR/HAR_DC.HAR.sum() * 100
ERA5_DC.ERA5 = ERA5_DC.ERA5/ERA5_DC.ERA5.sum() * 100
GPM_DC.GPM = GPM_DC.GPM/GPM_DC.GPM.sum() * 100

DC_col = pd.concat([WRF_DC, HAR_DC, ERA5_DC, GPM_DC],axis=1)
DC = DC_col.melt(id_vars=['Hour'], value_vars=['WRF', 'HAR', 'ERA5', 'GPM'])
DC.rename(columns={'variable':'Model', 'value':'Count'}, inplace=True)

del WRF_DC; del HAR_DC; del ERA5_DC; del GPM_DC

#%% Plotting (%)
sns.barplot(x='Hour', y='Count', hue='Model', data=DC)
plt.ylabel('Count(%)')
plt.grid()
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/DC(%).png')
#%% Plotting (total)
sns.barplot(x='Hour', y='Count', hue='Model', data=DC_tot)
plt.grid()
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/DC(total).png')
#%%
"""
#%%
# JJA Daily counts

"""
WRFJJA_DC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRFJJA_DC.npy')
HARJJA_DC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HARJJA_DC.npy')
ERA5JJA_DC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5JJA_DC.npy')
GPMJJA_DC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPMJJA_DC.npy')

#%% Total count JJA
WRFJJA_DC = pd.DataFrame(WRFJJA_DC)
WRFJJA_DC.insert(0, 'Hour', range(1, 25))
WRFJJA_DC.rename(columns={ 0:'WRF'}, inplace=True)

HARJJA_DC = pd.DataFrame(HARJJA_DC)
HARJJA_DC.rename(columns={0:'HAR'}, inplace=True)

ERA5JJA_DC = pd.DataFrame(ERA5JJA_DC)
ERA5JJA_DC.rename(columns={0:'ERA5'}, inplace=True)

GPMJJA_DC = pd.DataFrame(GPMJJA_DC)
GPMJJA_DC.rename(columns={0:'GPM'}, inplace=True)

DCJJA_tot_col = pd.concat([WRFJJA_DC, HARJJA_DC, ERA5JJA_DC, GPMJJA_DC],axis=1)
DCJJA_tot = DCJJA_tot_col.melt(id_vars=['Hour'], value_vars=['WRF', 'HAR', 'ERA5', 'GPM'])
DCJJA_tot.rename(columns={'variable':'Model', 'value':'Count'}, inplace=True)

#%% Normalize JJA
WRFJJA_DC.WRF = WRFJJA_DC.WRF/WRFJJA_DC.WRF.sum() * 100
HARJJA_DC.HAR = HARJJA_DC.HAR/HARJJA_DC.HAR.sum() * 100
ERA5JJA_DC.ERA5 = ERA5JJA_DC.ERA5/ERA5JJA_DC.ERA5.sum() * 100
GPMJJA_DC.GPM = GPMJJA_DC.GPM/GPMJJA_DC.GPM.sum() * 100

DCJJA_col = pd.concat([WRFJJA_DC, HARJJA_DC, ERA5JJA_DC, GPMJJA_DC],axis=1)
DC_JJA = DCJJA_col.melt(id_vars=['Hour'], value_vars=['WRF', 'HAR', 'ERA5', 'GPM'])
DC_JJA.rename(columns={'variable':'Model', 'value':'Count'}, inplace=True)

del WRFJJA_DC; del HARJJA_DC; del ERA5JJA_DC; del GPMJJA_DC

#%% Plotting JJA (%)
sns.barplot(x='Hour', y='Count', hue='Model', data=DC_JJA)
plt.ylabel('Count(%)')
plt.grid()
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/DC_JJA(%).png', dpi=100)
#%% Plotting JJA (total)
sns.barplot(x='Hour', y='Count', hue='Model', data=DCJJA_tot)
plt.grid()

plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/DC_JJA(total).png', dpi=100)
#%% JJA + All year (%)

plt.subplot(2,1,1)
sns.barplot(x='Hour', y='Count', hue='Model', data=DC)
plt.ylabel('Count(%)')
plt.grid()
plt.title('Annual')
plt.ylim(0,14)

plt.subplot(2,1,2)
sns.barplot(x='Hour', y='Count', hue='Model', data=DC_JJA)
plt.ylabel('Count(%)')
plt.grid()
plt.title('JJA')
plt.ylim(0,14)

fig = plt.gcf()
fig.set_size_inches(12, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/DC_DCJJA(%).png', dpi=100)
#%% JJA + All year (total)

plt.subplot(2,1,1)
sns.barplot(x='Hour', y='Count', hue='Model', data=DC_tot)
plt.grid()
plt.title('Annual')
plt.ylim(0,1450)

plt.subplot(2,1,2)
sns.barplot(x='Hour', y='Count', hue='Model', data=DCJJA_tot)
plt.grid()
plt.title('JJA')
plt.ylim(0,1450)

fig = plt.gcf()
fig.set_size_inches(12, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/DC_DCJJA(total).png', dpi=100)
#%%
"""
# Monthly count

"""

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

#%% Running and saving
# WRF
monthly_precip, bins = get_monthly_init(WRF_tracks, 2001, 2016)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_MC.npy', monthly_precip)
WRF_MC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_MC.npy')
del monthly_precip; del bins

#%% HAR
monthly_precip, bins = get_monthly_init(HAR_tracks, 2001, 2016)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_MC.npy', monthly_precip)
HAR_MC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_MC.npy')
del monthly_precip; del bins

#%% ERA5
monthly_precip, bins = get_monthly_init(ERA5_tracks, 2001, 2016)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_MC.npy', monthly_precip)
ERA5_MC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_MC.npy')
del monthly_precip; del bins

#%% GPM
monthly_precip, bins = get_monthly_init(GPM_tracks, 2001, 2016)
np.save('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_MC.npy', monthly_precip)
GPM_MC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_MC.npy')
del monthly_precip; del bins

#%%
# Monthly count files from above

WRF_MC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_MC.npy')
HAR_MC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_MC.npy')
ERA5_MC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_MC.npy')
GPM_MC = np.load('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_MC.npy')


#%% Total count
WRF_MC = pd.DataFrame(WRF_MC)
WRF_MC.insert(0, 'Month', range(1, 13))
WRF_MC.rename(columns={0:'WRF'}, inplace=True)

HAR_MC = pd.DataFrame(HAR_MC)
HAR_MC.rename(columns={0:'HAR'}, inplace=True)

ERA5_MC = pd.DataFrame(ERA5_MC)
ERA5_MC.rename(columns={0:'ERA5'}, inplace=True)

GPM_MC = pd.DataFrame(GPM_MC)
GPM_MC.rename(columns={0:'GPM'}, inplace=True)

MC_tot_col = pd.concat([WRF_MC, HAR_MC, ERA5_MC, GPM_MC],axis=1)
MC_tot = MC_tot_col.melt(id_vars=['Month'], value_vars=['WRF', 'HAR', 'ERA5', 'GPM'])
MC_tot.rename(columns={'variable':'Model', 'value':'Count'}, inplace=True)
#%%
# Normalize 

WRF_MC.WRF = WRF_MC.WRF/WRF_MC.WRF.sum() * 100
HAR_MC.HAR = HAR_MC.HAR/HAR_MC.HAR.sum() * 100
ERA5_MC.ERA5 = ERA5_MC.ERA5/ERA5_MC.ERA5.sum() * 100
GPM_MC.GPM = GPM_MC.GPM/GPM_MC.GPM.sum() * 100

MC_col = pd.concat([WRF_MC, HAR_MC, ERA5_MC, GPM_MC],axis=1)
MC = pd.concat([WRF_MC, HAR_MC, ERA5_MC, GPM_MC],axis=1)
MC = MC.melt(id_vars=['Month'], value_vars=['WRF', 'HAR', 'ERA5', 'GPM'])
MC.rename(columns={'variable':'Model', 'value':'Count'}, inplace=True)

del WRF_MC; del HAR_MC; del ERA5_MC; del GPM_MC


#%% Plotting (%)
sns.barplot(x='Month', y='Count', hue='Model', data=MC)
plt.ylabel('Count(%)')
plt.grid()
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/MC(%).png', dpi=100)

#%% Plotting (total)
sns.barplot(x='Month', y='Count', hue='Model', data=MC_tot)
plt.grid()
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/MC(total).png', dpi=100)
#%%
"""
SEASONS

"""

DJF = MC_col.loc[[11,0,1],:]; MAM = MC_col.loc[[2,3,4],:]; JJA = MC_col.loc[[5,6,7],:]; SON = MC_col.loc[[8,9,10],:]
DJF_tot = MC_tot_col.loc[[11,0,1],:]; MAM_tot = MC_tot_col.loc[[2,3,4],:]; JJA_tot = MC_tot_col.loc[[5,6,7],:]; SON_tot = MC_tot_col.loc[[8,9,10],:]

DJF = DJF.melt(id_vars=['Month'], value_vars=['WRF', 'HAR', 'ERA5', 'GPM'])
DJF.rename(columns={'variable':'Model', 'value':'Count'}, inplace=True)
DJF_tot = DJF_tot.melt(id_vars=['Month'], value_vars=['WRF', 'HAR', 'ERA5', 'GPM'])
DJF_tot.rename(columns={'variable':'Model', 'value':'Count'}, inplace=True)

MAM = MAM.melt(id_vars=['Month'], value_vars=['WRF', 'HAR', 'ERA5', 'GPM'])
MAM.rename(columns={'variable':'Model', 'value':'Count'}, inplace=True)
MAM_tot = MAM_tot.melt(id_vars=['Month'], value_vars=['WRF', 'HAR', 'ERA5', 'GPM'])
MAM_tot.rename(columns={'variable':'Model', 'value':'Count'}, inplace=True)

JJA = JJA.melt(id_vars=['Month'], value_vars=['WRF', 'HAR', 'ERA5', 'GPM'])
JJA.rename(columns={'variable':'Model', 'value':'Count'}, inplace=True)
JJA_tot = JJA_tot.melt(id_vars=['Month'], value_vars=['WRF', 'HAR', 'ERA5', 'GPM'])
JJA_tot.rename(columns={'variable':'Model', 'value':'Count'}, inplace=True)

SON = SON.melt(id_vars=['Month'], value_vars=['WRF', 'HAR', 'ERA5', 'GPM'])
SON.rename(columns={'variable':'Model', 'value':'Count'}, inplace=True)
SON_tot = SON_tot.melt(id_vars=['Month'], value_vars=['WRF', 'HAR', 'ERA5', 'GPM'])
SON_tot.rename(columns={'variable':'Model', 'value':'Count'}, inplace=True)
#%% Plotting seasons (%)
plt.subplot(2,2,1)
sns.barplot(x='Month', y='Count', hue='Model', data=DJF, order=[12, 1, 2])
plt.ylabel('Count(%)')
plt.grid()
plt.ylim(0,28)
plt.title('DJF')

plt.subplot(2,2,2)
sns.barplot(x='Month', y='Count', hue='Model', data=MAM)
plt.ylabel('Count(%)')
plt.grid()
plt.ylim(0,28)
plt.title('MAM')

plt.subplot(2,2,3)
sns.barplot(x='Month', y='Count', hue='Model', data=JJA)
plt.ylabel('Count(%)')
plt.grid()
plt.ylim(0,28)
plt.title('JJA')

plt.subplot(2,2,4)
sns.barplot(x='Month', y='Count', hue='Model', data=SON)
plt.ylabel('Count(%)')
plt.grid()
plt.ylim(0,28)
plt.title('SON')

fig = plt.gcf()
fig.set_size_inches(12, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/MC_seasons(%).png')

#%% Plotting seasons (total)
plt.subplot(2,2,1)
sns.barplot(x='Month', y='Count', hue='Model', data=DJF_tot, order=[12, 1, 2])
plt.grid()
plt.ylim(0,5000)
plt.title('DJF')

plt.subplot(2,2,2)
sns.barplot(x='Month', y='Count', hue='Model', data=MAM_tot)
plt.grid()
plt.ylim(0,5000)
plt.title('MAM')

plt.subplot(2,2,3)
sns.barplot(x='Month', y='Count', hue='Model', data=JJA_tot)
plt.grid()
plt.ylim(0,5000)
plt.title('JJA')

plt.subplot(2,2,4)
sns.barplot(x='Month', y='Count', hue='Model', data=SON_tot)
plt.grid()
plt.ylim(0,5000)
plt.title('SON')

fig = plt.gcf()
fig.set_size_inches(12, 10)
plt.savefig('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/plots/MC_seasons(total).png', dpi=100)