# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 13:13:36 2022

@author: gusbenjaho
"""

import pandas as pd
import numpy as np

#%%
WRF_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/WRF_filtered.pkl')
HAR_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/HAR_filtered.pkl')
GPM_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/ERA5_filtered.pkl')
ERA5_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/GPM_filtered.pkl')
GPM_tracks['time'] = pd.to_datetime(GPM_tracks.timestr)



#%% Get number of cells

def get_cell_count(preciptracks, start, end):
    preciptracks['hour'] = preciptracks.time.dt.hour
    count = 0
    for y in np.arange(start, end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        ycells = np.unique(ytracks.cell.values)
        count +=  len(ycells)
    return count

#%%

WRF_count = get_cell_count(WRF_tracks, 2001, 2017)
HAR_count = get_cell_count(HAR_tracks, 2001, 2017)
ERA5_count = get_cell_count(ERA5_tracks, 2001, 2017)
GPM_count = get_cell_count(GPM_tracks, 2001, 2017)
del WRF_tracks, HAR_tracks, ERA5_tracks, GPM_tracks

#%%

WRF_unfiltered = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_WRF_filtered_x2.pkl')
HAR_unfiltered = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_HAR_filtered_x2.pkl')
GPM_unfiltered = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_GPM_filtered_x2.pkl')
ERA5_unfiltered = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_ERA5_filtered_x2.pkl')
GPM_unfiltered['time'] = pd.to_datetime(GPM_unfiltered.timestr)

WRF_filtered = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_WRF_filtered.pkl')
HAR_filtered = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_HAR_filtered.pkl')
ERA5_filtered = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_ERA5_filtered.pkl')
GPM_filtered = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/TP_GPM_filtered.pkl')
GPM_filtered['time'] = pd.to_datetime(GPM_filtered.timestr)

#%%
WRF_unfiltered = get_cell_count(WRF_unfiltered, 2001, 2017); WRF_filtered = get_cell_count(WRF_filtered, 2001, 2017)
HAR_unfiltered = get_cell_count(HAR_unfiltered, 2001, 2017); HAR_filtered = get_cell_count(HAR_filtered, 2001, 2017)
ERA5_unfiltered = get_cell_count(ERA5_unfiltered, 2001, 2017); ERA5_filtered = get_cell_count(ERA5_filtered, 2001, 2017)
GPM_unfiltered = get_cell_count(GPM_unfiltered, 2001, 2017); GPM_filtered = get_cell_count(GPM_filtered, 2001, 2017)

#%%
WRF_lost = (WRF_unfiltered - WRF_filtered) / WRF_unfiltered *100
HAR_lost = (HAR_unfiltered - HAR_filtered) / HAR_unfiltered *100
ERA5_lost = (ERA5_unfiltered - ERA5_filtered) / ERA5_unfiltered *100
GPM_lost = (GPM_unfiltered - GPM_filtered) / GPM_unfiltered *100

del WRF_filtered, WRF_unfiltered, HAR_filtered, HAR_unfiltered, ERA5_filtered, ERA5_unfiltered, GPM_filtered, GPM_unfiltered
