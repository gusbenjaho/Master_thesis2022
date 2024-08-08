# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from datetime import datetime

#%%

"""
Note: for the TP only region, I have used the files called TP_tracks_filtered_x2 instead of TP_tracks_ws
This since I want to have tracks already filtered out above three hours for the entire study area
to have an accurate estimation of how many cells that are actually lost when subsetting TP only region
compared to the entire study area. If I dont do this, the calculations for cells lost after subsetting will
be significantly higher because then cells from outside the study area will be counted as well.

"""


WRF_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/WRF_tracks.pkl')
HAR_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/HAR_tracks.pkl')
GPM_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/GPM_tracks.pkl')
ERA5_tracks = pd.read_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/ERA5_tracks.pkl')
GPM_tracks['time'] = pd.to_datetime(GPM_tracks.timestr)

#%%

def get_lifetime_cell(preciptracks, start, end):
    FMT = "%Y-%m-%d %H:%M:%S"
    tracks_lt = pd.DataFrame()
    lifetime = []
    cell_id =[]
    year = []
    for y in np.arange(start, end):
        ytracks = preciptracks[preciptracks.time.dt.year == y]
        for cell in np.unique(ytracks.cell.values):
            tdelta = datetime.strptime(ytracks[ytracks.cell == cell].timestr.iloc[-1], FMT) - datetime.strptime(ytracks[ytracks.cell == cell].timestr.iloc[0], FMT)
            days, seconds = tdelta.days, tdelta.seconds
            hours = (days*24) + (seconds/3600)
            lifetime.append(hours)
            cell_id.append(cell)
            year.append(y)
    tracks_lt['lifetime'] = pd.DataFrame(lifetime)
    tracks_lt['cell'] = pd.DataFrame(cell_id)
    tracks_lt['year'] = pd.DataFrame(year)
    return tracks_lt

#%%
WRF = get_lifetime_cell(WRF_tracks, 2001, 2017); WRF = WRF[WRF.lifetime >= 3]
HAR = get_lifetime_cell(HAR_tracks, 2001, 2017); HAR = HAR[HAR.lifetime >= 3]
ERA5 = get_lifetime_cell(ERA5_tracks, 2001, 2017); ERA5 = ERA5[ERA5.lifetime >= 3]
GPM = get_lifetime_cell(GPM_tracks, 2001, 2017); GPM = GPM[GPM.lifetime >= 3]

#%%
def tracks_below_three(tracks, tracks_lt, start, end):    
    filtered = []
    for y in np.arange(start, end):
        ytracks = tracks[tracks.time.dt.year == y]
        for cell in tracks_lt[tracks_lt.year == y].cell:
            cell_track = ytracks[ytracks.cell == cell]
            filtered.append(cell_track)
    tracks_filtered = pd.concat(filtered)
    return tracks_filtered

#%%
WRF_filtered = tracks_below_three(WRF_tracks, WRF, 2001, 2017)
HAR_filtered = tracks_below_three(HAR_tracks, HAR, 2001, 2017)
ERA5_filtered = tracks_below_three(ERA5_tracks, ERA5, 2001, 2017)
GPM_filtered = tracks_below_three(GPM_tracks, GPM, 2001, 2017)

#%%
WRF_filtered.to_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/WRF_tracks_filtered.pkl')
HAR_filtered.to_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/HAR_tracks_filtered.pkl')
GPM_filtered.to_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/GPM_tracks_filtered.pkl')
ERA5_filtered.to_pickle('D:/Benjamin/MasterThesis2021-2022/data/Python_files/ERA5_tracks_filtered.pkl')

#%%
WRF_lost = (len(WRF_tracks) - len(WRF_filtered)) / len(WRF_tracks) *100
HAR_lost = (len(HAR_tracks) - len(HAR_filtered)) / len(HAR_tracks) *100
ERA5_lost = (len(ERA5_tracks) - len(ERA5_filtered)) / len(ERA5_tracks) *100
GPM_lost = (len(GPM_tracks) - len(GPM_filtered)) / len(GPM_tracks) *100

del WRF, HAR, ERA5, GPM, WRF_tracks, HAR_tracks, ERA5_tracks, GPM_tracks