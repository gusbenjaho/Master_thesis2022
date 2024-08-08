# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Checking HAR and ERA5 of their 1 and 19 hour peaks
import pandas as pd
WRF_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/WRF_tracks.pkl')
HAR_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/HAR_tracks.pkl')
GPM_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/GPM_tracks.pkl')
ERA5_tracks = pd.read_pickle('C:/Users/benja/Documents/Skola/MasterThesis2021-2022/data/Python_files/ERA5_tracks.pkl')

HAR01 = HAR_tracks[HAR_tracks.time.dt.year == 2015]
HAR01['Hour'] = HAR01.time.dt.hour
HAR01 = HAR01.drop_duplicates(subset='cell')
HAR01 = HAR01.cell.groupby([HAR01.time.dt.hour]).count().to_frame()

ERA01 = ERA5_tracks[ERA5_tracks.time.dt.year == 2011]
ERA01 = ERA01.drop_duplicates(subset='cell')
ERA01 = ERA01.cell.groupby([ERA01.time.dt.hour]).count().to_frame()

WRF = WRF_tracks[WRF_tracks.time.dt.year == 2011]
WRF = WRF.drop_duplicates(subset='cell')
WRF = WRF.cell.groupby([WRF.time.dt.hour]).count().to_frame()

HAR02 = HAR_tracks[HAR_tracks.time.dt.year == 2002]
HAR02 = HAR02.cell.groupby([HAR02.time.dt.hour]).count().to_frame()

HAR03 = HAR_tracks[HAR_tracks.time.dt.year == 2002]
HAR03 = HAR03.cell.groupby([HAR02.time.dt.hour]).count().to_frame()