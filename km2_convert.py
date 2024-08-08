# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 09:41:23 2021

@author: benja
"""
from numpy import meshgrid, deg2rad, gradient, cos
from xarray import DataArray



lon = ERA5_tracks.longitude.values
lat = ERA5_tracks.latitude.values
xlon, ylat = meshgrid(lon,lat)
R = 6371
dlat = deg2rad(gradient(lat))
dlon = deg2rad(gradient(lon))
dy = dlat * R
dx = dlon * R * cos(dlat)
area = dy * dx



deltalon=0.25
area3 = []
for idx, val in enumerate(np.arange(26,45,1)):  
    area_test=(pi/180)*R**2*((math.sin(lat[idx] - math.sin(lat[idx+1])*deltalon)))
    area3.append(area_test)

## Från David and 
## https://towardsdatascience.com/the-correct-way-to-average-the-globe-92ceecd172b7 & https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
lat_weight2 = [] ## weighting latitude
for lat in ERA5_tracks.latitude:
    weights = math.cos(deg2rad(lat))
    lat_weight2.append(weights)

## Converting the 31km ERA5 grid size at the equator to latitude dependent by multiplying 312 by latitude radian (see above)
weighted_gridsize = []
grid_size = 31
for rad in lat_weight:
    weighted_lon = rad * grid_size
    weighted_gridsize.append(weighted_lon)



dlon = deg2rad(lon) ## from grad to radians
dlat = deg2rad(lat)
dy = dlat *R ## radians times earth radius
dx = dlon * R * lat_weight
area = (dy * dx) / 1000  ## km2

dlat2 = deg2rad(gradient(lat))
dlon2 = deg2rad(gradient(lon))
dy2 = dlat * R * math.pi/180
dx2 = (dlon/180) * math.pi * R * lat_weight
area2 = (dy2*dx2)/1000

## From   https://www.pmel.noaa.gov/maillists/tmap/ferret_users/fu_2004/msg00023.html       
h = R*(1-math.sin(ERA_meanlat))
area = (pi/180)*R**2(math.sin)