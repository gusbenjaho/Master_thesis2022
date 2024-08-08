# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 10:16:53 2022

@author: benja
"""

from tabulate import tabulate
import matplotlib.pyplot as plt

#%%
table = [['Dataset', 'TP region features', 'TP features (unfiltered)', 'TP features (filtered', 'Diff features'],
         ['WRF_GU', '21170', '9256','8537','719 (7.8%)'], ['HAR', '46455', '9619', '8638', '981 (10.2%)'],
         ['ERA5', '45513', '1283', '815', '468 (36.5%)'], ['GPM', '49806', '3834', '2903', '913 (24.3%)']]

table = tabulate(table, headers='firstrow', tablefmt='fancy_grid')
plt.savefig('D:/Benjamin/MasterThesis2021-2022/data/plots/table.png')