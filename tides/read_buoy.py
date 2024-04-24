#!/usr/bin/env python 
USAGE = \
"""
Get current buoy speed and direction timeseries in dataframe
This script is specifically for accessing data from a current buoy 
and was created with buoy `cb1501` as a template.
  
Current direction is assumed to be in degrees clockwise from north. 

INPUTS: 
  buoy_id: string with NOAA buoy ID (default = "cb1501")
  start_date: string with start date "YYYYMMDD" (default = "20240401")
  end_date: string with end date "YYYYMMDD" (default ="20240415")

EXAMPLE: 
  from read_buoy import read_buoy
  df = read_buoy(buoy_id="cb1501", start_date="20240401", end_date="20240415")

Date created: 18 Apr 2024
Created by : Rachael D. Mueller
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt 

def read_buoy(buoy_id="cb1501", start_date="20240401", end_date="20240415"):
    # load data
    api = f'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date={start_date}&end_date={end_date}&station={buoy_id}&product=currents&time_zone=gmt&units=english&format=csv'
    r = requests.get(api)
    
    # split string into lines with \n delimiter
    data_list = r.text.splitlines()
    # create initial dataframe (N obs x 1, string entries)
    df = pd.DataFrame(data_list)
    
    # create a more useful dataframe with datetime index and columns with
    # numbers for speed and direction
    buoy = {}
    index_skipfirst = np.arange(len(df[:-1]))+1
    
    # read in date separately to provide as index
    date = [
        datetime.strptime(df[0][row].split(',')[0],'%Y-%m-%d %H:%M') 
        for row in index_skipfirst
    ]
    
    # speed is first column and direction is second
    # the string names are automated here in case this order changes
    # or if the variables are named differently elsewhere
    speed_string = df[0][0].split(',')[1].strip()
    direction_string = df[0][0].split(',')[2].strip()
    
    # read in speed and direction with conversion from str to float
    buoy[speed_string] = [
        np.float_(df[0][row].split(',')[1]) for row in index_skipfirst
    ]
    
    # original current direction in clockwise from north
    buoy[f'{direction_string} (original)'] = [
        np.float_(df[0][row].split(',')[2]) for row in index_skipfirst
    ]
    # convert from CW from north -> CCW from east
    from_east = [90 - direction for direction in buoy[f'{direction_string} (original)']]
    buoy[f'{direction_string} (radians, CCW from East)'] = np.deg2rad(from_east)
    
    # calculate u-,v-velocities from speed and direction
    buoy['v'] = buoy[speed_string] * np.sin(buoy[f'{direction_string} (radians, CCW from East)'])
    buoy['u'] = buoy[speed_string] * np.cos(buoy[f'{direction_string} (radians, CCW from East)'])

    # create dataframe from dictionary
    df_buoy = pd.DataFrame(buoy, index = date)

    # create vector of hours from start of record 
    time = df_buoy.index.minute/60 + df_buoy.index.hour + 24*df_buoy.index.day 
    df_buoy['hours from start'] = time-time[0]

    # reorder dataframe so "hours from start" is first 
    column_names = list(df_buoy.columns[0:-1])
    column_names.insert(0,'hours from start')
    df_buoy = df_buoy[column_names]
    
    return df_buoy



#--------------------------------------------------
# An example of how to plot the results
# # plot results
# ax1=df_buoy["speed"].plot(color = 'blue', label='speed')
# ax2=df_buoy['direction'].plot(secondary_y=True, color='green', label='direction')
# ax2.set_ylabel('direction [$^\circ$E]')
# ax1.set_ylabel('speed [m/s]')
# ax1.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
# ax2.legend(bbox_to_anchor=(1.1, .9), loc='upper left')
# plt.show()
#--------------------------------------------------
if __name__=='__main__':
    """
    This script is specifically for accessing data from a current buoy and was created with 
    buoy `cb1501` as a template.  
    
    buoy_id: string with NOAA buoy ID (default = "cb1501")
    start_date: string with start date "YYYYMMDD" (default = "20240401")
    end_date: string with end date "YYYYMMDD" (default ="20240415")
    """
    args = sys.argv[1:]
    buoy_id=args[0]
    start_date=args[1]
    end_date=args[2]
    
    read_buoy(buoy_id, start_date, end_date)
    
