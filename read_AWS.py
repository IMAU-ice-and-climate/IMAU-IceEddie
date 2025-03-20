#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Filename: read_iWS_daily.py
#Description: Some useful functions used to read abd process AWS data 
"""
Created on Fri Jan 11 14:42:36 2019

Maurice van Tiggelen
IMAU, Utrecht University, the Netherlands
m.vantiggelen@uu.nl
"""

####### LIBRARIES ##########
import glob, os
import pandas as pd
import numpy as np
import itertools
import xarray as xr
import datetime
import read_EC as rec
import utils 
import Bulk_turbulence as bulk
import calc_footprint_FFP_climatology as fp_clim
from scipy import stats 
from functools import reduce
import re
import suncalc

####### CUSTOM FUNCTIONS ##########

#_______________________________________________________
def compose_date(years, months=1, days=1, weeks=None, hours=None, minutes=None,
                 seconds=None, milliseconds=None, microseconds=None, nanoseconds=None):
    ## Convert time format, adapted to raw iWS data
    
    years  = np.asarray(years) - 1970
    months = np.asarray(months) - 1
    days   = np.asarray(days) - 1
    types  = ('<M8[Y]', '<m8[M]', '<m8[D]', '<m8[W]', '<m8[h]',
             '<m8[m]', '<m8[s]', '<m8[ms]', '<m8[us]', '<m8[ns]')
    vals   = (years, months, days, weeks, hours, minutes, seconds,
            milliseconds, microseconds, nanoseconds)
    
    return sum(np.asarray(v, dtype=t) for t, v in zip(types, vals)
               if v is not None)
   #_______________________________________________________
def vardetrend(x):
    """
    Compute residuals variance of detrended timeseries x(t) by removing least squares linear fit to original data
    
    Input
    ----------
    x: pandas time indexed series
        Data to detrend (e.g temperature)
  
    Output
    ----------
    pandas series
        detrended x data
    
    Example
    ----------
    
    
    Required packages
    ----------
    numpy, pandas, numpy stats
    
    Required functions
    ----------
    none
    
    """   
    
    # Create time vector (in seconds since start)
    t = np.append([0] ,[np.cumsum(np.diff(x.index).astype(int) / 1E9)])
    
    # Find nan values
    not_nan_ind = ~np.isnan(x)
    
    # Check if there is data
    if np.all(np.isnan(x)):
        return np.nan
    
    # Perform least-sques regression
    m, b, r_val, p_val, std_err = stats.linregress(t[not_nan_ind],x[not_nan_ind])
    
    # Return regression and residuals
    return  np.std(x - (m*t + b))

#_______________________________________________________
def load_iWS_daily(file,nml):
    """
    Load single daily iWS datafile in a data frame
    
    Input
    ----------
    file: str
        filename
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
    WDoffset           = nml['L0toL1A']['WDoffset']
    gillfactor         = nml['L0toL1A']['gillfactor']
        
    # Load data in a data frame
        
    column_names       = ['time','ID','TC1avg','TC1std',	'TC2avg','TC2std',\
                          'CJavg','TSTR1','TSTR2','TSTR3','TSTR4','TSTR5','TSTR6','TSTR7',\
                          'TSTR8','THUTavg','RHWavg'	,'RHTavg','HWSavg','HWSstd','HWSmax',\
                          'HWDavg','VWSavg',	'VWSstd','NRUavg','NRUstd','NRLavg',\
                          'NRIUavg',	'NRILavg','NRTavg','NRUcal',	'NRLcal','NRIUcal','NRILcal',\
                          'NRID','BAP','SSH','ADW','TBRG','MCH','TILTX',	'TILTY',	'LON','LAT',	'HMSL',\
                          'PACC','SATS','SPARE1','SPARE2','VBAT'	,'LBUT',\
                          'STATUS','xID'	,'xTC1avg','xTC2avg','xCJavg','xTHUTavg','xRHWavg',\
                          'xRHTavg','xHWSavg','xHWDavg',	'xVWSavg','xSSH','xSPARE1','xVBAT','xSTATUS']


   
    df_raw             = pd.read_csv(file,delimiter = '\t', names = column_names,skiprows=4,encoding = "utf-8",index_col=False)
    df_raw             = df_raw.set_index('time')
    df_raw.index       = pd.to_datetime(df_raw.index).round(freq='30min')
    
    # Remove duplicated timestamps
    df_raw                     = df_raw[~df_raw.index.duplicated()]
    
    df_raw = df_raw.apply(pd.to_numeric, errors='coerce')

    df_raw.drop(['STATUS','xSTATUS'], axis=1, inplace=True)
    
    # Rename variables
    df_raw.rename(columns={'THUTavg': 'T0', 'HWSavg': 'U', 'HWSstd': 'Ustd','RHWavg': 'RH','SSH': 'zm', 'BAP': 'p0', 'NRUavg': 'SWd',\
                            'NRLavg': 'SWu','NRUstd': 'SWdstd', 'NRIUavg': 'LWd', 'NRILavg': 'LWu', 'NRTavg': 'NRT', 'HWDavg': 'WD','VWSavg': 'w','VWSstd': 'wstd'}, inplace=True)
    # Add time columns
    df_raw['Year'] = pd.to_datetime(df_raw.index.values).year
    df_raw['Month'] = pd.to_datetime(df_raw.index.values).month
    df_raw['Day'] = pd.to_datetime(df_raw.index.values).day
    df_raw['Hour'] = pd.to_datetime(df_raw.index.values).hour
    df_raw['Minute'] = pd.to_datetime(df_raw.index.values).minute
    
    
    # Convert temperatures to Kelvin
    df_raw['T0']     = df_raw['T0']     + 273.15
    df_raw['TC1avg'] = df_raw['TC1avg'] + 273.15
    df_raw['TC2avg'] = df_raw['TC2avg'] + 273.15
    df_raw['NRT']    = df_raw['NRT'] + 273.15
    
    # Convert Pressure from mbar to Pascal
    df_raw['p0'] = df_raw['p0']*100
    
    # Rotate wind direction in azimuth reference
    df_raw.WD = df_raw.WD + WDoffset - 180
    
    # Keep wind direction in [0;360[ range
    df_raw.WD[df_raw.WD < 0]   = df_raw.WD[df_raw.WD < 0]   + 360 
    df_raw.WD[df_raw.WD > 360] = df_raw.WD[df_raw.WD > 360] - 360 

    # Calibration of Gill signal
    df_raw['w'] = gillfactor * df_raw.w
    df_raw['wstd'] = abs(gillfactor) * df_raw.wstd
    
    return df_raw


#_______________________________________________________
def load_iWS_daily_S21(file,nml):
    """
    Load single daily iWS datafile in a data frame
    
    Input
    ----------
    file: str
        filename
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
    WDoffset           = nml['L0toL1A']['WDoffset']
    gillfactor         = nml['L0toL1A']['gillfactor']
        
    # Load data in a data frame
    column_names       = ['ID','DATE','TIMESTAMP','TC1avg','TC1std',	'TC2avg','TC2std',\
                          'CJavg','TSTR1','TSTR2','TSTR3','TSTR4','TSTR5','TSTR6','TSTR7',\
                          'TSTR8','THUTavg','RHWavg'	,'RHTavg','HWSavg','HWSstd','HWSmax',\
                          'HWDavg','VWSavg',	'VWSstd','NRUavg','NRUstd','NRLavg',\
                          'NRIUavg',	'NRILavg','NRTavg','NRUcal',	'NRLcal','NRIUcal','NRILcal',\
                          'NRID','BAP','SSH','ADW','TBRG','MCH','TILTX',	'TILTY',	'LON','LAT',	'HMSL',\
                          'PACC','SATS','SPARE1','SPARE2','VBAT'	,'LBUT',\
                          'STATUS','xID'	,'xTC1avg','xTC2avg','xCJavg','xTHUTavg','xRHWavg',\
                          'xRHTavg','xHWSavg','xHWDavg',	'xVWSavg','xSSH','xSPARE1','xVBAT','xSTATUS']


    df_raw             = pd.read_csv(file,delimiter = '\t', names = column_names,skiprows=4,encoding = "utf-8",index_col=False)
    
    df_raw['time'] = df_raw['DATE'] + ' ' + df_raw['TIMESTAMP']
    
    
    df_raw             = df_raw.set_index('time')
    
    df_raw.index       = pd.to_datetime(df_raw.index).round(freq='30min')
    
    # Remove duplicated timestamps
    df_raw                     = df_raw[~df_raw.index.duplicated()]
    
    df_raw = df_raw.apply(pd.to_numeric, errors='coerce')
    
    # Remove rows with NAN in time information
    # df_raw = df_raw.dropna(subset=['time'])
    
    # Remove extra columns

    df_raw.drop(['STATUS','xSTATUS','DATE','TIMESTAMP'], axis=1, inplace=True)
    
    # Rename variables
    df_raw.rename(columns={'THUTavg': 'T0', 'HWSavg': 'U', 'HWSstd': 'Ustd','RHWavg': 'RH','SSH': 'zm', 'BAP': 'p0', 'NRUavg': 'SWd',\
                            'NRLavg': 'SWu','NRUstd': 'SWdstd', 'NRIUavg': 'LWd', 'NRILavg': 'LWu', 'NRTavg': 'NRT', 'HWDavg': 'WD','VWSavg': 'w','VWSstd': 'wstd'}, inplace=True)
    # Add time columns
    df_raw['Year'] = pd.to_datetime(df_raw.index.values).year
    df_raw['Month'] = pd.to_datetime(df_raw.index.values).month
    df_raw['Day'] = pd.to_datetime(df_raw.index.values).day
    df_raw['Hour'] = pd.to_datetime(df_raw.index.values).hour
    df_raw['Minute'] = pd.to_datetime(df_raw.index.values).minute
    
    
    # Convert temperatures to Kelvin
    df_raw['T0']     = df_raw['T0']     + 273.15
    df_raw['TC1avg'] = df_raw['TC1avg'] + 273.15
    df_raw['TC2avg'] = df_raw['TC2avg'] + 273.15
    df_raw['NRT']    = df_raw['NRT'] + 273.15
    
    # Convert Pressure from mbar to Pascal
    df_raw['p0'] = df_raw['p0']*100
    
    # Rotate wind direction in azimuth reference
    df_raw.WD = df_raw.WD + WDoffset - 180
    
    # Keep wind direction in [0;360[ range
    df_raw.WD[df_raw.WD < 0]   = df_raw.WD[df_raw.WD < 0]   + 360 
    df_raw.WD[df_raw.WD > 360] = df_raw.WD[df_raw.WD > 360] - 360 

    # Calibration of Gill signal
    df_raw['w'] = gillfactor * df_raw.w
    df_raw['wstd'] = abs(gillfactor) * df_raw.wstd
    
    return df_raw
 
#_______________________________________________________
def load_iWS_Carleen(file,nml):
    """
    Load single yearly iWS datafile from Carleen format in a data frame
    
    Input
    ----------
    file: str
        filename
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
    
    WDoffset           = nml['L0toL1A']['WDoffset']
    
    # Load data in a data frame
    column_names = ['Date' , 'Hour' , 'Day' , 'Tth1', 'Tth2', 'Tsurf', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8',\
                     'T', 'T2m', 'Tpot', 'RH', 'q', 'WS', 'WSm', 'WD', 'Stoa', 'Sin', 'Sout', 'Lin', 'Lout',\
                    'Trad', 'P', 'H', 'M',  'Compas', 'A1', 'A2', 'Lon', 'Lat', 'HMSL', 'Vel', 'Bat', 'BatDays',  'Qual', 'AWSid']


    df_raw             = pd.read_csv(file,delim_whitespace=True, names = column_names,na_values = -9999,skiprows=1)
    # Use first columns as dataframe index
    # df_raw['time'] = df_raw['Date'] + ' ' +  df_raw['Hour']
    
    # Convert index to datetime
    df_raw.index               = pd.to_datetime(df_raw['Date'] + ' ' +  df_raw['Hour'])
    
     # Remove duplicated timestamps
    df_raw                     = df_raw[~df_raw.index.duplicated()]
    
    # Fill skipped timestamps with NaNs
    df_raw                     = df_raw.reindex(pd.date_range(min(df_raw.index), max(df_raw.index), freq='30min'))
    
    # Rename variables
    df_raw.index.names         = ['time']
    
    
    # Rename variables
    df_raw.rename(columns={'T': 'T0', 'WS': 'U', 'P': 'p0','Sin': 'SWd','Sout': 'SWu', 'Lin': 'LWd', 'Lout': 'LWu',\
                            'H': 'zm', 'A1': 'TILTX' , 'A2': 'TILTY', 'M': 'ADW', 'Lon': 'LON', 'Lat': 'LAT','Compas': 'yaw',\
                                'Trad': 'NRT','Bat': 'VBAT'}, inplace=True)
    
    # Convert temperatures to Kelvin
    df_raw['T0']     = df_raw['T0']     + 273.15

    # Duplicate Compas variable
    df_raw['MCH']     = df_raw['yaw'] 
    
    # Convert Pressure from mbar to Pascal
    df_raw['p0'] = df_raw['p0']*100
    
    # Rotate wind direction in azimuth reference
    df_raw.WD = df_raw.WD + WDoffset - 180
    
    # Keep wind direction in [0;360[ range
    df_raw.WD[df_raw.WD < 0]   = df_raw.WD[df_raw.WD < 0]   + 360 
    df_raw.WD[df_raw.WD > 360] = df_raw.WD[df_raw.WD > 360] - 360 

    for column in df_raw:
        if isinstance(df_raw[column][0], float):
            df_raw[column][df_raw[column] < -300] = np.nan
    return df_raw
#_______________________________________________________
def load_irridium_AWS18(file,nml):
    """
    Load single AWS18 irridium datafile
    
    Input
    ----------
    file: str
        filename
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
    
    WDoffset           = nml['L0toL1A']['WDoffset']
    
    # Load data in a data frame
    df_raw = pd.read_csv(file,sep='\t',header=0,na_values = -9999)
  
    # Convert index to datetime
    df_raw.index       = rec.compose_date(df_raw['Year'], df_raw['Month'], df_raw['Day'], hours=df_raw['Hour'], minutes=10*(round(df_raw['Minute']/10)))
    
    # Remove duplicated timestamps
    df_raw                     = df_raw[~df_raw.index.duplicated()]
    
    # Fill skipped timestamps with NaNs
    df_raw                     = df_raw.reindex(pd.date_range(min(df_raw.index), max(df_raw.index), freq='1H'))
    
    # Rename variables
    df_raw.index.names         = ['time']
        
    # Rename variables
    df_raw.rename(columns={'T': 'T0', 'WSP': 'U', 'Pr': 'p0','SWin': 'SWd','SWrefl': 'SWu', 'LWin': 'LWd', 'LWrefl': 'LWu',\
                            'SR50-H': 'zm', 'Tilt1x': 'TILTX' , 'Tilt1y': 'TILTY', 'Lon': 'LON', 'Lat': 'LAT',\
                                'T-kipp': 'NRT','Batt': 'VBAT','Wdir': 'WD','Pwr-use':'Pwruse','CSAT-X':'CSATX',\
                                    'CSAT-Y':'CSATY','CSAT-Z':'CSATZ','CSAT-T':'CSATT','AC150-H2O':'AC150H2O','TC-CJ':'TCCJ','Wsp-Gill':'WspGill',\
                                        'T-PT1k':'TPT1k'}, inplace=True)
    
    df_raw.columns = df_raw.columns.str.replace("-", "_")
        
    # Convert temperatures to Kelvin
    df_raw['T0']     = df_raw['T0']     + 273.15
    df_raw['NRT']     = df_raw['NRT']     + 273.15
    
    # Correct longwave measurements
    df_raw['LWd'] =  df_raw['LWd'] + 5.67e-8*(df_raw['NRT']**4)
    df_raw['LWu'] =  df_raw['LWu'] + 5.67e-8*(df_raw['NRT']**4)
    
    # Convert Pressure from mbar to Pascal
    df_raw['p0'] = df_raw['p0']*100
    
    # Rotate wind direction in azimuth reference
    df_raw.WD = df_raw.WD + WDoffset - 180
    
    # Cooridnates
    df_raw.LAT = df_raw.LAT/1e7
    df_raw.LON = df_raw.LON/1e7
    
    # Keep wind direction in [0;360[ range
    df_raw.WD[df_raw.WD < 0]   = df_raw.WD[df_raw.WD < 0]   + 360 
    df_raw.WD[df_raw.WD > 360] = df_raw.WD[df_raw.WD > 360] - 360 

    return df_raw
#_______________________________________________________
def load_iWS_yearly(file,nml):
    """
    Load single yearly iWS datafile in a data frame
    
    Input
    ----------
    file: str
        filename
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
    
    gillfactor         = nml['L0toL1A']['gillfactor']
    
    # Load data in a data frame
        
    column_names       = ['yyyy','mm','dd','hh','MM','doy','TC1avg','TC1std','TC2avg','TC2std',\
                          'CJavg','TSTR1','TSTR2','TSTR3','TSTR4','TSTR5','TSTR6','TSTR7',\
                          'TSTR8','THUTavg','RHWavg'	,'RHTavg','HWSavg','HWSstd','HWSmax',\
                          'HWDavg','VWSavg',	'VWSstd','NRUavg','NRUstd','NRLavg',\
                          'NRIUavg',	'NRILavg','NRTavg','NRUcal',	'NRLcal','NRIUcal','NRILcal',\
                          'NRID','BAP','SSH','ADW','TBRG','MCH','TILTX',	'TILTY',	'LON','LAT',	'HMSL',\
                          'PACC','STATS','SPARE1','SPARE2','VBAT','LBUT',\
                          'STATUS','ID','xID','xTC1avg','xTC2avg','xCJavg','xTHUTavg','xRHWavg',\
                          'xRHTavg','xHWSavg','xHWDavg',	'xVWSavg','xSSH','xSPARE1','xVBAT','xSTATUS',\
                          'Thutcorr','Rhnew','LWin','LWout','dL','tmp']

    df_raw             = pd.read_csv(file,delimiter = ',', names = column_names,na_values = -9999)
    df_raw.index       = rec.compose_date(df_raw['yyyy'], df_raw['mm'], df_raw['dd'], hours=df_raw['hh'], minutes=10*(round(df_raw['MM']/10)))   
    df_raw.index.names = ['time']
    
    # Remove duplicated timestamps
    df_raw             = df_raw[~df_raw.index.duplicated()]
    
    df_raw.drop(['LBUT'] , axis=1, inplace=True)   
    
    # Rename variables
    df_raw.rename(columns={'Thutcorr': 'T0', 'HWSavg': 'U', 'HWSstd': 'Ustd','Rhnew': 'RH','SSH': 'zm', 'BAP': 'p0', 'NRUavg': 'SWd',\
                            'NRLavg': 'SWu','NRUstd': 'SWdstd', 'LWin': 'LWd', 'LWout': 'LWu', 'NRTavg': 'NRT', 'HWDavg': 'WD','VWSavg': 'w','VWSstd': 'wstd'}, inplace=True)
    
    # Add time columns
    df_raw['Year']   = pd.to_datetime(df_raw.index.values).year
    df_raw['Month']  = pd.to_datetime(df_raw.index.values).month
    df_raw['Day']    = pd.to_datetime(df_raw.index.values).day
    df_raw['Hour']   = pd.to_datetime(df_raw.index.values).hour
    df_raw['Minute'] = pd.to_datetime(df_raw.index.values).minute
    
    # Convert temperatures to Kelvin
    df_raw['T0']     = df_raw['T0']     + 273.15
    df_raw['TC1avg'] = df_raw['TC1avg'] + 273.15
    df_raw['TC2avg'] = df_raw['TC2avg'] + 273.15
    df_raw['NRT']    = df_raw['NRT'] + 273.15
    
    # Convert Pressure from mbar to Pascal
    df_raw['p0'] = df_raw['p0']*100
    
    
    # Keep wind direction in [0;360[ range
    df_raw.WD[df_raw.WD < 0]   = df_raw.WD[df_raw.WD < 0]   + 360 
    df_raw.WD[df_raw.WD > 360] = df_raw.WD[df_raw.WD > 360] - 360 

    # Calibration of Gill signal
    df_raw['w'] = gillfactor * df_raw.w
    df_raw['wstd'] = abs(gillfactor) * df_raw.w
    
    return df_raw 

#_______________________________________________________
def load_iWS_yearly_Paul(file,nml):
    """
    Load single yearly iWS datafile in a data frame
    
    Input
    ----------
    file: str
        filename
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
    
    gillfactor         = nml['L0toL1A']['gillfactor']
    
    # Load data in a data frame

    column_names       = ['year','doy','hhmm','WD','U','Umax','SWd','SWu','LWd','LWu',\
                          'NRT','T0','RH','p0','zm','T1a','T2a','T3a',\
                          'T4a','T5a','T1b'	,'T2b','T3b','LON','LAT',\
                          'TILTX','TILTY',	'Vbat','Tlogger','ADW','quality']  ## !!! CAREFUL, LAT AND LON MAY BE FLIPPED

    df_raw             = pd.read_csv(file,delimiter = ',', names = column_names,na_values = -9999)

    # Remove non numeric data
    df_raw = df_raw.apply(pd.to_numeric, errors='coerce')
    
    # Remove rows with NAN in time information
    df_raw = df_raw.dropna(subset=['year','doy','hhmm'])
    
    # Make time index
    df_raw.index       = compose_date(df_raw['year'], days=df_raw['doy'], hours=(df_raw['hhmm']/100).astype(int),minutes = (df_raw['hhmm']%100).astype(int)) 
    df_raw.index.names = ['time']
    
    # Remove duplicated timestamps
    df_raw             = df_raw[~df_raw.index.duplicated()]
        
    # Add time columns
    df_raw['Year']   = pd.to_datetime(df_raw.index.values).year
    df_raw['Month']  = pd.to_datetime(df_raw.index.values).month
    df_raw['Day']    = pd.to_datetime(df_raw.index.values).day
    df_raw['Hour']   = pd.to_datetime(df_raw.index.values).hour
    df_raw['Minute'] = pd.to_datetime(df_raw.index.values).minute
    
    # Convert temperatures to Kelvin
    df_raw['T0']     = df_raw['T0']     + 273.15
    df_raw['NRT']    = df_raw['NRT'] + 273.15
    df_raw['Tlogger'] = df_raw['Tlogger'] + 273.15
    df_raw['T1a']    = df_raw['T1a'] + 273.15
    df_raw['T2a']    = df_raw['T2a'] + 273.15
    df_raw['T3a']    = df_raw['T3a'] + 273.15
    df_raw['T4a']    = df_raw['T4a'] + 273.15
    df_raw['T5a']    = df_raw['T5a'] + 273.15
    df_raw['T1b']    = df_raw['T1b'] + 273.15
    df_raw['T2b']    = df_raw['T2b'] + 273.15
    df_raw['T3b']    = df_raw['T3b'] + 273.15
    
    # Convert Pressure from mbar to Pascal
    df_raw['p0'] = df_raw['p0']*100
    
    # Keep wind direction in [0;360[ range
    df_raw.WD[df_raw.WD < 0]   = df_raw.WD[df_raw.WD < 0]   + 360 
    df_raw.WD[df_raw.WD > 360] = df_raw.WD[df_raw.WD > 360] - 360 
    
    return df_raw 

#_______________________________________________________
def load_iWS_yearly_Paul_corr(file,nml):
    """
    Load single yearly iWS datafile in a data frame which includes different LW variables 
    
    Input
    ----------
    file: str
        filename
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
    
    gillfactor         = nml['L0toL1A']['gillfactor']
    
    # Load data in a data frame

    # column_names       = ['year','doy','hhmm','WD','U','Umax','SWd','SWu','LWd','LWu',\
    #                       'NRT','T0','RH','p0','zm','T1a','T2a','T3a',\
    #                       'T4a','T5a','T1b'	,'T2b','T3b','LON','LAT',\
    #                       'TILTX','TILTY',	'Vbat','Tlogger','ADW','quality','LWd_raw','LWu_raw','dL']  ## !!! CAREFUL, LAT AND LON MAY BE FLIPPED
    
    column_names       = ['year','doy','hhmm','WD','U','Umax','SWd','SWu','LWd','LWu',\
                          'NRT','T0','RH','p0','zm','T1a','T2a','T3a',\
                          'T4a','T5a','T1b'	,'T2b','T3b','T4b','T5b',\
                          'TILTX','TILTY',	'Vbat','Tlogger','sumint','quality','LWd_raw','LWu_raw','dL']
    df_raw             = pd.read_csv(file,delimiter = ',', names = column_names,na_values = -9999)

    # Remove non numeric data
    df_raw = df_raw.apply(pd.to_numeric, errors='coerce')
    
    # Remove rows with NAN in time information
    df_raw = df_raw.dropna(subset=['year','doy','hhmm'])
    
    # Make time index
    df_raw.index       = compose_date(df_raw['year'], days=df_raw['doy'], hours=(df_raw['hhmm']/100).astype(int),minutes = (df_raw['hhmm']%100).astype(int)) 
    df_raw.index.names = ['time']
    
    # Remove duplicated timestamps
    df_raw             = df_raw[~df_raw.index.duplicated()]
        
    # Add time columns
    df_raw['Year']   = pd.to_datetime(df_raw.index.values).year
    df_raw['Month']  = pd.to_datetime(df_raw.index.values).month
    df_raw['Day']    = pd.to_datetime(df_raw.index.values).day
    df_raw['Hour']   = pd.to_datetime(df_raw.index.values).hour
    df_raw['Minute'] = pd.to_datetime(df_raw.index.values).minute
    
    # Convert temperatures to Kelvin
    df_raw['T0']     = df_raw['T0']     + 273.15
    df_raw['NRT']    = df_raw['NRT'] + 273.15
    df_raw['Tlogger'] = df_raw['Tlogger'] + 273.15
    df_raw['T1a']    = df_raw['T1a'] + 273.15
    df_raw['T2a']    = df_raw['T2a'] + 273.15
    df_raw['T3a']    = df_raw['T3a'] + 273.15
    df_raw['T4a']    = df_raw['T4a'] + 273.15
    df_raw['T5a']    = df_raw['T5a'] + 273.15
    df_raw['T1b']    = df_raw['T1b'] + 273.15
    df_raw['T2b']    = df_raw['T2b'] + 273.15
    df_raw['T3b']    = df_raw['T3b'] + 273.15
    df_raw['T4b']    = df_raw['T4b'] + 273.15
    df_raw['T5b']    = df_raw['T5b'] + 273.15
    
    # Convert Pressure from mbar to Pascal
    df_raw['p0'] = df_raw['p0']*100
    
    # Keep wind direction in [0;360[ range
    df_raw.WD[df_raw.WD < 0]   = df_raw.WD[df_raw.WD < 0]   + 360 
    df_raw.WD[df_raw.WD > 360] = df_raw.WD[df_raw.WD > 360] - 360 
    
    return df_raw 

#_______________________________________________________
def load_iWS_yearly_Paul_corr_v2(file,nml):
    """
    Load single yearly iWS datafile in a data frame which includes different LW variables 
    
    Input
    ----------
    file: str
        filename
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
    
    gillfactor         = nml['L0toL1A']['gillfactor']


    column_names        = ['year','doy','hhmm','WD','U','Umax','SWd','SWu','LWd','LWu',\
                          'NRT','T0','RH','p0','zm','T1a','T2a','T3a',\
                          'T4a','T5a','T1b'	,'T2b','T3b','LON','LAT',\
                          'TILTX','TILTY',	'Vbat','Tlogger','ADW','quality','sumint','LWd_raw','LWu_raw','T0_raw']

    df_raw             = pd.read_csv(file,delimiter = ',', names = column_names,na_values = -9999)
    # print(df_raw)

    # Remove non numeric data
    df_raw = df_raw.apply(pd.to_numeric, errors='coerce')
    
    # Remove rows with NAN in time information
    df_raw = df_raw.dropna(subset=['year','doy','hhmm'])
    
    # Make time index
    df_raw.index       = compose_date(df_raw['year'], days=df_raw['doy'], hours=(df_raw['hhmm']/100).astype(int),minutes = (df_raw['hhmm']%100).astype(int)) 
    df_raw.index.names = ['time']
    
    # Remove duplicated timestamps
    df_raw             = df_raw[~df_raw.index.duplicated()]
        
    # Add time columns
    df_raw['Year']   = pd.to_datetime(df_raw.index.values).year
    df_raw['Month']  = pd.to_datetime(df_raw.index.values).month
    df_raw['Day']    = pd.to_datetime(df_raw.index.values).day
    df_raw['Hour']   = pd.to_datetime(df_raw.index.values).hour
    df_raw['Minute'] = pd.to_datetime(df_raw.index.values).minute
    
    # Convert temperatures to Kelvin
    df_raw['T0']     = df_raw['T0']     + 273.15
    df_raw['T0_raw']     = df_raw['T0_raw']     + 273.15
    df_raw['NRT']    = df_raw['NRT'] + 273.15
    df_raw['Tlogger'] = df_raw['Tlogger'] + 273.15
    df_raw['T1a']    = df_raw['T1a'] + 273.15
    df_raw['T2a']    = df_raw['T2a'] + 273.15
    df_raw['T3a']    = df_raw['T3a'] + 273.15
    df_raw['T4a']    = df_raw['T4a'] + 273.15
    df_raw['T5a']    = df_raw['T5a'] + 273.15
    df_raw['T1b']    = df_raw['T1b'] + 273.15
    df_raw['T2b']    = df_raw['T2b'] + 273.15
    df_raw['T3b']    = df_raw['T3b'] + 273.15
    
    # Convert Pressure from mbar to Pascal
    df_raw['p0'] = df_raw['p0']*100
    
    # Keep wind direction in [0;360[ range
    df_raw.WD[df_raw.WD < 0]   = df_raw.WD[df_raw.WD < 0]   + 360 
    df_raw.WD[df_raw.WD > 360] = df_raw.WD[df_raw.WD > 360] - 360 
    
    return df_raw 

#_______________________________________________________
def load_iWS_yearly_Paul_corr_v6(file,nml):
    """
    Load single yearly iWS datafile in a data frame which includes different LW variables 
    
    Input
    ----------
    file: str
        filename
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
    
    gillfactor         = nml['L0toL1A']['gillfactor']
    

    column_names        = ['year','doy','hhmm','WD','U','Umax','SWd','SWu','LWd','LWu',\
                          'NRT','T0','RH','p0','zm','T1a','T2a','T3a',\
                          'T4a','T5a','T1b'	,'T2b','T3b','LON','LAT',\
                          'TILTX','TILTY',	'Vbat','Tlogger','ADW','quality','sumint','LWd_raw','LWu_raw','T0_raw']

    df_raw             = pd.read_csv(file,delimiter = ',', names = column_names,na_values = -9999)
    # print(df_raw)

    # Remove non numeric data
    df_raw = df_raw.apply(pd.to_numeric, errors='coerce')
    
    # Remove rows with NAN in time information
    df_raw = df_raw.dropna(subset=['year','doy','hhmm'])
    
    # Make time index
    df_raw.index       = compose_date(df_raw['year'], days=df_raw['doy'], hours=(df_raw['hhmm']/100).astype(int),minutes = (df_raw['hhmm']%100).astype(int)) 
    df_raw.index.names = ['time']
    
    # Remove duplicated timestamps
    df_raw             = df_raw[~df_raw.index.duplicated()]
        
    # Add time columns
    df_raw['Year']   = pd.to_datetime(df_raw.index.values).year
    df_raw['Month']  = pd.to_datetime(df_raw.index.values).month
    df_raw['Day']    = pd.to_datetime(df_raw.index.values).day
    df_raw['Hour']   = pd.to_datetime(df_raw.index.values).hour
    df_raw['Minute'] = pd.to_datetime(df_raw.index.values).minute
    
    # Convert temperatures to Kelvin
    df_raw['T0']     = df_raw['T0']     + 273.15
    df_raw['T0_raw']     = df_raw['T0_raw']     + 273.15
    df_raw['NRT']    = df_raw['NRT'] + 273.15
    df_raw['Tlogger'] = df_raw['Tlogger'] + 273.15
    df_raw['T1a']    = df_raw['T1a'] + 273.15
    df_raw['T2a']    = df_raw['T2a'] + 273.15
    df_raw['T3a']    = df_raw['T3a'] + 273.15
    df_raw['T4a']    = df_raw['T4a'] + 273.15
    df_raw['T5a']    = df_raw['T5a'] + 273.15
    df_raw['T1b']    = df_raw['T1b'] + 273.15
    df_raw['T2b']    = df_raw['T2b'] + 273.15
    df_raw['T3b']    = df_raw['T3b'] + 273.15
    
    # Convert Pressure from mbar to Pascal
    df_raw['p0'] = df_raw['p0']*100
    
    # Keep wind direction in [0;360[ range
    df_raw.WD[df_raw.WD < 0]   = df_raw.WD[df_raw.WD < 0]   + 360 
    df_raw.WD[df_raw.WD > 360] = df_raw.WD[df_raw.WD > 360] - 360 
    
    return df_raw 


#_______________________________________________________
def load_iWS_yearly_Paul_corr_v7(file,nml):
    """
    Load single yearly iWS datafile in a data frame which includes different LW variables 
    
    Input
    ----------
    file: str
        filename
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
    
    gillfactor         = nml['L0toL1A']['gillfactor']
    
# % iWS column format na inlezen in metlab
# % 1 year
# % 2 mm
# % 3 dd
# % 4 hh
# % 5 minutes
# % 6 doy
# % 7 TC1avg (?C) -9.96 thermocouple data
# % 8 TC1std (?C) 0.28 thermocouple data standard deviations
# % 9 TC2avg (?C) -10.31 thermocouple data
# % 10 TC2std (?C) 0.26 thermocouple data standard deviations
# % 11 CJavg (?C) -10.36
# % 12 xTSTR1 (?C) -10.14
# % 13 xTSTR2 (?C) -10.11
# % 14 xTSTR3 (?C) -10.11
# % 15 xTSTR4 (?C) -10.11
# % 16 xTSTR5 (?C) -10.09
# % 17 xTSTR6 (?C) -10.25
# % 18 xTSTR7 (?C) -7.52
# % 19 xTSTR8 (?C) -4.02
# % 20 THUTavg (?C) -10.28 raw Tair
# % 21 RHWavg (%) 91.24 raw RHair
# % 22 RHTavg (?C) -10.44 raw RH sensor temperature data
# % 23 HWSavg (m/s) 6.39 wind speed
# % 24 HWSstd (m/s) 0.78 wind speed standard deviation
# % 25 HWSmax (m/s) 8.47 wind speed max
# % 26 HWDavg (?) 81.65 wind direction
# % 27 VWSavg (m/s) 0.00
# % 28 VWSstd (m/s) 0.00
# % 29 NRUavg (W) 2.4 short wave in
# % 30 NRUstd (W) 0.2
# % 31 NRLavg (W) 3.0 short wave out
# % 32 NRIUavg (W) 263.8 long wave in
# % 33 NRILavg (W) 273.8 long wave out
# % 34 NRTavg (?C) -10.34 temp from radiation sensor
# % 35 NRUcal (?V) 13.73
# % 36 NRLcal (?V) 13.37
# % 37 NRIUcal (?V) 6.91
# % 38 NRILcal (?V) 8.55
# % 39 NRID (n) 121096
# % 40 BAP (hPa) 794.7 air pressure
# % 41 SSH (m) 3.083 sensor heights / changing snow height
# % 42 xADW (m) 2.332
# % 43 xTBRG (mm) 0.0
# % 44 MCH (?) 342.20
# % 45 TILTX (?) 1.15
# % 46 TILTY (?) -1.27pl
# % 47 LON (?) -470.228.291
# % 48 LAT (?) 670.003.638
# % 49 HMSL (m) 1.848.078
# % 50 PACC (m) 2.43
# % 51 SATS (n) 11
# % 52 spare 1
# % 53 spare 2
# % 54 VBAT (V) 5.147
# % 55 IDENTIFIER card=1, datalogger=2 or ARGOS=3 data (used to be LBUT (days) )
# % 56 STATUS (hex) 0
# % 57 iwsID
# % 58 xID (n) 58 identifier X is for slave unit data
# % 59 xTC1avg (?C) 326.45
# % 60 xTC2avg (?C) 326.51
# % 61 xCJavg (?C) -10.44
# % 62 xTHUTavg (?C) 327.67`
# % 63 xRHWavg (%) 100.00
# % 64 xRHTavg (?C) 128.86
# % 65 xHWSavg (m/s) 0.00
# % 66 xHWDavg (?) 0.52
# % 67 xVWSavg (m/s) 0.00
# % 68 xSSH (m) 2.074
# % 69 xSPARE1 (-) 12op
# % 70 xVBAT (V) 3.468
# % 71 xSTATUS (hex) 02 station visit indicator !!!
# % 72 Thutcorr - radiation corrected Thut (former spare column)
# % 73 Rhnew - radiation corrected Rh (former spare column)
# % 74 Lin corrected for window heating etc
# % 75 Lout corrected for window heating etc
# % 76 dL additional correction for Lin and Lout related to radiation from air layer close to the surface

    column_names        = ['year','mm','dd','hh','minutes','doy',\
                           'TC1avg','TC1std','TC2avg','TC2std','CJavg',\
                           'T1a','T2a','T3a','T4a','T5a','T1b','T2b','T3b',\
                           'T0_raw','RH_raw','Tlogger','U','HWSstd','Umax','WD','VWSavg','VWSstd',\
                           'SWd','NRUstd','SWu','LWd_raw','LWu_raw',\
                          'NRT','NRUcal','NRLcal','NRIUcal','NRILcal','NRID',\
                            'p0','zm','ADW','xTBRG','MCH','TILTX','TILTY','LON','LAT',\
                            'HMSL','PACC','SATS','spare1','spare2','Vbat','IDENTIFIER','STATUS','iwsID',\
                            'xID','xTC1avg','xTC2avg','xCJavg','xTHUTavg','xRHWavg','xRHTavg','xHWSavg','xHWDavg',\
                            'xVWSavg','xSSH','xSPARE1','xVBAT','xSTATUS',\
                            'T0','RH','LWd','LWu','dL']

    df_raw             = pd.read_csv(file,delimiter = ',', names = column_names,na_values = -9999)
    # df_raw2             = pd.read_csv(file,delimiter = ',',na_values = -9999)
    # print(df_raw2.head())


    # Remove non numeric data
    # df_raw = df_raw.apply(pd.to_numeric, errors='coerce')
    
    # Remove rows with NAN in time information
    # df_raw = df_raw.dropna(subset=['year','mm','hh'])
    
    # Make time index
    df_raw.index       = rec.compose_date(df_raw['year'], df_raw['mm'], df_raw['dd'], hours=df_raw['hh'], minutes=10*(round(df_raw['minutes']/10)))   
    df_raw.index.names = ['time']
    
    # Remove duplicated timestamps
    df_raw             = df_raw[~df_raw.index.duplicated()]
        
    # Add time columns
    df_raw['Year']   = pd.to_datetime(df_raw.index.values).year
    df_raw['Month']  = pd.to_datetime(df_raw.index.values).month
    df_raw['Day']    = pd.to_datetime(df_raw.index.values).day
    df_raw['Hour']   = pd.to_datetime(df_raw.index.values).hour
    df_raw['Minute'] = pd.to_datetime(df_raw.index.values).minute
    
    # Convert temperatures to Kelvin
    df_raw['T0']     = df_raw['T0']     + 273.15
    df_raw['T0_raw']     = df_raw['T0_raw']     + 273.15
    df_raw['NRT']    = df_raw['NRT'] + 273.15
    df_raw['Tlogger'] = df_raw['Tlogger'] + 273.15
    df_raw['T1a']    = df_raw['T1a'] + 273.15
    df_raw['T2a']    = df_raw['T2a'] + 273.15
    df_raw['T3a']    = df_raw['T3a'] + 273.15
    df_raw['T4a']    = df_raw['T4a'] + 273.15
    df_raw['T5a']    = df_raw['T5a'] + 273.15
    df_raw['T1b']    = df_raw['T1b'] + 273.15
    df_raw['T2b']    = df_raw['T2b'] + 273.15
    df_raw['T3b']    = df_raw['T3b'] + 273.15
    
    # Convert Pressure from mbar to Pascal
    df_raw['p0'] = df_raw['p0']*100
    
    # Keep wind direction in [0;360[ range
    df_raw.WD[df_raw.WD < 0]   = df_raw.WD[df_raw.WD < 0]   + 360 
    df_raw.WD[df_raw.WD > 360] = df_raw.WD[df_raw.WD > 360] - 360 
    
    return df_raw 


#_______________________________________________________
def load_AWS_yearly(file,nml):
    """
    Load single yearly AWS datafile in a data frame
    
    Input
    ----------
    file: str
        filename
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
               
    # Load data in a data frame
    column_names       =  ['year','DOY1','hhmm','u1','u2','u1max','u2max', 'dd1','dd2',\
                           'Sin','Sout','Lin','Lout','Tcnr1','T1','T2','Rh1',\
                           'Rh2','P','z1','z2','quality','Snet','acc_albedo','z6','Tsubs1',\
                           'Tsubs2','Tsubs3','Tsubs4','Tsubs5','Angle1','Angle2','Ubattery','compass','Tlogger']

    df_raw             = pd.read_csv(file,delimiter = ',', names = column_names,na_values = -9999)
    
    # Remove non numeric data
    df_raw = df_raw.apply(pd.to_numeric, errors='coerce')
    
    # Make time index
    df_raw.index       = compose_date(df_raw['year'], days=df_raw['DOY1'], hours=(df_raw['hhmm']/100).astype(int),minutes = (df_raw['hhmm']%100).astype(int)) 
    df_raw.index.names = ['time']
    
    
    
    # Remove duplicated timestamps
    df_raw             = df_raw[~df_raw.index.duplicated()]
   
    # df_raw.drop(['year','DOY1','hhmm','u1max','u2max',\
    #                        'Tcnr1',\
    #                        'z2','quality','Snet','acc_albedo','z6','Tsubs1',\
    #                        'Tsubs2','Tsubs3','Tsubs4','Tsubs5','Angle1','Angle2','Ubattery','compass','Tlogger'] , axis=1, inplace=True)
    
    
    df_raw.rename(columns={'z1': 'zm', 'P': 'p0', 'Sin': 'SWd',\
                            'Sout': 'SWu', 'Lin': 'LWd', 'Lout': 'LWu'}, inplace=True)
    
    #Use highest level as '0' level
    df_raw['T0'] = df_raw['T2']
    df_raw['WD'] = df_raw['dd2']
    df_raw['RH'] = df_raw['Rh2']
    df_raw['U']  = df_raw['u2']
    
    # 
    df_raw['zstakes']  = df_raw['zm']
    
    # Convert temperature from to Kelvin
    df_raw['T0'] = df_raw['T0'] + 273.15
    df_raw['T1'] = df_raw['T1'] + 273.15
    df_raw['T2'] = df_raw['T2'] + 273.15
    
    # Convert Pressure from mbar to Pascal
    df_raw['p0'] = df_raw['p0']*100
    
    return df_raw

#_______________________________________________________
def load_AWS_yearly_Paul(file,nml):
    """
    Load single yearly AWS datafile in a data frame
    
    Input
    ----------
    file: str
        filename
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
   
    # Load data in a data frame
    column_names       = ['year','doy','hhmm','WD','U','Umax','SWd','SWu','LWd','LWu',\
                          'NRT','T0','RH','p0','zm','T1a','T2a','T3a',\
                          'T4a','T5a','T1b'	,'T2b','T3b','T4b','T5b',\
                          'TILTX','TILTY',	'Vbat','Tlogger','sumint','quality','error']
    
    # % 30 sum of all interpolated values for all columns (formerly - compass heading of the CNR1)
    # % 31 tp data quality parameter (1 = datalogger data, 2 and 3 ARGOS type data, 4 interpolated data)
    # % 32 in the case of a quality issue with the data a number*10 is added that identifies the type of issu
    
    df_raw             = pd.read_csv(file,delimiter = ',', names = column_names,na_values = -9999)
    
    # Remove non numeric data
    df_raw = df_raw.apply(pd.to_numeric, errors='coerce')
    
    # Remove rows with NAN in time information
    df_raw = df_raw.dropna(subset=['year','doy','hhmm'])
    
    # Make time index
    df_raw.index       = compose_date(df_raw['year'], days=df_raw['doy'], hours=(df_raw['hhmm']/100).astype(int),minutes = (df_raw['hhmm']%100).astype(int)) 
    df_raw.index.names = ['time']
    
    # Remove duplicated timestamps
    df_raw             = df_raw[~df_raw.index.duplicated()]
        
    # Add time columns
    df_raw['Year']   = pd.to_datetime(df_raw.index.values).year
    df_raw['Month']  = pd.to_datetime(df_raw.index.values).month
    df_raw['Day']    = pd.to_datetime(df_raw.index.values).day
    df_raw['Hour']   = pd.to_datetime(df_raw.index.values).hour
    df_raw['Minute'] = pd.to_datetime(df_raw.index.values).minute
    
    # Convert temperatures to Kelvin
    df_raw['T0']     = df_raw['T0']     + 273.15
    df_raw['NRT']    = df_raw['NRT'] + 273.15
    df_raw['Tlogger'] = df_raw['Tlogger'] + 273.15
    df_raw['T1a']    = df_raw['T1a'] + 273.15
    df_raw['T2a']    = df_raw['T2a'] + 273.15
    df_raw['T3a']    = df_raw['T3a'] + 273.15
    df_raw['T4a']    = df_raw['T4a'] + 273.15
    df_raw['T5a']    = df_raw['T5a'] + 273.15
    df_raw['T1b']    = df_raw['T1b'] + 273.15
    df_raw['T2b']    = df_raw['T2b'] + 273.15
    df_raw['T3b']    = df_raw['T3b'] + 273.15
    df_raw['T4b']    = df_raw['T4b'] + 273.15
    df_raw['T5b']    = df_raw['T5b'] + 273.15
    
    # Convert Pressure from mbar to Pascal
    df_raw['p0'] = df_raw['p0']*100
    
    # Keep wind direction in [0;360[ range
    df_raw.WD[df_raw.WD < 0]   = df_raw.WD[df_raw.WD < 0]   + 360 
    df_raw.WD[df_raw.WD > 360] = df_raw.WD[df_raw.WD > 360] - 360 
    
    return df_raw 


#_______________________________________________________
def load_snowfox(file,nml):
    """
    Load snowfox datafiles in a dataframe
    
    Input
    ----------
    file: str
        filename
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
   
    # Load data in a data frame
    column_names       = ['RecordNum','Datetime','P1_mb','P4_mb','T1_C','RH1','Vbat','N1Cts','N1ET_sec','N1T_C','N1RH']
    df_raw             = pd.read_csv(file,delimiter = ',', names = column_names,na_values = -9999,skiprows=29)      
    
    # Make time index
    df_raw.index  = pd.to_datetime(df_raw.Datetime)
    df_raw.index.names = ['time']
    
    return df_raw 
#_______________________________________________________
def load_snowfox_irridium(file,nml):
    """
    Load snowfox datafiles in a dataframe from irridium 
    
    Input
    ----------
    file: str
        filename
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
   
    # Load data in a data frame
    column_names       = ['Datetime','N1Cts','N2Cts','T1_C','RH1','Vbat','P4_mb','NMcounts','fbar','fsol','SWE']
    
    df_raw             = pd.read_csv(file,delimiter = ',',na_values = -9999,skiprows=2,names = column_names)      
    
    # Make time index
    df_raw.index  = pd.to_datetime(df_raw.Datetime)
    df_raw.index.names = ['time']
    # df_raw.drop(['UTC'] , axis=1, inplace=True)
    
    return df_raw 

#_______________________________________________________
def load_snowfox_irridium_14(file,nml):
    """
    Load snowfox datafiles in a dataframe from irridium 
    
    Input
    ----------
    file: str
        filename
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
   
    # Load data in a data frame
    column_names       = ['Datetime','N1Cts','N2Cts','T1_C','RH1','Vbat','RH7','P4_mb','T7_C','A1','B1','NMcounts','fbar','fsol','SWE']
    
    df_raw             = pd.read_csv(file,delimiter = ',',na_values = -9999,skiprows=2,names = column_names)      
    
    # Make time index
    df_raw.index  = pd.to_datetime(df_raw.Datetime)
    df_raw.index.names = ['time']

    return df_raw 

#_______________________________________________________
def load_snowfox_irridium_09(file,nml):
    """
    Load snowfox datafiles in a dataframe from irridium 
    
    Input
    ----------
    file: str
        filename
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
   
    # Load data in a data frame
    column_names       = ['Datetime','N1Cts','N2Cts','T1_C','RH1','Vbat','P4_mb','NMcounts','fbar','fsol','SWE']    
    
    df_raw             = pd.read_csv(file,delimiter = ',',na_values = -9999,skiprows=2,names = column_names)      
    
    # Make time index
    df_raw.index  = pd.to_datetime(df_raw.Datetime)
    df_raw.index.names = ['time']
    # df_raw.drop(['UTC'] , axis=1, inplace=True)
    
    return df_raw 

#_______________________________________________________
def load_AWS_yearly_Paul_corr(file,nml):
    """
    Load single yearly AWS datafile in a data frame which includes different LW variables
    
    Input
    ----------
    file: str
        filename
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
   
    # Load data in a data frame
    column_names       = ['year','doy','hhmm','WD','U','Umax','SWd','SWu','LWd','LWu',\
                          'NRT','T0','RH','p0','zm','T1a','T2a','T3a',\
                          'T4a','T5a','T1b'	,'T2b','T3b','T4b','T5b',\
                          'TILTX','TILTY',	'Vbat','Tlogger','sumint','quality','error','LWd_raw','LWu_raw','dL']

    df_raw             = pd.read_csv(file,delimiter = ',', names = column_names,na_values = -9999)
    
    # Remove non numeric data
    df_raw = df_raw.apply(pd.to_numeric, errors='coerce')
    
    # Remove rows with NAN in time information
    df_raw = df_raw.dropna(subset=['year','doy','hhmm'])
    
    # Make time index
    df_raw.index       = compose_date(df_raw['year'], days=df_raw['doy'], hours=(df_raw['hhmm']/100).astype(int),minutes = (df_raw['hhmm']%100).astype(int)) 
    df_raw.index.names = ['time']
    
    # Remove duplicated timestamps
    df_raw             = df_raw[~df_raw.index.duplicated()]
        
    # Add time columns
    df_raw['Year']   = pd.to_datetime(df_raw.index.values).year
    df_raw['Month']  = pd.to_datetime(df_raw.index.values).month
    df_raw['Day']    = pd.to_datetime(df_raw.index.values).day
    df_raw['Hour']   = pd.to_datetime(df_raw.index.values).hour
    df_raw['Minute'] = pd.to_datetime(df_raw.index.values).minute
    
    # Convert temperatures to Kelvin
    df_raw['T0']     = df_raw['T0']     + 273.15
    df_raw['NRT']    = df_raw['NRT'] + 273.15
    df_raw['Tlogger'] = df_raw['Tlogger'] + 273.15
    df_raw['T1a']    = df_raw['T1a'] + 273.15
    df_raw['T2a']    = df_raw['T2a'] + 273.15
    df_raw['T3a']    = df_raw['T3a'] + 273.15
    df_raw['T4a']    = df_raw['T4a'] + 273.15
    df_raw['T5a']    = df_raw['T5a'] + 273.15
    df_raw['T1b']    = df_raw['T1b'] + 273.15
    df_raw['T2b']    = df_raw['T2b'] + 273.15
    df_raw['T3b']    = df_raw['T3b'] + 273.15
    df_raw['T4b']    = df_raw['T4b'] + 273.15
    df_raw['T5b']    = df_raw['T5b'] + 273.15
    
    # Convert Pressure from mbar to Pascal
    df_raw['p0'] = df_raw['p0']*100
    
    # Keep wind direction in [0;360[ range
    df_raw.WD[df_raw.WD < 0]   = df_raw.WD[df_raw.WD < 0]   + 360 
    df_raw.WD[df_raw.WD > 360] = df_raw.WD[df_raw.WD > 360] - 360 
    
    return df_raw 

#_______________________________________________________
def load_PIG_ENV(file,nml):
    """
    Load single yearly AWS datafile from AWS PIG from USAP in a data frame 
    
    Input
    ----------
    file: str
        filename
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
   
    # Load data in a data frame
    column_names = ["TIMESTAMP","RECORD","Air_Pressure_Avg","Air_Temp_HMP45C_Avg","Air_Temp_RTD_Avg",\
                    "Rel_Humidity","Ground_Temp_Avg","Snow_Avg","Wind_Speed_WVc1","Wind_Speed_WVc2",\
                    "Wind_Speed_WVc3","Wind_Speed_WVc4","Wind_Speed_Max","Wind_Speed_TMx","Short_Down_Avg",\
                    "Short_Up_Avg","Long_DownCo_Avg","Long_UpCo_Avg","Radiometer_Temp_C_Avg"]
    
    df_raw                     = pd.read_csv(file,delimiter = ',', na_values = 'NAN',skiprows=[0,1,2,3],names=column_names, encoding = "ISO-8859-1")

    # Use first columns as dataframe index
    df_raw                     = df_raw.set_index(df_raw.columns[0])
    
    # Convert index to datetime
    df_raw.index               = pd.to_datetime(df_raw.index) #,format="ISO8601"

    # Remove duplicated timestamps
    df_raw                     = df_raw[~df_raw.index.duplicated()]
    
    # Rename variables
    df_raw.index.names         = ['time']
            
    # Convert temperatures to Kelvin
    df_raw['T0']     = df_raw['Air_Temp_HMP45C_Avg']     + 273.15
    df_raw['NRT']    = df_raw['Radiometer_Temp_C_Avg'] + 273.15
    
    # Convert Pressure from mbar to Pascal
    df_raw['p0'] = df_raw['Air_Pressure_Avg']*1000
    
    # Keep wind direction in [0;360[ range

    df_raw['WD'] = df_raw['Wind_Speed_WVc3']
    # print(df_raw)
    df_raw.drop(['Wind_Speed_TMx'], axis=1, inplace=True)
    # print(df_raw)
    grouped = df_raw.groupby(pd.Grouper(freq='30min',label='right'), group_keys=False)
    df_raw   = grouped.mean()
    return df_raw 


#_______________________________________________________
def load_AWS_yearly_Paul_corr_v6(file,nml):
    """
    Load single yearly AWS datafile in a data frame which includes different LW variables
    
    Input
    ----------
    file: str
        filename
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
   
    # Load data in a data frame
    column_names       = ['year','doy','hhmm','WD','U','Umax','SWd','SWu','LWd','LWu',\
                          'NRT','T0','RH','p0','zm','T1a','T2a','T3a',\
                          'T4a','T5a','T1b'	,'T2b','T3b','T4b','T5b',\
                          'TILTX','TILTY',	'Vbat','Tlogger','sumint','quality','error','LWd_raw','LWu_raw','T0_raw','RH_raw']

    df_raw             = pd.read_csv(file,delimiter = ',', names = column_names,na_values = -9999)
    
    # Remove non numeric data
    df_raw = df_raw.apply(pd.to_numeric, errors='coerce')
    
    # Remove rows with NAN in time information
    df_raw = df_raw.dropna(subset=['year','doy','hhmm'])
    
    # Make time index
    df_raw.index       = compose_date(df_raw['year'], days=df_raw['doy'], hours=(df_raw['hhmm']/100).astype(int),minutes = (df_raw['hhmm']%100).astype(int)) 
    df_raw.index.names = ['time']
    
    # Remove duplicated timestamps
    df_raw             = df_raw[~df_raw.index.duplicated()]
        
    # Add time columns
    df_raw['Year']   = pd.to_datetime(df_raw.index.values).year
    df_raw['Month']  = pd.to_datetime(df_raw.index.values).month
    df_raw['Day']    = pd.to_datetime(df_raw.index.values).day
    df_raw['Hour']   = pd.to_datetime(df_raw.index.values).hour
    df_raw['Minute'] = pd.to_datetime(df_raw.index.values).minute
    
    # Convert temperatures to Kelvin
    df_raw['T0']     = df_raw['T0']     + 273.15
    df_raw['T0_raw']     = df_raw['T0_raw']     + 273.15
    df_raw['NRT']    = df_raw['NRT'] + 273.15
    df_raw['Tlogger'] = df_raw['Tlogger'] + 273.15
    df_raw['T1a']    = df_raw['T1a'] + 273.15
    df_raw['T2a']    = df_raw['T2a'] + 273.15
    df_raw['T3a']    = df_raw['T3a'] + 273.15
    df_raw['T4a']    = df_raw['T4a'] + 273.15
    df_raw['T5a']    = df_raw['T5a'] + 273.15
    df_raw['T1b']    = df_raw['T1b'] + 273.15
    df_raw['T2b']    = df_raw['T2b'] + 273.15
    df_raw['T3b']    = df_raw['T3b'] + 273.15
    df_raw['T4b']    = df_raw['T4b'] + 273.15
    df_raw['T5b']    = df_raw['T5b'] + 273.15
    
    # Convert Pressure from mbar to Pascal
    df_raw['p0'] = df_raw['p0']*100
    
    # Keep wind direction in [0;360[ range
    df_raw.loc[df_raw.WD > 360, "WD"] -= 360
    df_raw.loc[df_raw.WD < 0, "WD"] += 360
    
    return df_raw 

#_______________________________________________________
def load_AWS_yearly_Paul_corr_v6_merged(file,nml):
    """
    Load single yearly AWS datafile that contains data from 2 separate AWS in a data frame which includes different LW variables
    
    Input
    ----------
    file: str
        filename
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
   
    # Load data in a data frame
    column_names       = ['year','doy','hhmm','WD','U','Umax','SWd','SWu','LWd','LWu',\
                          'NRT','T0','RH','p0','zm','AWSID','T2a','T3a',\
                          'T4a','T5a','T1b'	,'T2b','T3b','T4b','T5b',\
                          'TILTX','TILTY',	'Vbat','Tlogger','sumint','quality']

    df_raw             = pd.read_csv(file,delimiter = ',', names = column_names,na_values = -9999)
    
    # Remove non numeric data
    df_raw = df_raw.apply(pd.to_numeric, errors='coerce')
    
    # Remove rows with NAN in time information
    df_raw = df_raw.dropna(subset=['year','doy','hhmm'])
    
    # Make time index
    df_raw.index       = compose_date(df_raw['year'], days=df_raw['doy'], hours=(df_raw['hhmm']/100).astype(int),minutes = (df_raw['hhmm']%100).astype(int)) 
    df_raw.index.names = ['time']
    
        
    # Add time columns
    df_raw['Year']   = pd.to_datetime(df_raw.index.values).year
    df_raw['Month']  = pd.to_datetime(df_raw.index.values).month
    df_raw['Day']    = pd.to_datetime(df_raw.index.values).day
    df_raw['Hour']   = pd.to_datetime(df_raw.index.values).hour
    df_raw['Minute'] = pd.to_datetime(df_raw.index.values).minute
    
    # Convert temperatures to Kelvin
    df_raw['T0']     = df_raw['T0']     + 273.15
    df_raw['NRT']    = df_raw['NRT'] + 273.15
    df_raw['Tlogger'] = df_raw['Tlogger'] + 273.15
    df_raw['T2a']    = df_raw['T2a'] + 273.15
    df_raw['T3a']    = df_raw['T3a'] + 273.15
    df_raw['T4a']    = df_raw['T4a'] + 273.15
    df_raw['T5a']    = df_raw['T5a'] + 273.15
    df_raw['T1b']    = df_raw['T1b'] + 273.15
    df_raw['T2b']    = df_raw['T2b'] + 273.15
    df_raw['T3b']    = df_raw['T3b'] + 273.15
    df_raw['T4b']    = df_raw['T4b'] + 273.15
    df_raw['T5b']    = df_raw['T5b'] + 273.15
    
    # Convert Pressure from mbar to Pascal
    df_raw['p0'] = df_raw['p0']*100
    
    # Keep wind direction in [0;360[ range
    df_raw.WD[df_raw.WD < 0]   = df_raw.WD[df_raw.WD < 0]   + 360 
    df_raw.WD[df_raw.WD > 360] = df_raw.WD[df_raw.WD > 360] - 360 

    # SPlit dataframe
    df_raw_2 = df_raw[df_raw['AWSID'] == 3]
    df_raw = df_raw[df_raw['AWSID'] == 1]

    return df_raw, df_raw_2



#_______________________________________________________
def load_AWS_final_year(file,nml):
    """
    Load single yearly AWS datafile in a data frame
    
    Input
    ----------
    file: str
        filename
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
               
    # Load data in a data frame
    column_names       =  ['year','DOY1','hhmm','u1','u2','u1max','u2max', 'dd1','dd2',\
                           'Sin','Sout','Lin','Lout','Tcnr1','T1','T2','Rh1',\
                           'Rh2','P','z1','z2','quality','Snet','acc_albedo','z6','Tsubs1',\
                           'Tsubs2','Tsubs3','Tsubs4','Tsubs5','Angle1','Angle2','Ubattery','compass','Tlogger']

    df_raw             = pd.read_csv(file,delimiter = ',', names = column_names,na_values = -9999)
    # Remove non numeric data
    df_raw = df_raw.apply(pd.to_numeric, errors='coerce')
    # Make time index
    df_raw.index       = compose_date(df_raw['year'], days=df_raw['DOY1'], hours=(df_raw['hhmm']/100).astype(int),minutes = (df_raw['hhmm']%100).astype(int)) 
    df_raw.index.names = ['time']
    
    
    # Remove duplicated timestamps
    df_raw             = df_raw[~df_raw.index.duplicated()]
   
    df_raw.rename(columns={'z1': 'zm', 'P': 'p0', 'Sin': 'SWd',\
                            'Sout': 'SWu', 'Lin': 'LWd', 'Lout': 'LWu'}, inplace=True)
    
    #Use highest level as '0' level
    df_raw['T0'] = df_raw['T2']
    df_raw['WD'] = df_raw['dd2']
    df_raw['RH'] = df_raw['Rh2']
    df_raw['U']  = df_raw['u2']
    
    # 
    df_raw['zstakes']  = df_raw['zm']
    
    # Convert temperature from to Kelvin
    df_raw['T0'] = df_raw['T0'] + 273.15
    df_raw['T1'] = df_raw['T1'] + 273.15
    df_raw['T2'] = df_raw['T2'] + 273.15
    
    # Convert Pressure from mbar to Pascal
    df_raw['p0'] = df_raw['p0']*100
    
    return df_raw

#_______________________________________________________
def load_AWS_final_year_ant(file,nml):
    """
    Load single yearly AWS datafile in a data frame
    
    Input
    ----------
    file: str
        filename
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
               
    # Load data in a data frame
    column_names       =  ['year','DOY1','hhmm','dd1','u1','u1max',\
                           'Sin','Sout','Lin','Lout','Tcnr1','T1','Rh1',\
                           'P','z1','Tsubs1',\
                           'Tsubs2','Tsubs3','Tsubs4','Tsubs5','Tsubs6','Tsubs7','Tsubs8','Tsubs9','Tsubs10','Tsubs11',\
                          'Angle1','Angle2','Ubattery','compass','Tlogger']

    df_raw             = pd.read_csv(file,delimiter = ',', names = column_names,na_values = -9999)
    # Remove non numeric data
    df_raw = df_raw.apply(pd.to_numeric, errors='coerce')
    # Make time index
    df_raw.index       = compose_date(df_raw['year'], days=df_raw['DOY1'], hours=(df_raw['hhmm']/100).astype(int),minutes = (df_raw['hhmm']%100).astype(int)) 
    df_raw.index.names = ['time']
    
    
    
    # Remove duplicated timestamps
    df_raw             = df_raw[~df_raw.index.duplicated()]
    
    df_raw.rename(columns={'z1': 'zm', 'P': 'p0', 'Sin': 'SWd',\
                            'Sout': 'SWu', 'Lin': 'LWd', 'Lout': 'LWu'}, inplace=True)
    
    #Use highest level as '0' level
    df_raw['T0'] = df_raw['T1']
    df_raw['WD'] = df_raw['dd1']
    df_raw['RH'] = df_raw['Rh1']
    df_raw['U']  = df_raw['u1']
    
    # 
    df_raw['zstakes']  = df_raw['zm']
    
    # Convert temperature from to Kelvin
    df_raw['T0'] = df_raw['T0'] + 273.15
    df_raw['T1'] = df_raw['T1'] + 273.15
    
    # Convert Pressure from mbar to Pascal
    df_raw['p0'] = df_raw['p0']*100
    
    return df_raw
    
#_______________________________________________________
def EBM_to_csv(nml):
    """
    Merge SEB model output and writes to single csv file in NEAD format
    
    Input
    ----------
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    """
    EBMdir  = nml['AWScorr']['EBMdir2']
    odir  = nml['AWScorr']['DIR'] + '../FINAL/'
    AWSfile = nml['AWScorr']['file_AWS_locations']
    ID		= nml['global']['ID']    
    LOC		= nml['global']['LOC']   

    column_names_AWS = ['date','hour','year','day','t_a','t_2m','th_a','t_s_mod','t_s_obs','dT0', \
        'Tinv', 'q_a','q_2m','q_s','rh_2m','wspd_a','wspd_10m','p_a','cc','alb_mod','alb_mod_m','sza_mod', \
            'rest','melt_mod','swnet','swd_m','swu_m','spen_mod','spen_sin','lwnet','lwd_m', \
                'lwu_mod','lwu_m','rnet','shf_mod','lhf_mod','ghf_mod','ch','cq','ustar', \
                    'thstar','qstar','psim','psih','psiq','zt','zm_m','z0m','z0h','z0q','Hice','dHice','errorflag','awsid','paramerror_AWS']

    column_names_MB = ['date','hour','hsnow_obs','precip_obs','drift_obs', \
        'acc_obs','icemelt_obs','cumicemelt_obs','hsnowmod_mod','precip_mod', \
            'drift_mod','accsnow_mod', 'icemelt_mod','cumicemelt','hmass', 'totwater', 'topwater', 'topsnow', 'topmass', \
                'cum_totmelt','totmelt','cum_surfmelt','surfmelt','runoff','subl','slushdepth','surfwater','error_MB','errorobs']

    column_names_SNOW = ['date','hour','year','il','depth','thickness','temp','dens','mass','water', \
        'ice', 'Spenetration','cp','irrwater','energy','grainsize','id','hsnowmod','hsnow','error']
    
    if os.path.exists(EBMdir):
        os.chdir(EBMdir)
        file_AWS = (glob.glob('*AWS.txt'))[0]
        file_MB = (glob.glob('*MB.txt'))[0]
        print(file_AWS)
        # file_SNOW = (glob.glob('*SNOW.txt'))[0]

        df_AWS           = pd.read_csv(file_AWS,delim_whitespace = True, names = column_names_AWS,na_values = -998,skiprows=1)
        df_MB            = pd.read_csv(file_MB,delim_whitespace = True, names = column_names_MB,na_values = -998,skiprows=1)
        # df_SNOW          = pd.read_csv(file_SNOW,delim_whitespace = True, names = column_names_SNOW,na_values = -998,skiprows=1)

        df_AWS.index              = pd.to_datetime(df_AWS['date'] + ' ' +  df_AWS['hour'])
        df_MB.index               = pd.to_datetime(df_MB['date'] + ' ' +  df_MB['hour'])
        # df_SNOW.index             = pd.to_datetime(df_SNOW['date'] + ' ' +  df_SNOW['hour'])

        df_AWS.index.names         = ['time']
        df_MB.index.names         = ['time']

        df_MB = df_MB.where(df_MB.error_MB == 0)
        df_AWS = df_AWS.where(df_AWS.errorflag == 0)
        # df_SNOW = df_AWS.where(df_SNOW.erroflag == 0)
        # df_SNOW.index.names         = ['time']    
    else:
        print('WARNING: no EBM data found, continuing without...')

    if not os.path.exists(odir):
        os.makedirs(odir)
    os.chdir(odir)

    df = pd.concat([df_AWS,df_MB],axis=1)

    df.rename(columns={'t_a': 't'}, inplace=True)
    df['topwater'] = 0.1*df['topwater']/df['topsnow']
    df['topwater'][df['topwater'] > df['totwater']] = df['totwater'][df['topwater'] > df['totwater']]

    column_names_out = ['t','t_2m','t_s_obs','surfmelt','totmelt','LWC','LWC_top10cm','cum_surfmelt','cum_totmelt']

    df['t'] = np.round(df['t']-273.15,decimals=3)
    df['t_2m'] = np.round(df['t_2m']-273.15,decimals=3)
    df['t_s_obs'] = np.round(df['t_s_obs']-273.15,decimals=2)
    df['surfmelt'] = np.round(df['surfmelt'],decimals=2)
    df['totmelt'] = np.round(df['totmelt'],decimals=2)
    df['cum_surfmelt'] = np.round(df['cum_surfmelt'],decimals=2)
    df['cum_totmelt'] = np.round(df['cum_totmelt'],decimals=2)
    df['LWC'] = np.round(df['totwater'],decimals=2)
    df['LWC_top10cm'] = np.round(df['topwater'],decimals=2)

    df_out = df[column_names_out]
    file_out_nead  = odir + LOC + '_' + ID + '_hourly_all' + ".csv"

    # Get AWS location
    AWSloc = pd.read_csv(AWSfile,skiprows=0,delim_whitespace=True)
    AWSid =  AWSloc.loc[AWSloc['STATION'] == ID]['STATION'].values[0]
    AWSlat =  AWSloc.loc[AWSloc['STATION'] == ID]['LATITUDE'].values[0]
    AWSlon =  AWSloc.loc[AWSloc['STATION'] == ID]['LONGITUDE'].values[0]
    AWSalt =  AWSloc.loc[AWSloc['STATION'] == ID]['ELEVATION'].values[0]

    header = '\n'.join(
        ['# NEAD 1.0 UTF-8',
        '# [METADATA]',
        '# station_id       = ' + str(AWSid),
        '# latitude         = ' + str(AWSlat), 
        '# longitude        = ' + str(AWSlon),
        '# altitude         = ' + str(AWSalt),
        '# nodata           = ',
        '# field_delimiter  = ,',
        '# tz               = 0', 
        '# doi              = ', 
        '# [FIELDS]',
        '# fields           = time,t,t2m,t_s_obs,surfmelt,totmelt,LWC,LWC_top10cm,cumulative_surfmelt,cumulative_totmelt',
        '# units            = -,DegreeC,DegreeC,DegreeC,kg/m^2/hour,kg/m^2/hour,kg/m^2,kg/m^2,kg/m^2,kg/m^2',
        '# [DATA]',
        '',
        ]
    )
    with open(file_out_nead, 'w') as ict:
        for line in header:
            ict.write(line)
        df_out.to_csv(ict,index=True,header=False,sep=',',float_format='%6.4f')

#_______________________________________________________
def EBM_AWS_to_csv(nml):
    """
    Merge SEB model output and AWS data, and writes to single csv file in NEAD format. 
    Also save a different file with selected data and with daily averages
    
    Input
    ----------
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    """
    
    ID		= nml['global']['ID']    
    LOC		= nml['global']['LOC']    
    sensor  = nml['global']['sensor']  

    EBMdir  = nml['AWScorr']['EBMdir']
    EBMdir2  = nml['AWScorr']['EBMdir2']
    L1Adir  = nml['AWScorr']['DIR'] + '/L1A/'
    L1Bdir  = nml['AWScorr']['DIR'] + '/L1B/'
    odir  = nml['AWScorr']['DIR'] + '../FINAL/'
    lshift_time_EBM = nml['AWScorr']['lshift_time_EBM']
    
    yyyymmdd_start = nml['AWScorr']['yyyymmdd_start']
    yyyymmdd_end = nml['AWScorr']['yyyymmdd_end']
    AWSfile = nml['AWScorr']['file_AWS_locations']
    dT = nml['AWS_to_SEB']['dT']


    column_names_AWS = ['date','hour','year','day','t_a','t_2m','th_a','t_s_mod','t_s_obs','dT0', \
        'Tinv', 'q_a','q_2m','q_s','rh_2m','wspd_a','wspd_10m','p_a','cc','alb_mod','alb_mod_m','sza_mod', \
            'rest','melt_mod','swnet','swd_m','swu_m','spen_mod','spen_sin','lwnet','lwd_m', \
                'lwu_mod','lwu_m','rnet','shf_mod','lhf_mod','ghf_mod','ch','cq','ustar', \
                    'thstar','qstar','psim','psih','psiq','zt','zm_m','z0m','z0h','z0q','Hice','dHice','errorflag','awsid','paramerror_AWS']

    column_names_MB = ['date','hour','hsnow_obs','precip_obs','drift_obs', \
        'acc_obs','icemelt_obs','cumicemelt_obs','hsnowmod_mod','precip_mod', \
            'drift_mod','accsnow_mod', 'icemelt_mod','cumicemelt','hmass', 'totwater', 'topwater', 'topsnow', 'topmass', \
                'cum_totmelt','totmelt','cum_surfmelt','surfmelt','runoff','subl','slushdepth','surfwater','error_MB','errorobs']

    column_names_SNOW = ['date','hour','year','il','depth','thickness','temp','dens','mass','water', \
        'ice', 'Spenetration','cp','irrwater','energy','grainsize','id','hsnowmod','hsnow','error']
    
    column_names_AWS_2 = [s + '_2' for s in column_names_AWS]
    column_names_MB_2 = [s + '_2' for s in column_names_MB]


    os.chdir(L1Adir)
    file_L1A = L1Adir 
    file_L1A = (glob.glob('*L1A*' + "*nc"))[0]
    
    os.chdir(L1Bdir)
    file_L1B = L1Bdir 
    file_L1B = (glob.glob('*L1B*' + "*nc"))[0]

    os.chdir(L1Bdir)
    ds_L1B           = xr.open_dataset(file_L1B)
    df_L1B           = ds_L1B.to_dataframe()


    # Removed filled data
    df_L1B['RH'] = df_L1B['RH'].where(df_L1B.lhum==1)
    df_L1B['qv0'] = df_L1B['qv0'].where(df_L1B.lhum==1)
    df_L1B['p0'] = df_L1B['p0'].where(df_L1B.lbaro==1)
    df_L1B['U'] = df_L1B['U'].where(df_L1B.lanemo==1)
    df_L1B['Umax'] = df_L1B['Umax'].where(df_L1B.lanemo==1)
    df_L1B['WD'] = df_L1B['WD'].where(df_L1B.lanemo==1)
    if ID == 'AWS13':
        df_L1B['U'] = np.full(len(df_L1B.U),np.nan)

    os.chdir(L1Adir)
    ds_L1A           = xr.open_dataset(file_L1A)

    column_names_out_L1A = ['zm','U_L1A','swd_uncorr','swu_uncorr','LAT_L1A','LON_L1A']
    

    # Only keep data after start time
    # ds_L1A  = ds_L1A.sel(time=slice(yyyymmdd_start, yyyymmdd_end))
    df_L1A           = ds_L1A.to_dataframe()
    df_L1A.rename(columns={'U': 'U_L1A','SWd': 'swd_uncorr','SWu': 'swu_uncorr'}, inplace=True)
    if not 'zm' in list(df_L1A.keys()):
        df_L1A['zm'] = df_L1A['z_boom_u']
    if (ID == 'AWS11') | (ID == 'AWS14') | (ID == 'AWS16') | (ID == 'AWS17') | (ID == 'AWS18') | (ID == 'AWS19') :
        df_L1A['LAT'] =  df_L1A['T5b'] -273.15
        df_L1A['LON'] =  df_L1A['T4b']-273.15
        if ID == 'AWS11':
            cutoff_date = pd.to_datetime('2015-03-05')
        if ID == 'AWS14':
            cutoff_date = pd.to_datetime('2015-01-24')
        if ID == 'AWS16':
            cutoff_date = pd.to_datetime('2015-12-19')
        if ID == 'AWS17':
            cutoff_date = pd.to_datetime('2015-01-19')
        if ID == 'AWS18':
            cutoff_date = pd.to_datetime('2000-01-24')
        if ID == 'AWS19':
            cutoff_date = pd.to_datetime('2000-01-24')
        df_L1A.LAT[df_L1A.index < cutoff_date]   = np.nan
        df_L1A.LON[df_L1A.index < cutoff_date]   = np.nan

    if not 'LAT' in list(df_L1A.keys()):
        df_L1A['LAT'] = np.full(len(df_L1A.zm),np.nan)
        df_L1A['LON'] = np.full(len(df_L1A.zm),np.nan)
    df_L1A['LAT'][df_L1A['LAT'] < -90] = np.nan
    df_L1A['LAT'][df_L1A['LAT'] > 90] = np.nan
    df_L1A['LON'][df_L1A['LON'] < -180] = np.nan
    df_L1A['LON'][df_L1A['LON'] > 360] = np.nan
    df_L1A['LAT'][(abs(df_L1A['LAT'] - np.nanmean(df_L1A['LAT']))) > 3*np.nanstd(df_L1A['LAT']) ]  = np.nan
    df_L1A['LON'][(abs(df_L1A['LON'] - np.nanmean(df_L1A['LON']))) > 3*np.nanstd(df_L1A['LON']) ]  = np.nan
    df_L1A.rename(columns={'LON': 'LON_L1A','LAT': 'LAT_L1A'}, inplace=True)
    df_L1A = df_L1A[column_names_out_L1A]
    df_L1A.zm[df_L1A.zm < -900]   = np.nan
    df_L1A.rename(columns={'zm': 'z_boom'}, inplace=True)

    if os.path.exists(EBMdir):
        os.chdir(EBMdir)
        file_AWS = (glob.glob('*AWS.txt'))[0]
        file_MB = (glob.glob('*MB.txt'))[0]
        # file_SNOW = (glob.glob('*SNOW.txt'))[0]

        df_AWS           = pd.read_csv(file_AWS,delim_whitespace = True, names = column_names_AWS,na_values = [-998,-998.8,-999,-999.9],skiprows=1)
        df_MB            = pd.read_csv(file_MB,delim_whitespace = True, names = column_names_MB,na_values = [-998,-998.8,-999,-999.9],skiprows=1)
        # df_SNOW          = pd.read_csv(file_SNOW,delim_whitespace = True, names = column_names_SNOW,na_values = -998,skiprows=1)

        df_AWS.index              = pd.to_datetime(df_AWS['date'] + ' ' +  df_AWS['hour'])
        df_MB.index               = pd.to_datetime(df_MB['date'] + ' ' +  df_MB['hour'])
        # df_SNOW.index             = pd.to_datetime(df_SNOW['date'] + ' ' +  df_SNOW['hour'])

        df_AWS.index.names         = ['time']
        df_MB.index.names         = ['time']

        df_MB = df_MB.where(df_MB.error_MB == 0)
        df_AWS = df_AWS.where(df_AWS.errorflag == 0)
        # df_SNOW = df_AWS.where(df_SNOW.erroflag == 0)
        # df_SNOW.index.names         = ['time']    

        os.chdir(EBMdir2)
        file_AWS_2 = (glob.glob('*AWS.txt'))[0]
        file_MB_2 = (glob.glob('*MB.txt'))[0]
        # file_SNOW = (glob.glob('*SNOW.txt'))[0]

        df_AWS_2           = pd.read_csv(file_AWS_2,delim_whitespace = True, names = column_names_AWS_2,na_values = [-998,-998.8,-999,-999.9],skiprows=1)
        df_MB_2            = pd.read_csv(file_MB_2,delim_whitespace = True, names = column_names_MB_2,na_values = [-998,-998.8,-999,-999.9],skiprows=1)
        # df_SNOW          = pd.read_csv(file_SNOW,delim_whitespace = True, names = column_names_SNOW,na_values = -998,skiprows=1)

        df_AWS_2.index              = pd.to_datetime(df_AWS_2['date_2'] + ' ' +  df_AWS_2['hour_2'])
        df_MB_2.index               = pd.to_datetime(df_MB_2['date_2'] + ' ' +  df_MB_2['hour_2'])
        # df_SNOW.index             = pd.to_datetime(df_SNOW['date'] + ' ' +  df_SNOW['hour'])

        df_AWS_2.index.names         = ['time']
        df_MB_2.index.names         = ['time']

        df_MB_2 = df_MB_2.where(df_MB_2.error_MB_2 < 30)
        df_AWS_2 = df_AWS_2.where(df_AWS_2.errorflag_2 < 30)
        # df_SNOW = df_AWS.where(df_SNOW.erroflag == 0)
        # df_SNOW.index.names         = ['time']   
        if lshift_time_EBM:
            if dT == '1H':
                if ID == 'AWS05':
                    cutoff_date = pd.to_datetime('2003-02-01')
                    df_MB_2.index = df_MB_2.index.map(lambda x: x + pd.Timedelta(hours=0) if x < cutoff_date else x + pd.Timedelta(hours=1))
                    df_AWS_2.index = df_AWS_2.index.map(lambda x: x + pd.Timedelta(hours=0) if x < cutoff_date else x + pd.Timedelta(hours=1))
                    df_MB.index = df_MB.index.map(lambda x: x + pd.Timedelta(hours=0) if x < cutoff_date else x + pd.Timedelta(hours=1))
                    df_AWS.index = df_AWS.index.map(lambda x: x + pd.Timedelta(hours=0) if x < cutoff_date else x + pd.Timedelta(hours=1))
                    df_MB_2 = df_MB_2.drop_duplicates(keep='first')
                    df_AWS_2 = df_AWS_2.drop_duplicates(keep='first')
                    df_MB = df_MB.drop_duplicates(keep='first')
                    df_AWS = df_AWS.drop_duplicates(keep='first')   
                elif ID == 'AWS06':
                    cutoff_date = pd.to_datetime('2003-01-09')
                    df_MB_2.index = df_MB_2.index.map(lambda x: x + pd.Timedelta(hours=0) if x < cutoff_date else x + pd.Timedelta(hours=1))
                    df_AWS_2.index = df_AWS_2.index.map(lambda x: x + pd.Timedelta(hours=0) if x < cutoff_date else x + pd.Timedelta(hours=1))
                    df_MB.index = df_MB.index.map(lambda x: x + pd.Timedelta(hours=0) if x < cutoff_date else x + pd.Timedelta(hours=1))
                    df_AWS.index = df_AWS.index.map(lambda x: x + pd.Timedelta(hours=0) if x < cutoff_date else x + pd.Timedelta(hours=1))
                    df_MB_2 = df_MB_2.drop_duplicates(keep='first')
                    df_AWS_2 = df_AWS_2.drop_duplicates(keep='first')
                    df_MB = df_MB.drop_duplicates(keep='first')
                    df_AWS = df_AWS.drop_duplicates(keep='first')
                elif ID == 'AWS09':
                    cutoff_date = pd.to_datetime('2003-01-13')
                    df_MB_2.index = df_MB_2.index.map(lambda x: x + pd.Timedelta(hours=0) if x < cutoff_date else x + pd.Timedelta(hours=1))
                    df_AWS_2.index = df_AWS_2.index.map(lambda x: x + pd.Timedelta(hours=0) if x < cutoff_date else x + pd.Timedelta(hours=1))
                    df_MB.index = df_MB.index.map(lambda x: x + pd.Timedelta(hours=0) if x < cutoff_date else x + pd.Timedelta(hours=1))
                    df_AWS.index = df_AWS.index.map(lambda x: x + pd.Timedelta(hours=0) if x < cutoff_date else x + pd.Timedelta(hours=1))
                    df_MB_2 = df_MB_2[~df_MB_2.index.duplicated()]
                    df_AWS_2 = df_AWS_2[~df_AWS_2.index.duplicated()]
                    df_AWS = df_AWS[~df_AWS.index.duplicated()]
                    df_MB = df_MB[~df_MB.index.duplicated()]
                else:
                    df_MB_2 = df_MB_2.shift(periods=1)
                    df_AWS_2 = df_AWS_2.shift(periods=1)
                    df_MB = df_MB.shift(periods=1)
                    df_AWS = df_AWS.shift(periods=1)
            elif dT == '2H':
                df_MB_2 = df_MB_2.shift(2,'h')
                df_AWS_2 = df_AWS_2.shift(2,'h')
                df_MB = df_MB.shift(2,'h')
                df_AWS = df_AWS.shift(2,'h')
    else:
        print('WARNING: no EBM data found, continuing without...')
        df_MB  = pd.DataFrame(np.nan,index=df_L1A.index, columns=column_names_MB)  
        df_AWS = pd.DataFrame(np.nan,index=df_L1A.index, columns=column_names_AWS)
        df_MB_2  = pd.DataFrame(np.nan,index=df_L1A.index, columns=column_names_MB_2)  
        df_AWS_2 = pd.DataFrame(np.nan,index=df_L1A.index, columns=column_names_AWS_2)
        df_SNOW = pd.DataFrame(np.nan,index=df_L1A.index, columns=column_names_SNOW)

    if not os.path.exists(odir):
        os.makedirs(odir)
    os.chdir(odir)

    df = pd.concat([df_AWS,df_MB,df_L1B,df_L1A,df_AWS_2,df_MB_2],axis=1)

    if ID == 'AWS13':
        df['U'] = df['U_L1A']


    if 'LON' not in list(df.keys()):
        df['LON'] = np.full(len(df.T0),np.nan)
    if 'LAT' not in list(df.keys()):
        df['LAT'] = np.full(len(df.T0),np.nan)
    if 'T4b' not in list(df.keys()):
        df['T4b'] = np.full(len(df.T0),np.nan)
    if 'T5b' not in list(df.keys()):
        df['T5b'] = np.full(len(df.T0),np.nan)
    
    df.rename(columns={'albedo_acc': 'alb', 'zm': 'z_boom_filtered', 'WD': 'wdir', 'T0': 't', 'T0_raw': 't_uncorr','qv0': 'q','RH_raw': 'rh_uncorr','RH': 'rh','U': 'wspd','Umax': 'wspd_max', 'p0': 'p', \
        'SWd': 'swd_notiltcorr','SWu': 'swu','LWd_raw': 'lwd_uncorr','LWd': 'lwd','LWu_raw': 'lwu_uncorr','LWu': 'lwu','SWd_acc': 'swd','TILTX': 'tiltx','TILTY': 'tilty', \
            'Vbat': 'vbat','Tcnr': 't_cnr','Tlogger': 't_logger','T1a':'t1a','T2a':'t2a','T3a':'t3a','T4a':'t4a','T5a':'t5a', \
                'T1b':'t1b','T2b':'t2b','T3b':'t3b','T4b':'t4b','T5b':'t5b','quality':'data_source','SWin_TOA': 'swd_toa','SWin_max': 'swd_max'}, inplace=True)
    df = df[(df['t'].notna()) | df['rh'].notna() | df['wspd'].notna() | df['p'].notna() | df['swd'].notna() | df['swu'].notna() | df['lwd'].notna() | df['lwu'].notna() | df['z_boom'].notna()]

    df['swd'] = df['swd'].fillna(df['swd_notiltcorr'])
    df['l_mo'] = (df['ustar']**2) / ( (0.4*9.81/df['t']) * (df['thstar'] + df['t'] * ((1.-0.622)/0.622) * df['qstar'] ) )
    df['topwater_2'] = 0.1*df['topwater_2']/df['topsnow_2']
    df['topwater_2'][df['topwater_2'] > df['totwater_2']] = df['totwater_2'][df['topwater_2'] > df['totwater_2']]

    column_names_out = ['t','t_2m','t_uncorr','q','q_2m','rh','rh_2m','rh_uncorr','wspd','wspd_10m','wspd_max','wdir','p','z_boom','z_boom_filtered','surface_level','surface_level_zm', \
        'swd_notiltcorr','swd','swu','lwd_uncorr','lwd','lwu_uncorr','lwu','lwu_mod','shf_mod','lhf_mod','ghf_mod','spen_mod','melt_mod_2','lwu_mod_2','shf_mod_2','lhf_mod_2','ghf_mod_2','t_s_obs','t_s_mod_2', \
        'surfmelt_2','totmelt_2','subl_2','cum_surfmelt_2','cum_totmelt_2', 'totwater_2', 'topwater_2', \
        'ustar','thstar','qstar','l_mo','z0m','alb','cc','sza','az','tiltx','tilty','vbat','t_logger','t_cnr', 'z',\
        't1a','t2a','t3a','t4a','t5a','t1b','t2b','t3b','t4b','t5b', 'errorflag','data_source','paramerror','LAT_L1A','LON_L1A','precip_mod','swd_uncorr','swu_uncorr','swd_max','swd_toa']

    df_out = df[column_names_out]


    df_out['LWC_2'] = df_out['totwater_2']
    df_out['LWC_top10cm_2'] = df_out['topwater_2']
    df_out['errorflag'] = df_out['paramerror']
    df_out['q'] = 1e3*df_out['q']
    df_out['p'] = df_out['p']/1e2
    df_out['t'] = df_out['t']-273.15
    df_out['t_uncorr'] = df_out['t_uncorr']-273.15
    df_out['t_s_obs'] = df_out['t_s_obs']-273.15
    df_out['t_s_mod_2'] = df_out['t_s_mod_2']-273.15
    df_out['t_2m'] = df_out['t_2m']-273.15
    df_out['z0m'] = 1000*df_out['z0m']
    df_out['t1a'] = df_out['t1a']-273.15
    df_out['t2a'] = df_out['t2a']-273.15
    df_out['t3a'] = df_out['t3a']-273.15
    df_out['t4a'] = df_out['t4a']-273.15
    df_out['t5a'] = df_out['t5a']-273.15
    df_out['t1b'] = df_out['t1b']-273.15
    df_out['t2b'] = df_out['t2b']-273.15
    df_out['t3b'] = df_out['t3b']-273.15
    df_out['t4b'] = df_out['t4b']-273.15
    df_out['t5b'] = df_out['t5b']-273.15
    df_out['t_cnr'] = df_out['t_cnr']-273.15
    df_out['t_logger'] = df_out['t_logger']-273.15
    df_out['alb'] = df_out['alb'].where(df_out.sza<=70)

    columns_out_nead = ['t','t_2m','t_uncorr','q','q_2m','rh','rh_2m','rh_uncorr','wspd','wspd_10m','wspd_max','wdir','p', \
        'swd','swd_notiltcorr','swd_uncorr','swd_max','swd_toa','swu','swu_uncorr','lwd','lwd_uncorr','lwu','lwu_uncorr', \
        't1a','t2a','t3a','t4a','t5a','t1b','t2b','t3b','t4b','t5b', \
        'z_boom','z_boom_filtered','surface_level_zm','z', \
        'vbat','t_cnr','t_logger','LAT_L1A','LON_L1A','alb','sza','t_s_obs',\
        'shf_mod','lhf_mod','ghf_mod','cc','lwu_mod_2','shf_mod_2','lhf_mod_2','ghf_mod_2','t_s_mod_2','melt_mod_2','totmelt_2','cum_totmelt_2','subl_2','precip_mod', 'LWC_2' ,'LWC_top10cm_2','errorflag']

    columns_out_nead_sub = ['t','t_2m','q','q_2m','rh','rh_2m','wspd','wspd_10m','wspd_max','wdir','p', \
        'swd','swu','lwd','lwu', \
        'z_boom','z_boom_filtered','surface_level_zm', \
        'LAT_L1A','LON_L1A','alb','t_s_obs','errorflag']
    
    df_out_nead = df_out[columns_out_nead]
    df_out_nead_sub = df_out[columns_out_nead_sub]

    file_out_nead  = odir + LOC + '_' + ID + '_hourly_all' + ".csv"
    file_out_nead_sub  = odir + LOC + '_' + ID + '_hour' + ".csv"

    # Get AWS location
    AWSloc = pd.read_csv(AWSfile,skiprows=0,delim_whitespace=True)
    AWSid =  AWSloc.loc[AWSloc['STATION'] == ID]['STATION'].values[0]
    AWSlat =  AWSloc.loc[AWSloc['STATION'] == ID]['LATITUDE'].values[0]
    AWSlon =  AWSloc.loc[AWSloc['STATION'] == ID]['LONGITUDE'].values[0]
    AWSalt =  AWSloc.loc[AWSloc['STATION'] == ID]['ELEVATION'].values[0]

    header = '\n'.join(
        ['# NEAD 1.0 UTF-8',
        '# [METADATA]',
        '# station_id       = ' + str(AWSid),
        '# latitude         = ' + str(AWSlat), 
        '# longitude        = ' + str(AWSlon),
        '# altitude         = ' + str(AWSalt),
        '# nodata           = ',
        '# field_delimiter  = ,',
        '# tz               = 0', 
        '# doi              = ', 
        '# [FIELDS]',
        '# fields           = time,t,t2m,t_uncorr,q,q2m,rh,rh2m,rh_uncorr,wspd,wspd10m,wspdmax,wdir,p,SWd,SWd_notiltcorr,SWd_uncorr,SWd_max,SWd_toa,SWu,SWu_uncorr,LWd,LWd_uncorr,LWu,LWu_uncorr,Tsub1a,Tsub2a,Tsub3a,Tsub4a,Tsub5a,Tsub1b,Tsub2b,Tsub3b,Tsub4b,Tsub5b,z_surf,z_surf_filtered,cum_surface_height_zboom,z_u,Vbat,TCNR,Tlogger,LAT,LON,alb,sza,Ts_obs,SHFdown,LHFdown,GHFup,cloud_cover,LWu_mod,SHFdown_mod,LHFdown_mod,GHFup_mod,Ts_mod,meltE,melt,cumulative_melt,sublimation,accumulation,LWC,LWC_10cm,errorflag',
        '# units            = -,DegreeC,DegreeC,DegreeC,g/kg,g/kg,%,%,%,m/s,m/s,m/s,deg,hPa,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,m,m,m,m,V,DegreeC,DegreeC,deg,deg,-,deg,DegreeC,W/m^2,W/m^2,W/m^2,-,W/m^2,W/m^2,W/m^2,W/m^2,DegreeC,W/m^2,kg/m^2/dt,kg/m^2,kg/m^2/dt,kg/m^2/dt,kg/m^2,kg/m^2,-',
        '# [DATA]',
        '',
        ]
    )
    df_out_nead_write = df_out_nead
    df_out_nead_write = df_out_nead_write.dropna(subset='errorflag')
    
    for name in list(df_out_nead_write.keys()):
        if name in ['LAT_L1A', 'LON_L1A']:
            df_out_nead_write[name] = df_out_nead_write[name].map(lambda x: f'{float(x):.6f}')
        elif name in ['wdir']:
            df_out_nead_write[name] = df_out_nead_write[name].map(lambda x: f'{float(x):.2f}')
        elif name in ['errorflag']:
             df_out_nead_write[name] = df_out_nead_write[name].astype(int) 
        else:
            df_out_nead_write[name] = df_out_nead_write[name].map(lambda x: f'{float(x):.4f}')

    df_out_nead_write = df_out_nead_write.replace('nan', np.NaN)
    with open(file_out_nead, 'w') as ict:
        for line in header:
            ict.write(line)
        df_out_nead_write.to_csv(ict,index=True,header=False,sep=',',na_rep='')

    header = '\n'.join(
        ['# NEAD 1.0 UTF-8',
        '# [METADATA]',
        '# station_id       = ' + str(AWSid),
        '# latitude         = ' + str(AWSlat), 
        '# longitude        = ' + str(AWSlon),
        '# altitude         = ' + str(AWSalt),
        '# nodata           = ',
        '# field_delimiter  = ,',
        '# tz               = 0', 
        '# doi              = ', 
        '# [FIELDS]',
        '# fields           = time,t,t2m,q,q2m,rh,rh2m,wspd,wspd10m,wspdmax,wdir,p,SWd,SWu,LWd,LWu,z_surf,z_surf_filtered,cum_surface_height_zboom,LAT,LON,alb,Ts_obs,errorflag',
        '# units            = -,DegreeC,DegreeC,g/kg,g/kg,%,%,m/s,m/s,m/s,deg,hPa,W/m^2,W/m^2,W/m^2,W/m^2,m,m,m,deg,deg,-,DegreeC,-',
        '# [DATA]',
        '',
        ]
    )

    # df_out_nead_sub = df_out_nead_sub.map(lambda x: '%2.1f' % x)
    float_format_sub = ['%.4f', '%.4f', '%.4f' ,'%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.2f' ,'%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.6f', '%.6f', '%.4f', '%.4f', '%i' ]
    rounding_sub = [4, 4 ,4, 4, 4, 4, 4, 4, 4, 2 ,4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 4, 4 , 1]
    df_out_nead_sub = df_out_nead_sub.round({c:r for c, r in zip(df_out_nead_sub.columns, rounding_sub)})
    df_out_nead_sub = df_out_nead_sub.dropna(subset='errorflag')
    for name in list(df_out_nead_sub.keys()):
        if name in ['LAT_L1A', 'LON_L1A']:
            df_out_nead_sub[name] = df_out_nead_sub[name].map(lambda x: f'{float(x):.6f}')
        elif name in ['wdir']:
            df_out_nead_sub[name] = df_out_nead_sub[name].map(lambda x: f'{float(x):.2f}')
        elif name in ['errorflag']:
             df_out_nead_sub[name] = df_out_nead_sub[name].astype(int) 
        else:
            df_out_nead_sub[name] = df_out_nead_sub[name].map(lambda x: f'{float(x):.4f}')
    df_out_nead_sub = df_out_nead_sub.replace('nan', np.NaN)

    with open(file_out_nead_sub, 'w') as ict:
        for line in header:
            ict.write(line)
        df_out_nead_sub.to_csv(ict,index=True,header=False,sep=',',na_rep='') #,float_format='%.6f'

    # Compute daily file

    columns_out_nead_daily = ['t','t_2m','q','q_2m','rh','rh_2m','wspd','wspd_10m','wspd_max','wdir','p', \
        'swd','swd_max','swd_toa','swu','lwd','lwu', \
        't1a','t2a','t3a','t4a','t5a','t1b','t2b','t3b','t4b','t5b', \
        'z_boom','z_boom_filtered','surface_level_zm','z', \
        'vbat','t_cnr','t_logger','LAT_L1A','LON_L1A','alb','t_s_obs',\
        'shf_mod','lhf_mod','ghf_mod','cc','lwu_mod_2','shf_mod_2','lhf_mod_2','ghf_mod_2','t_s_mod_2','melt_mod_2','totmelt_2','cum_totmelt_2','subl_2',\
            'precip_mod', 'LWC_2' ,'LWC_top10cm_2']

    df_out_nead = df_out_nead[columns_out_nead_daily]
    df_out_nead_daily = df_out_nead.resample("1D",label='left').apply(np.mean)   
    df_out_nead_daily_pnan = df_out_nead.resample("1D",label='left').count()
    df_out_nead_daily_len = df_out_nead.resample("1D",label='left').size()
    df_out_nead_daily['Tsmax'] = df_out_nead.resample("1D",label='left').apply(np.max)['t_s_obs']
    df_out_nead_daily['Tmax'] = df_out_nead.resample("1D",label='left').apply(np.mean)['t']
    df_out_nead_daily['validsamples_SEB_per'] = 100*df_out_nead_daily_pnan['shf_mod']/df_out_nead_daily_len
    df_out_nead_daily['validsamples_AWS_per'] = 100*df_out_nead_daily_pnan['t']/df_out_nead_daily_len

    file_out_nead = odir + LOC + '_' + ID + '_daily_all' + ".csv"

    # Get AWS location
    AWSloc = pd.read_csv(AWSfile,skiprows=0,delim_whitespace=True)
    AWSid =  AWSloc.loc[AWSloc['STATION'] == ID]['STATION'].values[0]
    AWSlat =  AWSloc.loc[AWSloc['STATION'] == ID]['LATITUDE'].values[0]
    AWSlon =  AWSloc.loc[AWSloc['STATION'] == ID]['LONGITUDE'].values[0]
    AWSalt =  AWSloc.loc[AWSloc['STATION'] == ID]['ELEVATION'].values[0]

    header = '\n'.join(
        ['# NEAD 1.0 UTF-8',
        '# [METADATA]',
        '# station_id       = ' + str(AWSid),
        '# latitude         = ' + str(AWSlat), 
        '# longitude        = ' + str(AWSlon),
        '# altitude         = ' + str(AWSalt),
        '# nodata           = ',
        '# field_delimiter  = ,',
        '# tz               = 0', 
        '# doi              = ', 
        '# [FIELDS]',
        '# fields           = time,t,t2m,q,q2m,rh,rh2m,wspd,wspd10m,wspdmax,wdir,p,SWd,SWd_max,SWd_toa,SWu,LWd,LWu,Tsub1a,Tsub2a,Tsub3a,Tsub4a,Tsub5a,Tsub1b,Tsub2b,Tsub3b,Tsub4b,Tsub5b,z_surf,z_surf_filtered,cum_surface_height_zboom,z_u,Vbat,TCNR,Tlogger,LAT,LON,alb,Ts_obs,SHFdown,LHFdown,GHFup,cloud_cover,LWu_mod,SHFdown_mod,LHFdown_mod,GHFup_mod,Ts_mod,meltE,melt,cumulative_melt,sublimation,accumulation,LWC,LWC_10cm,Tsmax,Tmax,validsamples_SEB_per,validsamples_AWS_per',
        '# units            = -,DegreeC,DegreeC,g/kg,g/kg,%,%,m/s,m/s,m/s,deg,hPa,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,m,m,m,m,V,DegreeC,DegreeC,deg,deg,-,DegreeC,W/m^2,W/m^2,W/m^2,-,W/m^2,W/m^2,W/m^2,W/m^2,DegreeC,W/m^2,kg/m^2/dt,kg/m^2,kg/m^2/dt,kg/m^2/dt,kg/m^2,kg/m^2,degreeC,degreeC,-,-',
        '# [DATA]',
        '',
        ]
    )
    df_out_nead_daily = df_out_nead_daily.dropna(subset='validsamples_SEB_per')
    for name in list(df_out_nead_daily.keys()):
        if name in ['LAT_L1A', 'LON_L1A']:
            df_out_nead_daily[name] = df_out_nead_daily[name].map(lambda x: f'{float(x):.6f}')
        elif name in ['wdir']:
            df_out_nead_daily[name] = df_out_nead_daily[name].map(lambda x: f'{float(x):.2f}')
        elif name in ['errorflag','validsamples_SEB_per','validsamples_AWS_per']:
             df_out_nead_daily[name] = df_out_nead_daily[name].astype(int) 
        else:
            df_out_nead_daily[name] = df_out_nead_daily[name].map(lambda x: f'{float(x):.4f}')

    df_out_nead_daily = df_out_nead_daily.replace('nan', np.NaN)

    
    with open(file_out_nead, 'w') as ict:
        for line in header:
            ict.write(line)
        df_out_nead_daily.to_csv(ict,index=True,header=False,sep=',',na_rep='')

#_______________________________________________________
def EBM_AWS_to_csv_GRL(nml):
    """
    Merge SEB model output, AWS data and flux data for Greenland, and writes to single csv file in NEAD format. 
    Also save a different file with selected data and with daily averages
    
    Input
    ----------
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    """
    
    ID		= nml['global']['ID']    
    LOC		= nml['global']['LOC']    
    sensor  = nml['global']['sensor']  

    EBMdir  = nml['AWScorr']['EBMdir']
    EBMdir2  = nml['AWScorr']['EBMdir2']
    lshift_time_EBM = nml['AWScorr']['lshift_time_EBM']
    L1Adir  = nml['AWScorr']['DIR'] + '/L1A/'
    L1Bdir  = nml['AWScorr']['DIR'] + '/L1B/'
    odir  = nml['AWScorr']['DIR'] + '../../FINAL/'
    
    yyyymmdd_start = nml['AWScorr']['yyyymmdd_start']
    yyyymmdd_end = nml['AWScorr']['yyyymmdd_end']
    AWSfile = nml['AWScorr']['file_AWS_locations']
    heights_dir = nml['AWScorr']['heights_dir']
    L3B_dir = nml['AWScorr']['L3B_dir']
    lmetafile = nml['AWScorr']['lmetafile']

    toffset_min_L3B = nml['AWScorr']['toffset_min_L3B']


    column_names_AWS = ['date','hour','year','day','t_a','t_2m','th_a','t_s_mod','t_s_obs','dT0', \
        'Tinv', 'q_a','q_2m','q_s','rh_2m','wspd_a','wspd_10m','p_a','cc','alb_mod','alb_mod_m','sza_mod', \
            'rest','melt_mod','swnet','swd_m','swu_m','spen_mod','spen_sin','lwnet','lwd_m', \
                'lwu_mod','lwu_m','rnet','shf_mod','lhf_mod','ghf_mod','ch','cq','ustar_mod', \
                    'thstar','qstar','psim','psih','psiq','zt','zm_m','z0m','z0h','z0q','Hice','dHice','errorflag','awsid','paramerror_AWS']

    column_names_MB = ['date','hour','hsnow_obs','precip_obs','drift_obs', \
        'acc_obs','icemelt_obs','cumicemelt_obs','hsnowmod_mod','precip_mod', \
            'drift_mod','accsnow_mod', 'icemelt_mod','cumicemelt','hmass', 'totwater', 'topwater', 'topsnow', 'topmass', \
                'cum_totmelt','totmelt','cum_surfmelt','surfmelt','runoff','subl','slushdepth','surfwater','error_MB','errorobs']

    column_names_SNOW = ['date','hour','year','il','depth','thickness','temp','dens','mass','water', \
        'ice', 'Spenetration','cp','irrwater','energy','grainsize','id','hsnowmod','hsnow','error']
    

    column_names_LAY1 = ['date','hour','year','day','thickness','temp','T0','dens','mass','water','ice','Spenetration','cp','K','irrwater','energy','grainsize','id']

    column_names_AWS_2 = [s + '_2' for s in column_names_AWS]
    column_names_MB_2 = [s + '_2' for s in column_names_MB]


    os.chdir(L1Adir)
    file_L1A = L1Adir 
    file_L1A = (glob.glob('*L1A*' + "*nc"))[0]
    
    os.chdir(L1Bdir)
    file_L1B = L1Bdir 
    file_L1B = (glob.glob('*L1B*' + "*nc"))[0]

    os.chdir(L1Bdir)
    ds_L1B           = xr.open_dataset(file_L1B)
    df_L1B           = ds_L1B.to_dataframe()
    grouped = df_L1B.groupby(pd.Grouper(freq='1h',label='right'), group_keys=False)
    df_L1B   = grouped.mean()

    # Removed filled data
    if not 'lhum' in list(df_L1B.keys()):
        df_L1B['lhum'] = np.full(len(df_L1B.T0),1)
    if not 'lbaro' in list(df_L1B.keys()):
        df_L1B['lbaro'] = np.full(len(df_L1B.T0),1)
    if not 'lanemo' in list(df_L1B.keys()):
        df_L1B['lanemo'] = np.full(len(df_L1B.T0),1)
    if not 'Umax' in list(df_L1B.keys()):
        df_L1B['Umax'] = np.full(len(df_L1B.T0),np.nan)

    df_L1B['RH'] = df_L1B['RH'].where(df_L1B.lhum==1)
    df_L1B['qv0'] = df_L1B['qv0'].where(df_L1B.lhum==1)
    df_L1B['p0'] = df_L1B['p0'].where(df_L1B.lbaro==1)
    df_L1B['U'] = df_L1B['U'].where(df_L1B.lanemo==1)
    df_L1B['Umax'] = df_L1B['Umax'].where(df_L1B.lanemo==1)
    df_L1B['WD'] = df_L1B['WD'].where(df_L1B.lanemo==1)
    if ID == 'AWS13':
        df_L1B['U'] = np.full(len(df_L1B.U),np.nan)




    if 'Vbat'  not in list(df_L1B.keys()):
        df_L1B['Vbat']  = np.full(len(df_L1B),np.nan)
    if 'Tlogger'  not in list(df_L1B.keys()):
        df_L1B['Tlogger']  = np.full(len(df_L1B),np.nan)
    if 'Tcnr'  not in list(df_L1B.keys()):
        df_L1B['Tcnr']  = np.full(len(df_L1B),np.nan)
    if 'T1a'  not in list(df_L1B.keys()):
        df_L1B['T1a']  = np.full(len(df_L1B),np.nan)
    if 'T2a'  not in list(df_L1B.keys()):
        df_L1B['T2a']  = np.full(len(df_L1B),np.nan)
    if 'T3a'  not in list(df_L1B.keys()):
        df_L1B['T3a']  = np.full(len(df_L1B),np.nan)
    if 'T4a'  not in list(df_L1B.keys()):
        df_L1B['T4a']  = np.full(len(df_L1B),np.nan)
    if 'T5a'  not in list(df_L1B.keys()):
        df_L1B['T5a']  = np.full(len(df_L1B),np.nan)
    if 'T1b'  not in list(df_L1B.keys()):
        df_L1B['T1b']  = np.full(len(df_L1B),np.nan)
    if 'T2b'  not in list(df_L1B.keys()):
        df_L1B['T2b']  = np.full(len(df_L1B),np.nan)
    if 'T3b'  not in list(df_L1B.keys()):
        df_L1B['T3b']  = np.full(len(df_L1B),np.nan)
    if 'T4b'  not in list(df_L1B.keys()):
        df_L1B['T4b']  = np.full(len(df_L1B),np.nan) 
    if 'T5b'  not in list(df_L1B.keys()):
        df_L1B['T5b']  = np.full(len(df_L1B),np.nan)
    if 'quality'  not in list(df_L1B.keys()):
        df_L1B['quality']  = np.full(len(df_L1B),np.nan)

    os.chdir(L1Adir)
    ds_L1A           = xr.open_dataset(file_L1A)

    column_names_out_L1A = ['zm','LAT','LON','U_L1A','swd_uncorr','swu_uncorr']
    

    # Only keep data after start time
    ds_L1A  = ds_L1A.sel(time=slice(yyyymmdd_start, yyyymmdd_end))
    df_L1A           = ds_L1A.to_dataframe()

    if 'U'  not in list(df_L1A.keys()):
        df_L1A['U']  = np.full(len(df_L1A),np.nan)
    if sensor == 'PROMICE_v04':
        df_L1A.rename(columns={'wspd_u': 'U_L1A','dsr': 'swd_uncorr','usr': 'swu_uncorr'}, inplace=True)
    elif sensor == 'GCNet':
        df_L1A.rename(columns={'VW2': 'U_L1A','D_GLOBAL': 'swd_uncorr','U_GLOBAL': 'swu_uncorr'}, inplace=True)
    else:
        df_L1A.rename(columns={'U': 'U_L1A','SWd': 'swd_uncorr','SWu': 'swu_uncorr'}, inplace=True)
    if not 'zm' in list(df_L1A.keys()):
        if not 'z_boom_u' in list(df_L1A.keys()):
            df_L1A['zm'] = np.full(len(df_L1A.HW2),np.nan)
        else:
            df_L1A['zm'] = df_L1A['z_boom_u']
    if not 'zm_factor' in list(df_L1A.keys()):
        df_L1A['zm_factor'] = np.full(len(df_L1A.zm),1.0)
    df_L1A['zm'] = df_L1A['zm']*df_L1A['zm_factor']
    if not 'LAT' in list(df_L1A.keys()):
        df_L1A['LAT'] = np.full(len(df_L1A.zm),np.nan)
        df_L1A['LON'] = np.full(len(df_L1A.zm),np.nan)

        
    df_L1A = df_L1A[column_names_out_L1A]
    grouped = df_L1A.groupby(pd.Grouper(freq='1h',label='right'), group_keys=False)
    df_L1A   = grouped.mean()
    df_L1A.zm[df_L1A.zm < -900]   = np.nan
    df_L1A.rename(columns={'zm': 'z_boom'}, inplace=True)

    if os.path.exists(EBMdir):
        os.chdir(EBMdir)
        file_AWS = (glob.glob('*AWS.txt'))[0]
        file_MB = (glob.glob('*MB.txt'))[0]
        file_LAY1 = glob.glob('*LAY1.txt')[0]


        # file_SNOW = (glob.glob('*SNOW.txt'))[0]

        df_AWS           = pd.read_csv(file_AWS,delim_whitespace = True, names = column_names_AWS,na_values = -998,skiprows=1)
        df_MB            = pd.read_csv(file_MB,delim_whitespace = True, names = column_names_MB,na_values = -998,skiprows=1)
        # df_SNOW          = pd.read_csv(file_SNOW,delim_whitespace = True, names = column_names_SNOW,na_values = -998,skiprows=1)

        df_AWS.index              = pd.to_datetime(df_AWS['date'] + ' ' +  df_AWS['hour'])
        df_MB.index               = pd.to_datetime(df_MB['date'] + ' ' +  df_MB['hour'])
        # df_SNOW.index             = pd.to_datetime(df_SNOW['date'] + ' ' +  df_SNOW['hour'])

        df_AWS.index.names         = ['time']
        df_MB.index.names         = ['time']

        df_MB = df_MB.where(df_MB.error_MB == 0)
        df_AWS = df_AWS.where(df_AWS.errorflag == 0)
        # df_SNOW = df_AWS.where(df_SNOW.erroflag == 0)
        # df_SNOW.index.names         = ['time']   

        df_4           = pd.read_csv(file_LAY1,delim_whitespace = True, names = column_names_LAY1,na_values = -998,skiprows=1)
        df_4.index               = pd.to_datetime(df_4['date'] + ' ' +  df_4['hour'])
        df_4.index.names         = ['time']
        df_4 = df_4[['dens']] 

        os.chdir(EBMdir2)
        file_AWS_2 = (glob.glob('*AWS.txt'))[0]
        file_MB_2 = (glob.glob('*MB.txt'))[0]
        # file_SNOW = (glob.glob('*SNOW.txt'))[0]

        df_AWS_2           = pd.read_csv(file_AWS_2,delim_whitespace = True, names = column_names_AWS_2,na_values = -998,skiprows=1)
        df_MB_2            = pd.read_csv(file_MB_2,delim_whitespace = True, names = column_names_MB_2,na_values = -998,skiprows=1)
        # df_SNOW          = pd.read_csv(file_SNOW,delim_whitespace = True, names = column_names_SNOW,na_values = -998,skiprows=1)

        df_AWS_2.index              = pd.to_datetime(df_AWS_2['date_2'] + ' ' +  df_AWS_2['hour_2'])
        df_MB_2.index               = pd.to_datetime(df_MB_2['date_2'] + ' ' +  df_MB_2['hour_2'])
        # df_SNOW.index             = pd.to_datetime(df_SNOW['date'] + ' ' +  df_SNOW['hour'])

        df_AWS_2.index.names         = ['time']
        df_MB_2.index.names         = ['time']


        df_MB_2 = df_MB_2.where(df_MB_2.error_MB_2 == 0)
        df_AWS_2 = df_AWS_2.where(df_AWS_2.errorflag_2 == 0)
        # df_SNOW = df_AWS.where(df_SNOW.erroflag == 0)
        # df_SNOW.index.names         = ['time']    
        if lshift_time_EBM:
            df_MB_2.index = df_MB_2.index + np.timedelta64(60,'m')
            df_AWS_2.index = df_AWS_2.index + np.timedelta64(60,'m')
            df_MB.index = df_MB.index + np.timedelta64(60,'m')
            df_AWS.index = df_AWS.index + np.timedelta64(60,'m')
    else:
        print('WARNING: no EBM data found, continuing without...')
        df_MB  = pd.DataFrame(np.nan,index=df_L1A.index, columns=column_names_MB)  
        df_AWS = pd.DataFrame(np.nan,index=df_L1A.index, columns=column_names_AWS)
        df_MB_2  = pd.DataFrame(np.nan,index=df_L1A.index, columns=column_names_MB_2)  
        df_AWS_2 = pd.DataFrame(np.nan,index=df_L1A.index, columns=column_names_AWS_2)
        df_SNOW = pd.DataFrame(np.nan,index=df_L1A.index, columns=column_names_SNOW)


    # Heights file
    if os.path.exists(heights_dir):
        os.chdir(heights_dir)
        file = glob.glob('*txt')[0]
        df_1 = pd.read_csv(file,delimiter = ',', na_values = 'NAN',header=0)
        df_1 = df_1.set_index(df_1.columns[0])
        df_1.index = pd.to_datetime(df_1.index)
        df_1.index.names = ['time']
        df_1 = df_1.replace(-999,np.NaN)
        grouped = df_1.groupby(pd.Grouper(freq='1h',label='right'), group_keys=False)
        df_1   = grouped.mean()
        if 'ADW'  in list(df_1.keys()): # z_adw_geus zm DT_Avg zstakes z_pt_cor
            df_1['z_adw'] = df_1['ADW']
        else:
            df_1['z_adw'] =  np.full(len(df_1),np.nan)
        if 'ADW_GEUS'  in list(df_1.keys()):
            df_1['z_adw_geus'] = df_1['ADW_GEUS']
        else:
            df_1['z_adw_geus'] = np.full(len(df_1),np.nan)
        if 'zm'  in list(df_1.keys()):
            df_1['z_boom_2'] = df_1['zm']
            df_1.drop(['zm'], axis=1, inplace=True)
        else:
            if 'z_boom_u'  in list(df_1.keys()):
                df_1['z_boom_2'] = df_1['z_boom_u']
            else:
                df_1['z_boom_2'] = np.full(len(df_1),np.nan)
        if 'zstakes'  in list(df_1.keys()):
            df_1['z_stakes_2'] = df_1['zstakes']
        else:
            if 'z_stake'  in list(df_1.keys()):
                df_1['z_stakes_2'] = df_1['z_stake']
            else:
                df_1['z_stakes_2'] = np.full(len(df_1),np.nan)
        if 'z_pt_cor'  in list(df_1.keys()):
            df_1['z_pt_cor'] = df_1['z_pt_cor']
        else:
            df_1['z_pt_cor'] = np.full(len(df_1),np.nan)
        if 'DT_Avg'  in list(df_1.keys()):
            print('DT_Avg ignored') #df_1['z_stakes'] = pd.concat([df_1['z_stakes'], df_1['DT_Avg']])
    else:
        df_1  = pd.DataFrame(np.nan,index=df_L1A.index, columns=['z_adw','z_adw_geus','z_stakes_2','z_pt_cor','z_boom_2'])  


        

    if os.path.exists(L3B_dir):
        os.chdir(L3B_dir)
        file = glob.glob('*.nc')[0]
        if not file:
            print('WARNING: no L3B data found, continuing without...')
            df_3  = pd.DataFrame(np.nan,index=df_L1A.index, columns=['H1_EC','H2_EC','LE_EC','T1','T2','U_EC','WD_EC','z_EC','flag_EC','ustar']) 
        else:
            ds = xr.open_dataset(file)
            ds = ds.drop_dims(['fp_WDbins', 'fp_r','x2D_FP_clim','y2D_FP_clim','tmp2'],errors='ignore') 
            ds = ds.drop_dims(['z0'])
            df_3 = ds.to_dataframe()
            if not 'H1_EC'  in list(df_3.keys()):
                df_3['H1_EC'] = np.full(len(df_3),np.nan)
            if not 'H2_EC'  in list(df_3.keys()):
                df_3['H2_EC'] = np.full(len(df_3),np.nan)
            if not 'LE_EC'  in list(df_3.keys()):
                df_3['LE_EC'] = np.full(len(df_3),np.nan)
            df_3 = df_3[['H1_EC','H2_EC','LE_EC','T1','T2','U_EC','WD_EC','z_EC','flag_EC','ustar']]
            # df_3.drop(['ADW','zm'] , axis=1, inplace=True)
    else:
        df_3  = pd.DataFrame(np.nan,index=df_L1A.index, columns=['H1_EC','H2_EC','LE_EC','T1','T2','U_EC','WD_EC','z_EC','flag_EC','ustar'])  
    

    
    grouped = df_3.groupby(pd.Grouper(freq='1h',label='right'), group_keys=False)
    
    df_3   = grouped.mean()
    df_3.index = df_3.index + np.timedelta64(toffset_min_L3B,'m')

    if not os.path.exists(odir):
        os.makedirs(odir)
    os.chdir(odir)

    df = pd.concat([df_AWS,df_MB,df_L1B,df_L1A,df_AWS_2,df_MB_2,df_1,df_4,df_3],axis=1)



    df['zbmin'] = np.full(len(df.index),0.2)
    df['zsmin']  = np.full(len(df.index),0.2)
    df['zbmax']  = np.full(len(df.index),5.0)
    df['zsmax']  = np.full(len(df.index),10.0)
    df['ldata']  = np.full(len(df.index),1)
    if lmetafile:
        metafile = glob.glob(L1Adir + '*meta.csv*')
        meta = pd.read_csv(metafile[0],delimiter=',',header=0,decimal=".",na_values=-999)
        meta = meta.apply(pd.to_numeric, errors='coerce')
        meta.index       = compose_date(meta['year'], meta['month'],meta['day'])
        meta.index.names = ['time']
        meta             = meta[~meta.index.duplicated()]
        meta = meta.loc[meta.index.notnull()]
        # Find metadat for each time stamp
        idx = np.full(len(df.index),0)
        for i in range(len(df.index.values)):
            try:
                idx[i] = meta.index.get_indexer(df.index.values,method='ffill')[i]

                # idx[i] = meta.index.get_loc(ds.time.values[i], method='ffill')
            except:
                idx[i] = meta.index.get_indexer(df.index.values,method='bfill')[i]

                # idx[i] = meta.index.get_loc(ds.time.values[i], method='bfill')
            df['zbmin'].values[i] = meta['zbmin'].iloc[idx[i]]
            df['zsmin'].values[i] = meta['zsmin'].iloc[idx[i]]
            df['zbmax'].values[i] = meta['zbmax'].iloc[idx[i]]
            df['zsmax'].values[i] = meta['zsmax'].iloc[idx[i]]
            df['ldata'].values[i] = meta['ldata'].iloc[idx[i]]

    idx = 24
    q = 1
    # filer sonic ranger
    # dif = abs(df.z_boom_2 - df.z_boom_2.rolling(window=idx).median())
    # MAD = abs(df.z_boom_2  -  df.z_boom_2.rolling(window = idx).median()).rolling(window = idx).median()
    # df.z_boom_2[dif > (q/0.6745)*MAD] = np.nan
    # df.z_boom_2 = df.z_boom_2.interpolate(method='linear', axis=0)

    # dif = abs(df.z_stakes_2 - df.z_stakes_2.rolling(window=idx).median())
    # MAD = abs(df.z_stakes_2  -  df.z_stakes_2.rolling(window = idx).median()).rolling(window = idx).median()
    # df.z_stakes_2[dif > (q/0.6745)*MAD] = np.nan
    # df.z_stakes_2 = df.z_stakes_2.interpolate(method='linear', axis=0)

    # dif = abs(df.z_pt_cor - df.z_pt_cor.rolling(window=idx).median())
    # MAD = abs(df.z_pt_cor  -  df.z_pt_cor.rolling(window = idx).median()).rolling(window = idx).median()
    # df.z_pt_cor[dif > (q/0.6745)*MAD] = np.nan
    # df.z_pt_cor = df.z_pt_cor.interpolate(method='linear', axis=0)

    # dif = abs(df.z_adw - df.z_adw.rolling(window=idx).median())
    # MAD = abs(df.z_adw  -  df.z_adw.rolling(window = idx).median()).rolling(window = idx).median()
    # df.z_adw[dif > (q/0.6745)*MAD] = np.nan
    # df.z_adw = df.z_adw.interpolate(method='linear', axis=0)

    # dif = abs(df.z_adw_geus - df.z_adw_geus.rolling(window=idx).median())
    # MAD = abs(df.z_adw_geus  -  df.z_adw.rolling(window = idx).median()).rolling(window = idx).median()
    # df.z_adw_geus[dif > (q/0.6745)*MAD] = np.nan
    # df.z_adw_geus = df.z_adw_geus.interpolate(method='linear', axis=0)

    if 'zbmin' in df.keys():
        df['z_boom_2'][(df['z_boom_2'])>df['zbmax']] = np.nan     
        df['z_boom_2'][(df['z_boom_2'])<df['zbmin']] = np.nan  
        df['z_boom_2'][(df['z_boom_2'])<0.1] = np.nan  
        df['z_boom_2'][(df['z_boom_2'])>30] = np.nan  
    if 'zsmax' in df.keys():
        df['z_stakes_2'][(df['z_stakes_2'])>df['zsmax']] = np.nan     
        df['z_stakes_2'][(df['z_stakes_2'])<df['zsmin']] = np.nan  
        df['z_stakes_2'][(df['z_stakes_2'])<0.1] = np.nan  
        df['z_stakes_2'][(df['z_stakes_2'])>30] = np.nan  

    if 'lzboom' in df.keys():
        df['z_boom_2'][df['lzboom'] < 1] = np.nan
    if 'lzstake' in df.keys():
        df['z_stakes_2'][df['lzstake'] < 1] = np.nan

    m = (np.gradient(df.T0) < 0.6) & (df.U > 1) & (df.flag_EC == 0) & (df.ustar > 0.1)  & (df.H1_EC > -500) & (df.H1_EC < 100)
    df['SHF_obs']  = -df.H1_EC.where(m,np.nan)
    df['SHF_obs_c']  = -df.H2_EC.where(m,np.nan)
    m2 = (np.gradient(df.T0) < 0.6) & (df.U > 1) & (df.flag_EC == 0) & (df.ustar > 0.1)  & (df.H1_EC > -500) & (df.H1_EC < 100) & (df.LE_EC > -500) & (df.LE_EC < 500)
    df['LE_obs']  = -df.LE_EC.where(m2,np.nan)

    if 'LON' not in list(df.keys()):
        df['LON'] = np.full(len(df.T0),np.nan)
    if 'LAT' not in list(df.keys()):
        df['LAT'] = np.full(len(df.T0),np.nan)
    if 'T4b' not in list(df.keys()):
        df['T4b'] = np.full(len(df.T0),np.nan)
    if 'T5b' not in list(df.keys()):
        df['T5b'] = np.full(len(df.T0),np.nan)
    if 'T0_raw' not in list(df.keys()):
        df['T0_raw'] = np.full(len(df.T0),np.nan)
    
    df.rename(columns={'albedo_acc': 'alb','zm': 'z_boom_filtered', 'WD': 'wdir', 'T0': 't', 'T0_raw': 't_uncorr','qv0': 'q','RH': 'rh','U': 'wspd','Umax': 'wspd_max', 'p0': 'p', \
        'SWd': 'swd_notiltcorr','SWu': 'swu','LWd_raw': 'lwd_uncorr','LWd': 'lwd','LWu_raw': 'lwu_uncorr','LWu': 'lwu','SWd_acc': 'swd','TILTX': 'tiltx','TILTY': 'tilty', \
            'Vbat': 'vbat','Tcnr': 't_cnr','Tlogger': 't_logger','T1a':'t1a','T2a':'t2a','T3a':'t3a','T4a':'t4a','T5a':'t5a', \
                'T1b':'t1b','T2b':'t2b','T3b':'t3b','T4b':'t4b','T5b':'t5b','quality':'data_source'}, inplace=True)



    df['l_mo'] = (df['ustar_mod']**2) / ( (0.4*9.81/df['t']) * (df['thstar'] + df['t'] * ((1.-0.622)/0.622) * df['qstar'] ) )
    df['topwater_2'] = 0.1*df['topwater_2']/df['topsnow_2']
    df['topwater_2'][df['topwater_2'] > df['totwater_2']] = df['totwater_2'][df['topwater_2'] > df['totwater_2']]

    column_names_out = ['t','t_2m','t_uncorr','q','q_2m','rh','rh_2m','wspd','wspd_10m','wspd_max','wdir','p','z_boom','z_boom_filtered','surface_level','surface_level_zm','z', \
        'swd_notiltcorr','swd','swu','lwd_uncorr','lwd','lwu_uncorr','lwu','lwu_mod','shf_mod','lhf_mod','ghf_mod','spen_mod','melt_mod_2','lwu_mod_2','shf_mod_2','lhf_mod_2','ghf_mod_2','t_s_obs','t_s_mod_2', \
        'surfmelt_2','totmelt_2','subl_2','cum_surfmelt_2','cum_totmelt_2', 'totwater_2', 'topwater_2', \
        'ustar_mod','thstar','qstar','l_mo','z0m','alb','cc','sza','az','tiltx','tilty','vbat','t_logger','t_cnr', \
        't1a','t2a','t3a','t4a','t5a','t1b','t2b','t3b','t4b','t5b', 'errorflag','data_source','paramerror','LAT','LON','precip_mod','z_boom_2','z_stakes_2','z_adw','z_adw_geus','z_pt_cor','dens',\
        'SHF_obs','LE_obs','SHF_obs_c','flag_EC','U_EC','WD_EC','ustar','T1','T2','z_EC','swd_uncorr','swu_uncorr']

    df = df.where(df['ldata'] != 0.)  
    df = df.dropna(how='all')

    df_out = df[column_names_out]

    if not os.path.exists(EBMdir):
        print('WARNING: using uncorrected SWd ...')
        df_out['swd_corr'] = df_out['swd']

    df_out['t'] = df_out['t']-273.15
    df_out['t_2m'] = df_out['t_2m']-273.15
    df_out['q'] = 1000*df_out['q']
    # df_out['q_2m'] = 1000*df_out['q_2m'] 
    df_out['p'] = df_out['p']/100
    df_out['t_s_obs'] = df_out['t_s_obs']-273.15
    df_out['t_s_mod_2'] = df_out['t_s_mod_2']-273.15
    df_out['z0m'] = 1000*df_out['z0m']
    df_out['t1a'] = df_out['t1a']-273.15
    df_out['t2a'] = df_out['t2a']-273.15
    df_out['t3a'] = df_out['t3a']-273.15
    df_out['t4a'] = df_out['t4a']-273.15
    df_out['t5a'] = df_out['t5a']-273.15
    df_out['t1b'] = df_out['t1b']-273.15
    df_out['t2b'] = df_out['t2b']-273.15
    df_out['t3b'] = df_out['t3b']-273.15
    df_out['t4b'] = df_out['t4b']-273.15
    df_out['t5b'] = df_out['t5b']-273.15
    df_out['t_cnr'] = df_out['t_cnr']-273.15
    df_out['t_logger'] = df_out['t_logger']-273.15
    df_out['T1'] = df_out['T1']-273.15
    df_out['T2'] = df_out['T2']-273.15
    # df_out['alb'] = df_out['alb'].where(df_out.sza<=70)
    df_out['LWC_2'] = np.round(df_out['totwater_2'],decimals=2)
    df_out['LWC_top10cm_2'] = np.round(df_out['topwater_2'],decimals=2)

    columns_out_nead = ['t','t_2m','t_uncorr','q','q_2m','rh','rh_2m','wspd','wspd_10m','wspd_max','wdir','p', \
        'swd','swd_notiltcorr','swd_uncorr','swu','swu_uncorr','lwd','lwd_uncorr','lwu','lwu_uncorr', \
        't1a','t2a','t3a','t4a','t5a','t1b','t2b','t3b','t4b','t5b', \
        'z_boom','z_boom_filtered','surface_level_zm','surface_level','z', \
        'vbat','t_cnr','t_logger','LAT','LON','alb','sza','t_s_obs',\
        'shf_mod','lhf_mod','ghf_mod','cc','lwu_mod_2','shf_mod_2','lhf_mod_2','ghf_mod_2','t_s_mod_2','melt_mod_2','totmelt_2','cum_totmelt_2','subl_2',\
            'precip_mod', 'LWC_2' ,'LWC_top10cm_2','z_boom_2','z_stakes_2','z_adw','z_adw_geus','z_pt_cor',\
                'dens','SHF_obs','LE_obs','SHF_obs_c','U_EC','WD_EC','ustar','T1','T2','z_EC']
    variables = 'time,t,t2m,t_uncorr,q,q2m,rh,rh2m,ff,ff10m,ffmax,ffdir,p,SWd,SWd_notiltcorr,SWd_uncorr,SWu,SWu_uncorr,LWd,LWd_uncorr,LWu,LWu_uncorr,Tsub1a,Tsub2a,Tsub3a,Tsub4a,Tsub5a,Tsub1b,Tsub2b,Tsub3b,Tsub4b,Tsub5b,z_surf,z_surf_filtered,cum_surface_height_zboom,cum_surface_height,z_u,Vbat,TCNR,Tlogger,LAT,LON,alb,sza,Ts_obs,SHFdown,LHFdown,GHFup,cloud_cover,LWu_mod,SHFdown_mod,LHFdown_mod,GHFu_mod,Ts_mod,meltE,melt,cumulative_melt,sublimation,accumulation,LWC,LWC_10cm,z_boom,z_stakes,z_adw,z_adw_geus,z_pt_cor,dens,SHFdown_obs,LHFdown_obs,SHFdown_c,ff_EC,ffdir_EC,ustar_EC,Tsonic,Tcouple,z_EC'
    units = '-,DegreeC,DegreeC,DegreeC,g/kg,g/kg,%,%,m/s,m/s,m/s,deg,hPa,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,m,m,m,m,m,V,DegreeC,DegreeC,deg,deg,-,deg,DegreeC,W/m^2,W/m^2,W/m^2,-,W/m^2,W/m^2,W/m^2,W/m^2,DegreeC,W/m^2,kg/m^2/dt,kg/m^2,kg/m^2/dt,kg/m^2/dt,kg/m^2,kg/m^2,m,m,m,m,m,kg/m^3,W/m^2,W/m^2,W/m^2,m/s,deg,m/s,degC,degC,m' 
    rounding = [4, 4, 4 ,4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4 ,4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,4, 4, 4, 4,  4,4, 4, 4,  6, 6, 4, 4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]

    columns_out_nead_sub = ['t','t_2m','q','q_2m','rh','rh_2m','wspd','wspd_10m','wspd_max','wdir','p', \
        'swd','swu','lwd','lwu', \
        't1a','t2a','t3a','t4a','t5a','t1b','t2b','t3b','t4b','t5b', \
        'z_boom_2','z_boom_filtered','surface_level_zm','surface_level', 'z', \
        'LAT','LON','alb','t_s_obs']
    variables_sub = 'time,t,t2m,q,q2m,rh,rh2m,wspd,wspd10m,wspdmax,wdir,p,SWd,SWu,LWd,LWu,Tsub1a,Tsub2a,Tsub3a,Tsub4a,Tsub5a,Tsub1b,Tsub2b,Tsub3b,Tsub4b,Tsub5b,z_surf,z_surf_filtered,cum_surface_height_zboom,cum_surface_height,z_u,LAT,LON,alb,Ts_obs'
    units_sub = '-,DegreeC,DegreeC,g/kg,g/kg,%,%,m/s,m/s,m/s,deg,hPa,W/m^2,W/m^2,W/m^2,W/m^2,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,m,m,m,m,m,deg,deg,-,DegreeC'
    rounding_sub = [4, 4, 4 ,4, 4, 4, 4, 4, 4, 4, 2,4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,4, 4, 4, 4, 4, 6, 6, 4, 4, 4]
    if (sensor == 'iWS') | (sensor == 'AWS_iWS_GRL'):
        columns_out_nead_sub += ['z_adw']
        variables_sub += ',z_adw'
        units_sub += ',m'
        rounding_sub += [4]
    if sensor == 'PROMICE_v04':
        columns_out_nead_sub += ['z_pt_cor','z_stakes_2']
        variables_sub += ',z_pt_cor,z_stakes'
        units_sub += ',m,m'
        rounding_sub += [4,4]

    df_out_nead = df_out[columns_out_nead]
    df_out_nead = df_out_nead.round({c:r for c, r in zip(df_out_nead.columns, rounding)})

    print(list(df_out_nead.keys()))

    df_out_nead_sub = df_out[columns_out_nead_sub]

    df_out_nead_sub = df_out_nead_sub.round({c:r for c, r in zip(df_out_nead_sub.columns, rounding_sub)})

    file_out_nead  = odir + LOC + '_' + ID + '_hour_all' + ".csv"
    file_out_nead_sub  = odir + LOC + '_' + ID + '_hour' + ".csv"

    # Get AWS location

    AWSloc = pd.read_csv(AWSfile,skiprows=0,delim_whitespace=True)
    AWSid =  AWSloc.loc[AWSloc['STATION'] == ID]['STATION'].values[0]
    AWSlat =  AWSloc.loc[AWSloc['STATION'] == ID]['LATITUDE'].values[0]
    AWSlon =  AWSloc.loc[AWSloc['STATION'] == ID]['LONGITUDE'].values[0]
    AWSalt =  AWSloc.loc[AWSloc['STATION'] == ID]['ELEVATION'].values[0]

    header = '\n'.join(
        ['# NEAD 1.0 UTF-8',
        '# [METADATA]',
        '# station_id       = ' + str(AWSid),
        '# latitude         = ' + str(AWSlat), 
        '# longitude        = ' + str(AWSlon),
        '# altitude         = ' + str(AWSalt),
        '# nodata           = ',
        '# field_delimiter  = ,',
        '# tz               = 0', 
        '# doi              = ', 
        '# [FIELDS]',
        '# fields           = ' + variables,
        '# units            = ' + units,
        '# [DATA]',
        '',
        ]
    )
    with open(file_out_nead, 'w') as ict:
        for line in header:
            ict.write(line)
        df_out_nead.to_csv(ict,index=True,header=False,sep=',',float_format='%6.4f')

    header = '\n'.join(
        ['# NEAD 1.0 UTF-8',
        '# [METADATA]',
        '# station_id       = ' + str(AWSid),
        '# latitude         = ' + str(AWSlat), 
        '# longitude        = ' + str(AWSlon),
        '# altitude         = ' + str(AWSalt),
        '# nodata           = ',
        '# field_delimiter  = ,',
        '# tz               = 0', 
        '# doi              = ', 
        '# [FIELDS]',
        '# fields           = ' + variables_sub,
        '# units            = ' + units_sub,
        '# [DATA]',
        '',
        ]
    )
    with open(file_out_nead_sub, 'w') as ict:
        for line in header:
            ict.write(line)
        df_out_nead_sub.to_csv(ict,index=True,header=False,sep=',')

    # Compute daily file
    def MB(x): # mass balance function...
        if np.all(np.isnan(x)):
            return np.nan
        if len(x) < 1:
            return np.nan
        return x[np.isfinite(x)][-1] - x[np.isfinite(x)][0] #... Simplest measurement ever

    diff_ADW = df_out_nead['z_adw'].resample("1D",label='left').apply(MB)     
    diff_ADW_GEUS = df_out_nead['z_adw_geus'].resample("1D",label='left').apply(MB)     
    diff_zm = df_out_nead['z_boom_2'].resample("1D",label='left').apply(MB)     
    diff_PTA = df_out_nead['z_pt_cor'].resample("1D",label='left').apply(MB)     
    diff_z_stake = df_out_nead['z_stakes_2'].resample("1D",label='left').apply(MB)     

    df_merged_daily = df_out_nead.resample("1D",label='left').apply(np.nanmean)   
    df_merged_daily['Tmax'] = df_out_nead.resample("1D",label='left').apply(np.nanmax)['t']
    df_merged_daily['Tsmax'] = df_out_nead.resample("1D",label='left').apply(np.nanmax)['t_s_obs']
    df_merged_daily_pnan = df_out_nead.resample("1D",label='left').count()
    df_merged_daily['valid_samples_SEB'] = df_merged_daily_pnan['shf_mod']
    df_merged_daily['valid_samples_AWS'] = df_merged_daily_pnan['t']

    df_merged_daily['dz_adw'] = diff_ADW
    df_merged_daily['dz_adw_geus'] = diff_ADW_GEUS
    df_merged_daily['dz_boom'] = diff_zm
    df_merged_daily['dz_stakes'] = diff_z_stake
    df_merged_daily['dz_pt_cor'] = diff_PTA

    df_merged_daily.dz_adw[df_merged_daily.dz_adw > 0.20] = np.nan
    df_merged_daily.dz_adw[df_merged_daily.dz_adw < -0.20] = np.nan

    df_merged_daily.dz_adw_geus[df_merged_daily.dz_adw_geus > 0.20] = np.nan
    df_merged_daily.dz_adw_geus[df_merged_daily.dz_adw_geus < -0.20] = np.nan

    df_merged_daily.dz_boom[df_merged_daily.dz_boom > 0.20] = np.nan
    df_merged_daily.dz_boom[df_merged_daily.dz_boom < -0.20] = np.nan

    df_merged_daily.dz_stakes[df_merged_daily.dz_stakes > 0.20] = np.nan
    df_merged_daily.dz_stakes[df_merged_daily.dz_stakes < -0.20] = np.nan

    df_merged_daily.dz_pt_cor[df_merged_daily.dz_pt_cor > 0.20] = np.nan
    df_merged_daily.dz_pt_cor[df_merged_daily.dz_pt_cor < -0.20] = np.nan

    df_merged_daily['acc_day'] = df_merged_daily.dz_boom.where(df_merged_daily.dz_boom < 0,0)
    df_merged_daily['melt_day_SEB'] = df_merged_daily.swd - df_merged_daily.swu + df_merged_daily.lwd - df_merged_daily.lwu + df_merged_daily.shf_mod + df_merged_daily.lhf_mod + df_merged_daily.ghf_mod
    df_merged_daily['melt_day'] = (24*3600)*(df_merged_daily.melt_day_SEB)/(utils.lf*df_merged_daily.dens)
    df_merged_daily['subl_day'] = (24*3600)*(-df_merged_daily.lhf_mod)/(utils.ls0*df_merged_daily.dens)
    
    df_merged_daily['melt_day'] = df_merged_daily['melt_day'].where(df_merged_daily['Tsmax'] > -2,0)
    df_merged_daily['melt_day_SEB'] = df_merged_daily['melt_day_SEB'].where(df_merged_daily['Tsmax'] > -2,0)

    df_merged_daily['melt_day_dz_adw'] = (-df_merged_daily.dz_adw.where(df_merged_daily.dz_adw < 0) - df_merged_daily['subl_day'])*utils.lf*utils.rho_i/(24*3600)
    df_merged_daily['melt_day_dz_adw_geus'] = (-df_merged_daily.dz_adw_geus.where(df_merged_daily.dz_adw_geus < 0) - df_merged_daily['subl_day'])*utils.lf*utils.rho_i/(24*3600)
    df_merged_daily['melt_day_dz_boom'] = (df_merged_daily.dz_boom.where(df_merged_daily.dz_boom > 0.001) - df_merged_daily['subl_day'])*utils.lf*df_merged_daily.dens/(24*3600)
    df_merged_daily['melt_day_dz_stakes'] = (df_merged_daily.dz_stakes.where(df_merged_daily.dz_stakes > 0.001) - df_merged_daily['subl_day'])*utils.lf*df_merged_daily.dens/(24*3600)
    df_merged_daily['melt_day_dz_pt_cor'] = (-df_merged_daily.dz_pt_cor.where(df_merged_daily.dz_pt_cor < 0) - df_merged_daily['subl_day'])*utils.lf*utils.rho_i/(24*3600)

    df_merged_daily['melt_day_dz_adw'] = df_merged_daily['melt_day_dz_adw'].where((df_merged_daily.Tsmax > -2) & (df_merged_daily.acc_day == 0) & (df_merged_daily.dens > 900))

    df_merged_daily['melt_day_dz_adw'] = df_merged_daily['melt_day_dz_adw'].where((df_merged_daily.Tsmax > -2) & (df_merged_daily.acc_day == 0) & (df_merged_daily.dens > 900))
    df_merged_daily['melt_day_dz_adw_geus'] = df_merged_daily['melt_day_dz_adw_geus'].where((df_merged_daily.Tsmax > -2) & (df_merged_daily.acc_day == 0) & (df_merged_daily.dens > 900))
    df_merged_daily['melt_day_dz_boom'] = df_merged_daily['melt_day_dz_boom'].where((df_merged_daily.Tsmax > -2) & (df_merged_daily.acc_day == 0)  )
    df_merged_daily['melt_day_dz_stakes'] = df_merged_daily['melt_day_dz_stakes'].where((df_merged_daily.Tsmax > -2) & (df_merged_daily.acc_day == 0) )
    df_merged_daily['melt_day_dz_pt_cor'] = df_merged_daily['melt_day_dz_pt_cor'].where((df_merged_daily.Tsmax > -2) & (df_merged_daily.acc_day == 0) & (df_merged_daily.dens > 900))
    
    columns_out_nead_daily = ['t','t_2m','q','q_2m','rh','rh_2m','wspd','wspd_10m','wspd_max','wdir','p', \
        'swd','swu','lwd','lwu', \
        't1a','t2a','t3a','t4a','t5a','t1b','t2b','t3b','t4b','t5b', \
        'z_boom','z_boom_filtered','surface_level_zm','surface_level','z', \
        'vbat','t_cnr','t_logger','LAT','LON','alb','t_s_obs',\
        'shf_mod','lhf_mod','ghf_mod','cc','lwu_mod_2','shf_mod_2','lhf_mod_2','ghf_mod_2','t_s_mod_2','melt_mod_2','totmelt_2','cum_totmelt_2','subl_2',\
            'precip_mod', 'LWC_2', 'LWC_top10cm_2','z_boom_2','z_stakes_2','z_adw','z_adw_geus','z_pt_cor',\
                'dens','SHF_obs','LE_obs','SHF_obs_c','U_EC','WD_EC','ustar','T1','T2','z_EC',\
                    'melt_day_dz_adw','melt_day_dz_adw_geus','melt_day_dz_boom','melt_day_dz_stakes',\
                        'melt_day_dz_pt_cor','acc_day','melt_day_SEB','melt_day','subl_day','Tsmax','Tmax','dz_adw','dz_adw_geus','dz_boom','dz_stakes','dz_pt_cor','valid_samples_SEB','valid_samples_AWS']

    df_out_nead_daily = df_merged_daily[columns_out_nead_daily]
    file_out_nead = odir + LOC + '_' + ID + '_daily_all' + ".csv"

    # Get AWS location

    AWSloc = pd.read_csv(AWSfile,skiprows=0,delim_whitespace=True)
    AWSid =  AWSloc.loc[AWSloc['STATION'] == ID]['STATION'].values[0]
    AWSlat =  AWSloc.loc[AWSloc['STATION'] == ID]['LATITUDE'].values[0]
    AWSlon =  AWSloc.loc[AWSloc['STATION'] == ID]['LONGITUDE'].values[0]
    AWSalt =  AWSloc.loc[AWSloc['STATION'] == ID]['ELEVATION'].values[0]

    header = '\n'.join(
        ['# NEAD 1.0 UTF-8',
        '# [METADATA]',
        '# station_id       = ' + str(AWSid),
        '# latitude         = ' + str(AWSlat), 
        '# longitude        = ' + str(AWSlon),
        '# altitude         = ' + str(AWSalt),
        '# nodata           = ',
        '# field_delimiter  = ,',
        '# tz               = 0', 
        '# doi              = 10.1594/PANGAEA.970127', 
        '# [FIELDS]',
        '# fields           = time,t,t2m,q,q2m,rh,rh2m,ff,ff10m,ffmax,ffdir,p,SWd,SWu,LWd,LWu,Tsub1a,Tsub2a,Tsub3a,Tsub4a,Tsub5a,Tsub1b,Tsub2b,Tsub3b,Tsub4b,Tsub5b,z_surf,z_surf_filtered,cum_surface_height_zboom,cum_surface_height,z_u,Vbat,TCNR,Tlogger,LAT,LON,alb,Ts_obs,SHFdown,LHFdown,GHFup,cloud_cover,LWu_mod,SHFdown_mod,LHFdown_mod,GHFu_modp,Ts_mod,meltE,melt,cumulative_melt,sublimation,accumulation,LWC,LWC_10cm,z_boom,z_stakes,z_adw,z_adw_geus,z_pt_cor,dens,SHFdown_obs,LHFdown_obs,SHFdown_c,ff_EC,ffdir_EC,ustar_EC,Tsonic,Tcouple,z_EC,melt_day_dz_adw,melt_day_dz_adw_geus,melt_day_dz_boom,melt_day_dz_stakes,melt_day_dz_pt_cor,acc_day,melt_day_SEB,melt_day,subl_day,Tsmax,Tmax,dz_adw,dz_adw_geus,dz_boom,dz_stakes,dz_pt_cor,valid_samples_SEB,valid_samples_AWS',
        '# units            = -,DegreeC,DegreeC,g/kg,g/kg,%,%,m/s,m/s,m/s,deg,hPa,W/m^2,W/m^2,W/m^2,W/m^2,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,DegreeC,m,m,m,m,m,V,DegreeC,deg,deg,-,DegreeC,DegreeC,W/m^2,W/m^2,W/m^2,-,W/m^2,W/m^2,W/m^2,W/m^2,DegreeC,W/m^2,kg/m^2/dt,kg/m^2,kg/m^2/dt,kg/m^2/dt,kg/m^2,kg/m^2,m,m,m,m,m,kg/m^3,W/m^2,W/m^2,W/m^2,m/s,deg,m/s,degC,degC,m,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,m,W/m^2,m,m,degreeC,degreeC,m,m,m,m,m,-,-',
        '# [DATA]',
        '',
        ]
    )
        # '# fields           = timestamp,t,t2m,q,q2m,rh,rh2m,ff,ff10m,ffmax,ffdir,p,SWd,SWu,LWd,LWu,Tsub1a,Tsub2a,Tsub3a,Tsub4a,Tsub5a,Tsub1b,Tsub2b,Tsub3b,Tsub4b,Tsub5b,z_surf,z_surf_filtered,cum_surface_height_zboom,cum_surface_height,Vbat,TCNR,Tlogger,LAT,LON,alb,Ts_obs,Ts_mod,SHFdown,LHFdown,GHFup,Melt,Acc,LWCtot,LWC10cm',
        # '# units            = -,C,C,g/kg,g/kg,%,%,m/s,m/s,m/s,deg,hPa,W/m2,W/m2,W/m2,W/m2,C,C,C,C,C,C,C,C,C,C,m,m,m,m,V,C,C,deg,deg,-,C,C,W/m2,W/m2,W/m2,kg/m2/hour,kg/m2/hour,kg/m2,kg/m2',
    with open(file_out_nead, 'w') as ict:
        for line in header:
            ict.write(line)
        df_out_nead_daily.to_csv(ict,index=True,header=False,sep=',',float_format='%6.3f')
#_______________________________________________________
def lav_csv(nml):
    """
    Comoute 10-daily averages from merged SEB model and AWS data
    
    Input
    ----------
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    """

    LOC		= nml['global']['LOC']    

    csvdir  = nml['AWScorr']['csvdir']
    odir =   nml['AWScorr']['csvdir'] + '../10daily_all/'
    odir_sel =   nml['AWScorr']['csvdir'] + '../daily_shared/'
    AWSfile = nml['AWScorr']['file_AWS_locations']

    if not os.path.exists(odir):
        os.makedirs(odir) 
    if not os.path.exists(odir_sel):
        os.makedirs(odir_sel) 
 
    os.chdir(csvdir)
    files = (glob.glob("*daily_all.csv"))

    columns_out_nead_daily = ['time','t','t_2m','q','q_2m','rh','rh_2m','wspd','wspd_10m','wspd_max','wdir','p', \
        'swd','swu','lwd','lwu', \
        't1a','t2a','t3a','t4a','t5a','t1b','t2b','t3b','t4b','t5b', \
        'z_boom','z_boom_filtered','surface_level_zm','surface_level','z', \
        'vbat','t_cnr','t_logger','LAT','LON','alb','t_s_obs',\
        'shf_mod','lhf_mod','ghf_mod','cc','lwu_mod_2','shf_mod_2','lhf_mod_2','ghf_mod_2','t_s_mod_2','melt_mod_2','totmelt_2','cum_totmelt_2','subl_2',\
            'precip_mod', 'LWC_2', 'LWC_top10cm_2','z_boom_2','z_stakes_2','z_adw','z_adw_geus','z_pt_cor',\
                'dens','SHF_obs','LE_obs','SHF_obs_c','U_EC','WD_EC','ustar','T1','T2','z_EC',\
                    'melt_day_dz_adw','melt_day_dz_adw_geus','melt_day_dz_boom','melt_day_dz_stakes',\
                        'melt_day_dz_pt_cor','acc_day','melt_day_SEB','melt_day','subl_day','Tsmax','Tmax','dz_adw','dz_adw_geus','dz_boom','dz_stakes','dz_pt_cor','valid_samples_SEB','valid_samples_AWS']
    
    columns_out_nead_daily_sel  = ['t','t_2m','q','q_2m','rh','rh_2m','wspd','wspd_10m','wdir','p', \
        'swd','swu','lwd','lwu','shf_mod','lhf_mod','ghf_mod','shf_mod_2','lhf_mod_2','ghf_mod_2', 'z', 't_s_obs','t_s_mod_2','melt_mod_2',\
            'z_boom_2','z_stakes_2','z_adw','z_adw_geus','z_pt_cor',\
                'dens','SHF_obs','LE_obs',\
                    'melt_day_dz_adw','melt_day_dz_adw_geus','melt_day_dz_boom','melt_day_dz_stakes','melt_day_dz_pt_cor',\
                        'acc_day','melt_day_SEB','melt_day','subl_day','Tsmax','dz_adw','dz_adw_geus','dz_boom','dz_stakes','dz_pt_cor','valid_samples_SEB','valid_samples_AWS']
    
    columns_out_nead_10daily  = ['t','t_2m','q','q_2m','rh','rh_2m','wspd','wspd_10m','wdir','p', \
        'swd','swu','lwd','lwu','shf_mod','lhf_mod','ghf_mod','shf_mod_2','lhf_mod_2','ghf_mod_2', 'z', 't_s_obs','t_s_mod_2','melt_mod_2',\
            'z_boom_2','z_stakes_2','z_adw','z_adw_geus','z_pt_cor',\
                'dens','SHF_obs','LE_obs',\
                    'melt_day_dz_adw','melt_day_dz_adw_geus','melt_day_dz_boom','melt_day_dz_stakes','melt_day_dz_pt_cor',\
                        'acc_day','melt_day_SEB','melt_day','subl_day','Tsmax','dz_adw','dz_adw_geus','dz_boom','dz_stakes','dz_pt_cor','valid_samples_SEB','valid_samples_AWS']
    
    for file in files:
        os.chdir(csvdir)
        # df = pd.read_csv(file)
        print(file)
        df           = pd.read_csv(file,delimiter=',', names = columns_out_nead_daily,skiprows=14,index_col = 'time')
        df.index = pd.to_datetime(df.index)
        df.index.names = ['time']
        # print(df.index)
        df_out_nead_daily_sel = df[columns_out_nead_daily_sel]
        df_av = df.resample('10D',label='left').apply(np.mean)
        df_av_pnan = df.resample('10D',label='left').sum()
        df_av['valid_samples_SEB'] = df_av_pnan['valid_samples_SEB']
        df_av['valid_samples_AWS'] = df_av_pnan['valid_samples_AWS']

        df_out_nead_daily = df_av[columns_out_nead_10daily]
       
        # print(list(df_out_nead_daily.keys()))

        file_out_nead = file.replace('daily','10daily')

        # Get AWS location
        result = re.search(LOC + '_(.*)_daily*', file)

        ID = result.group(1)

        AWSloc = pd.read_csv(AWSfile,skiprows=0,delim_whitespace=True)
        AWSid =  AWSloc.loc[AWSloc['STATION'] == ID]['STATION'].values[0]
        AWSlat =  AWSloc.loc[AWSloc['STATION'] == ID]['LATITUDE'].values[0]
        AWSlon =  AWSloc.loc[AWSloc['STATION'] == ID]['LONGITUDE'].values[0]
        AWSalt =  AWSloc.loc[AWSloc['STATION'] == ID]['ELEVATION'].values[0]

        header = '\n'.join(
            ['# NEAD 1.0 UTF-8',
            '# [METADATA]',
            '# station_id       = ' + str(AWSid),
            '# latitude         = ' + str(AWSlat), 
            '# longitude        = ' + str(AWSlon),
            '# altitude         = ' + str(AWSalt),
            '# nodata           = ',
            '# field_delimiter  = ,',
            '# tz               = 0', 
            '# doi              = 10.1594/PANGAEA.970127', 
            '# [FIELDS]',
            '# fields           = time,t,t2m,q,q2m,rh,rh2m,ff,ff10m,ffdir,p,SWd,SWu,LWd,LWu,SHFdown,LHFdown,GHFup,SHFdown_mod,LHFdown_mod,GHFup_mod,z_u,Ts_obs,Ts_mod,melt_mod,z_boom,z_stakes,z_adw,z_adw_2,z_pt_cor,dens,SHFdown_obs,LHFdown_obs,melt_day_dz_adw,melt_day_dz_adw_2,melt_day_dz_boom,melt_day_dz_stakes,melt_day_dz_pt_cor,acc_day,melt_day_SEB,melt_day,subl_day,Tsmax,dz_adw,dz_adw_2,dz_boom,dz_stakes,dz_pt_cor,valid_samples_SEB,valid_samples_AWS',
            '# units            = -,DegreeC,DegreeC,g/kg,g/kg,%,%,m/s,m/s,Degrees,hPa,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,DegreeC,DegreeC,kg/m^2/day,m,m,m,m,m,kg/m^3,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,m,m,DegreeC,m,m,m,m,m,-,-',
            '# [DATA]',
            '',
            ]
        )


        os.chdir(odir)
        with open(file_out_nead, 'w') as ict:
            for line in header:
                ict.write(line)
            df_out_nead_daily.to_csv(ict,index=True,header=False,sep=',',float_format='%6.3f')
        file_out_nead = file.replace('daily_all','daily')

        header = '\n'.join(
            ['# NEAD 1.0 UTF-8',
            '# [METADATA]',
            '# station_id       = ' + str(AWSid),
            '# latitude         = ' + str(AWSlat), 
            '# longitude        = ' + str(AWSlon),
            '# altitude         = ' + str(AWSalt),
            '# nodata           = ',
            '# field_delimiter  = ,',
            '# tz               = 0', 
            '# doi              = 10.1594/PANGAEA.970127', 
            '# [FIELDS]',
            '# fields           = time,t,t2m,q,q2m,rh,rh2m,ff,ff10m,ffdir,p,SWd,SWu,LWd,LWu,SHFdown,LHFdown,GHFup,SHFdown_mod,LHFdown_mod,GHFup_mod,z_u,Ts_obs,Ts_mod,melt_mod,z_boom,z_stakes,z_adw,z_adw_2,z_pt_cor,dens,SHFdown_obs,LHFdown_obs,melt_day_dz_adw,melt_day_dz_adw_2,melt_day_dz_boom,melt_day_dz_stakes,melt_day_dz_pt_cor,acc_day,melt_day_SEB,melt_day,subl_day,Tsmax,dz_adw,dz_adw_2,dz_boom,dz_stakes,dz_pt_cor,valid_samples_SEB,valid_samples_AWS',
            '# units            = -,DegreeC,DegreeC,g/kg,g/kg,%,%,m/s,m/s,Degrees,hPa,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,DegreeC,DegreeC,kg/m^2/day,m,m,m,m,m,kg/m^3,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,W/m^2,m,m,DegreeC,m,m,m,m,m,-,-',
            '# [DATA]',
            '',
            ]
        )


        os.chdir(odir_sel)
        with open(file_out_nead, 'w') as ict:
            for line in header:
                ict.write(line)
            df_out_nead_daily_sel.to_csv(ict,index=True,header=False,sep=',',float_format='%6.3f')
#_______________________________________________________
def L1B_to_EBM(nml):
    """
    Converts  L1B netCDF file with AWS data to one .txt file in "HOUR_EBM" 
    format used as input for the SEB model
    Input
    ----------
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    """

    L1Bdir  = nml['AWScorr']['DIR'] + '/L1B/'
    sensor  = nml['GLOBAL']['sensor'] 
    z0fill  = nml['AWS_to_SEB']['z0fill'] 
    zfill  = nml['AWS_to_SEB']['zfill'] 
    lzfixed = nml['AWS_to_SEB']['lzfixed'] 
    zfixed = nml['AWS_to_SEB']['zfixed'] 
    LOC		= nml['global']['LOC']    
    ID		= nml['global']['ID']    
    yyyymmdd_start = nml['AWS_to_SEB']['yyyymmdd_start']
    yyyymmdd_end = nml['AWS_to_SEB']['yyyymmdd_end']
    lalbcorr = nml['AWS_to_SEB']['lalbcorr']
    lSWdcorr = nml['AWS_to_SEB']['lSWdcorr']
    lSWucorr = nml['AWS_to_SEB']['lSWucorr']
    lperturb_alb = nml['AWS_to_SEB']['lperturb_alb']
    perturb_alb = nml['AWS_to_SEB']['perturb_alb']
    z_snow = nml['AWS_to_SEB']['z_snow']
    dT = nml['AWS_to_SEB']['dT']
    serie = nml['AWS_to_SEB']['serie']
    luse_LWraw = nml['AWS_to_SEB']['luse_LWraw']
    lcorr_dL = nml['AWS_to_SEB']['lcorr_dL']
    luse_Traw = nml['AWS_to_SEB']['luse_Traw']
    luse_RHraw = nml['AWS_to_SEB']['luse_RHraw']
    luse_SWdraw = nml['AWS_to_SEB']['luse_SWdraw']

    os.chdir(L1Bdir)
    
    fid = 0

    file = (glob.glob('*L1B*' + "*nc"))[0]
        
    ds = xr.open_dataset(file)
    print(file)

    ds['z0m'] = xr.DataArray(np.full(len(ds.time),z0fill),dims=['time'])
    ds['z'] = ds.z.fillna(zfill)
    
    
    # Only keep data after start time
    ds  = ds.sel(time=slice(yyyymmdd_start, yyyymmdd_end))

    # offset serie
    ds['surface_level'] = ds['surface_level']- ds['surface_level'].values[0]
    
    # Correct SWu unsing corrected SWd and moving average albedo
    if lperturb_alb:
        ds['albedo_acc']  = ds['albedo_acc']  + perturb_alb
    if lSWdcorr:        
        ds['albedo_acc']   = (ds['SWu'].rolling(time=48,center=True,min_periods=4).reduce(np.nansum)) / (ds['SWd_corr'].rolling(time=48,center=True,min_periods=4).reduce(np.nansum))
        if lperturb_alb:
            ds['albedo_acc']  = ds['albedo_acc']  + perturb_alb
        # Fill nonphyscial values
        ds['albedo_acc'][ds['albedo_acc'] > 0.95] = 0.95
        ds['albedo_acc'][ds['albedo_acc'] < 0.1] = 0.1
        
        ds['SWu_acc'] = ds['SWd_corr'] * ds['albedo_acc'] 
            
    # Convert to dataframe
    df   = ds.to_dataframe()
    
    # Compute hourly averages
    grouped = df.groupby(pd.Grouper(freq=dT,label='left'), group_keys=False)
    df   = grouped.mean()
    df_sum   = grouped.sum()
    df_min = grouped.min()
    df['acc'] = df_sum['acc']      
    df['paramerror'] = df_min['paramerror']      
    # Make sure there is a line very dT = 30min
    df = df.reindex(pd.date_range(min(df.index), max(df.index), freq=dT)) # dT = '30min'   
    df["paramerror"] = df["paramerror"].ffill()

    # Define output data    
    columns_out = ['Date\tHour', 'Day', 'WS', 'Sin', 'Sout', 'Lin', 'Lout', 'T', \
                   'q', 'P', 'alb', 'zenith', 'Serie', 'precip', 'zt', 'zm', 'z0m', 'AWSid', 'Error']

    # Define new variables
    # Rename variables
    df.rename(columns={'U': 'WS', 'p0': 'P',\
                            'sza': 'zenith', 'acc': 'precip'}, inplace=True)
    
    if luse_SWdraw:
            df['alb'] = df['SWu']/df['SWd']
            df['alb'][ df['alb'] > 0.95] = 0.95
            df['alb'][ df['alb'] < 0.20] = 0.20
            df.rename(columns={'SWd': 'Sin'}, inplace=True)
            df.rename(columns={'SWu': 'Sout'}, inplace=True)
    else:
        if lalbcorr:
            df.rename(columns={'albedo_acc': 'alb'}, inplace=True)
            if lSWdcorr:
                df.rename(columns={'SWd_corr': 'Sin'}, inplace=True)
            else:
                df.rename(columns={'SWd_acc': 'Sin'}, inplace=True)
                
            if lSWucorr:
                df.rename(columns={'SWu_acc': 'Sout'}, inplace=True)
            else:
                df.rename(columns={'SWu': 'Sout'}, inplace=True)
        else:
            df['alb'] = df['SWu']/df['SWd']
            df['alb'][ df['alb'] > 0.95] = 0.95
            df['alb'][ df['alb'] < 0.20] = 0.20
            df.rename(columns={'SWd': 'Sin'}, inplace=True)
            df.rename(columns={'SWu': 'Sout'}, inplace=True)
        
    if luse_LWraw:
        df.rename(columns={'LWd_raw': 'Lin','LWu_raw': 'Lout'}, inplace=True)
    else:
        df.rename(columns={'LWd': 'Lin','LWu': 'Lout'}, inplace=True)

    if luse_Traw:
        df['T0_raw'][df['T0_raw'] > 273.15+30]  = np.nan
        df['T0_raw'][df['T0_raw'] < 273.15-90]  = np.nan
        df.rename(columns={'T0_raw': 'T'}, inplace=True)

    else:
        df.rename(columns={'T0': 'T'}, inplace=True)

    if luse_RHraw:
        df['RH_raw'][df['RH_raw'] > 150]  =  np.nan
        df['RH_raw'][df['RH_raw'] <= 0]    = np.nan
        df['RH_raw'][(abs(df['RH_raw'] - np.nanmean(df['RH_raw']))) > 5*np.nanstd(df['RH_raw']) ]  = np.nan
        df['qv0_raw'] = utils.RH2qv(df['RH_raw'],df['P'], df['T'])
        df.rename(columns={'qv0_raw': 'q'}, inplace=True)
    else:
        df.rename(columns={'qv0': 'q'}, inplace=True)



    if serie == 'surface_level':
        df.rename(columns={'surface_level': 'Serie'}, inplace=True)
    elif serie == 'surface_level_zm':
        df.rename(columns={'surface_level_zm': 'Serie'}, inplace=True)
    elif serie == 'surface_level_zs':
        df.rename(columns={'surface_level_zs': 'Serie'}, inplace=True)
        
        
    else:
        raise ValueError('Please select which serie to use in namoptions')

    # Define new variables
    df['Day']        = df.index.dayofyear + df.index.hour/24 + df.index.minute/(24*60)
    if sensor == 'AWI_neuma':
        df['zt']         = 2+0*df.z
        df['zm']         = 10+0*df.z
    else:
        if lzfixed:
            df['zt']         = zfixed+0*df.z
            df['zm']         = zfixed+0*df.z
        else:
            df['zt']         = df.z
            df['zm']         = df.z
        
    if lcorr_dL:
        df['Lin']          = df.Lin + df.dL
        df['Lout']         = df.Lout + df.dL
    df['AWSid']      = np.full(len(df.Day),float(1))
    df['z0m']        = df.z0m #np.full(len(df.Day),1.26e-3)
    df['Error']      = df.paramerror.astype(int)
    df['Date\tHour'] = df.index.strftime('     %Y-%m-%d\t%H:%M:%S')
    
    # Change units
    df['T'] = df['T']-273.15
    df['P'] = df['P']/100
    df['q'] = df['q']*1000
    df['zenith'] = df['zenith']*np.pi/180
    
    # add snow 
    df['Serie'] = df['Serie'] + z_snow
    
    # Write new dataframe
    df_out = df[columns_out]
    
    # iinterpolate NAN
    df_out = df_out.interpolate(method='linear')
    
    # Export to csv
    file_out =  LOC.lower() + '_' + ID.lower()  + '_HOUR-EBM' + '.txt'
    
    df_out.to_csv(file_out,index=False,header=True,sep='|',na_rep=float(-9999),float_format='%10.5f')

    # # Remove last blank line and replace spaces
    with open(file_out) as f:
        lines = f.readlines()
        lines = [l.replace('|', ' ') for l in lines]
        last = len(lines) - 1
        lines[last] = lines[last].replace('\r','').replace('\n','')
    with open(file_out, 'w') as wr:
        wr.writelines(lines)

#_______________________________________________________
def L1B_to_snowpack(nml):
    """
    Converts L1B netCDF file with AWS data to one .smet file used as input for SNOWPACK 
    Input
    ----------
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    """

    L1Bdir  = nml['AWScorr']['DIR'] + '/L1B/'
    ID		= nml['global']['ID']    
    LOC		= nml['global']['LOC']    
    yyyymmdd_start = nml['AWS_to_SEB']['yyyymmdd_start']
    yyyymmdd_end = nml['AWS_to_SEB']['yyyymmdd_end']
    dT = nml['AWS_to_SEB']['dT']
    TSG = nml['AWS_to_SEB']['TSG']
    AWSfile = nml['AWScorr']['file_AWS_locations']

    os.chdir(L1Bdir)
    file = (glob.glob('*L1B*' + "*nc"))[0]
    ds = xr.open_dataset(file)
    print(file)
    # Only keep data after start time
    ds  = ds.sel(time=slice(yyyymmdd_start, yyyymmdd_end))

    # Convert to dataframe
    df   = ds.to_dataframe()
    
    # Compute hourly averages
    grouped = df.groupby(pd.Grouper(freq=dT,label='right'), group_keys=False)
    df   = grouped.mean()
    df_sum   = grouped.sum()

    # Make sure there is a line very 30min
    df = df.reindex(pd.date_range(min(df.index), max(df.index), freq=dT)) 

    # Define output data    
    columns_out = ['timestamp', 'TA', 'RH', 'VW', 'ISWR', 'OSWR', 'ILWR', 'OLWR', 'PINT', 'TSG']

    # Rename variables
    df.rename(columns={'U': 'VW', 'T0': 'TA', 'acc': 'PINT', 'SWd_acc': 'ISWR', 'SWu': 'OSWR', 'LWd': 'ILWR', 'LWu': 'OLWR'}, inplace=True)

    # Define new variables
    df['timestamp'] = df.index.strftime('%Y-%m-%dT%H:%M:%S')
    df['TSG']      = np.full(len(df.VW),TSG)
    
    # Change units
    df['RH'] = df['RH']/100
    df['PINT'] = df['PINT'] # mm w.e.

    # Write new dataframe
    df_out = df[columns_out]
    
    # iinterpolate NAN
    df_out = df_out.interpolate(method='linear')
    
    # Export to csv
    file_out =  ID + '.smet'

    # Get AWS location
    AWSloc = pd.read_csv(AWSfile,skiprows=0,delim_whitespace=True)
    AWSid =  AWSloc.loc[AWSloc['STATION'] == ID]['STATION'].values[0]
    AWSlat =  AWSloc.loc[AWSloc['STATION'] == ID]['LATITUDE'].values[0]
    AWSlon =  AWSloc.loc[AWSloc['STATION'] == ID]['LONGITUDE'].values[0]
    AWSalt =  AWSloc.loc[AWSloc['STATION'] == ID]['ELEVATION'].values[0]
    header = '\n'.join(
        [ 'SMET 1.1 ASCII',
        '[HEADER]',
        'station_id       = '+ str(AWSid),
        'station_name     = '+ str(LOC) + ':'+ str(AWSid),
        'latitude         = '+ str(AWSlat), 
        'longitude        = '+ str(AWSlon),
        'altitude         = '+ str(AWSalt),
        'nodata           = -999',
        'tz               = 0', 
        'fields           = timestamp TA RH VW ISWR RSWR ILWR OLWR PSUM TSG',
        '[DATA]',
        '',
        ]
    )

    with open(file_out, 'w') as ict:
        for line in header:
            ict.write(line)
        df_out.to_csv(ict,index=False,header=False,sep='|',na_rep=float(-999),float_format='%6.3f')

    # # Remove last blank line and replace spaces
    with open(file_out) as f:
        lines = f.readlines()
        lines = [l.replace('|', '  ') for l in lines]
        last = len(lines) - 1
        lines[last] = lines[last].replace('\r','').replace('\n','')
    with open(file_out, 'w') as wr:
        wr.writelines(lines)

#_______________________________________________________
def Lejeune2019_to_EBM(nml):
    """
    Converts all files from Col de Porte (Lejeune et al 2019) to one .txt file in "HOUR_EBM" format
    Input
    ----------
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    """

    L0dir  = '/Volumes/ORION/05_Data/OTHER_AWS_DATA/ColdePorte_Lejeune2019/'

    ID = 'CDP'

    os.chdir(L0dir)

    file = 'CRYOBSCLIM.CDP.2018.MetInsitu.nc'
    file_snow = 'CRYOBSCLIM.CDP.2018.HourlySnow.nc'
    file_snow_daily = 'CRYOBSCLIM.CDP.2018.MetSnowDaily.nc'
        
    ds = xr.open_dataset(file)
    ds_snow = xr.open_dataset(file_snow)
    ds_snow_daily = xr.open_dataset(file_snow_daily)
    ds_snow_daily = ds_snow_daily.resample(time = "1H").ffill()
    
    ds = ds.isel(Number_of_points=0)

    ds['z0m'] = xr.DataArray(np.full(len(ds.time),1e-3),dims=['time'])
    ds['paramerror'] = xr.DataArray(np.full(len(ds.time),1000000000),dims=['time'])
    ds['sza'] = xr.DataArray(np.full(len(ds.time),0),dims=['time']) 
    ds['acc'] = ds.Snowf * 3600

    # Only keep data after start time
    ds  = ds.sel(time=slice('20050921', '20220801'))

    # offset serie
    ds['Serie'] = ds_snow.Snow_depth/10 #convert to m

    # get albedo
    ds['albedo'] = ds_snow_daily.albedo_daily 

    # get SWd & SWu
    ds['Sin'] = ds.DIR_SWdown + ds.SCA_SWdown
    ds['Sout'] = ds.Sin*ds.albedo
    ds['Lin'] = ds.LWdown
    ds['Lout'] = ds_snow.Surface_temperature**4 * 5.67e-8 
    
    # Convert to dataframe
    df   = ds.to_dataframe()
    
    # Make sure there is a line very 30min
    df = df.reindex(pd.date_range(min(df.index), max(df.index), freq='1H'))
    
    
    # Define output data    
    columns_out = ['Date\tHour', 'Day', 'WS', 'Sin', 'Sout', 'Lin', 'Lout', 'T', \
                   'q', 'P', 'alb', 'zenith', 'Serie', 'precip', 'zt', 'zm', 'z0m', 'AWSid', 'Error']

    # Define new variables
    # Rename variables
    df.rename(columns={'Wind': 'WS', 'Tair': 'T', 'Qair': 'q', 'PSurf': 'P',\
                            'albedo': 'alb','sza': 'zenith', 'acc': 'precip'}, inplace=True)
        

    # Define new variables
    df['Day']        = df.index.dayofyear + df.index.hour/24 + df.index.minute/(24*60)

    df['zt']         = np.full(len(df.Day),float(1.5))
    df['zm']         = np.full(len(df.Day),float(10))
        
    df['AWSid']      = np.full(len(df.Day),float(1))
    df['z0m']        = df.z0m #np.full(len(df.Day),1.26e-3)
    df['Error']      = df.paramerror.astype(int)
    df['Date\tHour'] = df.index.strftime('     %Y-%m-%d\t%H:%M:%S')
    
    # Change units
    df['T'] = df['T']-273.15
    df['P'] = df['P']/100
    df['q'] = df['q']*1000
    df['zenith'] = df['zenith']*np.pi/180
    
    # Write new dataframe
    df_out = df[columns_out]
    
    # iinterpolate NAN
    df_out = df_out.interpolate(method='linear')
    
    # Export to csv
    file_out =  'grl_aws' + ID + '_HOUR_EBM' + '.txt'
    
    df_out.to_csv(file_out,index=False,header=True,sep='|',na_rep=float(-9999),float_format='%10.5f')

    # # Remove last blank line and replace spaces
    with open(file_out) as f:
        lines = f.readlines()
        lines = [l.replace('|', ' ') for l in lines]
        last = len(lines) - 1
        lines[last] = lines[last].replace('\r','').replace('\n','')
    with open(file_out, 'w') as wr:
        wr.writelines(lines)
                        
#_______________________________________________________
def L1B_to_COSIPY(nml):
    """
    Converts  L1B netCDF files with AWS data to one .csv file in COSIPY format
    Input
    ----------
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    """
    ID = nml['GLOBAL']['ID'] 
    L1Bdir  = nml['AWScorr']['DIR'] + '/L1B/'
    os.chdir(L1Bdir)
    
    fid = 0

    for file in sorted(glob.glob('*L1B*' + "*nc")):
        print(file)
        tmp = xr.open_dataset(file)
        if fid == 0:
            ds    = tmp
        else:
            ds = xr.combine_nested([ds,tmp], 'time', compat='no_conflicts', data_vars='all', coords='all', join='outer')
        fid =+ 1
    
    # Convert to dataframe
    df   = ds.to_dataframe()
    
        
    # Make sure there is a line very 30min
    df = df.reindex(pd.date_range(min(df.index), max(df.index), freq='30min'))
    
    # Define output data    
    columns_out = ['TIMESTAMP','T2','PRES','N','U2','RH2','RRR','G','LWin']

    df.rename(columns={'U': 'U2', 'LWd': 'LWin','T0': 'T2', 'RH': 'RH2', 'p0': 'PRES',\
                                'clt': 'N','SWd': 'G', 'acc': 'RRR'}, inplace=True)
    # Define new variables
    df['Day']        = df.index.dayofyear + df.index.hour/24 + df.index.minute/(24*60)

    df['TIMESTAMP'] = df.index.strftime('%Y-%m-%d %H:%M:%S')
    
    # Change units
    df['T2'] = df['T2']
    df['PRES'] = df['PRES']/100
    
    # Write new dataframe
    df_out = df[columns_out]

    
    # iinterpolate NAN
    df_out = df_out.interpolate(method='linear')
    
    # Export to csv
    file_out =  'grl_aws' + ID + '_30MIN_COSIPY' + '.csv'
    
    df_out.to_csv(file_out,index=False,header=True,sep=',',na_rep=float(-9999),float_format='%10.5f')

    # # Remove last blank line and replace spaces
    with open(file_out) as f:
        lines = f.readlines()
        lines = [l.replace('|', ' ') for l in lines]
        last = len(lines) - 1
        lines[last] = lines[last].replace('\r','').replace('\n','')
    with open(file_out, 'w') as wr:
        wr.writelines(lines)

            
#_______________________________________________________
def PRY_to_NED(pitch,roll,yaw):
    """
    Convert iWS inclinomter pitch, roll and yaw angles to tilt angles in North-East reference frame
    
    """
    
    # Convert degree to radians
    pitch = pitch * np.pi/180
    roll  = roll * np.pi/180
    yaw   = yaw * np.pi/180

    # Reference mast position (with respect to g acceleration)
    X = np.array([0,0,1])
    
    # Roll rotation matrix (around old x-axis)
    A_roll = np.array([[1, 0, 0],[0,np.cos(roll),-np.sin(roll)],[0, np.sin(roll), np.cos(roll)]])

    # Pitch rotation matrix (around old y-axis)
    A_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],[0,1,0],[-np.sin(pitch), 0, np.cos(pitch)]])
    
    # Apply pitch and roll to mast
    Y =  np.dot(np.dot(A_pitch,A_roll),X)
    
    # Mast tilt angle (with respect to z-axis)
    beta = np.pi/2 - np.arctan2(Y[2],(Y[0]**2+Y[1]**2)**0.5)
    
    # Mast tilt orientation (azimuth, aka clockwise with respect to true north)
    aw = np.arctan2(Y[1],Y[0]) + yaw
    
    # Keep tilt oriantion in [0;360[ range
    if aw<0:
        aw = aw + 2*np.pi
    elif aw > 2*np.pi:
        aw = aw - 2*np.pi

    # convert radians to degrees
    beta = beta * 180/np.pi
    aw  = aw * 180/np.pi
    
    return aw, beta


#_______________________________________________________
def load_PROMICE_inst(file):
    """
    Load instantenaous 10-min PROMICE file 
    
    """
    
    df = pd.read_csv(file, delim_whitespace=True,na_values = [-999, -9999],header=0)  
    df.index = rec.compose_date(df['Year'], df['MonthOfYear'], df['DayOfMonth'], hours=df['HourOfDay(UTC)'], minutes=df['MinuteOfHour'])   
    df.index.names = ['time']
    df = df[~df.index.duplicated()]
    df = df.resample('30min',label = 'left').mean()
    df.rename(columns={'HourOfDay(UTC)': 'HourOfDayUTC','AirPressure(hPa)': 'P','AirTemperature(C)': 'AirTemperatureC', 'AirTemperatureHygroClip(C)': 'AirTemperatureHygroClipC', \
                       'RelativeHumidity_wrtWater(%)': 'RelativeHumidity_wrtWater%', 'RelativeHumidity(%)': 'RelativeHumidity%','WindSpeed(m/s)': 'WindSpeedms', 'ShortwaveRadiationDown(W/m2)': 'ShortwaveRadiationDownWm2', \
                       'WindDirection(d)': 'WindDirectiond', 'ShortwaveRadiationDown(W/m2)': 'ShortwaveRadiationDownWm2', 'ShortwaveRadiationDown_Cor(W/m2)': 'ShortwaveRadiationDown_CorWm2','ShortwaveRadiationUp(W/m2)': 'ShortwaveRadiationUpWm2',\
                           'ShortwaveRadiationUp_Cor(W/m2)': 'ShortwaveRadiationUp_CorWm2','LongwaveRadiationDown(W/m2)': 'LongwaveRadiationDownWm2','HeightSensorBoom(m)': 'HeightSensorBoomm', \
                               'HeightStakes(m)': 'HeightStakesm','DepthPressureTransducer(m)':'DepthPressureTransducerm','DepthPressureTransducer_Cor(m)':'DepthPressureTransducer_Corm',\
                               'LongwaveRadiationUp(W/m2)': 'LongwaveRadiationUpWm2',\
                            'albedo_acc': 'alb','sza': 'zenith', 'surface_level': 'Serie', 'acc': 'precip', 'ZenithAngleSun(d)': 'sza', 'DirectionSun(d)': 'az'}, inplace=True)
    return df
#_______________________________________________________
def load_PROMICE_hourly_txt(file):
    """
    Load hourly PROMICE file in txt format
    
    """
    
    df = pd.read_csv(file, delim_whitespace=True,na_values = [-999, -9999],header=0)  
    df.index = rec.compose_date(df['Year'], df['MonthOfYear'], df['DayOfMonth'], hours=df['HourOfDay(UTC)'])   
    df.index.names = ['time']
    df = df[~df.index.duplicated()]
    df = df.resample('1H',label = 'left').mean()
    df.rename(columns={'HourOfDay(UTC)': 'HourOfDayUTC','WindSpeed(m/s)': 'wspd', 'ShortwaveRadiationDown(W/m2)': 'fsds', \
                       'ShortwaveRadiationDown_Cor(W/m2)': 'fsds_cor','ShortwaveRadiationUp(W/m2)': 'fsus',\
                           'ShortwaveRadiationUp_Cor(W/m2)': 'fsus_cor','LongwaveRadiationDown(W/m2)': 'flds', \
                               'LongwaveRadiationUp(W/m2)': 'flus', 'AirPressure(hPa)': 'pa',\
                                   'WindDirection(d)': 'wdir',\
                            'Albedo_theta<70d ': 'alb','sza': 'zenith', 'surface_level': 'Serie', 'acc': 'precip', 'ZenithAngleSun(d)': 'sza', 'DirectionSun(d)': 'az'}, inplace=True)
    return df

#_______________________________________________________
def load_PROMICE_hourly_v04(file):
    """
    Load hourly PROMICE file version v04
    
    """
    
    df = pd.read_csv(file, delimiter=',',header=0)  
    df.index = pd.to_datetime(df['time'], infer_datetime_format=True)
    df.drop(['time'] , axis=1, inplace=True)   
    df.index.names = ['time']
    df = df[~df.index.duplicated()]

    return df

#_______________________________________________________
def load_GCnet(file):
    """
    Load hourly GCNet file 
    
    """
    column_names = ['time','ISWR','OSWR','NR','TA1','TA2','TA3','TA4',\
                    'RH1','RH2','VW1','VW2','DW1','DW2','P','HW1','HW2','V',\
                        'TA5','TS1','TS2','TS3','TS4','TS5','TS6','TS7','TS8','TS9','TS10',\
                            'HW1_adj_flag','HW2_adj_flag','OSWR_adj_flag','P_adj_flag','NR_cor',\
                                'HS1','HS2','HS_combined','SHF','LHF','TA2m','RH2m','VW10m','SZA',\
                                    'SAA','Alb','RH1_cor','Q1','RH2_cor','Q2','latitude','longitude',\
                                        'elevation','DTS1','DTS2','DTS3','DTS4','DTS5','DTS6','DTS7','DTS8','DTS9','DTS10','TS_10m']


    
    df = pd.read_csv(file, delimiter=',', names = column_names,skiprows=27)  

    df.index = pd.to_datetime(df['time'], infer_datetime_format=True)
    df = df.tz_localize(None)
    df.drop(['time'] , axis=1, inplace=True)   
    df.index.names = ['time']
    df = df[~df.index.duplicated()]

    return df

#_______________________________________________________
def load_RAD(file):
    """
    Load NOAA rad file from SUMMIT  
    
    """
    column_names = ['year','month','day','hour','minute','D_GLOBAL','D_IR','U_GLOBAL',\
                    'U_IR','Zenith']
    
    df = pd.read_csv(file, delimiter="\s+", names = column_names,skiprows=4)  
    df.index       = compose_date(df['year'], months=df['month'], days=df['day'], hours=df['hour'],minutes = df['minute']) 
    df.index.names = ['time']
    df = df[~df.index.duplicated()]
    df = df.resample('1H',label = 'right').mean()

    return df



#_______________________________________________________
def L0toL1A(nml):
    """
    Converts raw files to single netCDF 
    Input
    ----------
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    """
    L1Adir = nml['AWScorr']['DIR'] + 'L1A/'
    L0dir  = nml['AWScorr']['DIR'] + nml['AWScorr']['L0dir']
    
    ID			= nml['global']['ID']    
    LOC			= nml['global']['LOC']
    version		= nml['global']['version']
    Input_fname = nml['AWScorr']['Input_fname']
    input_type	= nml['global']['input_type']
    L0_raddir  = nml['AWScorr']['L0_raddir']

    lmetafile   = nml['AWScorr']['lmetafile']
    toffset_min = nml['AWScorr']['toffset_min']
 
    yyyymmdd_start = nml['AWScorr']['yyyymmdd_start']
    yyyymmdd_end = nml['AWScorr']['yyyymmdd_end']
    
    if not os.path.exists(L1Adir):
        os.makedirs(L1Adir) 
        
    # Move to input data directory
    if not L0dir:
        os.chdir(os.getcwd())  
    else:
        os.chdir(L0dir)  
    print(os.getcwd())
        
    if Input_fname == 'fluxstation18':
        file =  sorted(glob.glob("*fluxstation.txt"), key=lambda x: x[1:])   
        out = load_irridium_AWS18(file[0],nml)
        
        ds = out.to_xarray()
        
        file_out = L1Adir + LOC + '_' + ID + '_' + 'flux' + '_' + input_type + '_' + "L1A_" + version +  '_' + 'all' + ".nc"

        ds.to_netcdf(file_out, mode='w')
        
    elif Input_fname == 'grl_AWSID_final_year1hrYYYY':
        # Group files per {IDYYYY}
        files =  sorted(glob.glob("*.txt"), key=lambda x: x[-8:-4])
        # Loop over {IDYYYY} files
        fid = 0
        for file in files:
            # Load raw file in a dataframe
            print(file)
            if fid == 0:
                out = load_AWS_final_year(os.getcwd() + "/" + file,nml)
            else:
                tmp = load_AWS_final_year(os.getcwd() + "/" + file,nml)
                out = pd.concat([out,tmp])
            fid = fid + 1
        out = out[~out.index.duplicated(keep='last')]
        out.index = out.index + np.timedelta64(toffset_min,'m')
        file_out = L1Adir + LOC + '_' + ID + '_' + 'AWS' + '_' + input_type + '_' + "L1A_" + version +  '_' + 'all' + ".nc"
            
        # Convert to xarray
        ds = out.to_xarray()
        
        # Attributes
        ds                                    = utils.Add_dataset_attributes(ds,'AWS_WS_L1A.JSON')
        ds.attrs['location']                  = LOC + '_' + ID
        ds.attrs['file_creation_date_time']   = str(datetime.datetime.now())
        ds.attrs['IceEddie_namelist_GLOBAL']  = str(nml['global'])
        ds.attrs['IceEddie_namelist_AWScorr'] = str(nml['AWScorr'])
        
        ds.to_netcdf(file_out, mode='w')
        
    elif Input_fname == 'grl_AWSIDfinal_aug2augYYYY':
        # Group files per {IDYYYY}
        files =  sorted(glob.glob("*.txt"), key=lambda x: x[-8:-4])
        # Loop over {IDYYYY} files
        fid = 0
        for file in files:
            # Load raw file in a dataframe
            print(file)
            if fid == 0:
                out = load_AWS_yearly(os.getcwd() + "/" + file,nml)
            else:
                tmp = load_AWS_yearly(os.getcwd() + "/" + file,nml)
                out = pd.concat([out,tmp])
            fid = fid + 1
        out = out[~out.index.duplicated(keep='last')]
        out.index = out.index + np.timedelta64(toffset_min,'m')
        file_out = L1Adir + LOC + '_' + ID + '_' + 'AWS' + '_' + input_type + '_' + "L1A_" + version +  '_' + 'all' + ".nc"
            
        # Convert to xarray
        ds = out.to_xarray()
        ds = ds.assign_coords(time=ds.time.astype("datetime64[ns]"))
        
        # Attributes
        ds                                    = utils.Add_dataset_attributes(ds,'AWS_WS_L1A.JSON')
        ds.attrs['location']                  = LOC + '_' + ID
        ds.attrs['file_creation_date_time']   = str(datetime.datetime.now())
        ds.attrs['IceEddie_namelist_GLOBAL']  = str(nml['global'])
        ds.attrs['IceEddie_namelist_AWScorr'] = str(nml['AWScorr'])
        
        ds.to_netcdf(file_out, mode='w')
            
    elif Input_fname == 'AWS_yearly_Paul_corr':

        files =  sorted(glob.glob("*.txt"), key=lambda x: x[-8:-4])
        # Loop over {IDYYYY} files
        fid = 0
        for file in files:
            # Load raw file in a dataframe
            if fid == 0:
                print(file)
                out = load_AWS_yearly_Paul_corr(os.getcwd() + "/" + file,nml)
            else:
                tmp = load_AWS_yearly_Paul_corr(os.getcwd() + "/" + file,nml)
                out = pd.concat([out,tmp])
            fid = fid +1
        out = out[~out.index.duplicated(keep='last')]
        
        out.index = out.index + np.timedelta64(toffset_min,'m')
        file_out = L1Adir + LOC + '_' + ID + '_' + 'AWS' + '_' + input_type + '_' + "L1A_" + version +  '_' + 'all' + ".nc"

        # Convert to xarray
        ds = out.to_xarray()
        
        # Attributes
        ds                                    = utils.Add_dataset_attributes(ds,'AWS_WS_L1A.JSON')
        ds.attrs['location']                  = LOC + '_' + ID
        ds.attrs['file_creation_date_time']   = str(datetime.datetime.now())
        ds.attrs['IceEddie_namelist_GLOBAL']  = str(nml['global'])
        ds.attrs['IceEddie_namelist_AWScorr'] = str(nml['AWScorr'])
        
        ds.to_netcdf(file_out, mode='w')

    elif Input_fname == 'AWS_yearly_Paul_corr_v6':

            files =  sorted(glob.glob("*.txt"), key=lambda x: x[-8:-4])
            # Loop over {IDYYYY} files
            fid = 0
            for file in files:
                # Load raw file in a dataframe
                if fid == 0:
                    print(file)
                    out = load_AWS_yearly_Paul_corr_v6(os.getcwd() + "/" + file,nml)
                else:
                    tmp = load_AWS_yearly_Paul_corr_v6(os.getcwd() + "/" + file,nml)
                    out = pd.concat([out,tmp])
                fid = fid +1
            out = out[~out.index.duplicated(keep='last')]

            out.index = out.index + np.timedelta64(toffset_min,'m')
            file_out = L1Adir + LOC + '_' + ID + '_' + 'AWS' + '_' + input_type + '_' + "L1A_" + version +  '_' + 'all' + ".nc"

            # Convert to xarray
            ds = out.to_xarray()
            ds = ds.assign_coords(time=ds.time.astype("datetime64[ns]"))
            
            # Attributes
            ds                                    = utils.Add_dataset_attributes(ds,'AWS_WS_L1A.JSON')
            ds.attrs['location']                  = LOC + '_' + ID
            ds.attrs['file_creation_date_time']   = str(datetime.datetime.now())
            ds.attrs['IceEddie_namelist_GLOBAL']  = str(nml['global'])
            ds.attrs['IceEddie_namelist_AWScorr'] = str(nml['AWScorr'])
            
            ds.to_netcdf(file_out, mode='w')

    elif Input_fname == 'AWS_yearly_Paul_corr_v6_merged':

            files =  sorted(glob.glob("*.txt"), key=lambda x: x[-8:-4])
            # Loop over {IDYYYY} files
            fid = 0
            for file in files:
                # Load raw file in a dataframe
                if fid == 0:
                    print(file)
                    out, out2 = load_AWS_yearly_Paul_corr_v6_merged(os.getcwd() + "/" + file,nml)
                else:
                    tmp, tmp2 = load_AWS_yearly_Paul_corr_v6_merged(os.getcwd() + "/" + file,nml)
                    out = pd.concat([out,tmp])
                    out2 = pd.concat([out2,tmp2])
                fid = fid +1
            out = out[~out.index.duplicated(keep='last')]
            out2 = out2[~out2.index.duplicated(keep='last')]

            out.index = out.index + np.timedelta64(toffset_min,'m')
            # out2.index = out2.index + np.timedelta64(toffset_min,'m')

            file_out = L1Adir + LOC + '_' + ID + '_' + 'AWStypeI' + '_' + input_type + '_' + "L1A_" + version +  '_' + 'all' + ".nc"
            file_out2 = L1Adir + LOC + '_' + ID + '_' + 'AWStypeII' + '_' + input_type + '_' + "L1A_" + version +  '_' + 'all' + ".nc"

            # Convert to xarray

            ds = out.to_xarray()
            ds = ds.assign_coords(time=ds.time.astype("datetime64[ns]"))
            ds2 = out2.to_xarray()
            ds2 = ds2.assign_coords(time=ds2.time.astype("datetime64[ns]"))

            # Attributes
            ds.attrs['location']                  = LOC + '_' + ID
            ds.attrs['file_creation_date_time']   = str(datetime.datetime.now())
            ds.attrs['IceEddie_namelist_GLOBAL']  = str(nml['global'])
            ds.attrs['IceEddie_namelist_AWScorr'] = str(nml['AWScorr'])
            ds.to_netcdf(file_out, mode='w')

            ds2.attrs['location']                  = LOC + '_' + ID
            ds2.attrs['file_creation_date_time']   = str(datetime.datetime.now())
            ds2.attrs['IceEddie_namelist_GLOBAL']  = str(nml['global'])
            ds2.attrs['IceEddie_namelist_AWScorr'] = str(nml['AWScorr']) 
            ds2.to_netcdf(file_out2, mode='w')

    elif Input_fname == 'iWS_yearly_Paul_corr_v7':

            files =  sorted(glob.glob("*.txt"), key=lambda x: x[-8:-4])
            # Loop over {IDYYYY} files
            fid = 0
            for file in files:
                # Load raw file in a dataframe
                if fid == 0:
                    print(file)
                    out = load_iWS_yearly_Paul_corr_v7(os.getcwd() + "/" + file,nml)
                else:
                    tmp = load_iWS_yearly_Paul_corr_v7(os.getcwd() + "/" + file,nml)
                    out = pd.concat([out,tmp])
                fid = fid +1
            out = out[~out.index.duplicated(keep='last')]
            print(out)
            out.index = out.index + np.timedelta64(toffset_min,'m')
            file_out = L1Adir + LOC + '_' + ID + '_' + 'iWS' + '_' + input_type + '_' + "L1A_" + version +  '_' + 'all' + ".nc"

            # Convert to xarray
            ds = out.to_xarray()
            ds = ds.assign_coords(time=ds.time.astype("datetime64[ns]"))
            
            # Attributes
            ds                                    = utils.Add_dataset_attributes(ds,'iWS_WS_L1A.JSON')
            ds.attrs['location']                  = LOC + '_' + ID
            ds.attrs['file_creation_date_time']   = str(datetime.datetime.now())
            ds.attrs['IceEddie_namelist_GLOBAL']  = str(nml['global'])
            ds.attrs['IceEddie_namelist_AWScorr'] = str(nml['AWScorr'])
            
            ds.to_netcdf(file_out, mode='w')

    elif Input_fname == 'iWS_yearly_Paul_corr':

        files =  sorted(glob.glob("*.txt"), key=lambda x: x[-8:-4])
        # Loop over {IDYYYY} files
        fid = 0
        for file in files:
            # Load raw file in a dataframe
            if fid == 0:
                print(file)
                out = load_iWS_yearly_Paul_corr(os.getcwd() + "/" + file,nml)
            else:
                tmp = load_iWS_yearly_Paul_corr(os.getcwd() + "/" + file,nml)
                out = pd.concat([out,tmp])
            fid = fid +1
        out = out[~out.index.duplicated(keep='last')]
        
        out.index = out.index + np.timedelta64(toffset_min,'m')
        file_out = L1Adir + LOC + '_' + ID + '_' + 'AWS' + '_' + input_type + '_' + "L1A_" + version +  '_' + 'all' + ".nc"

        # Convert to xarray
        ds = out.to_xarray()
        
        # Attributes
        ds                                    = utils.Add_dataset_attributes(ds,'iWS_WS_L1A.JSON')
        ds.attrs['location']                  = LOC + '_' + ID
        ds.attrs['file_creation_date_time']   = str(datetime.datetime.now())
        ds.attrs['IceEddie_namelist_GLOBAL']  = str(nml['global'])
        ds.attrs['IceEddie_namelist_AWScorr'] = str(nml['AWScorr'])
        
        ds.to_netcdf(file_out, mode='w')
        
    elif Input_fname == 'iWS_yearly_Paul_corr_v2':

        files =  sorted(glob.glob("*.txt"), key=lambda x: x[-8:-4])
        # Loop over {IDYYYY} files
        fid = 0
        for file in files:
            # Load raw file in a dataframe
            if fid == 0:
                print(file)
                out = load_iWS_yearly_Paul_corr_v2(os.getcwd() + "/" + file,nml)
            else:
                tmp = load_iWS_yearly_Paul_corr_v2(os.getcwd() + "/" + file,nml)
                out = pd.concat([out,tmp])
            fid = fid +1
        out = out[~out.index.duplicated(keep='last')]
        
        out.index = out.index + np.timedelta64(toffset_min,'m')
        file_out = L1Adir + LOC + '_' + ID + '_' + 'AWS' + '_' + input_type + '_' + "L1A_" + version +  '_' + 'all' + ".nc"

        # Convert to xarray
        ds = out.to_xarray()
        ds = ds.assign_coords(time=ds.time.astype("datetime64[ns]"))
        # print(ds)
        
        # Attributes
        ds                                    = utils.Add_dataset_attributes(ds,'iWS_WS_L1A.JSON')
        ds.attrs['location']                  = LOC + '_' + ID
        ds.attrs['file_creation_date_time']   = str(datetime.datetime.now())
        ds.attrs['IceEddie_namelist_GLOBAL']  = str(nml['global'])
        ds.attrs['IceEddie_namelist_AWScorr'] = str(nml['AWScorr'])
        
        ds.to_netcdf(file_out, mode='w')

    elif Input_fname == 'ant_AWSID_final_year1hrYYYY':
        # Group files per {IDYYYY}
        files =  sorted(glob.glob("*.txt"), key=lambda x: x[-8:-4])
        fid = 0
        # Loop over {IDYYYY} files
        for file in files:
            print(file)
            # Load raw file in a dataframe
            if fid == 0:
                out = load_AWS_final_year_ant(os.getcwd() + "/" + file,nml)
            else:
                tmp = load_AWS_final_year_ant(os.getcwd() + "/" + file,nml)
                out = pd.concat([out,tmp])
            fid = fid +1
        out = out[~out.index.duplicated(keep='last')]
                                
        # Export monthly dataframe to net CDF    
        file_out = L1Adir + LOC + '_' + ID + '_' + 'AWS1hr' + '_' + input_type + '_' + "L1A_" + version +  '_' + 'all' + ".nc"
    
        # Convert to xarray
        ds = out.to_xarray()
        
        # Attributes
        ds                                    = utils.Add_dataset_attributes(ds,'AWS_WS_L1A.JSON')
        ds.attrs['location']                  = LOC + '_' + ID
        ds.attrs['file_creation_date_time']   = str(datetime.datetime.now())
        ds.attrs['IceEddie_namelist_GLOBAL']  = str(nml['global'])
        
        ds.attrs['IceEddie_namelist_AWScorr'] = str(nml['AWScorr'])
        # Export to net CDF
        ds.to_netcdf(file_out, mode='w')

    elif Input_fname == 'iWS_final_yearYYYY':
        # Group files per {IDYYYY}
        files =  sorted(glob.glob("*.txt"), key=lambda x: x[-8:-4])
        # Loop over {IDYYYY} files
        fid = 0
        for file in files:
            # Load raw file in a dataframe
            print(file)
            if fid == 0:
                out = load_iWS_yearly(os.getcwd() + "/" + file,nml)
            else:
                tmp = load_iWS_yearly(os.getcwd() + "/" + file,nml)
                out = pd.concat([out,tmp])
            fid = fid +1
        out = out[~out.index.duplicated(keep='last')]
         
        # Export monthly dataframe to net CDF    
        file_out = L1Adir + LOC + '_' + ID + '_' + 'iWS' + '_' + input_type + '_' + "L1A_" + version +  '_' + 'all' + ".nc"
        
        # Convert to xarray
        ds = out.to_xarray()
        
        # Attributes
        ds                                    = utils.Add_dataset_attributes(ds,'iWS_WS_L1A.JSON')
        ds.attrs['location']                  = LOC + '_' + ID
        ds.attrs['file_creation_date_time']   = str(datetime.datetime.now())
        ds.attrs['IceEddie_namelist_GLOBAL']  = str(nml['global'])
        
        ds.attrs['IceEddie_namelist_AWScorr'] = str(nml['AWScorr'])
        # Export to net CDF
        ds.to_netcdf(file_out, mode='w')
        
    if Input_fname == 'IDYYMMDD':
                
        files =  sorted(glob.glob("*.TXT"), key=lambda x: x[0:-4])

        # Load raw file in a dataframe
        fid = 0
        for file in files:
            print(file)
            if fid == 0:
                out = load_iWS_daily(os.getcwd() + "/" + file,nml)
            else:
                # print(file)
                tmp = load_iWS_daily(os.getcwd() + "/" + file,nml)
                # Load raw file in a dataframe
                out = pd.concat([out,tmp])
            fid = fid + 1
            
        out = out[~out.index.duplicated(keep='last')]


        file_out = L1Adir + LOC + '_' + ID + '_' + 'iWS' + '_' + input_type + '_' + "L1A_" + version + '_' + 'all' + ".nc"
            
        # Convert to xarray
        ds = out.to_xarray()
        
        # Attributes
        ds                                    = utils.Add_dataset_attributes(ds,'iWS_WS_L1A.JSON')
        ds.attrs['location']                  = LOC + '_' + ID
        ds.attrs['file_creation_date_time']   = str(datetime.datetime.now())
        ds.attrs['IceEddie_namelist_GLOBAL']  = str(nml['global'])
        
        ds.attrs['IceEddie_namelist_AWScorr'] = str(nml['AWScorr'])
        # Export to net CDF
        ds.to_netcdf(file_out, mode='w')

    if Input_fname == 'IDYYMMDD_S21':
                
        files =  sorted(glob.glob("*.TXT"), key=lambda x: x[0:-4])

        # Load raw file in a dataframe
        fid = 0
        for file in files:
            print(file)
            if fid == 0:
                out = load_iWS_daily_S21(os.getcwd() + "/" + file,nml)
            else:
                # print(file)
                tmp = load_iWS_daily_S21(os.getcwd() + "/" + file,nml)
                # Load raw file in a dataframe
                out = pd.concat([out,tmp])
            fid = fid + 1
            
        out = out[~out.index.duplicated(keep='last')]


        file_out = L1Adir + LOC + '_' + ID + '_' + 'iWS' + '_' + input_type + '_' + "L1A_" + version + '_' + 'all' + ".nc"
            
        # Convert to xarray
        ds = out.to_xarray()
        
        # Attributes
        ds                                    = utils.Add_dataset_attributes(ds,'iWS_WS_L1A.JSON')
        ds.attrs['location']                  = LOC + '_' + ID
        ds.attrs['file_creation_date_time']   = str(datetime.datetime.now())
        ds.attrs['IceEddie_namelist_GLOBAL']  = str(nml['global'])
        
        ds.attrs['IceEddie_namelist_AWScorr'] = str(nml['AWScorr'])
        # Export to net CDF
        ds.to_netcdf(file_out, mode='w')
        
        
    elif Input_fname == 'grl_iWSID_final_yearYYYY':
        # Group files per {IDYYYY}
        files =  sorted(glob.glob("*.txt"), key=lambda x: x[-8:-4])
        # Loop over {IDYYYY} files
        fid = 0
        for file in files:
            # Load raw file in a dataframe
            if fid == 0:
                print(file)
                out = load_iWS_yearly(os.getcwd() + "/" + file,nml)
            else:
                tmp = load_iWS_yearly(os.getcwd() + "/" + file,nml)
                out = pd.concat([out,tmp])
            fid = fid +1
        out = out[~out.index.duplicated(keep='last')]
        
        file_out = L1Adir + LOC + '_' + ID + '_' + 'iWS' + '_' + input_type + '_' + "L1A_" + version + '_' + 'all' + ".nc"

        # Convert to xarray
        ds = out.to_xarray()

        ds = ds.assign_coords(time=ds.time.astype("datetime64[ns]"))
        # print(ds.time)
        
        # Attributes
        ds                                    = utils.Add_dataset_attributes(ds,'iWS_WS_L1A.JSON')
        ds.attrs['location']                  = LOC + '_' + ID
        ds.attrs['file_creation_date_time']   = str(datetime.datetime.now())
        ds.attrs['IceEddie_namelist_GLOBAL']  = str(nml['global'])
        
        ds.attrs['IceEddie_namelist_AWScorr'] = str(nml['AWScorr'])
        # Export to net CDF
        ds.to_netcdf(file_out, mode='w')
        
    elif Input_fname == 'PROMICE_v04':
        file =  glob.glob("*csv")
        print(file)
        out = load_PROMICE_hourly_v04(file[0])
        
        file_out = L1Adir + LOC + '_' + ID + '_' + 'PROMICE' + '_' + input_type + '_' + "L1A_" + version + '_' + 'all' + ".nc"

        # Convert to xarray
        ds = out.to_xarray()
        ds.to_netcdf(file_out, mode='w')

    elif Input_fname == 'GCNet':
        file =  glob.glob("*csv")
        print(file)
        out = load_GCnet(file[0])
        if L0_raddir:
             os.chdir('../' + L0_raddir)  
             files =  glob.glob("*dat")
             fid = 0
             for file in files:
                if fid == 0:
                    out_rad = load_RAD(file)
                else:
                    tmp = load_RAD(file)
                    out_rad = pd.concat([out_rad,tmp])
                fid = fid + 1
             os.chdir(L0dir) 
             out_rad = out_rad[~out_rad.index.duplicated(keep='last')]
             
             out = pd.concat([out,out_rad],axis=1)
             print(out)
        

        file_out = L1Adir + LOC + '_' + ID + '_' + 'GCNet' + '_' + input_type + '_' + "L1A_" + version + '_' + 'all' + ".nc"

        # Convert to xarray
        ds = out.to_xarray()
        ds.to_netcdf(file_out, mode='w')

    if Input_fname == 'PIG_ENV':
                
        files =  sorted(glob.glob("*.dat"), key=lambda x: x[0:-4])

        # Load raw file in a dataframe
        fid = 0
        for file in files:
            print(file)
            if fid == 0:
                out = load_PIG_ENV(os.getcwd() + "/" + file,nml)
            else:
                # print(file)
                tmp = load_PIG_ENV(os.getcwd() + "/" + file,nml)
                # Load raw file in a dataframe
                out = pd.concat([out,tmp])
            fid = fid + 1
            
        out = out[~out.index.duplicated(keep='last')]


        file_out = L1Adir + LOC + '_' + ID + '_' + 'USAP' + '_' + input_type + '_' + "L1A_" + version + '_' + 'all' + ".nc"
            
        # Convert to xarray
        ds = out.to_xarray()
        
        # Attributes
        ds.attrs['location']                  = LOC + '_' + ID
        ds.attrs['file_creation_date_time']   = str(datetime.datetime.now())
        ds.attrs['IceEddie_namelist_GLOBAL']  = str(nml['global'])
        
        ds.attrs['IceEddie_namelist_AWScorr'] = str(nml['AWScorr'])
        # Export to net CDF
        ds.to_netcdf(file_out, mode='w')

    elif Input_fname == 'snowfox':
                
        files =  sorted(glob.glob("*.203"), key=lambda x: x[0:-4])

        # Load raw file in a dataframe
        fid = 0
        for file in files:
            # print(file)
            if fid == 0:
                out = load_snowfox(os.getcwd() + "/" + file,nml)
            else:
                # print(file)
                tmp = load_snowfox(os.getcwd() + "/" + file,nml)
                # Load raw file in a dataframe
                out = pd.concat([out,tmp])
            fid = fid + 1
            
        out = out[~out.index.duplicated(keep='last')]


        file_out = L1Adir + LOC + '_' + ID + '_' + 'snowfox' + '_' + "L1A_" + version + '_' + 'all' + ".nc"
            
        # Convert to xarray
        ds = out.to_xarray()
        
        # Export to net CDF
        ds.to_netcdf(file_out, mode='w')
        
    elif Input_fname == 'snowfox_irridium':
                
        files =  sorted(glob.glob("*snowfox*.txt"), key=lambda x: x[0:-4])

        # Load raw file in a dataframe
        fid = 0
        for file in files:
            # print(file)
            if fid == 0:
                out = load_snowfox_irridium(os.getcwd() + "/" + file,nml)
            else:
                # print(file)
                tmp = load_snowfox_irridium(os.getcwd() + "/" + file,nml)
                # Load raw file in a dataframe
                out = pd.concat([out,tmp])
            fid = fid + 1
            
        out = out[~out.index.duplicated(keep='last')]

        # print(out)
        file_out = L1Adir + LOC + '_' + ID + '_' + 'snowfox_irridium' + '_' + "L1A_" + version + '_' + 'all' + ".nc"
            
        # Convert to xarray
        ds = out.to_xarray()
        
        # Export to net CDF
        ds.to_netcdf(file_out, mode='w')
        
    elif Input_fname == 'snowfox_irridium_14':
                
        files =  sorted(glob.glob("*snowfox*.txt"), key=lambda x: x[0:-4])

        # Load raw file in a dataframe
        fid = 0
        for file in files:
            # print(file)
            if fid == 0:
                out = load_snowfox_irridium_14(os.getcwd() + "/" + file,nml)
            else:
                # print(file)
                tmp = load_snowfox_irridium_14(os.getcwd() + "/" + file,nml)
                # Load raw file in a dataframe
                out = pd.concat([out,tmp])
            fid = fid + 1
            
        out = out[~out.index.duplicated(keep='last')]

        # print(out)
        file_out = L1Adir + LOC + '_' + ID + '_' + 'snowfox_irridium' + '_' + "L1A_" + version + '_' + 'all' + ".nc"
            
        # Convert to xarray
        ds = out.to_xarray()
        
        # Export to net CDF
        ds.to_netcdf(file_out, mode='w')

    elif Input_fname == 'snowfox_irridium_09':
                
        files =  sorted(glob.glob("*snowfox*.txt"), key=lambda x: x[0:-4])

        # Load raw file in a dataframe
        fid = 0
        for file in files:
            # print(file)
            if fid == 0:
                out = load_snowfox_irridium_09(os.getcwd() + "/" + file,nml)
            else:
                # print(file)
                tmp = load_snowfox_irridium_09(os.getcwd() + "/" + file,nml)
                # Load raw file in a dataframe
                out = pd.concat([out,tmp])
            fid = fid + 1
            
        out = out[~out.index.duplicated(keep='last')]

        # print(out)
        file_out = L1Adir + LOC + '_' + ID + '_' + 'snowfox_irridium' + '_' + "L1A_" + version + '_' + 'all' + ".nc"
            
        # Convert to xarray
        ds = out.to_xarray()
        
        # Export to net CDF
        ds.to_netcdf(file_out, mode='w')


    elif Input_fname == 'iWS_Carleen':
        file =  sorted(glob.glob("*.txt"), key=lambda x: x[1:])[0]
        print(file)
        out = load_iWS_Carleen(file,nml)

        out = out[~out.index.duplicated(keep='last')]

        # print(out)
        file_out = L1Adir + LOC + '_' + ID + '_' + 'iWS' + '_' + "L1A_" + version + '_' + 'all' + ".nc"
            
        # Convert to xarray
        ds = out.to_xarray()
        
        # Export to net CDF
        ds.to_netcdf(file_out, mode='w')
                                 

#_______________________________________________________
def mergeL1A(nml):
    """
    Merges all L1A files from different sensors in one folder. Removes duplicates.
    
    Input
    ----------
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    """

    base_dir = nml['AWScorr']['DIR']
    merged_L1Adir = nml['AWScorr']['DIR'] + 'L1A/'
    ID			= nml['global']['ID']    
    LOC			= nml['global']['LOC']
    sensor     = nml['global']['sensor']
    version		= nml['global']['version']
    input_type	= nml['global']['input_type']
    
    os.chdir(base_dir)
    L1Adirs = sorted(glob.glob("*L1A*"))
    print(L1Adirs)
    fid = 0
    for L1Adir in  L1Adirs:
        os.chdir(L1Adir)
        if fid == 0:
            file = glob.glob("*L1A*.nc")[0]
            print(file)
            ds = xr.open_dataset(file)
            metafile = glob.glob('*meta.csv*')
            if metafile:
                print(metafile)
                meta = pd.read_csv(metafile[0],delimiter=',',header=0,decimal=".",na_values=-999)
                yyyymmdd_start = meta['yyyymmdd_start'][0]
                yyyymmdd_end = meta['yyyymmdd_end'][0]
                print('keeping data from ' + L1Adir + ' between ' + yyyymmdd_start + ' and ' + yyyymmdd_end)
                ds  = ds.sortby('time')
                ds  = ds.sel(time=slice(yyyymmdd_start, yyyymmdd_end))

        else:
            file = glob.glob("*L1A*.nc")[0]
            print(file)
            tmp = xr.open_dataset(file)
            metafile = glob.glob('*meta.csv*')
            if metafile:
                
                meta = pd.read_csv(metafile[0],delimiter=',',header=0,decimal=".",na_values=-999)
                yyyymmdd_start = meta['yyyymmdd_start'][0]
                yyyymmdd_end = meta['yyyymmdd_end'][0]
                print('keeping data from ' + L1Adir + ' between ' + yyyymmdd_start + ' and ' + yyyymmdd_end)
                tmp  = tmp.sortby('time')
                tmp  = tmp.sel(time=slice(yyyymmdd_start, yyyymmdd_end))

            ds = xr.merge([ds,tmp])

        fid = fid + 1
        os.chdir(base_dir)
        
    if not os.path.exists(merged_L1Adir):
        os.makedirs(merged_L1Adir) 
        
    os.chdir(merged_L1Adir)
    
    file_out = merged_L1Adir + LOC + '_' + ID + '_' + sensor + '_' + input_type + '_' + "L1A_" + version +  '_all.nc'

    if "Datetime" in list(ds.keys()):
        ds = ds.drop_vars(["Datetime"])
    
    ds.to_netcdf(file_out, mode='w')
    

   
#_______________________________________________________
def L1AtoL1B_snowfox(nml):
    """
    Reads all uncorrected snowfox data and applies correciton using pressure and reference neutron counts
    Input
    ----------
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    """  
    L1Adir = nml['AWScorr']['DIR'] + 'L1A/'
    L1Bdir = nml['AWScorr']['DIR'] + 'L1B/'
    L1B_REF_dir  = nml['AWScorr']['L1B_REF_dir'] 
    tref_start = nml['AWScorr']['tref_start'] 
    tref_end = nml['AWScorr']['tref_end'] 
    lref_method = nml['AWScorr']['lref_method'] 
    lref_method_var = nml['AWScorr']['lref_method_var'] 
    yyyymmdd_start = nml['AWScorr']['yyyymmdd_start']
    yyyymmdd_end = nml['AWScorr']['yyyymmdd_end']
    
    if not os.path.exists(L1Bdir):
        os.makedirs(L1Bdir) 
        
    # Move to input data directory
    if not L1Adir:
        os.chdir(os.getcwd())  
    else:
        os.chdir(L1Adir)  
    
    files = sorted(glob.glob("*L1A*nc"))
    for fid in range(len(files)):
        # Open file with raw snowfox data
        file = files[fid]
        print(file)
        ds = xr.open_dataset(file)
        # ds  = ds.sel(time=slice(yyyymmdd_start, yyyymmdd_end))

        ds['N1Cts'][ds['N1Cts'] < 50]  = np.nan
        if 'N2Cts'in list(ds.keys()):
            ds['N2Cts'][ds['N2Cts'] < 50]  = np.nan

        # print(ds.time)
        # Open file with reference, pressure corrected data downloaded from https://www.nmdb.eu/nest/search.php
        file_REF = ''
        if lref_method == 'L1B_ref_NMDB':
            if L1B_REF_dir:
                os.chdir(L1B_REF_dir) 
                file_REF = glob.glob("*txt") 
                if not file_REF:
                    file_REF = ''
                    print('no REF data found')
                    os.chdir(L1Adir)  
                else:
                    df_REF = pd.read_csv(file_REF[0],delimiter = ';',skiprows=26,names = ['start_date_time','RCORR_P'])
                    df_REF.index  = pd.to_datetime(df_REF.start_date_time)
                    df_REF.index.names = ['time']
                    ds_REF = df_REF.to_xarray()
                    # Inteporlate reference data on measurement times
                    ds_REF = ds_REF.interp(time=ds.time)
                    ds['RCORR_P'] = ds_REF.RCORR_P
                    ds['Nref'] = ds_REF.RCORR_P
                    os.chdir(L1Adir)  
        elif lref_method == 'L1B_ref':
            if L1B_REF_dir:
                os.chdir(L1B_REF_dir) 
                file_REF = glob.glob("*nc") 
                if not file_REF:
                    file_REF = ''
                    print('no REF data found')
                    os.chdir(L1Adir)  
                else:
                    ds_REF = xr.open_dataset(file_REF[0])
                    # Inteporlate reference data on measurement times
                    ds_REF = ds_REF.interp(time=ds.time)
                    ds['RCORR_P'] = ds_REF.RCORR_P
                    ds['Nref'] = ds_REF.RCORR_P
                    os.chdir(L1Adir) 
        elif lref_method == 'L1A':
            ds['RCORR_P'] = ds[lref_method_var]
            ds['Nref'] = ds[lref_method_var]
        else:
            print('no REF data found')
                        
        ds = ds.sortby('time')

        # compute reference values during snow free period
        P0  = np.nanmean(ds.P4_mb.sel(time=slice(tref_start, tref_end)))
        Fi0 = np.nanmean(ds.Nref.sel(time=slice(tref_start, tref_end)))
        print(Fi0)

        # 1. Correction for atmospheric pressure 
        L = 130
        fp = np.exp((ds.P4_mb-P0)/L)
        ds['fp'] = fp
        
        # apply pressure correction on reference data
        if lref_method == 'L1A':
            ds['RCORR_P'] = ds['RCORR_P']*ds['fp']
            ds['Nref']  = ds['RCORR_P']
            Fi0 = np.nanmean(ds.Nref.sel(time=slice(tref_start, tref_end)))
            print(Fi0)
            
        # 2. Correction for variations in incoming cosmic ray intensity
        beta = 1.0
        ds['Nref'] = ds.RCORR_P
        fi = 1+beta*((Fi0/ds['Nref'])-1)
        ds['fi'] = fi
        
        # 3. Apply corrections to measured neutron counting rate
        if (lref_method_var == 'N1Cts') & (lref_method == 'L1A'):
            ds['Ncorr'] = ds['N2Cts']*ds['fp']*ds['fi']
        else:
            ds['Ncorr'] = ds['N1Cts']*ds['fp']*ds['fi']

        # 4. Estimate SWE from corrected count rate
        N0 =  np.nanmean(ds.Ncorr.sel(time=slice(tref_start, tref_end))) # snow free corrected counts
        ds['Nrel'] = ds['Ncorr']/N0

        Lambda_max = 114.4
        Lambda_min = 14.11
        a1 = 0.313 # 0.355 # 3.133e-1 # 1.133e-13.133e-1;
        a2 = 0.082 # 8.268e-2
        a3 = 1.117
        Lambda =  (1/Lambda_max) + ((1/Lambda_min) -  (1/Lambda_max)) * (1+np.exp((-(ds['Nrel']-a1)/a2)))**(-a3) 

        ds['SWE'] = -1 * np.log(ds['Nrel'] ) / Lambda
        
        # 5. Save corrected data
        ds.to_netcdf(L1Bdir + file.replace('L1A','L1B')) 
        
        
    
#_______________________________________________________
def L1AtoL1B(nml):
    """
    Reads the raw L1A netCDF data files, removes outliers, removes spikes, applies corrections, compute solar angles and mass balance terms
    
    Input
    ----------
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    """

    L1Adir = nml['AWScorr']['DIR'] + 'L1A/'
    L1Bdir = nml['AWScorr']['DIR'] + 'L1B/'
    L1B_precip_dir  = nml['AWScorr']['L1B_precip_dir'] 
    L1B_WS_dir  = nml['AWScorr']['L1B_WS_dir'] 
    L1B_RH_dir  = nml['AWScorr']['L1B_RH_dir'] 
    H      = nml['AWScorr']['H'] 
    d      = nml['AWScorr']['d'] 
    SMB    = nml['AWScorr']['SMB'] 
    lcalCNR4 = nml['AWScorr']['lcalCNR4'] 
    ID			= nml['global']['ID']    
    LOC			= nml['global']['LOC']
    sensor  = nml['global']['sensor']    
    lsolar  = nml['AWScorr']['lsolar']    
    zrange = nml['AWScorr']['zrange']    
    lmetafile = nml['AWScorr']['lmetafile']    
    version		= nml['global']['version']
    input_type	= nml['global']['input_type']
    lcorr_LWheating = nml['AWScorr']['lcorr_LWheating']    
    pcorr_LWheating = nml['AWScorr']['pcorr_LWheating']  
    lcorr_SWcoldbias = nml['AWScorr']['lcorr_SWcoldbias']  
    pcorr_SWu_coldbias = nml['AWScorr']['pcorr_SWu_coldbias']  
    pcorr_SWd_coldbias = nml['AWScorr']['pcorr_SWd_coldbias']  
    lcorrect_SWd = nml['AWScorr']['lcorrect_SWd']  
    lfilter_height_histogram = nml['AWScorr']['lfilter_height_histogram']  
    zsnow_start = nml['AWScorr']['zsnow_start'] 
    ldrop_ABL =  nml['AWScorr']['ldrop_ABL'] 
    l_correctRH =  nml['AWScorr']['l_correctRH'] 
    zm_factor =  nml['AWScorr']['zm_factor']
    alb_ice_certain =  nml['AWScorr']['alb_ice_certain']
    
    pl = nml['AWScorr']['pl']
    pu = nml['AWScorr']['pu']
    lSMB = nml['AWScorr']['lSMB'] 
    p0_constant = nml['AWScorr']['p0_constant'] * 100
    yyyymmdd_start = nml['AWScorr']['yyyymmdd_start']
    yyyymmdd_end = nml['AWScorr']['yyyymmdd_end']
    zfill = nml['AWScorr']['zfill']
    sensor_height = nml['AWScorr']['sensor_height']

                               
    if not os.path.exists(L1Bdir):
        os.makedirs(L1Bdir) 
        
    # Move to input data directory
    if not L1Adir:
        os.chdir(os.getcwd())  
    else:
        os.chdir(L1Adir)  
        
 

    files = sorted(glob.glob("*L1A*nc"))
    for fid in range(len(files)):
        file = files[fid]
        print(file)
        get_LATLON = False
        ds = xr.open_dataset(file)
        
        # Remove duplicate indices if any
        _, index = np.unique(ds['time'], return_index=True)
        ds = ds.isel(time=index)
        
        # Only keep data after start time
        ds  = ds.sel(time=slice(yyyymmdd_start, yyyymmdd_end))

        # Sort in time
        ds = ds.sortby('time')
        
        if sensor == 'PROMICE_inst':
            # Rename variables
            ds['LAT'] = ds['LatitudeGPS(degN)']
            ds['LON'] = -ds['LongitudeGPS(degW)']
        if sensor == 'PROMICE_inst':
            # Rename variables
            ds['LAT'] = ds['gps_lat']
            ds['LON'] = ds['gps_lon']
        ds['zbmin'] = xr.DataArray(np.full(len(ds.time),zrange[0]),dims=['time'])
        ds['zsmin']  = xr.DataArray(np.full(len(ds.time),zrange[0]),dims=['time'])
        ds['zbmax']  = xr.DataArray(np.full(len(ds.time),zrange[1]),dims=['time'])
        ds['zsmax']  = xr.DataArray(np.full(len(ds.time),zrange[1]),dims=['time'])
        ds['ABL_sensor']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
        if sensor == 'AWI_neuma':
            ds['ACC_sensor']  = xr.DataArray(np.full(len(ds.time),4),dims=['time'])
        else:
            ds['ACC_sensor']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
        ds['yaw']  = xr.DataArray(np.full(len(ds.time),0.0),dims=['time'])
        ds['zt']  = xr.DataArray(np.full(len(ds.time),zfill),dims=['time'])
        ds['zu']  = xr.DataArray(np.full(len(ds.time),zfill),dims=['time'])
        ds['lzboom']  = xr.DataArray(np.full(len(ds.time),1.0),dims=['time'])
        ds['lzstake']  = xr.DataArray(np.full(len(ds.time),1.0),dims=['time'])
        ds['linterpzboom']  = xr.DataArray(np.full(len(ds.time),1.0),dims=['time'])
        ds['zm_factor']  = xr.DataArray(np.full(len(ds.time),zm_factor),dims=['time'])
        ds['U_factor']  = xr.DataArray(np.full(len(ds.time),1.0),dims=['time'])
        ds['lincl']  = xr.DataArray(np.full(len(ds.time),0.0),dims=['time'])
        ds['ldata']  = xr.DataArray(np.full(len(ds.time),1.0),dims=['time'])
        ds['lbaro']  = xr.DataArray(np.full(len(ds.time),1.0),dims=['time'])
        ds['lanemo']  = xr.DataArray(np.full(len(ds.time),1.0),dims=['time'])
        ds['l_correctRH']  = xr.DataArray(np.full(len(ds.time),0.0),dims=['time'])
        ds['l_correctT']  = xr.DataArray(np.full(len(ds.time),0.0),dims=['time'])
        ds['lhum']  = xr.DataArray(np.full(len(ds.time),1.0),dims=['time'])
        pcorr_SWd1 = xr.DataArray(np.full(len(ds.time),pcorr_SWd_coldbias[0]),dims=['time'])
        pcorr_SWd2 = xr.DataArray(np.full(len(ds.time),pcorr_SWd_coldbias[1]),dims=['time'])
        pcorr_SWu1 = xr.DataArray(np.full(len(ds.time),pcorr_SWu_coldbias[0]),dims=['time'])
        pcorr_SWu2 = xr.DataArray(np.full(len(ds.time),pcorr_SWu_coldbias[1]),dims=['time'])
        pcorr_LW1 = xr.DataArray(np.full(len(ds.time),pcorr_LWheating[0]),dims=['time'])
        pcorr_LW2 = xr.DataArray(np.full(len(ds.time),pcorr_LWheating[1]),dims=['time'])
        
        if not 'LON' in list(ds.keys()) or not 'LAT' in list(ds.keys()) :
            get_LATLON = True
            ds['LON'] = xr.DataArray(np.full(len(ds.time),0.0),dims=['time'])
            ds['LAT'] = xr.DataArray(np.full(len(ds.time),0.0),dims=['time'])
        if (np.isnan(ds['LON'].mean())) | (np.isnan(ds['LAT'].mean())):
            get_LATLON = True
            ds['LON'] = xr.DataArray(np.full(len(ds.time),0.0),dims=['time'])
            ds['LAT'] = xr.DataArray(np.full(len(ds.time),0.0),dims=['time'])

        if (sensor == 'PROMICE_hourly') | (sensor == 'PROMICE_inst') | (sensor == 'PROMICE_v04'):
            ds['ABL_sensor']  = xr.DataArray(np.full(len(ds.time),2),dims=['time'])
            
        if sensor == 'AWS':
            ds['ABL_sensor']  = xr.DataArray(np.full(len(ds.time),1),dims=['time'])
        
        #Open metadata
        if lmetafile:
            metafile = glob.glob(L1Adir + '*meta.csv*')
            meta = pd.read_csv(metafile[0],delimiter=',',header=0,decimal=".",na_values=-999)
            meta = meta.apply(pd.to_numeric, errors='coerce')
            meta.index       = compose_date(meta['year'], meta['month'],meta['day'])
            meta.index.names = ['time']
            meta             = meta[~meta.index.duplicated()]
            meta = meta.loc[meta.index.notnull()]
            # Find metadat for each time stamp
            idx = np.full(len(ds.time),0)
            for i in range(len(ds.time.values)):
                try:
                    idx[i] = meta.index.get_indexer(ds.time.values,method='ffill')[i]
                except:
                    idx[i] = meta.index.get_indexer(ds.time.values,method='bfill')[i]

                ds['zbmin'].values[i] = meta['zbmin'].iloc[idx[i]]
                ds['zsmin'].values[i] = meta['zsmin'].iloc[idx[i]]
                ds['zbmax'].values[i] = meta['zbmax'].iloc[idx[i]]
                ds['zsmax'].values[i] = meta['zsmax'].iloc[idx[i]]
                ds['ABL_sensor'].values[i] = meta['ABL_sensor'].iloc[idx[i]]
                ds['ACC_sensor'].values[i] = meta['ACC_sensor'].iloc[idx[i]]
                ds['yaw'].values[i] = meta['yaw'].iloc[idx[i]]
                pcorr_SWd1[i] = meta['pcorr_SWd1'].iloc[idx[i]]
                pcorr_SWd2[i] = meta['pcorr_SWd2'].iloc[idx[i]]
                pcorr_SWu1[i] = meta['pcorr_SWu1'].iloc[idx[i]]
                pcorr_SWu2[i] = meta['pcorr_SWu2'].iloc[idx[i]]
                pcorr_LW1[i] = meta['pcorr_LW1'].iloc[idx[i]]
                pcorr_LW2[i] = meta['pcorr_LW2'].iloc[idx[i]]
                ds['zt'].values[i] = meta['zt'].iloc[idx[i]]
                ds['zu'].values[i] = meta['zu'].iloc[idx[i]]
                ds['lzboom'].values[i] = meta['lzboom'].iloc[idx[i]]
                ds['lzstake'].values[i] = meta['lzstake'].iloc[idx[i]]
                ds['linterpzboom'].values[i] = meta['linterpzboom'].iloc[idx[i]]
                ds['zm_factor'].values[i] = meta['zm_factor'].iloc[idx[i]]
                ds['U_factor'].values[i] = meta['U_factor'].iloc[idx[i]]
                ds['lincl'].values[i] = meta['lincl'].iloc[idx[i]]
                ds['ldata'].values[i] = meta['ldata'].iloc[idx[i]]
                ds['lbaro'].values[i] = meta['lbaro'].iloc[idx[i]]
                ds['lanemo'].values[i] = meta['lanemo'].iloc[idx[i]]
                ds['l_correctRH'].values[i] = meta['l_correctRH'].iloc[idx[i]]
                ds['l_correctT'].values[i] = meta['l_correctT'].iloc[idx[i]]
                ds['lhum'].values[i] = meta['lhum'].iloc[idx[i]]
                
                
                if get_LATLON:
                    ds['LON'].values[i] = meta['LON'].iloc[idx[i]]
                    ds['LAT'].values[i] = meta['LAT'].iloc[idx[i]]
            
        if sensor == 'PROMICE_hourly':
            # Rename variables
            ds['U'] = ds['wspd']
            ds['WD'] = ds['wdir']
            ds['p0'] = ds['pa']
            ds['T0'] = ds['ta']
            ds['RH'] = ds['rh_wrtwater']
            ds['zm'] = ds['height_sensor_boom']
            ds['zstakes'] = ds['height_sensor_boom']
            ds['LWu'] = ds['flus']
            ds['SWu'] = ds['fsus']
            ds['SWu_corr'] = ds['fsus_cor']            
            ds['SWd_corr'] = ds['fsds_cor']
            ds['LWd'] = ds['flds']
            ds['SWd'] = ds['fsds']
            ds['depth_pressure_transducer_cor'] = ds['depth_pressure_transducer_cor']
            ds = ds.drop({'station_name','time_bounds'})
        
        if sensor == 'PROMICE_inst':
            # Rename variables
            ds['U'] = ds['WindSpeedms']
            ds['WD'] = ds['WindDirectiond']
            ds['p0'] = ds['P'] * 100
            ds['T0'] = ds['AirTemperatureC'] + 273.15
            ds['RH'] = ds['RelativeHumidity_wrtWater%']
            ds['zm'] = ds['HeightSensorBoomm']
            ds['zstakes'] = ds['HeightStakesm']
            ds['LWu'] = ds['LongwaveRadiationUpWm2']
            ds['SWu'] = ds['ShortwaveRadiationUpWm2']
            ds['SWu_corr'] = ds['ShortwaveRadiationUp_CorWm2']            
            ds['SWd_corr'] = ds['ShortwaveRadiationDown_CorWm2']
            ds['LWd'] = ds['LongwaveRadiationDownWm2']
            ds['SWd'] = ds['ShortwaveRadiationDownWm2']
            ds['depth_pressure_transducer_cor'] = ds['DepthPressureTransducer_Corm']
            
        if sensor == 'PROMICE_v04':
            # Rename variables
            ds['U'] = ds['wspd_u']
            ds['WD'] = ds['wdir_u']
            ds['p0'] = ds['p_u'] * 100
            ds['T0'] = ds['t_u'] + 273.15
            ds['RH'] = ds['rh_u_cor']
            ds['zm'] = ds['z_boom_u']
            if not 'z_stake' in list(ds.keys()):
                ds['z_stake'] = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
            ds['zstakes'] = ds['z_stake']
            ds['LWu'] = ds['ulr']
            ds['SWu'] = ds['usr']
            ds['SWu_corr'] = ds['usr_cor']            
            ds['SWd_corr'] = ds['dsr_cor']
            ds['LWd'] = ds['dlr']
            ds['SWd'] = ds['dsr']
            if not 'z_pt_cor' in list(ds.keys()):
                ds['z_pt_cor'] = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
            ds['depth_pressure_transducer_cor'] = ds['z_pt_cor']
            ds['TILTX'] = ds['tilt_x']
            ds['TILTY'] = ds['tilt_y']
            ds['yaw'] = ds['rot']

        if sensor == 'AWI_neuma':
            # Rename variables
            ds['U'] = ds['U10']
            ds['WD'] = ds['WD10']
            ds['p0'] = ds['P'] * 100
            ds['T0'] = ds['T2'] + 273.15
            ds['RH'] = ds['RH2']
            ds['zm'] = 10-ds['Snowhm']
            if not 'z_stake' in list(ds.keys()):
                ds['z_stake'] = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
            ds['zstakes'] = ds['z_stake']
            ds['LWu'] = ds['LWUWm2']
            ds['SWu'] = ds['SWUWm2']
            ds['LWd'] = ds['LWDWm2']
            ds['SWd'] = ds['SWDWm2']
            if not 'z_pt_cor' in list(ds.keys()):
                ds['z_pt_cor'] = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])

        if sensor == 'PIG_ENV':
            # Rename variables
            ds['U'] = ds['Wind_Speed_WVc1']
            ds['WD'] = ds['WD']
            ds['p0'] = ds['p0'] 
            ds['T0'] = ds['T0'] 
            ds['RH'] = ds['Rel_Humidity']
            ds['zm'] = ds['Snow_Avg']/100
            ds['z_stake'] = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
            ds['zstakes'] = ds['z_stake']
            ds['LWu'] = ds['Long_UpCo_Avg']
            ds['SWu'] = ds['Short_Up_Avg']
            ds['LWd'] = ds['Long_DownCo_Avg']
            ds['SWd'] = ds['Short_Down_Avg']
            if not 'z_pt_cor' in list(ds.keys()):
                ds['z_pt_cor'] = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])

        if sensor == 'GCNet':
            # Rename variables
            ds['U'] = ds['VW2']
            ds['WD'] = ds['DW2']
            ds['p0'] = ds['P'] * 100
            ds['T0'] = ds['TA2'] + 273.15
            ds['RH'] = ds['RH2_cor']
            ds['zm'] = ds['HS2']
            ds['zu'] = ds['HW2']
            if not 'z_stake' in list(ds.keys()):
                ds['z_stake'] = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
            ds['zstakes'] = ds['z_stake']
            ds['LWu'] = ds['U_IR']
            ds['SWu'] = ds['U_GLOBAL']
            ds['SWu_corr'] = ds['U_GLOBAL']            
            ds['SWd_corr'] = ds['D_GLOBAL']
            ds['LWd'] = ds['D_IR']
            ds['SWd'] = ds['D_GLOBAL']
            if not 'z_pt_cor' in list(ds.keys()):
                ds['z_pt_cor'] = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
            ds['depth_pressure_transducer_cor'] = ds['z_pt_cor']
            if not 'tilt_x' in list(ds.keys()):
                ds['tilt_x'] = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
                ds['tilt_y'] = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
                ds['rot'] = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
            ds['TILTX'] = ds['tilt_x']
            ds['TILTY'] = ds['tilt_y']
            ds['yaw'] = ds['rot']

        if not 'zstakes' in list(ds.keys()):
            ds['zstakes'] = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
            ds['T5a'] = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
            ds['T5b'] = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
            ds['quality'] = xr.DataArray(np.full(len(ds.time),1),dims=['time'])

        if sensor == 'AWS':
            ds['quality'] = xr.DataArray(np.full(len(ds.time),1),dims=['time'])
            
        # Initialize some output
        if (sensor != 'PROMICE_hourly') & (sensor != 'PROMICE_inst'):
            ds['sza'] = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
            ds['az']  = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
            ds['aw']  = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
            ds['beta']  = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
            ds['clt']  = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
            ds['fdif']  = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
            ds['SWd_corr']  = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
            ds['SWd_corr']  = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
            
        ds['acc']  = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
        ds['abl']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
        ds['snowmelt']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
        ds['snowmelt_zstakes']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
        ds['acc_zstakes']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
        ds['surface_level']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
        ds['surface_level_acc']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
        ds['surface_level_snowmelt']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
        ds['surface_level_acc_zstakes']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
        ds['surface_level_snowmelt_zstakes']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
        ds['surface_level_abl']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
        ds['surface_level_zm']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
        ds['surface_level_zs']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
        ds['snowheight']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
        ds['tau_cs']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
        ds['SWin_max']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
        ds['paramerror']  = xr.DataArray(np.full(len(ds.time),100000000),dims=['time'])
        

        surface_level  = np.full(len(ds.time),0)
        surface_level_snowmelt  = np.full(len(ds.time),0)
        surface_level_abl  = np.full(len(ds.time),0)
        surface_level_acc  = np.full(len(ds.time),0)

        # Make paramerror
        rr = np.ones(np.shape(ds.time),dtype=np.int64)  # rime if 2, 1 otherwise
        aa = np.zeros(np.shape(ds.time),dtype=np.int64) # wspd 
        bb = np.zeros(np.shape(ds.time),dtype=np.int64) # SWd
        cc = np.zeros(np.shape(ds.time),dtype=np.int64) # SWu
        dd = np.zeros(np.shape(ds.time),dtype=np.int64) # LWd
        ee = np.zeros(np.shape(ds.time),dtype=np.int64) # LWu
        ff = np.zeros(np.shape(ds.time),dtype=np.int64) # T
        gg = np.zeros(np.shape(ds.time),dtype=np.int64) # RH
        hh = np.zeros(np.shape(ds.time),dtype=np.int64) # p0
        ii = np.zeros(np.shape(ds.time),dtype=np.int64)  # zm

        # factor correction for wind speed
        ds['U'] = ds['U']*ds['U_factor']

        if len(ds.time)<144: continue
    
        # Time window for spike removal (14days for 30min files)
        T = 24*2 

        # factor correction for zm
        ds['zm'] = ds['zm']*ds['zm_factor']
        
        # Remove wrong data 
        ds = ds.where(ds['ldata'] != 0.)  
        
        # Use constant pressure if there is no barometer data
        ds['p0'][ds['lbaro'] == 0] = p0_constant
        hh[ds['lbaro'] == 0]  = 1 # p0
        
        # Correct measured relative humidity with respect to water 
        # if l_correctRH[i] == 1:
        # ds['qv0'][ds['l_correctRH'] == 1.] = utils.RH2qv_water(ds['RH'][ds['l_correctRH'] == 1.],ds['p0'][ds['l_correctRH'] == 1.],ds['T0'][ds['l_correctRH'] == 1.])
        # ds['RH'][ds['l_correctRH'] == 1.] = utils.qv2RH(ds['qv0'][ds['l_correctRH'] == 1.],ds['p0'][ds['l_correctRH'] == 1.], ds['T0'][ds['l_correctRH'] == 1.])
            
        # Fill wind speed data from other station
        aa[np.isnan(ds.U)] = 1 # wspd
        file_WS = ''
        if L1B_WS_dir:
            os.chdir(L1B_WS_dir) 
            file_WS = glob.glob("*WS*" + file[ -9:-3] + "*nc") 
            if not file_WS:
                file_WS = ''
                os.chdir(L1Adir)  
            else:
                ds_WS = xr.open_dataset(file_WS[0])
                os.chdir(L1Adir)  
        if file_WS:     
            ds['U'] = ds['U'].combine_first(ds_WS['U'])
            ds['WD'] = ds['WD'].combine_first(ds_WS['WD'])
            ds_WS.close()

        # Fill relative humidity data from other station
        gg[np.isnan(ds.RH)] = 1 # RH
        file_RH = ''
        if L1B_RH_dir:
            os.chdir(L1B_RH_dir) 
            file_RH = glob.glob("*WS*" + "*nc") 
            if not file_RH:
                file_RH = ''
                os.chdir(L1Adir)  
            else:
                ds_RH = xr.open_dataset(file_RH[0])
                os.chdir(L1Adir)  
        if file_RH:     
            ds_RH['lhum'] = ds['lhum']
            ds['RH'][ds['lhum'] == 0.] = ds_RH['RH'][ds_RH['lhum'] == 0.]
            ds_RH.close()
        else:
            ds['RH'][ds['lhum'] == 0.] = np.nanmedian(ds['RH'][ds['lhum'] == 1.])

        # Consistency checks
        if not get_LATLON:
            ds['LAT'][ds['LAT'] < -90] = np.nan
            ds['LAT'][ds['LAT'] > 90] = np.nan
            ds['LON'][ds['LON'] < -180] = np.nan
            ds['LON'][ds['LON'] > 360] = np.nan
            ds['LAT'][(abs(ds['LAT'] - np.nanmean(ds['LAT']))) > 3*np.nanstd(ds['LAT']) ]  = np.nan
            ds['LON'][(abs(ds['LON'] - np.nanmean(ds['LON']))) > 3*np.nanstd(ds['LON']) ]  = np.nan
        ds['zm'][(ds['zm'])>ds['zbmax']] = np.nan     
        ds['zm'][(ds['zm'])<ds['zbmin']] = np.nan  
        ds['zm'][(ds['zm'])<0.1] = np.nan  
        ds['zm'][(ds['zm'])>30] = np.nan  
        
        ds['zstakes'][(ds['zstakes'])>ds['zsmax']] = np.nan     
        ds['zstakes'][(ds['zstakes'])<ds['zsmin']] = np.nan  
        ds['zstakes'][(ds['zstakes'])<0.1] = np.nan  
        ds['zstakes'][(ds['zstakes'])>30] = np.nan  

        gg[ds['RH'] > 110] = 1 # RH
        ds['RH'][ds['RH'] > 150]  =  np.nan
        ds['RH'][ds['RH'] <= 0]    = np.nan
        ds['RH'][(abs(ds['RH'] - np.nanmean(ds['RH']))) > 5*np.nanstd(ds['RH']) ]  = np.nan
        
        ds['T0'][ds['T0'] > 273.15+30]  = np.nan
        ds['T0'][ds['T0'] < 273.15-90]  = np.nan
        ds['U'][ds['U'] < 0]  = np.nan
        ds['U'][ds['U'] > 100]  = np.nan
        ds['WD'][ds['WD'] < 0]  = np.nan
        ds['WD'][ds['WD'] > 360]  = np.nan
        ds['p0'][ds['p0'] < 50000]  = np.nan
        ds['p0'][ds['p0'] > 110000]  = np.nan
        ds['SWd'][ds['SWd'] < -20]  = np.nan
        ds['SWd'][ds['SWd'] > 1500]  = np.nan
        ds['SWu'][ds['SWu'] < -20]  = np.nan
        ds['SWu'][ds['SWu'] > 1000]  = np.nan
        ds['LWu'][ds['LWu'] < 50]  = np.nan
        ds['LWu'][ds['LWu'] > 500]  = np.nan
        ds['LWd'][ds['LWd'] < 50]  = np.nan
        ds['LWd'][ds['LWd'] > 500]  = np.nan
        if 'LWd_raw' in list(ds.keys()):
            ds['LWd_raw'][ds['LWd_raw'] < 50]  = np.nan
            ds['LWd_raw'][ds['LWd_raw'] > 500]  = np.nan
            ds['LWu_raw'][ds['LWu_raw'] < 50]  = np.nan
            ds['LWu_raw'][ds['LWu_raw'] > 500]  = np.nan
            if not 'dL' in list(ds.keys()):
                ds['dL']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
        else:   
            ds['LWd_raw'] = ds['LWd']
            ds['LWu_raw'] = ds['LWu']
            ds['dL']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])

        if not 'T0_raw' in list(ds.keys()):
                ds['T0_raw']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
        if not 'RH_raw' in list(ds.keys()):
                ds['RH_raw']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
        if not 'T1a' in list(ds.keys()):
                ds['T1a']  = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
        if 'Vbat' in list(ds.keys()) and not 'VBAT' in list(ds.keys()):
            ds = ds.rename({'Vbat': 'VBAT'})
        
        if 'TILTX' in list(ds.keys()):
                ds['TILTX'][ds['TILTX'] > 30]  = np.nan
                ds['TILTX'][ds['TILTX'] < -30] = np.nan
                ds['TILTX'][(abs(ds['TILTX'] - np.nanmean(ds['TILTX']))) > 3*np.nanstd(ds['TILTX']) ]  = np.nan
                ds['TILTY'][ds['TILTY'] > 30]  = np.nan
                ds['TILTY'][ds['TILTY'] < -30] = np.nan
                ds['TILTY'][(abs(ds['TILTY'] - np.nanmean(ds['TILTY']))) > 3*np.nanstd(ds['TILTY']) ]  = np.nan
                ds['TILTX'] = ds['TILTX'].rolling(time=48,center=True,min_periods=4).mean()
                ds['TILTY'] = ds['TILTY'].rolling(time=48,center=True,min_periods=4).mean()
                
        if 'ADW' in list(ds.keys()):
            if ldrop_ABL:
                ds['ADW'] = ds['ADW']*0
            else:
                ds['ADW'][ds['ADW'] > 20]  = np.nan
                ds['ADW'][ds['ADW'] < 0]  = np.nan
                ds['ADW'][(abs(ds['ADW'] - np.nanmean(ds['ADW']))) > 4*np.nanstd(ds['ADW']) ]  = np.nan
                ds['ADW']         = rec.despike(ds['ADW'],T=T,q=1)

        # filter PTA data from PROMICE
        elif (sensor == 'PROMICE_hourly') | (sensor == 'PROMICE_inst') | (sensor == 'PROMICE_v04'):
            ds['depth_pressure_transducer_cor'][(abs(ds['depth_pressure_transducer_cor'] - np.nanmean(ds['depth_pressure_transducer_cor']))) > 4*np.nanstd(ds['depth_pressure_transducer_cor']) ]  = np.nan
            ds['depth_pressure_transducer_cor'] = rec.despike(ds['depth_pressure_transducer_cor'],T=T,q=1)
            ds['depth_pressure_transducer_cor'] = ds['depth_pressure_transducer_cor'].rolling(time=48,center=True,min_periods=4).mean()
        else:
            ds['ADW'] = ds['T0']*0.0
            
        if lcalCNR4:
            # old and correct calbration coeeficients from CNR4
            cal_SWu_old = 9.40
            cal_SWd_old = 10.32
            cal_LWu_old = 10.85
            cal_LWd_old = 8.38
            
            cal_SWu_new = 13.36
            cal_SWd_new = 13.72
            cal_LWu_new = 8.54
            cal_LWd_new = 6.90
            
            # Pyranometer calibration correction
            ds['SWu'] = (cal_SWu_old /  cal_SWu_new) * ds['SWu']
            ds['SWd'] = (cal_SWd_old /  cal_SWd_new) * ds['SWd']
            
            # Pyrgeometer calibration correction
            ds['LWu'] = (cal_LWu_old /  cal_LWu_new) * (ds['LWu'] -  utils.boltz*ds['NRT']**4) + utils.boltz*ds['NRT']**4
            ds['LWd'] = (cal_LWd_old /  cal_LWd_new) * (ds['LWd'] -  utils.boltz*ds['NRT']**4) + utils.boltz*ds['NRT']**4  
        
        if lcorr_LWheating: # Correction of shortwave heating of pyrgeometer window (see Smeets 2018) 
            ds['LWu'] = ds['LWu'] - (pcorr_LW1 *  ds['SWu'] + pcorr_LW2)
            ds['LWd'] = ds['LWd'] - (pcorr_LW1 *  ds['SWd'] + pcorr_LW2)
            
        if lcorr_SWcoldbias: # Correction of longwave cooling of pyranometer (see Foken 2021) 
            ds['SWu_raw'] = ds['SWu']
            ds['SWd_raw'] = ds['SWd']
            ds['SWu'] = ds['SWu']- (pcorr_SWu1*  (ds['LWd']-ds['LWu']) + pcorr_SWu2)
            ds['SWd'] = ds['SWd']- (pcorr_SWd1*  (ds['LWd']-ds['LWu']) + pcorr_SWd2)
            ds['SWu'][pcorr_SWu1 > 0] = ds['SWu'][pcorr_SWu1 > 0] - (pcorr_SWu1[pcorr_SWu1 > 0] *  (ds['LWd'][pcorr_SWu1 > 0]-ds['LWu'][pcorr_SWu1 > 0]) + pcorr_SWu2[pcorr_SWu1 > 0])
            ds['SWd'][pcorr_SWu1 > 0] = ds['SWd'][pcorr_SWu1 > 0] - (pcorr_SWd1[pcorr_SWu1 > 0] *  (ds['LWd'][pcorr_SWu1 > 0]-ds['LWu'][pcorr_SWu1 > 0]) + pcorr_SWd2[pcorr_SWu1 > 0])
            ds['SWu'][np.isnan(ds['SWu'])] = ds['SWu_raw'][np.isnan(ds['SWu'])] 
            ds['SWd'][np.isnan(ds['SWd'])] = ds['SWd_raw'][np.isnan(ds['SWd'])]            
            
        ds['SWd'][ds['SWd'] < 0]      = 0
        ds['SWd'][ds['SWd'] > 1500]   = np.nan
        ds['SWu'][ds['SWu'] < 0]      = 0
        ds['SWu'][ds['SWu'] > 800]    = np.nan
        
        ds['LWd'][ds['LWd'] < 50]     = np.nan
        ds['LWd'][ds['LWd'] > 500]    = np.nan
        ds['LWu'][ds['LWu'] < 50]     = np.nan
        ds['LWu'][ds['LWu'] > 500]    = np.nan
        
        ### Filter sonic height ranger measurements
        # Remove wrong sonic ranger data 
        ds['zm'][ds['lzboom'] < 1] = np.nan
        ds['zstakes'][ds['lzstake'] < 1] = np.nan
        ## Remove secondary reflections
        if lfilter_height_histogram:
            x = ds['zm'].values
            hist, bin_edges = np.histogram(x,bins=np.arange(0,5,0.2))
            modes = np.sort(hist)
            N1 = modes[-1]
            N2 = modes[-2]
            max1 =  bin_edges[np.where(hist==N1)][0]+0.1
            max2 =  bin_edges[np.where(hist==N2)][0]+0.1
            if abs(max2-max1) > 0.6:
                x[x<np.max([max1,max2])-0.05] = np.nan
            elif abs(max2-max1) < 0.2:
                x = x
            else:
                x[x<np.min([max1,max2])-0.2] = np.nan
            ds['zm'].values = x
            # Spike removal
            ds['zm'][(abs(ds['zm'] - np.nanmean(ds['zm']))) > 4*np.nanstd(ds['zm']) ]  = np.nan
        ds['zm']         = rec.despike(ds['zm'],T=T,q=2)
        
        # Linear detrending in daily windows
        df  = ds['zm'].to_pandas()
        grouped = df.groupby(pd.Grouper(freq="1D",label='left'))
        var  =  grouped.apply(lambda x: vardetrend(x))
        var_m                     = var.reindex(pd.date_range(min(df.index), max(df.index), freq='0.5H')).ffill()
        df[var_m>0.1] = np.nan
        ds['zm'].values = df
         
        # interpolate
        x1        = ds['zm'].rolling(time=144,center=True,min_periods=1).mean()
        x         = ds['zm'].rolling(time=48,center=True,min_periods=4).mean()
        x[0:4] = x1[0:4]
        x[-4:-1] = x1[-4:-1]
        ds['zm'] = x.interpolate_na('time')
        ds['zm'] = ds['zm'].interpolate_na('time')
       
        # Filter stake measurements
        if 'zstakes' in list(ds.keys()):
            # Despiking
            ds['zstakes']  = rec.despike(ds['zstakes'],T=T,q=1)
            # Linear detrendog in daily window
            df  = ds['zstakes'].to_pandas()
            grouped = df.groupby(pd.Grouper(freq="1D",label='left'))
            var  =  grouped.apply(lambda x: vardetrend(x))
            var_m                     = var.reindex(pd.date_range(min(df.index), max(df.index), freq='0.5H')).ffill()
            df[var_m>0.1] = np.nan
            ds['zstakes'].values = df
            # interpolate
            x1        = ds['zstakes'].rolling(time=144,center=True,min_periods=1).mean()
            x         = ds['zstakes'].rolling(time=48,center=True,min_periods=4).mean()
            x[0:4] = x1[0:4]
            x[-4:-1] = x1[-4:-1]
            ds['zstakes'] = x.interpolate_na('time')
            
        # make paramerror
        aa[np.isnan(ds.U)] = 1 # wspd
        bb[np.isnan(ds.SWd)] = 1 # SWd
        cc[np.isnan(ds.SWu)] = 1 # SWu
        dd[np.isnan(ds.LWd)] = 1 # LWd
        ee[np.isnan(ds.LWu)] = 1 # LWd
        ff[np.isnan(ds.T0)] = 1 # T
        gg[np.isnan(ds.RH)] = 1 # RH
        gg[ds.lhum.values == 0] = 1 # RH
        hh[np.isnan(ds.p0)] = 1 # p0
        ii[ds.lzboom.values == 0] = 1
        ii[np.isnan(ds.zm)] = 1 # zm
       
        # Model for obstacle heightt
        ds['H']   = xr.DataArray(np.full(len(ds.time),H),dims=['time'])
        
        # Model for displacement height
        ds['d']   = xr.DataArray(np.full(len(ds.time),d),dims=['time'])
        
        # Effective height
        ds['z']   = ds['zm']  + ds['H'] - ds['d']
        ds['z'][ds['linterpzboom'] < 1] = ds['zu'][ds['linterpzboom'] < 1]  + ds['H'][ds['linterpzboom'] < 1] - ds['snowheight'][ds['linterpzboom'] < 1]
        
        # specific humidity 
        ds['qv0'] = utils.RH2qv(ds['RH'],ds['p0'], ds['T0'])
        
        # Potential temperature
        ds['th0'] = utils.T2thl(ds['T0'],ds['p0'],ql=0)

        # Surface temperature 
        ds['Ts']  = (ds['LWu'] / utils.boltz) ** (1/4)
        ds['Ts'][ds['Ts'] > 273.15] = 273.15
        if 'LWd_raw' in list(ds.keys()):
            ds['Ts_raw']  = (ds['LWu_raw'] / utils.boltz) ** (1/4)
            ds['Ts_raw'][ds['Ts_raw'] > 273.15] = 273.15

        # Air virtual temperature
        Tv0 = (ds['T0'])*(1+0.608*ds['qv0'])
        
        # Surface pressure
        ps = ds['p0']/np.exp(-utils.grav*ds['z']/(utils.rd*Tv0))
        
        # Surface potential temperature
        ds['ths'] = utils.T2thl(ds['Ts'],ps,ql=0)
        
        # Surface humidity assuming saturation
        ds['qvs'] = utils.RH2qv(100,ps, ds['Ts'])
        if 'LWd_raw' in list(ds.keys()):
            ds['qvs_raw'] = utils.RH2qv(100,ps, ds['Ts_raw'])
        
        # Air density and heat capacity
        ds['rho_a'] = utils.rho_air(ds.T0,ds.p0,ds.qv0)
        ds['Cp']    = utils.Cp(ds.qv0)

        if (sensor == 'PROMICE_hourly') | (sensor == 'PROMICE_inst'):
            if lcorrect_SWd:
                ds['SWu'][ds['sza'] >= 90] = 0
                ds['SWd'][ds['sza'] >= 90] = 0
            
        # Filter SWdown
        ds['albedo_acc']   = (ds['SWu'].rolling(time=48,center=True,min_periods=4).reduce(np.nansum)) / (ds['SWd'].rolling(time=48,center=True,min_periods=4).reduce(np.nansum))
        ds['SWd_acc'] = ds['SWu'] / ds['albedo_acc'] 
        
        # Fill nonphyscial values
        ds['albedo_acc'][ds['albedo_acc'] > 0.95] = 0.95
        ds['albedo_acc'][ds['albedo_acc'] < 0.1] = 0.1
        if sensor == 'ANT_AWS_Paul':
            ds['U'][(ds['U'] > ds['Umax'] ) & (ds['Umax'].notnull())] = np.nan
            ds['Umax'][np.isnan(ds['U'])] = np.nan
        
        # Flag suspicious wind speed data when wspd < 0.1 m/s for more than 6 consecutive hours
        condition = (ds['U'] < 0.1 ) | (np.isnan(ds['U']))
        rolling_sum = condition.rolling(time=6, center=True).sum()
        rolling_sum_2 = condition.rolling(time=6, center=False).sum()
        result = (rolling_sum >= 6) | (rolling_sum_2 >= 6) 
        indices_where_condition_met = np.where(result)[0]
        aa[indices_where_condition_met] = 1
        aa[np.isnan(ds.U)] = 1 # wspd
        ds['U'][ds['U'] < 0.1] = 0.1
        
        # Compute solar position
        if lsolar:
            print('Calculating solar position + radiometer tilt correction')
            ds['time_solar'] = ds['time'] - np.timedelta64(30,'m')
            angles = pd.DataFrame(suncalc.get_position(ds['time_solar'], ds['LON'].rolling(time=100, center=True).median(skipna=True).fillna(ds.LON.median(skipna=True)), ds['LAT'].rolling(time=100, center=True).median(skipna=True).fillna(ds.LAT.median(skipna=True))),index = ds.time)

            ds['az'] = xr.DataArray(angles['azimuth'].values*180/np.pi,dims=['time'])
            ds['sza'] =  xr.DataArray(90-angles['altitude'].values*180/np.pi,dims=['time'])

            ### Cloud fraction estimation
            LWdclear  = pl[2] + pl[1]* (ds['T0']-273.15) + pl[0]* (ds['T0']-273.15)**2
            LWdcloudy = pu[2] + pu[1]* (ds['T0']-273.15 + pu[0])* (ds['T0']-273.15)**2 
            ds['clt']= (ds['LWd']-LWdclear)/(LWdcloudy-LWdclear)

            # Fraction of diffuse radiation
            ds['fdif'] = 0.2 + (1-0.2)*(ds['LWd']-LWdclear)/(LWdcloudy-LWdclear)
            ds['fdif'][ds['fdif'] > 1] = 1
            ds['fdif'][ds['fdif'] <0.2] = 0.2
            ds['clt'][ds['clt'] > 1 ] = 1
            ds['clt'][ds['clt'] < 0 ] = 0

            #  clear sky shortwave absorption
            ds['tau_cs']=1.021-0.0824*((9.49e-6*ds['p0']+0.051)/np.cos(ds['sza']*np.pi/180))**0.5
            ds['tau_cs'][ds['tau_cs']< 0] =  0
            ds['tau_cs'][ds['sza'] > 90] = 0
            SWglobal = 1361 
            ds['SWin_TOA'] = SWglobal*np.sin((90-ds['sza'])*np.pi/180)
            ds['SWin_TOA'][ds['SWin_TOA'] < 0] = 0
            ds['SWin_max'] = ds['tau_cs']*ds['SWin_TOA']
            ds['SWd_corr'] = ds['SWd']

            if lcorrect_SWd:
                print('Radiometer tilt correction using geometric equations') 
                for i in range(len(ds.time.values)):
                    if ds['lincl'][i] == 1:
                        ### iWS tilt angles
                        ds['aw'][i], ds['beta'][i] = PRY_to_NED(-ds['TILTX'][i],-ds['TILTY'][i],ds['yaw'][i])
                        
                        ### Tilt correction
                        # Geometric relation
                        sza = ds['sza'][i]*np.pi/180
                        az = ds['az'][i]*np.pi/180
                        aw = ds['aw'][i]*np.pi/180
                        beta = ds['beta'][i]*np.pi/180
                        fdif = ds['fdif'][i] 
                        SWd = ds['SWd'][i] 
                        alpha = ds['albedo_acc'][i] 
                        
                        cosi = np.sin(sza)*np.cos(az-aw)*np.sin(beta) + np.cos(sza)*np.cos(beta)
                        
                        # correction factor 
                        if cosi < 0:
                            # no correction when sun below tilted plane
                            A = 1
                        else:
                            A = (np.cos(sza) + fdif)/(cosi+fdif*(1+np.cos(beta))/2+alpha*(np.cos(sza)+fdif)*(1-np.cos(beta))/2)
                        
                        # corrected SWd 
                        # SWd_corr = SWd * (A/(1-fdif+A*fdif))
                        SWd_corr = SWd * A
                        if np.isnan(SWd_corr):
                            SWd_corr = SWd
                            
                        ds['SWd_corr'][i] = SWd_corr
                    else:
                        ds['SWd_corr'][i] = ds['SWd'][i] 
                        
                ds['SWd_corr'][ds['SWd_corr'] > 1360] = 1360

            # flag daily data based on daily max SWd at surface
            SWdd = ds['SWd'].rolling(time=24,center=True,min_periods=4).reduce(np.nansum)
            SWmaxd =  ds['SWin_TOA'].rolling(time=24,center=True,min_periods=4).reduce(np.nansum)
            bb[(SWdd > SWmaxd) & (SWmaxd> 10)] = 1 #SWd
            cc[(SWdd > SWmaxd) & (SWmaxd> 10)] = 1 #SWu    

            # New 24h albedo
            ds['albedo_acc']   = (ds['SWu'].rolling(time=24,center=True,min_periods=4).reduce(np.nansum)) / (ds['SWd'].rolling(time=24,center=True,min_periods=4).reduce(np.nansum))
            ds['albedo_acc'][(ds['SWu'].rolling(time=24,center=True,min_periods=4).reduce(np.nansum) == 0) & (SWmaxd> 10)] = np.nan
            ds['albedo_acc'][(ds['SWd'].rolling(time=24,center=True,min_periods=4).reduce(np.nansum) == 0)  & (SWmaxd> 10)] = np.nan

            # Fill nonphyscial values
            ds['albedo_acc'][ds['albedo_acc'] > 0.95] = 0.95
            ds['albedo_acc'][ds['albedo_acc'] < 0.1] = 0.1
            ds['SWd_acc'] = ds['SWu'] / ds['albedo_acc'] 
            bb[(ds['SWd'].rolling(time=24,center=True,min_periods=4).reduce(np.nansum) == 0) & (SWmaxd> 10)] = 1 #SWd
            cc[(ds['SWu'].rolling(time=24,center=True,min_periods=4).reduce(np.nansum) == 0)  & (SWmaxd> 10)] = 1 #SWu

            # Remove unphysical measurements
            ds['SWu'][ds['SWd_acc']< ds['SWu']]  = ds['SWd_acc'][ds['SWd_acc']< ds['SWu']]
            ds['SWu'][ds['albedo_acc'].isnull()] == np.nan
        else:
            ds['SWin_max']  = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
            ds['SWin_TOA']  = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])

        # Flag suspicious data 
        dd[ds['LWu'] > 320]  = 1 # LWd
        ee[ds['LWu'] > 320]  = 1 # LWu
        dd[ds['T0'] < 273.15-60]  = 1 # LWd
        ee[ds['T0'] < 273.15-60]  = 1 # LWu

        if 'LWd_raw' not in list(ds.keys()):
            ds['LWd_raw']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
            ds['LWu_raw']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
            ds['dL']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
            ds['qvs_raw']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
            ds['Ts_raw']  = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
            
            

        # Find when there is riming of sensors are close to being burried (r = 2)
        df  = ds.to_pandas()
        # Define the condition: 
        condition = (((abs(df['LWd']-df['LWu']) < 2 ) & (df['RH'] > 90)  & (df['T0'] < 273.15)) | (df['LWd'].isna()) | (df['LWu'].isna()) | (df['RH'].isna()) | (df['T0'].isna()))
        rolling_sum_left  = condition.iloc[::-1].rolling('24h', center=False,min_periods = 1).sum()
        rolling_sum_right = condition.rolling('24h', center=False,min_periods = 1).sum()
        rolling_sumdata_left = condition.iloc[::-1].rolling('24h', center=False,min_periods = 1).count()
        rolling_sumdata_right = condition.rolling('24h', center=False,min_periods = 1).count()
        result = ((rolling_sum_left >= rolling_sumdata_left - 2)  | (rolling_sum_right >= rolling_sumdata_right - 2))
        indices_where_condition_met = np.where(result)[0]
        rr[indices_where_condition_met] = 2
        rr[ds.z.values < 0.2] = 2

        # Make paramerror from individual binary flags
        def concat(r,a,b,c,d,e,f,g,h,i):
            return int(f"{r}{a}{b}{c}{d}{e}{f}{g}{h}{i}")
        for j in range(len(aa)):
           ds['paramerror'][j] = concat(rr[j],ii[j],hh[j],gg[j],ff[j],ee[j],dd[j],cc[j],bb[j],aa[j])
        
        # Prepare level L1B output dataset
        if sensor == 'AWS_2L':
            ds['Rh2'][ds['Rh2'] > 100]  = 100
            ds['Rh2'][ds['Rh2'] < 0]    = 0
            ds['Rh2'][(abs(ds['Rh2'] - np.nanmean(ds['Rh2']))) > 3*np.nanstd(ds['Rh2']) ]  = np.nan

            ds['Rh1'][ds['Rh1'] > 100]  = 100
            ds['Rh1'][ds['Rh1'] < 0]    = 0
            ds['Rh1'][(abs(ds['Rh1'] - np.nanmean(ds['Rh1']))) > 3*np.nanstd(ds['Rh1']) ]  = np.nan

            ds['qv2'] = utils.RH2qv(ds['Rh2'],ds['p0'], ds['T2'])
            ds['qv1'] = utils.RH2qv(ds['Rh1'],ds['p0'], ds['T1'])
            
            ds['th2'] = utils.T2thl(ds['T2'],ds['p0'],ql=0)
            ds['th1'] = utils.T2thl(ds['T1'],ds['p0'],ql=0)
            

            # Make a new dataset with new variable names
            ds_out = xr.Dataset({'U': ds.U, 'U2': ds.u2,'U1': ds.u1, 'WD': ds.WD, 'WD2': ds.dd2,'WD1': ds.dd1,'p0': ds.p0, \
                             'T0': ds.T0,'T1': ds.T1,'T2': ds.T2,'th0': ds.th0, 'th1': ds.th1,'th2': ds.th2,'qv0': ds.qv0,'qv1': ds.qv1,'qv2': ds.qv2, \
                             'Ts': ds.Ts,'ths': ds.ths, 'qvs': ds.qvs, 'RH':ds.RH, 'rho_a':ds.rho_a, 'Cp':ds.Cp, \
                             'LWd': ds.LWd, 'LWu': ds.LWu, 'SWd': ds.SWd, 'SWd_acc': ds.SWd_acc,'SWu': ds.SWu, \
                             'albedo_acc': ds.albedo_acc,'z': ds.z,'zt': ds.zt,'zu': ds.zu,'zm': ds.zm, 'H': ds.H,'d': ds.d,\
                                 'sza': ds.sza, 'az':ds.az, 'acc': ds.acc, 'abl':ds.abl, 'snowmelt':ds.snowmelt, 'surface_level': ds.surface_level, \
                                     'snowheight': ds.snowheight, 'surface_level_acc': ds.surface_level_acc, 'surface_level_snowmelt': ds.surface_level_snowmelt, \
                                         'surface_level_acc_zstakes': ds.surface_level_acc_zstakes, 'surface_level_snowmelt_zstakes': ds.surface_level_snowmelt_zstakes, \
                                         'surface_level_abl': ds.surface_level_abl,'clt': ds.clt, 'fdif': ds.fdif, 'ABL_sensor': ds.ABL_sensor, 'ACC_sensor': ds.ACC_sensor, 'zstakes': ds.zstakes, 'surface_level_zm': ds.surface_level_zm, 'surface_level_zs': ds.surface_level_zs, \
                                             'lzboom':ds.lzboom,'ldata':ds.ldata, 'linterpzboom': ds.linterpzboom, 'ADW': ds.ADW,'TILTX': ds.TILTX, 'TILTY': ds.TILTY, 'aw': ds.aw, 'beta': ds.beta,'paramerror':ds.paramerror, \
                                             'LWd_raw': ds.LWd_raw,'LWu_raw': ds.LWu_raw,'dL': ds.dL,'Ts_raw': ds.Ts_raw, 'qvs_raw': ds.qvs_raw,'SWin_max':ds.SWin_max}) 
        elif sensor == 'AWS':

            # Make a new dataset with new variable names
            ds_out = xr.Dataset({'U': ds.U, 'WD': ds.WD,'p0': ds.p0, \
                             'T0': ds.T0,'th0': ds.th0,'qv0': ds.qv0, \
                             'Ts': ds.Ts,'ths': ds.ths, 'qvs': ds.qvs, 'RH':ds.RH,'rho_a':ds.rho_a, 'Cp':ds.Cp, \
                             'LWd': ds.LWd, 'LWu': ds.LWu, 'SWd': ds.SWd, 'SWd_acc': ds.SWd_acc,'SWu': ds.SWu, \
                             'albedo_acc': ds.albedo_acc,'z': ds.z,'zm': ds.zm, 'H': ds.H,'d': ds.d,\
                                 'sza': ds.sza, 'az':ds.az, 'acc': ds.acc, 'abl':ds.abl, 'snowmelt':ds.snowmelt, 'surface_level': ds.surface_level, \
                                     'snowheight': ds.snowheight, 'surface_level_acc': ds.surface_level_acc, 'surface_level_snowmelt': ds.surface_level_snowmelt, \
                                         'surface_level_acc_zstakes': ds.surface_level_acc_zstakes, 'surface_level_snowmelt_zstakes': ds.surface_level_snowmelt_zstakes, 'zstakes': ds.zstakes, 'surface_level_zm': ds.surface_level_zm, 'surface_level_zs': ds.surface_level_zs,\
                                         'surface_level_abl': ds.surface_level_abl,'clt': ds.clt, 'fdif': ds.fdif, 'ABL_sensor': ds.ABL_sensor, 'ACC_sensor': ds.ACC_sensor, \
                                             'lzboom':ds.lzboom,'lzstake':ds.lzstake,'ldata':ds.ldata,'linterpzboom': ds.linterpzboom,'TILTX': ds.Angle1, 'TILTY': ds.Angle2,'zu': ds.zu, 'tau_cs' : ds.tau_cs, 'aw': ds.aw, 'beta': ds.beta,'paramerror':ds.paramerror, \
                                             'LWd_raw': ds.LWd_raw,'LWu_raw': ds.LWu_raw,'dL': ds.dL,'Ts_raw': ds.Ts_raw, 'qvs_raw': ds.qvs_raw, \
                                                 'T1a': ds.Tsubs1,'T1b': ds.Tsubs2,'T2a': ds.Tsubs3,'T2b': ds.Tsubs4,'T3a': ds.Tsubs5, \
                                                     'Tlogger': ds.Tlogger, 'quality': ds.quality, 'Umax': ds.u1max,'lbaro':ds.lbaro,'lhum':ds.lhum,'lanemo':ds.lanemo, 'Vbat': ds.Ubattery,}) #'T3b': ds.Tsubs6,'T4a': ds.Tsubs7,'T4b': ds.Tsubs8, 'T5a': ds.Tsubs9, 'T5b': ds.Tsubs10,
        elif sensor == 'AWS_S9':

            # Make a new dataset with new variable names
            ds_out = xr.Dataset({'U': ds.U, 'WD': ds.WD,'p0': ds.p0, \
                             'T0': ds.T0,'th0': ds.th0,'qv0': ds.qv0, \
                             'Ts': ds.Ts,'ths': ds.ths, 'qvs': ds.qvs, 'RH':ds.RH,'rho_a':ds.rho_a, 'Cp':ds.Cp, \
                             'LWd': ds.LWd, 'LWu': ds.LWu, 'SWd': ds.SWd, 'SWd_acc': ds.SWd_acc,'SWu': ds.SWu, \
                             'albedo_acc': ds.albedo_acc,'z': ds.z,'zm': ds.zm, 'H': ds.H,'d': ds.d,\
                                 'sza': ds.sza, 'az':ds.az, 'acc': ds.acc, 'abl':ds.abl, 'snowmelt':ds.snowmelt, 'surface_level': ds.surface_level, \
                                     'snowheight': ds.snowheight, 'surface_level_acc': ds.surface_level_acc, 'surface_level_snowmelt': ds.surface_level_snowmelt, \
                                         'surface_level_acc_zstakes': ds.surface_level_acc_zstakes, 'surface_level_snowmelt_zstakes': ds.surface_level_snowmelt_zstakes, 'surface_level_zm': ds.surface_level_zm, 'surface_level_zs': ds.surface_level_zs,\
                                         'surface_level_abl': ds.surface_level_abl,'clt': ds.clt, 'fdif': ds.fdif, 'ABL_sensor': ds.ABL_sensor, 'ACC_sensor': ds.ACC_sensor, \
                                             'lzboom':ds.lzboom,'lzstake':ds.lzstake,'ldata':ds.ldata,'linterpzboom': ds.linterpzboom,'TILTX': ds.Angle1, 'TILTY': ds.Angle2,'zu': ds.zu, 'tau_cs' : ds.tau_cs, 'aw': ds.aw, 'beta': ds.beta,'paramerror':ds.paramerror, \
                                             'LWd_raw': ds.LWd_raw,'LWu_raw': ds.LWu_raw,'dL': ds.dL,'Ts_raw': ds.Ts_raw, 'qvs_raw': ds.qvs_raw, \
                                                 'T1a': ds.Tsubs1,'T2a': ds.Tsubs2,'T3a': ds.Tsubs3,'T4a': ds.Tsubs4,'T5a': ds.Tsubs5,'T1b': ds.TSTR1,'T2b': ds.TSTR2,'T3b': ds.TSTR3, 'T4b': ds.TSTR4, 'T5b': ds.TSTR4, \
                                                     'T6b': ds.TSTR6,'T7b': ds.TSTR7,'T8b': ds.TSTR8,\
                                                     'Tlogger': ds.Tlogger, 'quality': ds.quality, 'Umax': ds.u1max,'lbaro':ds.lbaro,'lhum':ds.lhum,'lanemo':ds.lanemo, 'Vbat': ds.Ubattery,})

        elif (sensor == 'PROMICE_hourly') | (sensor == 'PROMICE_inst')  | (sensor == 'PROMICE_v04') | (sensor == 'GCNet'):
            # Make a new dataset with new variable names
            ds_out = xr.Dataset({'U': ds.U, 'WD': ds.WD, 'p0': ds.p0, \
                            'T0': ds.T0,'th0': ds.th0, 'qv0': ds.qv0, \
                            'Ts': ds.Ts,'ths': ds.ths, 'qvs': ds.qvs, 'RH':ds.RH,'rho_a':ds.rho_a, 'Cp':ds.Cp, \
                            'LWd': ds.LWd, 'LWu': ds.LWu, 'SWd': ds.SWd, 'SWd_acc': ds.SWd_acc,'SWu': ds.SWu, \
                            'albedo_acc': ds.albedo_acc,'z': ds.z,'zm': ds.zm, 'H': ds.H,'d': ds.d, 'depth_pressure_transducer_cor': ds.depth_pressure_transducer_cor, \
                                'sza': ds.sza, 'az':ds.az, \
                                     'acc': ds.acc, 'abl':ds.abl, 'snowmelt':ds.snowmelt, 'surface_level': ds.surface_level, \
                                         'snowheight': ds.snowheight, 'surface_level_acc': ds.surface_level_acc, 'surface_level_snowmelt': ds.surface_level_snowmelt, \
                                             'surface_level_abl': ds.surface_level_abl,'SWd_corr': ds.SWd_corr, 'zstakes': ds.zstakes, \
                                                 'surface_level_acc_zstakes': ds.surface_level_acc_zstakes, 'surface_level_snowmelt_zstakes': ds.surface_level_snowmelt_zstakes, 'surface_level_zm': ds.surface_level_zm, 'surface_level_zs': ds.surface_level_zs, \
                                                     'ABL_sensor': ds.ABL_sensor, 'ACC_sensor': ds.ACC_sensor,'lzboom':ds.lzboom,'lzstake':ds.lzstake,'ldata':ds.ldata,'linterpzboom': ds.linterpzboom,'zu': ds.zu,'TILTX': ds.TILTX, 'TILTY': ds.TILTY, 'yaw': ds.yaw,'paramerror':ds.paramerror, \
                                             'LWd_raw': ds.LWd_raw,'LWu_raw': ds.LWu_raw,'dL': ds.dL,'Ts_raw': ds.Ts_raw, 'qvs_raw': ds.qvs_raw,'LAT': ds.LAT,'LON': ds.LON, 'SWin_TOA': ds.SWin_TOA})    
                
        elif (sensor == 'fluxstation') :
            # Make a new dataset with new variable names
            ds_out = xr.Dataset({'U': ds.U, 'WD': ds.WD, 'p0': ds.p0, \
                                 'T0': ds.T0,'th0': ds.th0, 'qv0': ds.qv0, \
                                 'Ts': ds.Ts,'ths': ds.ths, 'qvs': ds.qvs, 'RH': ds.RH, 'Tcnr': ds.NRT, 'Ubattery': ds.VBAT, \
                                     'TILTX': ds.TILTX, 'TILTY': ds.TILTY, 'rho_a':ds.rho_a, 'Cp':ds.Cp, \
                                 'LWd': ds.LWd, 'LWu': ds.LWu, 'SWd': ds.SWd, 'SWd_acc': ds.SWd_acc,'SWu': ds.SWu, \
                                 'albedo_acc': ds.albedo_acc,'z': ds.z,'zm': ds.zm, 'H': ds.H,'d': ds.d, 'sza': ds.sza, 'az':ds.az, \
                                     'acc': ds.acc, 'abl':ds.abl, 'snowmelt':ds.snowmelt, 'surface_level': ds.surface_level, \
                                        'snowheight': ds.snowheight,  'surface_level_acc': ds.surface_level_acc,\
                                           'surface_level_acc_zstakes': ds.surface_level_acc_zstakes, 'surface_level_snowmelt_zstakes': ds.surface_level_snowmelt_zstakes,'surface_level_zm': ds.surface_level_zm, 'surface_level_zs': ds.surface_level_zs, \
                                               'surface_level_snowmelt': ds.surface_level_snowmelt, 'surface_level_abl': ds.surface_level_abl,'clt': ds.clt, 'fdif': ds.fdif, \
                                                   'aw': ds.aw, 'beta': ds.beta, 'SWd_corr': ds.SWd_corr, 'yaw': ds.yaw, 'ABL_sensor': ds.ABL_sensor, 'ACC_sensor': ds.ACC_sensor,'lzboom':ds.lzboom,'lzstake':ds.lzstake,'ldata':ds.ldata,'linterpzboom': ds.linterpzboom,'zu': ds.zu,'paramerror':ds.paramerror, \
                                             'LWd_raw': ds.LWd_raw,'LWu_raw': ds.LWu_raw,'dL': ds.dL,'Ts_raw': ds.Ts_raw, 'qvs_raw': ds.qvs_raw})
        elif sensor == 'ANT_AWS_Paul':

            # Make a new dataset with new variable names
            ds_out = xr.Dataset({'U': ds.U, 'WD': ds.WD,'p0': ds.p0, \
                             'T0': ds.T0,'T0_raw': ds.T0_raw,'RH_raw': ds.RH_raw,'th0': ds.th0,'qv0': ds.qv0, \
                             'Ts': ds.Ts,'ths': ds.ths, 'qvs': ds.qvs, 'RH':ds.RH,'rho_a':ds.rho_a, 'Cp':ds.Cp, \
                             'LWd': ds.LWd, 'LWu': ds.LWu, 'SWd': ds.SWd, 'SWd_acc': ds.SWd_acc,'SWu': ds.SWu, \
                             'albedo_acc': ds.albedo_acc,'z': ds.z,'zm': ds.zm, 'H': ds.H,'d': ds.d,\
                                 'sza': ds.sza, 'az':ds.az, 'acc': ds.acc, 'abl':ds.abl, 'snowmelt':ds.snowmelt, 'surface_level': ds.surface_level, \
                                     'snowheight': ds.snowheight, 'surface_level_acc': ds.surface_level_acc, 'surface_level_snowmelt': ds.surface_level_snowmelt, \
                                         'surface_level_acc_zstakes': ds.surface_level_acc_zstakes, 'surface_level_snowmelt_zstakes': ds.surface_level_snowmelt_zstakes, 'surface_level_zm': ds.surface_level_zm, 'surface_level_zs': ds.surface_level_zs,\
                                         'surface_level_abl': ds.surface_level_abl,'clt': ds.clt, 'fdif': ds.fdif, 'ABL_sensor': ds.ABL_sensor, 'ACC_sensor': ds.ACC_sensor, \
                                             'lzboom':ds.lzboom,'lzstake':ds.lzstake,'ldata':ds.ldata,'linterpzboom': ds.linterpzboom,'zu': ds.zu, 'tau_cs' : ds.tau_cs, 'aw': ds.aw, 'beta': ds.beta,'paramerror':ds.paramerror, \
                                             'LWd_raw': ds.LWd_raw,'LWu_raw': ds.LWu_raw,'dL': ds.dL,'Ts_raw': ds.Ts_raw, 'qvs_raw': ds.qvs_raw,'quality': ds.quality,'Vbat': ds.VBAT,'TILTX': ds.TILTX, 'TILTY': ds.TILTY, \
                                                 'Tcnr': ds.NRT, 'T1a': ds.T1a, 'T2a': ds.T2a, 'T3a': ds.T3a, 'T4a': ds.T4a, 'T5a': ds.T5a, \
                                                     'T1b': ds.T1b, 'T2b': ds.T2b, 'T3b': ds.T3b, 'Tlogger': ds.Tlogger, 'Umax': ds.Umax,'lbaro':ds.lbaro,'lhum':ds.lhum,'lanemo':ds.lanemo,'SWin_max':ds.SWin_max,'LAT': ds.LAT,'LON': ds.LON, 'SWin_TOA': ds.SWin_TOA})            

        elif sensor == 'AWS_iWS_GRL':
            # Make a new dataset with new variable names
            ds_out = xr.Dataset({'U': ds.U, 'WD': ds.WD, 'p0': ds.p0, \
                                 'T0': ds.T0,'th0': ds.th0, 'qv0': ds.qv0, \
                                 'Ts': ds.Ts,'ths': ds.ths, 'qvs': ds.qvs, 'RH': ds.RH, 'Tcnr': ds.NRT, 'Vbat': ds.VBAT, \
                                     'TILTX': ds.TILTX, 'TILTY': ds.TILTY, 'rho_a':ds.rho_a, 'Cp':ds.Cp, \
                                 'LWd': ds.LWd, 'LWu': ds.LWu, 'SWd': ds.SWd, 'SWd_acc': ds.SWd_acc,'SWu': ds.SWu, \
                                 'albedo_acc': ds.albedo_acc,'z': ds.z,'zm': ds.zm, 'H': ds.H,'d': ds.d, 'sza': ds.sza, 'az':ds.az, 'ADW': ds.ADW, \
                                     'acc': ds.acc, 'abl':ds.abl, 'snowmelt':ds.snowmelt, 'surface_level': ds.surface_level, \
                                        'snowheight': ds.snowheight,  'surface_level_acc': ds.surface_level_acc, 'zstakes': ds.zstakes,\
                                           'surface_level_acc_zstakes': ds.surface_level_acc_zstakes, 'surface_level_snowmelt_zstakes': ds.surface_level_snowmelt_zstakes, 'surface_level_zm': ds.surface_level_zm, 'surface_level_zs': ds.surface_level_zs,\
                                               'surface_level_snowmelt': ds.surface_level_snowmelt, 'surface_level_abl': ds.surface_level_abl,'clt': ds.clt, 'fdif': ds.fdif, \
                                                   'aw': ds.aw, 'beta': ds.beta, 'SWd_corr': ds.SWd_corr, 'yaw': ds.yaw, 'ABL_sensor': ds.ABL_sensor, 'ACC_sensor': ds.ACC_sensor,'lzboom':ds.lzboom,'lzstake':ds.lzstake,'ldata':ds.ldata,'linterpzboom': ds.linterpzboom,'zu': ds.zu, 'tau_cs' : ds.tau_cs,'paramerror':ds.paramerror, \
                                             'LWd_raw': ds.LWd_raw,'LWu_raw': ds.LWu_raw,'dL': ds.dL,'Ts_raw': ds.Ts_raw, 'qvs_raw': ds.qvs_raw, 'lbaro':ds.lbaro,'lhum':ds.lhum,'lanemo':ds.lanemo,'SWin_max':ds.SWin_max,'LAT': ds.LAT,'LON': ds.LON, 'SWin_TOA': ds.SWin_TOA})         
        elif sensor == 'AWI_neuma':

            # Make a new dataset with new variable names
            ds_out = xr.Dataset({'U': ds.U, 'WD': ds.WD,'p0': ds.p0, \
                             'T0': ds.T0,'th0': ds.th0,'qv0': ds.qv0, \
                             'Ts': ds.Ts,'ths': ds.ths, 'qvs': ds.qvs, 'RH':ds.RH,'rho_a':ds.rho_a, 'Cp':ds.Cp, \
                             'LWd': ds.LWd, 'LWu': ds.LWu, 'SWd': ds.SWd, 'SWd_acc': ds.SWd_acc,'SWu': ds.SWu, \
                             'albedo_acc': ds.albedo_acc,'z': ds.z,'zm': ds.zm, 'H': ds.H,'d': ds.d,\
                                 'sza': ds.sza, 'az':ds.az, 'acc': ds.acc, 'abl':ds.abl, 'snowmelt':ds.snowmelt, 'surface_level': ds.surface_level, \
                                     'snowheight': ds.snowheight, 'surface_level_acc': ds.surface_level_acc, 'surface_level_snowmelt': ds.surface_level_snowmelt, \
                                         'surface_level_acc_zstakes': ds.surface_level_acc_zstakes, 'surface_level_snowmelt_zstakes': ds.surface_level_snowmelt_zstakes, 'zstakes': ds.zstakes, 'surface_level_zm': ds.surface_level_zm, 'surface_level_zs': ds.surface_level_zs,\
                                         'surface_level_abl': ds.surface_level_abl,'clt': ds.clt, 'fdif': ds.fdif, 'ABL_sensor': ds.ABL_sensor, 'ACC_sensor': ds.ACC_sensor, \
                                             'lzboom':ds.lzboom,'lzstake':ds.lzstake,'ldata':ds.ldata,'linterpzboom': ds.linterpzboom,'zu': ds.zu, 'tau_cs' : ds.tau_cs, 'aw': ds.aw, 'beta': ds.beta,'paramerror':ds.paramerror, \
                                                     'lbaro':ds.lbaro,'lhum':ds.lhum,'lanemo':ds.lanemo}) #'T3b': ds.Tsubs6,'T4a': ds.Tsubs7,'T4b': ds.Tsubs8, 'T5a': ds.Tsubs9, 'T5b': ds.Tsubs10,
        elif sensor == 'PIG_ENV':

            # Make a new dataset with new variable names
            ds_out = xr.Dataset({'U': ds.U, 'WD': ds.WD,'p0': ds.p0, \
                             'T0': ds.T0,'th0': ds.th0,'qv0': ds.qv0, \
                             'Ts': ds.Ts,'ths': ds.ths, 'qvs': ds.qvs, 'RH':ds.RH,'rho_a':ds.rho_a, 'Cp':ds.Cp, \
                             'LWd': ds.LWd, 'LWu': ds.LWu, 'SWd': ds.SWd, 'SWd_acc': ds.SWd_acc,'SWu': ds.SWu, \
                             'albedo_acc': ds.albedo_acc,'z': ds.z,'zm': ds.zm, 'H': ds.H,'d': ds.d,\
                                 'sza': ds.sza, 'az':ds.az, 'acc': ds.acc, 'abl':ds.abl, 'snowmelt':ds.snowmelt, 'surface_level': ds.surface_level, \
                                     'snowheight': ds.snowheight, 'surface_level_acc': ds.surface_level_acc, 'surface_level_snowmelt': ds.surface_level_snowmelt, \
                                         'surface_level_acc_zstakes': ds.surface_level_acc_zstakes, 'surface_level_snowmelt_zstakes': ds.surface_level_snowmelt_zstakes, 'zstakes': ds.zstakes, 'surface_level_zm': ds.surface_level_zm, 'surface_level_zs': ds.surface_level_zs,\
                                         'surface_level_abl': ds.surface_level_abl,'clt': ds.clt, 'fdif': ds.fdif, 'ABL_sensor': ds.ABL_sensor, 'ACC_sensor': ds.ACC_sensor, \
                                             'lzboom':ds.lzboom,'lzstake':ds.lzstake,'ldata':ds.ldata,'linterpzboom': ds.linterpzboom,'zu': ds.zu, 'tau_cs' : ds.tau_cs, 'aw': ds.aw, 'beta': ds.beta,'paramerror':ds.paramerror, \
                                                     'lbaro':ds.lbaro,'lhum':ds.lhum,'lanemo':ds.lanemo}) #'T3b': ds.Tsubs6,'T4a': ds.Tsubs7,'T4b': ds.Tsubs8, 'T5a': ds.Tsubs9, 'T5b': ds.Tsubs10,
        else:
            # Make a new dataset with new variable names
            ds_out = xr.Dataset({'U': ds.U, 'WD': ds.WD, 'p0': ds.p0, \
                                 'T0': ds.T0,'th0': ds.th0, 'qv0': ds.qv0, \
                                 'Ts': ds.Ts,'ths': ds.ths, 'qvs': ds.qvs, 'RH': ds.RH, 'Tcnr': ds.NRT, 'Vbat': ds.VBAT, \
                                     'TILTX': ds.TILTX, 'TILTY': ds.TILTY, 'rho_a':ds.rho_a, 'Cp':ds.Cp, \
                                 'LWd': ds.LWd, 'LWu': ds.LWu, 'SWd': ds.SWd, 'SWd_acc': ds.SWd_acc,'SWu': ds.SWu, \
                                 'albedo_acc': ds.albedo_acc,'z': ds.z,'zm': ds.zm, 'H': ds.H,'d': ds.d, 'sza': ds.sza, 'az':ds.az, 'ADW': ds.ADW, \
                                     'acc': ds.acc, 'abl':ds.abl, 'snowmelt':ds.snowmelt, 'surface_level': ds.surface_level, \
                                        'snowheight': ds.snowheight,  'surface_level_acc': ds.surface_level_acc,\
                                           'surface_level_acc_zstakes': ds.surface_level_acc_zstakes, 'surface_level_snowmelt_zstakes': ds.surface_level_snowmelt_zstakes, 'surface_level_zm': ds.surface_level_zm, 'surface_level_zs': ds.surface_level_zs,\
                                               'surface_level_snowmelt': ds.surface_level_snowmelt, 'surface_level_abl': ds.surface_level_abl,'clt': ds.clt, 'fdif': ds.fdif, \
                                                   'aw': ds.aw, 'beta': ds.beta, 'SWd_corr': ds.SWd_corr, 'yaw': ds.yaw, 'ABL_sensor': ds.ABL_sensor, 'ACC_sensor': ds.ACC_sensor,'lzboom':ds.lzboom,'lzstake':ds.lzstake,'ldata':ds.ldata,'linterpzboom': ds.linterpzboom,'zu': ds.zu, 'tau_cs' : ds.tau_cs,'paramerror':ds.paramerror, \
                                             'LWd_raw': ds.LWd_raw,'LWu_raw': ds.LWu_raw,'dL': ds.dL,'Ts_raw': ds.Ts_raw, 'qvs_raw': ds.qvs_raw, 'lbaro':ds.lbaro,'lhum':ds.lhum,'lanemo':ds.lanemo, 'SWin_TOA': ds.SWin_TOA})
        
        # Compute SMB components : ice ablation, snowmelt/erosion and snow accumulation
        if lSMB:
            
            def MB(x, axis): # mass balance function...
                if np.all(np.isnan(x)):
                    return 0
                if len(x) < 1:
                    return 0
                return x[np.isfinite(x)][-1] - x[np.isfinite(x)][0] #... Simplest measurement ever
            
            # Open file with precipitation data
            file_precip = ''
            if L1B_precip_dir:
                os.chdir(L1B_precip_dir) 
                file_precip = glob.glob("*WS*" + "*nc") 
                if not file_precip:
                    print('no precip file found')
                    file_precip = ''
                    os.chdir(L1Adir)  
                else:
                    print('precip file found')
                    ds_precip = xr.open_dataset(file_precip[0])
                    ds_precip = ds_precip.where(ds['ldata'] != 0.)  
                    os.chdir(L1Adir)  
                
            time_min = min(ds_out.time)
            time_max = max(ds_out.time)
            
            if file_precip:       
                diff_zm_precip = ds_precip['zm'].resample(time="1D",label='left',closed='left').reduce(MB)  
                diff_zm_precip_flag = ds_precip['ldata'].resample(time="1D",label='left',closed='left').mean() 
                ds_precip.close()

            print('Calculating SMB components')
    
            # Remove wrong sonic ranger data 
            ds_out['ldata'] = ds_out['ldata'].fillna(0) 
            ds_out['zm'][ds_out['ldata'] < 1] = np.nan
            if 'zstakes' in list(ds_out.keys()):
                ds_out['zstakes'][ds_out['ldata'] < 1] = np.nan

            # keep sonic ranger data if interpolated
            ds_out['zm'][ds_out['lzboom']+ds_out['linterpzboom'] < 1] = np.nan
            if 'zstakes' in list(ds_out.keys()):
                ds_out['zstakes'][ds_out['lzstake'] < 1] = np.nan
                
            if (sensor == 'PROMICE_hourly') | (sensor == 'PROMICE_inst')  | (sensor == 'PROMICE_v04') | (sensor == 'GCNet'):
                if SMB == 'daily':
                    diff_zm  = ds_out['zm'].resample(time="1D",label='left',closed='left').reduce(MB)   
                    diff_zstakes  = ds_out['zstakes'].resample(time="1D",label='left',closed='left').reduce(MB)   
                    diff_ADW = ds_out['depth_pressure_transducer_cor'].resample(time="1D",label='left').reduce(MB)     
                elif SMB == '3hourly':
                    diff_zm  = ds_out['zm'].resample(time="3H",label='left',closed='left').reduce(MB)    
                    diff_zstakes  = ds_out['zstakes'].resample(time="3H",label='left',closed='left').reduce(MB)
                    diff_ADW = ds_out['depth_pressure_transducer_cor'].resample(time="3H",label='left').reduce(MB)   
            else:
                if SMB == 'daily':
                    diff_zm  = ds_out['zm'].resample(time="1D",label='left',closed='left').reduce(MB)   
                    if 'ADW' in list(ds_out.keys()):
                        diff_ADW = ds_out['ADW'].resample(time="1D",label='left').reduce(MB)     
                    else:
                        diff_ADW = diff_zm * 0
                    if 'zstakes' in list(ds_out.keys()):
                        diff_zstakes  = ds_out['zstakes'].resample(time="1D",label='left',closed='left').reduce(MB)   
                    else:
                        diff_zstakes  = diff_zm * 0 
                elif SMB == '3hourly':
                    diff_zm  = ds_out['zm'].resample(time="3H",label='left',closed='left').reduce(MB)   
                    if 'ADW' in list(ds_out.keys()):
                        diff_ADW = ds_out['ADW'].resample(time="3H",label='left').reduce(MB)   
                    else:
                        diff_ADW = diff_zm * 0
                    if 'zstakes' in list(ds_out.keys()):
                        diff_zstakes  = ds_out['zstakes'].resample(time="3H",label='left',closed='left').reduce(MB)   
                    else:
                        diff_zstakes = diff_zm * 0
                    
                    
            surface_level_acc = ds_out['surface_level_acc'].resample(time="1D",label='left',closed='left').reduce(MB).ffill(dim = 'time')
            surface_level_abl = ds_out['surface_level_abl'].resample(time="1D",label='left',closed='left').reduce(MB).ffill(dim = 'time')
            surface_level_snowmelt = ds_out['surface_level_snowmelt'].resample(time="1D",label='left',closed='left').reduce(MB).ffill(dim = 'time')
            surface_level_acc_zstakes = ds_out['surface_level_acc_zstakes'].resample(time="1D",label='left',closed='left').reduce(MB).ffill(dim = 'time')
            surface_level_snowmelt_zstakes = ds_out['surface_level_snowmelt_zstakes'].resample(time="1D",label='left',closed='left').reduce(MB).ffill(dim = 'time')
            surface_level = ds_out['surface_level'].resample(time="1D",label='left',closed='left').reduce(MB).ffill(dim = 'time')
            surface_level_zm = ds_out['surface_level_zm'].resample(time="1D",label='left',closed='left').reduce(MB).ffill(dim = 'time')
            surface_level_zs = ds_out['surface_level_zs'].resample(time="1D",label='left',closed='left').reduce(MB).ffill(dim = 'time')
            T0_d = ds_out['T0'].resample(time="1D",label='left',closed='left').reduce(np.nanmax).ffill(dim = 'time')
            alb_d = ds_out['albedo_acc'].resample(time="1D",label='left',closed='left').reduce(np.nanmean).ffill(dim = 'time')
            SWD_d = ds_out['SWd_acc'].resample(time="1D",label='left',closed='left').reduce(np.nanmean).ffill(dim = 'time')
            sza_d = ds_out['sza'].resample(time="1D",label='left',closed='left').reduce(np.nanmean).ffill(dim = 'time')
            ABL_sensor_d = np.rint(ds_out['ABL_sensor'].resample(time="1D",label='left',closed='left').reduce(np.nanmean)).ffill(dim = 'time')
            ACC_sensor_d = np.rint(ds_out['ACC_sensor'].resample(time="1D",label='left',closed='left').reduce(np.nanmean)).ffill(dim = 'time')
            lzboom_d =  np.rint(ds_out['lzboom'].resample(time="1D",label='left',closed='left').reduce(np.nanmean)).ffill(dim = 'time')
            lzstake_d =  np.rint(ds_out['lzstake'].resample(time="1D",label='left',closed='left').reduce(np.nanmean)).ffill(dim = 'time')

            if file_precip:
                acc      = -diff_zm_precip.where(diff_zm_precip <= 0,0) # snow accumulation in m per time period
                acc2     = -diff_zm.where(diff_zm <= 0,0) # snow accumulation in m per time period
                snowmelt =  diff_zm.where(diff_zm > 0,0) # snow melt (or erosion or sublimation) in m per time period
                acc = acc.combine_first(acc2)
            else:
                acc      = -diff_zm.where(diff_zm <= 0.0,0.0) # snow accumulation in m per time period
                snowmelt =  diff_zm.where(diff_zm > 0.0,0) # snow melt (or erosion or sublimation) in m per time period
            abl      = -diff_ADW.where(diff_ADW < 0,0)# ice ablation in in m per time period
            
            if 'zstakes' in list(ds.keys()):
                acc_zstakes = -diff_zstakes.where(diff_zstakes <= 0,0) # snow accumulation in m per time period
                snowmelt_zstakes =  diff_zstakes.where(diff_zstakes > 0,0) # snow melt (or erosion or sublimation) in m per time period
            else:
                acc_zstakes = diff_zm * 0
                snowmelt_zstakes  = diff_zm * 0
            dzm = -diff_zm
            dzs = -diff_zstakes

            
            # remove small and large values
            if (sensor == 'PROMICE_hourly') | (sensor == 'PROMICE_inst') | (sensor == 'PROMICE_v04') | (sensor == 'GCNet'):
                acc = acc.where((acc > 0.001) & (acc < 0.8),0).fillna(0) # 0.01 0.2
                snowmelt = snowmelt.where((snowmelt > 0.001) & (snowmelt < 0.4),0).fillna(0) #0.01 0.4
                acc_zstakes = acc_zstakes.where((acc > 0.001) & (acc_zstakes < 0.8),0).fillna(0) # 0.01 0.2
                snowmelt_zstakes = snowmelt_zstakes.where((snowmelt_zstakes > 0.001) & (snowmelt_zstakes < 0.8),0).fillna(0) #0.01 0.4
                abl = abl.where((T0_d > 273.15) & (alb_d <  alb_ice_certain) & (abl > 0.01 ) & (abl < 0.2),0).fillna(0) # 0.003 0.2
            elif (sensor == 'AWS') | (sensor == 'AWS_1L') | (sensor == 'AWS_S9'):
                acc = acc.where((acc > 0.001) & (acc < 0.8),0).fillna(0) # 0.01 0.2
                snowmelt = snowmelt.where((snowmelt > 0.001) & (snowmelt < 0.4),0).fillna(0) #0.01 0.4
                acc_zstakes = acc.where((acc_zstakes > 0.001) & (acc_zstakes < 0.8),0).fillna(0) # 0.01 0.2
                snowmelt_zstakes = snowmelt_zstakes.where((snowmelt_zstakes > 0.001) & (snowmelt_zstakes < 0.4),0).fillna(0) #0.01 0.4
                abl = abl.where((T0_d > 273.15) & (alb_d <  alb_ice_certain) & (abl > 0.01 ) & (abl < 0.2),0).fillna(0) # 0.003 0.2
            elif (sensor == 'fluxstation'):
                acc = acc.where((acc > 0.001) & (acc < 0.8),0).fillna(0) # 0.01 0.2
                snowmelt = snowmelt.where((snowmelt > 0.001) & (snowmelt < 0.4),0).fillna(0) #0.01 0.4
                abl = abl.where((T0_d > 273.15) & (alb_d <  0.6) & (abl > 0.01 ) & (abl < 0.2),0).fillna(0) # 0.003 0.2
            else:
                acc = acc.where((acc > 0.003) & (acc < 0.2),0).fillna(0) # 0.01 0.2
                snowmelt = snowmelt.where((snowmelt > 0.003) & (snowmelt < 0.4),0).fillna(0) #0.01 0.4
                abl = abl.where((T0_d > 273.15) & (abl > 0.003 ) & (abl < 0.2),0).fillna(0) # 0.003 0.2
            
            # Remove erroneous ablation
            abl = abl.where((ABL_sensor_d > -1),0).fillna(0) 
            
            # Remove accumulation when there is ablation
            ds_d = xr.Dataset({'acc': acc, 'abl': abl, 'snowmelt': snowmelt, 'acc_zstakes': acc_zstakes, 'snowmelt_zstakes': snowmelt_zstakes, 'dzm': dzm, 'dzs': dzs}).fillna(0)
            ds_d['acc'] = ds_d['acc'].where(ds_d['abl']==0,0).fillna(0)
            # snowmelt = snowmelt.where(abl==0,0).fillna(0)
            acc = ds_d['acc'] 
            abl = ds_d['abl'] 
            snowmelt = ds_d['snowmelt'] 
            snowmelt_zstakes = ds_d['snowmelt_zstakes'] 
            acc_zstakes = ds_d['acc_zstakes'] 
            dzm = ds_d['dzm']
            dzs = ds_d['dzs']
            
            acc = acc.fillna(0)
            abl = abl.fillna(0.0)
            snowmelt = snowmelt.fillna(0)
            snowmelt_zstakes = snowmelt_zstakes.fillna(0)
            acc_zstakes = acc_zstakes.fillna(0)
            dzm = dzm.fillna(0)
            dzs = dzs.fillna(0)
            

            print('Setting Serie to 0')
            # start serie from 0 (or from zsnow_start)
            old_snowheight = 0
            old_level = 0
            old_level_acc = zsnow_start
            old_level_snowmelt = 0
            old_level_acc_zstakes = zsnow_start
            old_level_snowmelt_zstakes = 0
            old_level_abl = 0
            old_level_zm = 0
            old_level_zs = 0

            print('old_level_acc = ',old_level_acc)
            print('old_level_snowmelt = ',old_level_snowmelt)
            print('old_level_acc_zstakes = ',old_level_acc_zstakes)
            print('old_level = ',old_level)
                
            # Relative change of surface height
            for k in range(len(acc)):
                if k == 0:
                    surface_level_acc.values[k] = old_level_acc
                    surface_level_snowmelt.values[k] = old_level_snowmelt
                    surface_level_abl.values[k] = old_level_abl
                    surface_level_acc_zstakes.values[k] = old_level_acc_zstakes
                    surface_level_snowmelt_zstakes.values[k] = old_level_snowmelt_zstakes
                    surface_level_zm.values[k] = old_level_zm
                    surface_level_zs.values[k] = old_level_zs
                else:
                    if (ACC_sensor_d[k] == 1): # sonic ranger on stake
                        surface_level_acc.values[k] = surface_level_acc.values[k-1] + acc_zstakes[k]
                        surface_level_snowmelt.values[k] = surface_level_snowmelt.values[k-1] + snowmelt_zstakes[k]        
                        surface_level_acc_zstakes.values[k] = surface_level_acc_zstakes.values[k-1] + acc_zstakes[k]
                        surface_level_snowmelt_zstakes.values[k] = surface_level_snowmelt_zstakes.values[k-1] + snowmelt_zstakes[k]
                        surface_level_zm.values[k] = surface_level_zm.values[k-1] + dzm[k]
                        surface_level_zs.values[k] = surface_level_zs.values[k-1] + dzs[k]
                    elif (ACC_sensor_d[k] == 4): # sonic ranger on boom
                        surface_level_acc.values[k] = surface_level_acc.values[k-1] + acc[k]
                        surface_level_snowmelt.values[k] = surface_level_snowmelt.values[k-1] + snowmelt[k]
                        surface_level_acc_zstakes.values[k] = surface_level_acc_zstakes.values[k-1] + acc_zstakes[k]
                        surface_level_snowmelt_zstakes.values[k] = surface_level_snowmelt_zstakes.values[k-1] + snowmelt_zstakes[k]
                        surface_level_zm.values[k] = surface_level_zm.values[k-1] + dzm[k]
                        surface_level_zs.values[k] = surface_level_zs.values[k-1] + dzs[k]
                    else: #  no data
                        surface_level_acc.values[k] = surface_level_acc.values[k-1] 
                        surface_level_snowmelt.values[k] = surface_level_snowmelt.values[k-1]
                        surface_level_acc_zstakes.values[k] = surface_level_acc_zstakes.values[k-1] 
                        surface_level_snowmelt_zstakes.values[k] = surface_level_snowmelt_zstakes.values[k-1] 
                        surface_level_zm.values[k] = surface_level_zm.values[k-1]
                        surface_level_zs.values[k] = surface_level_zs.values[k-1]
                    
                    # No more snowmelt than snow if height ranger is on AWS boom 
                    if (surface_level_snowmelt.values[k] > surface_level_acc.values[k]):
                        if (ABL_sensor_d[k] == 4):
                            if (sensor == 'iWS') | (sensor == 'AWS') | (sensor == 'AWS_S9'):
                                abl[k] = snowmelt[k] # snow melt is in fact ice ablation
                        surface_level_snowmelt.values[k] = surface_level_acc.values[k]

                    # Artificial 1cm snow when albedo > 0.6
                    if (surface_level_acc.values[k] - surface_level_snowmelt.values[k] < 0.01) & (alb_d.values[k] > 0.6):
                        surface_level_acc.values[k] = surface_level_acc.values[k] + 0.01
                        
                    # No snow when albedo < alb_ice_certain
                    if (alb_d.values[k] < alb_ice_certain) & (sza_d.values[k] < 70):
                        surface_level_acc.values[k] = surface_level_snowmelt.values[k]
                        surface_level_acc_zstakes.values[k] = surface_level_snowmelt_zstakes.values[k]
                        
                    # no snow when albedo is < alb_ice_certain when there is no ablation measurement & no sza estimate
                    if (ABL_sensor_d[k] == -1) & (alb_d.values[k] < alb_ice_certain) & (SWD_d.values[k] > 200):
                        surface_level_acc.values[k] = surface_level_snowmelt.values[k]
                        surface_level_acc_zstakes.values[k] = surface_level_snowmelt_zstakes.values[k]
                    
                    if abl.values[k] > 0:
                        surface_level_snowmelt.values[k] = surface_level_acc.values[k]
                        surface_level_snowmelt_zstakes.values[k] = surface_level_acc_zstakes.values[k]
                        
                    # Ablation measurement
                    if (ABL_sensor_d[k] == 2) | (ABL_sensor_d[k] == 0): # ADW (0) or PTA (2)
                        surface_level_abl.values[k] = surface_level_abl.values[k-1] + abl[k]
                    elif (ABL_sensor_d[k] == 1): # sonic ranger on stake (1)
                        if 'zstakes' in list(ds.keys()):
                            if (sensor == 'PROMICE_hourly') | (sensor == 'PROMICE_inst') | (sensor == 'PROMICE_v04') | (sensor == 'AWS_iWS_GRL'):
                                if (alb_d.values[k] < alb_ice_certain): # not sure if this is needed sicne abl is already computed abvove
                                    surface_level_abl.values[k] = surface_level_abl.values[k-1] + snowmelt_zstakes[k]
                                else:
                                    surface_level_abl.values[k] = surface_level_abl.values[k-1]
                            else: surface_level_abl.values[k] = surface_level_abl.values[k-1] + abl[k]
                        else:
                            surface_level_abl.values[k] = surface_level_abl.values[k-1] + abl[k]
                    elif (ABL_sensor_d[k] == 4): # sonic ranger on boom (1)
                        if (alb_d.values[k] < alb_ice_certain): 
                            surface_level_abl.values[k] = surface_level_abl.values[k-1] + snowmelt_zstakes[k]
                    elif (ABL_sensor_d[k] == -1): # no sensor
                        surface_level_abl.values[k] = surface_level_abl.values[k-1]
                    
            # this is basically the most important thing !
            surface_level  =   surface_level_acc - surface_level_snowmelt  - surface_level_abl
            snowheight =  surface_level_acc - surface_level_snowmelt
            
            # Resample to 30min
            acc      = acc.resample(time = "0.5H").pad(0)
            abl      = abl.resample(time = "0.5H").pad(0)
            snowmelt = snowmelt.resample(time = "0.5H").pad(0)
            acc_zstakes      = acc_zstakes.resample(time = "0.5H").pad(0)
            snowmelt_zstakes = snowmelt_zstakes.resample(time = "0.5H").pad(0)
            
            snowheight = snowheight.resample(time = "0.5H").ffill()
            surface_level = surface_level.resample(time = "0.5H").ffill()
            surface_level_acc = surface_level_acc.resample(time = "0.5H").ffill()
            surface_level_snowmelt = surface_level_snowmelt.resample(time = "0.5H").ffill()
            surface_level_abl = surface_level_abl.resample(time = "0.5H").ffill()
            surface_level_acc_zstakes = surface_level_acc_zstakes.resample(time = "0.5H").ffill()
            surface_level_snowmelt_zstakes = surface_level_snowmelt_zstakes.resample(time = "0.5H").ffill()
            surface_level_zm = surface_level_zm.resample(time = "0.5H").ffill()
            surface_level_zs = surface_level_zs.resample(time = "0.5H").ffill()
            
            ds_out['acc'] = acc
            ds_out['snowmelt'] = snowmelt
            ds_out['abl'] = abl
            ds_out['snowheight'] = snowheight
            ds_out['surface_level'] = surface_level
            ds_out['surface_level_acc'] = surface_level_acc
            ds_out['surface_level_snowmelt'] = surface_level_snowmelt
            ds_out['surface_level_abl'] = surface_level_abl
            ds_out['surface_level_acc_zstakes'] = surface_level_acc_zstakes
            ds_out['surface_level_snowmelt_zstakes'] = surface_level_snowmelt_zstakes
            ds_out['surface_level_zm'] = surface_level_zm
            ds_out['surface_level_zs'] = surface_level_zs
            
            # Merge surface level from zm and surfce level from mixed zm/ADW together
            ds_out['surface_level_merged'] = ds_out['surface_level_zm'].where(ds_out['ABL_sensor'] == 1,0) + ds_out['surface_level'].where(ds_out['ABL_sensor'] == 0 | (ds_out['ABL_sensor'] == 2) ,0)
            
            # Add last SMB measruements interval to output 
            ds_out['snowheight'] = ds_out['snowheight'].fillna(snowheight[-1].values)
            ds_out['acc'] = ds_out['acc'].fillna(0)
            ds_out['snowmelt'] = ds_out['snowmelt'].fillna(0)
            ds_out['abl'] = ds_out['abl'].fillna(0)
            ds_out['surface_level'] = ds_out['surface_level'].fillna(surface_level[-1].values)
            ds_out['surface_level_acc'] = ds_out['surface_level_acc'].fillna(surface_level_acc[-1].values)
            ds_out['surface_level_snowmelt'] = ds_out['surface_level_snowmelt'].fillna(surface_level_snowmelt[-1].values)
            ds_out['surface_level_abl'] = ds_out['surface_level_abl'].fillna(surface_level_abl[-1].values)
            ds_out['surface_level_acc_zstakes'] = ds_out['surface_level_acc_zstakes'].fillna(surface_level_acc_zstakes[-1].values)
            ds_out['surface_level_snowmelt_zstakes'] = ds_out['surface_level_snowmelt_zstakes'].fillna(surface_level_snowmelt_zstakes[-1].values)
            ds_out['surface_level_zm'] = ds_out['surface_level_zm'].fillna(surface_level_zm[-1].values)
            ds_out['surface_level_zs'] = ds_out['surface_level_zs'].fillna(surface_level_zs[-1].values)
            ds_out['surface_level_merged'] = ds_out['surface_level_merged'].fillna(ds_out['surface_level_merged'][-1].values)

            # Convert to mm w.e per 30 min
            if SMB == 'daily':
                ds_out['acc'] *= (1000)*(utils.rho_s/utils.rho_w) # snow accumulation in mm w.e per day
                ds_out['snowmelt'] *= (1000)*(utils.rho_s/utils.rho_w) # snow accumulation in mm w.e per day
                ds_out['abl'] *= (1000)*(utils.rho_i/utils.rho_w) # ice ablation in mm w.e hour-1 per day
            else:
                print('error: use daily SMB or add routine here for other timesteps')

                
        
        # # Set to fixed heihgt if sonic ranger is not working (CHECK IF THIS IS REALLY NECESSARY ! NOT USED)
        if sensor_height == 'surface_level_zm':
            corr_snow = 0
            for k in range(len(ds_out.time)):
                # if ds_out['lzboom'][k] == 0:
                #     ds_out['z'][k] = ds_out['zu'][k]
                if ds_out['linterpzboom'][k] == 0: # we cant use snow ranger 
                # Overwrite height of sensor using snowheight
                    if k > 0:
                        if ds_out['zu'][k]!= ds_out['zu'][k-1]: # station chnaged height during maintenance
                            # corr_snow = ds_out['snowheight'][k]
                            corr_snow = ds_out['surface_level_zm'][k]
                    else:
                        corr_snow = 0
                # ds_out['z'][k] = ds_out['zu'][k]  + ds_out['H'][k] - ds_out['d'][k] - ds_out['snowheight'][k] + corr_snow
                ds_out['z'][k] = ds_out['zu'][k]  + ds_out['H'][k] - ds_out['d'][k] - ds_out['surface_level_zm'][k] + corr_snow
                if ds_out['z'][k] < 0:
                    ds_out['z'][k] = 0.2
        else:
            ds_out['z'] = ds_out['zu'] + ds_out['H'] - ds_out['d'] - ds_out['snowheight']

        if sensor == 'GCNet':
            ds_out['z'] = ds_out['zu']
            ds_out['surface_level_zm'] = ds_out['zm']
                
        ds_out = ds_out.where(ds['ldata'] != 0.)  

        # Write current month
        # Attributes
        ds_out                                    = utils.Add_dataset_attributes(ds_out,'iWS_WS_L1B.JSON')
        ds_out.attrs['location']                  = LOC + '_' + ID
        ds_out.attrs['file_creation_date_time']   = str(datetime.datetime.now())
        ds_out.attrs['IceEddie_namelist_GLOBAL']  = str(nml['global'])
        ds_out.attrs['IceEddie_namelist_AWScorr'] = str(nml['AWScorr'])
 
        # Export to net CDF
        print(L1Bdir + file.replace('L1A','L1B'))
        ds_out.to_netcdf(L1Bdir + file.replace('L1A','L1B')) 
        

        ds_out.close()

#_______________________________________________________
def L1BtoL2(nml):
    """
    Reads the corrected L1B netCDF data files and computes turbulent fluxes using bulk model
    
    Input
    ----------
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
    """



    
    L1Bdir = nml['AWScorr']['DIR'] + 'L1B/'
    L2dir = nml['AWScorr']['DIR'] + 'L2/'
    L3dir = nml['global']['dir'] + 'L3/'
    ID = nml['global']['ID']    
    LOC = nml['global']['LOC']
    SvdB = nml['AWScorr']['lSvdB']
    LUT_file = nml['AWScorr']['z0_WD_table']
    ldownsample = nml['AWScorr']['ldownsample']
    
    
    if not os.path.exists(L2dir):
        os.makedirs(L2dir) 
        
    # Move to input data directory
    if not L1Bdir:
        os.chdir(os.getcwd())  
    else:
        os.chdir(L1Bdir)  
        
        
    for file in sorted(glob.glob("*nc")):
        print(file)
      
        ds = xr.open_dataset(file)
               
        # Open Look-up table (LUT)
        if LUT_file:
            LUT       = pd.read_csv(LUT_file,delimiter = ';',skiprows=0)
            LUT       = LUT.set_index('Angle')
            # Save headers
            ds['z0'] = ['0.1mm','1mm','1cm','10cm','measured_raw','measured_12h','measured_24h','measured_1w','measured_4w','icesnow_RACMO','icesnow_PROMICE'] + LUT.columns.values.tolist()
            ds.set_coords(['z0'])
            # Prepare output
            N_LUT     = np.shape(LUT.columns)[0]; # Amount of different z0 columns in LUT
            N_z0      = np.shape(ds.z0)[0] # Total Amount of different z0 columns
            N_nLUT    = N_z0 - N_LUT
            ds['z0m'] = xr.DataArray(np.full((len(ds.time),N_z0),np.nan),dims=['time','z0'])
            
            # Build array
            for i in range(len(ds.time)):
                idx            = np.argmin(np.abs(ds.WD[i].values - LUT.index))
                ds['z0m'][i,N_nLUT:N_z0] = LUT.loc[LUT.index[idx],:].values
                
        else:
            # Save headers
            ds['z0'] = ['0.1mm','1mm','1cm','10cm','measured_raw','measured_12h','measured_24h','measured_1w','measured_4w','icesnow_RACMO','icesnow_PROMICE']
            ds.set_coords(['z0'])
            # Prepare output
            N_z0      = np.shape(ds.z0)[0] # Total Amount of different z0 columns
            ds['z0m'] = xr.DataArray(np.full((len(ds.time),N_z0),np.nan),dims=['time','z0'])
        
        # Build array            
        ds.z0m[:,0]   = xr.DataArray(np.full(len(ds.time),1.0e-4),dims=['time'])
        ds.z0m[:,1]   = xr.DataArray(np.full(len(ds.time),1.0e-3),dims=['time'])
        ds.z0m[:,2]   = xr.DataArray(np.full(len(ds.time),1.0e-2),dims=['time'])
        ds.z0m[:,3]   = xr.DataArray(np.full(len(ds.time),1.0e-1),dims=['time'])
        ds.z0m[:,4]   = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
        ds.z0m[:,5]   = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
        ds.z0m[:,6]   = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
        ds.z0m[:,7]   = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
        ds.z0m[:,8]   = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
        ds.z0m[:,9]   = xr.DataArray(np.full(len(ds.time),5.0e-3),dims=['time'])
        ds.z0m[:,10]  = xr.DataArray(np.full(len(ds.time),1.0e-3),dims=['time'])
        
        # Import measured roughness length from L3 files, if available
        if os.path.isdir(L3dir):
            os.chdir(L3dir) 
            file_L3 = glob.glob("*iWS*L3*" + file[-9:-3] + "*nc")         
            if file_L3: 
                print('iWS L3 fluxes file found')
                ds_L3   = xr.open_dataset(file_L3[0])
                try:
                    # remove flagged data (obstructed or mising meas.) and low wind speeds
                    ds_L3.z0_EC[(ds_L3.flagUW == 1) | ((ds_L3.u**2 + ds_L3.v**2 )**(0.5)< 5) | (ds_L3.ustar[:,0,0,0] < 0.1) | ((ds_L3.ww[:,0,0,0]**0.5/ds_L3.ustar[:,0,0,0]) < 0.2) | ((ds_L3.ww[:,0,0,0]**0.5/ds_L3.ustar[:,0,0,0]) > 2),0,0,0] = np.nan
                    # Add roughness length to dataset
                    ds.z0m[:,4] =  ds.z0m[:,4].combine_first(ds_L3.where(ds.time == ds_L3.time).z0_EC[:,0,0,0])
                except:                 
                    # remove flagged data (obstructed or mising meas.) and low wind speeds
                    ds_L3.z0_EC[(ds_L3.flagUW == 1) | ((ds_L3.u**2 + ds_L3.v**2 )**(0.5)< 5) | (ds_L3.ustar < 0.1) | ((ds_L3.ww**0.5/ds_L3.ustar) < 0.2) | ((ds_L3.ww**0.5/ds_L3.ustar) > 2)] = np.nan
                    # Add roughness length to dataset
                    ds.z0m[:,4] =  ds.z0m[:,4].combine_first(ds_L3.where(ds.time == ds_L3.time).z0_EC)
                os.chdir(L1Bdir) 
                
                
                
            elif glob.glob("*CSAT*L3*" + "*nc"):
                file_L3 = glob.glob("*CSAT*L3*" + "*nc")
                print('CSAT L3 fluxes file found')
                ds_L3   = xr.open_dataset(file_L3[0])
                _, index = np.unique(ds_L3['time'], return_index=True)
                ds_L3 = ds_L3.isel(time=index)
                # remove flagged data (obstructed or mising meas.) and low wind speeds
                try:
                    ds_L3.z0_EC[(ds_L3.flagUW == 1) | ((ds_L3.u**2 + ds_L3.v**2 )**(0.5)< 5) | (ds_L3.ustar < 0.1) | ((ds_L3.ww**0.5/ds_L3.ustar) < 0.2) | ((ds_L3.ww**0.5/ds_L3.ustar) > 2)]  = np.nan
                except:
                    ds_L3.z0_EC[((ds_L3.u**2 + ds_L3.v**2 )**(0.5)< 5) | (ds_L3.ustar < 0.1) | ((ds_L3.ww**0.5/ds_L3.ustar) < 0.2) | ((ds_L3.ww**0.5/ds_L3.ustar) > 2)]  = np.nan
                # Add roughness length to dataset
                ds.z0m[:,4] =  ds.z0m[:,4].combine_first(ds_L3.where(ds.time == ds_L3.time).z0_EC)    
                os.chdir(L1Bdir) 
                
            else:
                print('No L3 fluxes file found')
                # Go back to L1B dir
                os.chdir(L1Bdir) 
                    
            # moving average s
            ds.z0m[:,5] = np.log(np.exp(ds.z0m[:,4]).rolling(time=24,center=True,min_periods=4).reduce(np.nanmean).interpolate_na('time'))
            ds.z0m[:,6] = np.log(np.exp(ds.z0m[:,4]).rolling(time=48,center=True,min_periods=4).reduce(np.nanmean).interpolate_na('time'))
            ds.z0m[:,7] = np.log(np.exp(ds.z0m[:,4]).rolling(time=336,center=True,min_periods=4).reduce(np.nanmean).interpolate_na('time'))
            ds.z0m[:,8] = np.log(np.exp(ds.z0m[:,4]).rolling(time=1344,center=True,min_periods=4).reduce(np.nanmean).interpolate_na('time'))

            
            # Replace nan in the end
            if not ds.z0m[:,4].isnull().values.any():
                ds.z0m[-1:,5] = ds.z0m[:,5].loc[ds.z0m[:,5].to_dataframe()['z0m'].last_valid_index()].values
                ds.z0m[-1:,6] = ds.z0m[:,6].loc[ds.z0m[:,6].to_dataframe()['z0m'].last_valid_index()].values
                ds.z0m[-1:,7] = ds.z0m[:,7].loc[ds.z0m[:,7].to_dataframe()['z0m'].last_valid_index()].values
                ds.z0m[-1:,8] = ds.z0m[:,8].loc[ds.z0m[:,8].to_dataframe()['z0m'].last_valid_index()].values
                
                ds.z0m[:,5]  = ds.z0m[:,5].interpolate_na('time')
                ds.z0m[:,6]  = ds.z0m[:,6].interpolate_na('time')
                ds.z0m[:,7]  = ds.z0m[:,7].interpolate_na('time')
                ds.z0m[:,8]  = ds.z0m[:,8].interpolate_na('time')   
               
                      
            # Find when surface is snow covered
            bulk.is_snow(ds)
            ds.z0m[ds.issnow,9]  = 1.0e-3
            ds.z0m[ds.issnow,10] = 1.0e-4
          
        # Calculate turbulent flux
        bulk.bulkflux(ds,SvdB=SvdB)
        
        if ldownsample:
            ds_3h = ds.resample(time="3H",label = 'left').mean()
            ds_daily = ds.resample(time="1D",label = 'left').mean()
            
            if 'ADW' in list(ds.keys()):
                Melt_3H_Jm2 = utils.lf * utils.rho_i * (ds['ADW'].resample(time="3H",label = 'left').last() - ds['ADW'].resample(time="3H",label = 'left').first())
                ds_3h['melt_ADW'] = Melt_3H_Jm2/(3*3600) # total melt in Wm-2
                Melt_DAILY_Jm2 = utils.lf * utils.rho_i * (ds['ADW'].resample(time="1D",label = 'left').last() - ds['ADW'].resample(time="1D",label = 'left').first())
                ds_daily['melt_ADW'] = Melt_DAILY_Jm2/(24*3600) # total melt in Wm-2
                
            if 'depth_pressure_transducer_cor' in list(ds.keys()):
                Melt_3H_Jm2 = utils.lf * utils.rho_i * (ds['depth_pressure_transducer_cor'].resample(time="3H",label = 'left').last() - ds['depth_pressure_transducer_cor'].resample(time="3H",label = 'left').first())
                ds_3h['melt_DPT'] = Melt_3H_Jm2/(3*3600) # total melt in Wm-2
                Melt_DAILY_Jm2 = utils.lf * utils.rho_i * (ds['depth_pressure_transducer_cor'].resample(time="1D",label = 'left').last() - ds['depth_pressure_transducer_cor'].resample(time="1D",label = 'left').first())
                ds_daily['melt_DPT'] = Melt_DAILY_Jm2/(24*3600) # total melt in Wm-2  
            
            ds_3h.to_netcdf(L2dir + file.replace('L1B','3H_L2'),encoding={'time': \
                 {'units':'minutes since ' + file[-9:-5] + '-' +  file[-5:-3] + '-01'}})
            ds_daily.to_netcdf(L2dir + file.replace('L1B','DAILY_L2'),encoding={'time': \
                 {'units':'minutes since ' + file[-9:-5] + '-' +  file[-5:-3] + '-01'}})
                
        # Attributes
        ds                                    = utils.Add_dataset_attributes(ds,'iWS_WS_L2.JSON')
        ds.attrs['location']                  = LOC + '_' + ID
        ds.attrs['file_creation_date_time']   = str(datetime.datetime.now())
        ds.attrs['IceEddie_namelist_GLOBAL']  = str(nml['global'])
        ds.attrs['IceEddie_namelist_AWScorr'] = str(nml['AWScorr'])
        
        # Export to net CDF
        print(L2dir + file.replace('L1B','L2'))
        ds.to_netcdf(L2dir + file.replace('L1B','L2')) 
        
        ds.close()
    
    
#_______________________________________________________
def L3toL3B(nml):
    """
    Merges AWS data with EC data, calculates flux footprint and scalar roughness length
    
    Input
    ----------
    nml: python f90nml namelist 
        IceEddie namelist containing all parameters
 
    
    """ 

    L3_ECdir = nml['global']['DIR']  + 'L3/'
    L2_WSdir = nml['AWScorr']['DIR'] + 'L2/'
    L3B_dir  = nml['global']['DIR']  + 'L3B/'
    ID	     = nml['global']['ID']    
    LOC	     = nml['global']['LOC']
    
    L3sensor  = nml['L3toL3B']['L3sensor']
    lfp       = nml['L3toL3B']['lfp']
    fp_dx     = nml['L3toL3B']['fp_dx']
    fp_domain = nml['L3toL3B']['fp_domain']
    fp_WDbins = nml['L3toL3B']['fp_WDbins']
    lbin_WD = nml['L3toL3B']['lbin_WD']
    fp_dWDbins = nml['L3toL3B']['fp_dWDbins']
    fp_r = nml['L3toL3B']['fp_r']
    lsave_all_fp = nml['L3toL3B']['lsave_all_fp']
    toffset_min = nml['L3toL3B']['toffset_min']
    EC_hgt_offset  = nml['L3toL3B']['EC_hgt_offset']
    ldownsample = nml['L3toL3B']['ldownsample']
    
    if not os.path.exists(L3B_dir):
        os.makedirs(L3B_dir) 
        
    # Move to input data directory
    if not L3_ECdir:
        os.chdir(os.getcwd())  
    else:
        os.chdir(L3_ECdir)  
             
    for file in sorted(glob.glob('*' + L3sensor + "*nc")):
        print(file)
        
        # Open L3 EC data
        ds_EC = xr.open_dataset(file)
        # Remove duplicate indices if any
        _, index = np.unique(ds_EC['time'], return_index=True)
        ds_EC = ds_EC.isel(time=index)
        # Set time to start of averaging interval
        ds_EC['time'] = ds_EC.time.values + np.timedelta64(toffset_min,'m')
        # Average to 30min
        ds_EC =  ds_EC.resample(time='30min').mean()
        
        # Import L2 weather station data
        os.chdir(L2_WSdir) 
        print(L2_WSdir)
        file_L2_WS = glob.glob("*L2*nc")        
        ds_WS = xr.open_dataset(file_L2_WS[0])
        os.chdir(L3_ECdir)  
        ds = ds_WS


        
        # Import Eddy-covariance data (only keep one timeseries of corrected fluxes)
        if L3sensor == 'iWS':
            try:
                ds['H_EC'] = ds.rho_a * ds.Cp * ds_EC['wT'][:,0,0,0]
                ds['U_EC'] = (ds_EC['u']**2+ds_EC['v']**2)**0.5
                ds['WD_EC'] = ds_EC['WD']
                ds['ustar'] = ds_EC['ustar'][:,0,0,0]
                ds['tstar'] = ds_EC['tstar'][:,0,0,0]
                ds['obl_EC'] = ds_EC['obh'][:,0,0,0]
                ds['z0_EC'] = ds_EC['z0_EC'][:,0,0,0]
                ds['zeta'] = ds_EC['zeta'][:,0,0,0]
                ds['sigmau'] = (ds_EC['uu'][:,0,0,0])**0.5
                ds['sigmav'] = (ds_EC.vv/ds_EC.Auu[:,0,0,0])**0.5
                ds['sigmaw'] = (ds_EC['ww'][:,0,0,0])**0.5
                ds['uw'] = (ds_EC['uw'][:,0,0,0])
                ds['vw'] = (ds_EC['vw'][:,0,0,0])
                ds['uv'] = (ds_EC['uv'][:,0,0,0])
                ds['uT1'] = (ds_EC['uT1'][:,0,0,0])
                ds['T1T1'] = (ds_EC['T1T1'][:,0,0,0])
                ds['uT2'] = (ds_EC['uT2'][:,0,0,0])
                ds['T2T2'] = (ds_EC['T2T2'][:,0,0,0])
                ds['DRyaw'] = (ds_EC['yaw'])
                ds['DRpitch'] = (ds_EC['pitch'])
                ds['T1'] = ds_EC['T1']
                ds['T2'] = ds_EC['T2']
                ds['T_intern'] = ds_EC['T_intern']
                ds['flag_EC'] = ds_EC['flagH1']
                ds['z_EC'] = ds_EC['z']
            except:
                ds['H_EC'] = ds.rho_a * ds.Cp * ds_EC['wT']
                ds['U_EC'] = (ds_EC['u']**2+ds_EC['v']**2)**0.5
                ds['WD_EC'] = ds_EC['WD']
                ds['ustar'] = ds_EC['ustar']
                ds['tstar'] = ds_EC['tstar']
                ds['obl_EC'] = ds_EC['obh']
                ds['z0_EC'] = ds_EC['z0_EC']
                ds['zeta'] = ds_EC['zeta']
                ds['sigmau'] = (ds_EC['uu'])**0.5
                ds['sigmav'] = (ds_EC.vv/ds_EC.Auu)**0.5
                ds['sigmaw'] = (ds_EC['ww'])**0.5
                ds['uw'] = (ds_EC['uw'])
                ds['vw'] = (ds_EC['vw'])
                ds['uv'] = (ds_EC['uv'])
                ds['uT1'] = (ds_EC['uT1'])
                ds['T1T1'] = (ds_EC['T1T1'])
                ds['uT2'] = (ds_EC['uT2'])
                ds['T2T2'] = (ds_EC['T2T2'])
                ds['DRyaw'] = (ds_EC['yaw'])
                ds['DRpitch'] = (ds_EC['pitch'])
                ds['T1'] = ds_EC['T1']
                ds['T2'] = ds_EC['T2']
                ds['T_intern'] = ds_EC['T_intern']
                ds['flag_EC'] = ds_EC['flagH1']
                ds['z_EC'] = ds_EC['z']
                
        if L3sensor == 'Young':
            try:
                ds['H1_EC'] = ds.rho_a * ds.Cp * ds_EC['wT1'][:,0,0,0]
                ds['U_EC'] = (ds_EC['u']**2+ds_EC['v']**2)**0.5
                ds['WD_EC'] = ds_EC['WD']
                ds['ustar'] = ds_EC['ustar'][:,0,0,0]
                ds['tstar'] = ds_EC['tstar'][:,0,0,0]
                ds['obl_EC'] = ds_EC['obh'][:,0,0,0]
                ds['z0_EC'] = ds_EC['z0_EC'][:,0,0,0]
                ds['zeta'] = ds_EC['zeta'][:,0,0,0]
                ds['sigmau'] = (ds_EC['uu'][:,0,0,0])**0.5
                ds['sigmav'] = (ds_EC.vv/ds_EC.Auu[:,0,0,0])**0.5
                ds['sigmaw'] = (ds_EC['ww'][:,0,0,0])**0.5
                ds['uw'] = (ds_EC['uw'][:,0,0,0])
                ds['vw'] = (ds_EC['vw'][:,0,0,0])
                ds['uv'] = (ds_EC['uv'][:,0,0,0])
                ds['uT'] = (ds_EC['uT'][:,0,0,0])
                ds['TT'] = (ds_EC['TT'][:,0,0,0])
                ds['DRyaw'] = (ds_EC['DRyaw'])
                ds['DRpitch'] = (ds_EC['DRpitch'])
                ds['T1'] = ds_EC['T1']
                ds['flag_EC'] = ds_EC['flagH1']
                ds['z_EC'] = ds_EC['z']
            except:
                ds['H1_EC'] = ds.rho_a * ds.Cp * ds_EC['wT1']
                ds['U_EC'] = (ds_EC['u']**2+ds_EC['v']**2)**0.5
                ds['WD_EC'] = ds_EC['WD']
                ds['ustar'] = ds_EC['ustar']
                ds['tstar'] = ds_EC['tstar']
                ds['obl_EC'] = ds_EC['obh']
                ds['z0_EC'] = ds_EC['z0_EC']
                ds['zeta'] = ds_EC['zeta']
                ds['sigmau'] = (ds_EC['uu'])**0.5
                ds['sigmav'] = (ds_EC.vv/ds_EC.Auu)**0.5
                ds['sigmaw'] = (ds_EC['ww'])**0.5
                ds['uw'] = (ds_EC['uw'])
                ds['vw'] = (ds_EC['vw'])
                ds['uv'] = (ds_EC['uv'])
                ds['uT'] = (ds_EC['uT'])
                ds['TT'] = (ds_EC['TT'])
                ds['DRyaw'] = (ds_EC['DRyaw'])
                ds['DRpitch'] = (ds_EC['DRpitch'])
                ds['T1'] = ds_EC['T1']
                ds['flag_EC'] = ds_EC['flagH1']
                ds['z_EC'] = ds_EC['z']     
        if L3sensor == 'CSAT':
            if 'wq' in list(ds_EC.keys()):
                if 'H2O' in list(ds_EC.keys()):
                    ds['q_EC'] = ds_EC['H2O']
                else:
                    ds['q_EC'] = ds_EC['q']
                    ds['sigmaq'] = (ds_EC['qq'])**0.5
                    ds['qstar'] = ds_EC['qstar']
                if 'CO2' in list(ds_EC.keys()):
                    ds['CO2_EC'] = ds_EC['CO2']
                if 'wCO2' in list(ds_EC.keys()):
                    ds['wCO2'] = ds_EC['wCO2'] 
                ds['wq_EC'] = ds_EC['wq']
                ds['LE_EC'] = ds.rho_a * utils.Ls() * ds_EC['wq']
                if 'wq_Bo' in list(ds_EC.keys()):
                    ds['LE_EC_wqBo'] =  ds.rho_a * utils.Ls() * ds_EC['wq_Bo']
                if 'wgT2' in list(ds_EC.keys()):
                    ds['LE_EC_wgqBo'] =  ds.rho_a * utils.Ls() * ds_EC['wgq_Bo']
                    ds['H1_EC_g'] =  ds.rho_a * ds.Cp * ds_EC['wgT2']
                    ds['ustar_y'] = ds_EC['ustary']
                    ds['tstar_y'] = ds_EC['tstar_g']
                    ds['obl_EC_y'] = ds_EC['obhy']
                    ds['zeta_y'] = ds_EC['zetay']
                    ds['z0_EC_y'] = ds_EC['z0_ECy']
                    ds['sigmau_y'] =  (ds_EC['uyuy'])**0.5
                    ds['sigmaw_g'] = (ds_EC['wgwg'])**0.5
                    ds['T2T2'] = (ds_EC['T2T2'])
                    ds['DRyaw_y'] = (ds_EC['yawy'])
                    ds['DRpitch_y'] = (ds_EC['pitchy'])
                    ds['U_y'] = (ds_EC['Uy'])
                    ds['WD_y'] = (ds_EC['WDy'])
                    ds['T2'] = (ds_EC['T_couple2'])
                    ds['z_y'] = (ds_EC['zy'])
                    ds['z0h_EC_y'] = ds['z_y']/(np.exp((utils.kappa*(ds['th0']-ds['ths'])/(ds['tstar_y']))  +  Psih(ds['zeta_y'])  ))  

            ds['H1_EC'] = ds.rho_a * ds.Cp * ds_EC['wTs']
            ds['H2_EC'] = ds.rho_a * ds.Cp * ds_EC['wTc']
            ds['U_EC'] = (ds_EC['u']**2+ds_EC['v']**2)**0.5
            ds['WD_EC'] = ds_EC['WD']
            ds['ustar'] = ds_EC['ustar']
            ds['tstar'] = ds_EC['tstar']
            ds['tstar_c'] = ds_EC['tstar_c']
            ds['obl_EC'] = ds_EC['obh']
            ds['z0_EC'] = ds_EC['z0_EC']
            ds['zeta'] = ds_EC['zeta']
            ds['sigmau'] = (ds_EC['uu'])**0.5
            ds['sigmav'] = (ds_EC['vv'])**0.5
            ds['sigmaw'] = (ds_EC['ww'])**0.5
            ds['uw'] = (ds_EC['uw'])
            ds['vw'] = (ds_EC['vw'])
            ds['uv'] = (ds_EC['uv'])
            ds['uTs'] = (ds_EC['uTs'])
            ds['uTc'] = (ds_EC['uTc'])
            ds['TsTs'] = (ds_EC['TsTs'])
            ds['TcTc'] = (ds_EC['TcTc'])
            try:
                ds['DRyaw'] = (ds_EC['yaw'])
                ds['DRpitch'] = (ds_EC['pitch'])
            except:
                try:
                    ds['DRyaw'] = (ds_EC['DRyaw'])
                    ds['DRpitch'] = (ds_EC['DRpitch'])
                except:
                    ds['DRyaw'] = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
                    ds['DRpitch'] = xr.DataArray(np.full(len(ds.time),np.nan),dims=['time'])
                    
            ds['Ts'] = ds_EC['T_sonic']
            ds['T1'] = ds_EC['T_couple']
            try:
                ds['flag_EC'] = ds_EC['flagHs']
            except:
                try:
                    ds['flag_EC'] = ds_EC['flag']
                except: 
                    ds['flag_EC'] = xr.DataArray(np.full(len(ds.time),0),dims=['time'])
            ds['z_EC'] = ds_EC['z']
            ds['z0h_EC'] = ds['z_EC']/(np.exp((utils.kappa*(ds['th0']-ds['ths'])/(ds['tstar']))  +  Psih(ds['zeta'])  ))  
            ds['z0h_EC_c'] = ds['z_EC']/(np.exp((utils.kappa*(ds['th0']-ds['ths'])/(ds['tstar_c']))  +  Psih(ds['zeta'])  ))
            nu = utils.nu_air(ds.T0,ds.p0,ds.qv0) 
        
            # Roughness Reynolds number
            ds['Restar_EC'] = ds['ustar']*ds['z0_EC']/nu
            if 'wgT2' in list(ds_EC.keys()):
                ds['Restar_EC_y'] = ds['ustar_y']*ds['z0_EC_y']/nu

            if 'wq' in list(ds_EC.keys()):
                ds['z0q_EC'] = ds['z_EC']/(np.exp((utils.kappa*(ds['qv0']-ds['qvs'])/(ds['qstar']))  +  Psih(ds['zeta'])  ))  
# EC_L3.uv EC_L3.uTs EC_L3.vTs EC_L3.TsTs EC_L3.TcTc EC_L3.DRyaw EC_L3.DRpitch ..

        
        # Remove obsolete coordinates and variables
        if 'issnow' in list(ds.keys()):
            ds = ds.drop(['issnow'])

        # Roughness length for temperature
        ds['z0h_EC'] = ds['z_EC']/(np.exp((utils.kappa*(ds['th0']-ds['ths'])/(ds['tstar']))  +  Psih(ds['zeta'])  ))  
        
        if ldownsample:
            ds_3h = ds.resample(time="3H",label = 'left').mean()
            ds_daily = ds.resample(time="1D",label = 'left').mean()
            
            if 'ADW' in list(ds.keys()):
                Melt_3H_Jm2 = utils.lf * utils.rho_i * (ds['ADW'].resample(time="3H",label = 'left').last() - ds['ADW'].resample(time="3H",label = 'left').first())
                ds_3h['melt_ADW'] = Melt_3H_Jm2/(3*3600) # total melt in Wm-2
                Melt_DAILY_Jm2 = utils.lf * utils.rho_i * (ds['ADW'].resample(time="1D",label = 'left').last() - ds['ADW'].resample(time="1D",label = 'left').first())
                ds_daily['melt_ADW'] = Melt_DAILY_Jm2/(24*3600) # total melt in Wm-2
                
            if 'depth_pressure_transducer_cor' in list(ds.keys()):
                Melt_3H_Jm2 = utils.lf * utils.rho_i * (ds['depth_pressure_transducer_cor'].resample(time="3H",label = 'left').last() - ds['depth_pressure_transducer_cor'].resample(time="3H",label = 'left').first())
                ds_3h['melt_DPT'] = Melt_3H_Jm2/(3*3600) # total melt in Wm-2
                Melt_DAILY_Jm2 = utils.lf * utils.rho_i * (ds['depth_pressure_transducer_cor'].resample(time="1D",label = 'left').last() - ds['depth_pressure_transducer_cor'].resample(time="1D",label = 'left').first())
                ds_daily['melt_DPT'] = Melt_DAILY_Jm2/(24*3600) # total melt in Wm-2  
            
            ds_3h.to_netcdf(L3B_dir + file.replace('L3','3H_L3B'),encoding={'time': \
                 {'units':'minutes since ' + file[-9:-5] + '-' +  file[-5:-3] + '-01'}})
            ds_daily.to_netcdf(L3B_dir + file.replace('L3','DAILY_L3B'),encoding={'time': \
                 {'units':'minutes since ' + file[-9:-5] + '-' +  file[-5:-3] + '-01'}})
        if lfp:
            # Estimate footprint for each wind direction band after Kljun(2015)
            Nx = abs(fp_domain[1] - fp_domain[0]) + 1
            Ny = abs(fp_domain[3] - fp_domain[2]) + 1
            WDbins = np.linspace(fp_WDbins[0],fp_WDbins[-1],int(1+abs(fp_WDbins[-1]-fp_WDbins[0])/fp_dWDbins))+fp_dWDbins/2
            ds['fp_WDbins'] = WDbins
            ds['fp_r']      = fp_r
            f_FP = np.full((Nx,Ny,len(WDbins)),np.nan)
            
            z0m_EC_binned = np.full(len(WDbins),np.nan)
            H1_EC_binned = np.full(len(WDbins),np.nan)
            H2_EC_binned = np.full(len(WDbins),np.nan)
            SWd_binned = np.full(len(WDbins),np.nan)
            LWd_binned = np.full(len(WDbins),np.nan)
            SWu_binned = np.full(len(WDbins),np.nan)
            LWu_binned = np.full(len(WDbins),np.nan)
            
            x_FP = np.full((5000,len(WDbins),len(fp_r)),np.nan)
            y_FP = np.full((5000,len(WDbins),len(fp_r)),np.nan)
            
            x_FP_clim = np.full((5000,len(fp_r)),np.nan)
            y_FP_clim = np.full((5000,len(fp_r)),np.nan)  
            
            print('Calculating footprints ...')
            
            if lbin_WD:
                    
                for i in range(len(ds.fp_WDbins)):
                    # Select valid data
                    idx = (ds.flag_EC.values == 0) & \
                     (~np.isnan(ds['ustar'].values)) & \
                     (~np.isnan(ds['obl_EC'].values)) &  \
                     (~np.isnan(ds['z0_EC'].values)) & \
                     (~np.isnan(ds['z_EC'].values)) & \
                     (~np.isnan(ds['U_EC'].values)) & \
                     (~np.isnan(ds['sigmav'].values)) & \
                     (~np.isnan(ds['WD_EC'].values)) & \
                     (ds['WD_EC'].values > ds.fp_WDbins.values[i] - fp_dWDbins/2) & (ds['WD_EC'].values <ds.fp_WDbins.values[i] + fp_dWDbins/2)
                     
                    if np.count_nonzero(idx)==0: continue
                    # Convert dataset to lists
                    zm       = ds['z_EC'].values[idx].tolist()
                    umean    = ds['U_EC'].values[idx].tolist()
                    h        = np.full(np.count_nonzero(idx),200).tolist() #Ignore influece of BL height
                    ol       = ds['obl_EC'].values[idx].tolist()
                    sigmav   = ds['sigmav'].values[idx].tolist()
                    ustar    = ds['ustar'].values[idx].tolist()
                    wind_dir = ds['WD_EC'].values[idx].tolist()
        
                    FP = fp_clim.FFP_climatology(zm=zm,\
                                        z0=None,\
                                        umean=umean,\
                                        h=h,\
                                        ol=ol,\
                                        sigmav=sigmav,\
                                        ustar=ustar,\
                                        wind_dir=wind_dir,\
                                        domain= fp_domain, \
                                        dx = fp_dx, dy = fp_dx, \
                                        nx=None, ny=None, \
                                        rs=fp_r, \
                                        rslayer=0, \
                                        smooth_data=1, \
                                        crop=False, pulse=None, verbosity=0, \
                                        fig=False)
                    
                    if FP.get('n') == 0: continue
                
                    # if lsave_all_fp:
                        # f_FP[:,:,i] = FP.get('fclim_2d')  
                    for j in range(len(ds.fp_r)):
                        if not FP.get('xr')[j]: continue
                        x_FP[:len(FP.get('xr')[j]),i,j] = FP.get('xr')[j]
                        y_FP[:len(FP.get('yr')[j]),i,j] = FP.get('yr')[j]

                    if L3sensor == 'CSAT' :
                        z0m_EC_binned[i] = 10**(np.log10(ds.z0_EC[idx]).mean())
                        H1_EC_binned[i]  = ds.H1_EC[idx].mean()
                        H2_EC_binned[i]  = ds.H2_EC[idx].mean()
                        SWd_binned[i]  = ds.SWd[idx].mean()
                        LWd_binned[i]  = ds.LWd[idx].mean()
                        SWu_binned[i]  = ds.SWu[idx].mean()
                        LWu_binned[i]  = ds.LWu[idx].mean()
                    if L3sensor == 'iWS' :
                        z0m_EC_binned[i] = 10**(np.log10(ds.z0_EC[idx]).mean())
                        H1_EC_binned[i]  = ds.H_EC[idx].mean()
                        SWd_binned[i]  = ds.SWd[idx].mean()
                        LWd_binned[i]  = ds.LWd[idx].mean()
                        SWu_binned[i]  = ds.SWu[idx].mean()
                        LWu_binned[i]  = ds.LWu[idx].mean()
                    if L3sensor == 'Young':
                        z0m_EC_binned[i] = 10**(np.log10(ds.z0_EC[idx]).mean())
                        H1_EC_binned[i]  = ds.H1_EC[idx].mean()
                        SWd_binned[i]  = ds.SWd[idx].mean()
                        LWd_binned[i]  = ds.LWd[idx].mean()
                        SWu_binned[i]  = ds.SWu[idx].mean()
                        LWu_binned[i]  = ds.LWu[idx].mean()
                        
                # Save indivudual footprint to dataset
                ds['x_FP'] = xr.DataArray(x_FP, coords=[np.full(5000,np.nan), ds.fp_WDbins,ds.fp_r], dims=['tmp', 'fp_WDbins','fp_r'] )
                ds['y_FP'] = xr.DataArray(y_FP, coords=[np.full(5000,np.nan), ds.fp_WDbins,ds.fp_r], dims=['tmp', 'fp_WDbins','fp_r'] )
                if lsave_all_fp:
                    ds['x2D_FP'] = FP.get('x_2d')[0,:]
                    ds['y2D_FP'] = FP.get('y_2d')[:,0]
                    # ds['f2D_FP'] = xr.DataArray(f_FP, coords=[ds.y2D_FP, ds.x2D_FP,ds.fp_WDbins], dims=['y2D_FP', 'x2D_FP','fp_WDbins'] )
        
                
                # Caculate bin averaged variables
                if (L3sensor == 'CSAT') | (L3sensor == 'iWS'):
                    ds['z0m_binned'] = xr.DataArray(z0m_EC_binned, coords=[ds.fp_WDbins], dims=['fp_WDbins'] )
                    ds['H1_EC_binned'] = xr.DataArray(H1_EC_binned, coords=[ds.fp_WDbins], dims=['fp_WDbins'] )
                    ds['H2_EC_binned'] = xr.DataArray(H2_EC_binned, coords=[ds.fp_WDbins], dims=['fp_WDbins'] )
                    ds['SWd_binned'] = xr.DataArray(SWd_binned, coords=[ds.fp_WDbins], dims=['fp_WDbins'] )
                    ds['LWd_binned'] = xr.DataArray(LWd_binned, coords=[ds.fp_WDbins], dims=['fp_WDbins'] )
                    ds['SWu_binned'] = xr.DataArray(SWu_binned, coords=[ds.fp_WDbins], dims=['fp_WDbins'] )
                    ds['LWu_binned'] = xr.DataArray(LWu_binned, coords=[ds.fp_WDbins], dims=['fp_WDbins'] )
                elif L3sensor == 'Young':
                    ds['z0m_binned'] = xr.DataArray(z0m_EC_binned, coords=[ds.fp_WDbins], dims=['fp_WDbins'] )
                    ds['H1_EC_binned'] = xr.DataArray(H1_EC_binned, coords=[ds.fp_WDbins], dims=['fp_WDbins'] )
                    ds['SWd_binned'] = xr.DataArray(SWd_binned, coords=[ds.fp_WDbins], dims=['fp_WDbins'] )
                    ds['LWd_binned'] = xr.DataArray(LWd_binned, coords=[ds.fp_WDbins], dims=['fp_WDbins'] )
                    ds['SWu_binned'] = xr.DataArray(SWu_binned, coords=[ds.fp_WDbins], dims=['fp_WDbins'] )
                    ds['LWu_binned'] = xr.DataArray(LWu_binned, coords=[ds.fp_WDbins], dims=['fp_WDbins'] )
    #        # Convert dataset to lists
            idx = (ds.flag_EC.values == 0) & \
             (~np.isnan(ds['ustar'].values)) & \
             (~np.isnan(ds['obl_EC'].values)) &  \
             (~np.isnan(ds['z0_EC'].values)) & \
             (~np.isnan(ds['z_EC'].values)) & \
             (~np.isnan(ds['U'].values)) & \
             (~np.isnan(ds['sigmav'].values)) & \
             (~np.isnan(ds['WD'].values))
            if np.count_nonzero(idx)==0: continue
            zm       = ds['z_EC'].values[idx].tolist()
            umean    = ds['U_EC'].values[idx].tolist()
            h        = np.full(np.count_nonzero(idx),200).tolist() #Ignore influece of BL height
            ol       = ds['obl_EC'].values[idx].tolist()
            sigmav   = ds['sigmav'].values[idx].tolist()
            ustar    = ds['ustar'].values[idx].tolist()
            wind_dir = ds['WD_EC'].values[idx].tolist()
         
    #        # Estimate footprint climatology after Kljun(2015)
            FP_clim = fp_clim.FFP_climatology(zm=zm,\
                                z0=None,\
                                umean=umean,\
                                h=h,\
                                ol=ol,\
                                sigmav=sigmav,\
                                ustar=ustar,\
                                wind_dir=wind_dir,\
                                domain= fp_domain, \
                                dx = fp_dx, dy = fp_dx, \
                                nx=None, ny=None, \
                                rs=fp_r, \
                                rslayer=0, \
                                smooth_data=1, \
                                crop=False, pulse=None, verbosity=2, \
                                fig=False)
            
            for j in range(len(ds.fp_r)):
                if not FP_clim.get('xr')[j]: continue
                x_FP_clim[:len(FP_clim.get('xr')[j]),j] = FP_clim.get('xr')[j]
                y_FP_clim[:len(FP_clim.get('yr')[j]),j] = FP_clim.get('yr')[j]
            
            ds['x2D_FP_clim'] = FP_clim.get('x_2d')[0,:]
            ds['y2D_FP_clim'] = FP_clim.get('y_2d')[:,0]
            ds['f2D_FP_clim'] = xr.DataArray(FP_clim.get('fclim_2d'), coords=[ds.y2D_FP_clim, ds.x2D_FP_clim], dims=['y2D_FP_clim', 'x2D_FP_clim'] )
            
            ds['x_FP_clim'] = xr.DataArray(x_FP_clim, coords=[np.full(5000,np.nan),ds.fp_r], dims=['tmp2','fp_r'] )
            ds['y_FP_clim'] = xr.DataArray(y_FP_clim, coords=[np.full(5000,np.nan),ds.fp_r], dims=['tmp2','fp_r'] )   
                
        # Add new attributes
        ds                                      = utils.Add_dataset_attributes(ds,'iWS_EC_L3B.JSON')
        ds.attrs['location']                    = LOC + '_' + ID
        ds.attrs['file_creation_date_time']     = str(datetime.datetime.now())
        ds.attrs['IceEddie_namelist_L3toL3B']   = str(nml['L3toL3B'])
       
        os.chdir(L3B_dir)       
        ds.to_netcdf(L3B_dir + file.replace('L3','L3B'))
    
        os.chdir(L3_ECdir)      
                  
#_______________________________________________________        
def Psih(zeta,unstable='Högström',stable='BeljaarsHoltslag'):
    """
    Returns the integrated flux-profile function for heat evaluated at zeta
    
    Input
    ----------
    zeta: float or numpy array
        stability parameter
    unstable: str
        function to use in case of unstable stratification
    stable: str
        function to use in case of stable stratification
    
    Output
    ----------
    float or numpy array
    
    Example
    ----------
    Psim(z/L)
    
    Required packages
    ----------
    glob, os, itertools, pandas
    
    Required functions
    ----------
    """
    
    
    out = np.full(np.shape(zeta), np.nan)
        
    if unstable == 'Högström':     
        x             = (1 - 16*zeta)**(1/4)
        out[zeta<0]  = np.log( ((1 + x[zeta<0]**2)/2) * ((1 + x[zeta<0])/2)**2 ) \
        - 2*np.arctan(x[zeta<0]) + np.pi/2
           
    if stable == 'BeljaarsHoltslag':
        a = 1
        b = 2/3
        c = 5
        d = 0.35
        out[zeta>=0] = -b * (zeta[zeta>=0] - c/d)*np.exp(d * zeta[zeta>=0]) -  \
        (1 + b * a * zeta[zeta>=0]) ** (1.5) - (b*c) / d + 1.
   
    return out