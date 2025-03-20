#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Filename: readrawdata.py
#Description: Some useful functions used to read raw data from a VPEC system or from a CSAT sonic anemometer
"""
Created on Fri Nov  9 15:58:19 2018

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
import math
import utils
import f90nml
from scipy import stats 
import utils 
import Bulk_turbulence as bulk
####### CUSTOM FUNCTIONS ##########

#_______________________________________________________
def compose_date(years=1, months=1, days=1, weeks=None, hours=None, minutes=None,
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
def merge_files(folder,nml):
    """
    Merge all monthly datasets, remove duplicates and save in monthly files
    
    Input
    ----------
    folder: str
        folder containing files
        
    """   

    sensor     = nml['global']['sensor']
    input_type     = nml['global']['input_type']
    version = nml['global']['version']
    LOC        = nml['global']['LOC']
    ID         = nml['global']['ID']
    level         = 'L1A'
    
    os.chdir(folder)
    files = glob.glob("*nc*")
    fid = 0
    for file in files:
        if fid == 0:
            ds = xr.open_dataset(file)
        else:
            tmp = xr.open_dataset(file)
            ds = xr.concat([ds,tmp],dim='time')
        fid = fid + 1
    ds = ds.sortby('time')
    _, index = np.unique(ds['time'], return_index=True)
    ds = ds.isel(time=index)
    odir = folder + 'merged'
    os.makedirs(odir) 
    os.chdir(odir)
    for monthid,out_monthly in ds.resample(time='1M'):
        file_out = LOC + '_' + ID + '_' + sensor + '_' + input_type + '_' + level + "_" + version + '_' + str(pd.to_datetime((monthid)).year) + '{:02d}'.format(pd.to_datetime((monthid)).month)  + ".nc"
        out_monthly.to_netcdf(file_out, mode='w',encoding={'time': {'units':'minutes since ' + str(pd.to_datetime((monthid)).year) + \
                                                               '-' +  str(pd.to_datetime((monthid)).month) + '-01'}})
        

#_______________________________________________________
def mergeL3(nml):
    """
    Merges all L3 files from different years in one file. Removes duplicates.
    
    Input
    ----------
    L1Adir: str
        path of L3 data
    L1Bdir: str
        path of L1B data
    
    Output
    ----------
    monthly nc files containing raw corrected data L1B
    
    Example
    ----------
    L1AtoL1B()
    
    Required packages
    ----------
    glob, os, itertools, pandas
    
    Required functions
    ----------
    
    
    """


    base_dir = nml['global']['DIR']
    merged_L3dir = nml['global']['DIR'] + 'L3/'
    ID			= nml['global']['ID']    
    LOC			= nml['global']['LOC']
    sensor     = nml['global']['sensor']
    version		= nml['global']['version']
    input_type	= nml['global']['input_type']
    
    os.chdir(base_dir)
    L3dirs = sorted(glob.glob("*L3*"))
    fid = 0
    for L3dir in L3dirs:
        os.chdir(L3dir)
        if fid == 0:
            ds = xr.open_mfdataset("*L3*.nc")
            metafile = glob.glob('*meta.csv*')
            if metafile:
                print(metafile)
                meta = pd.read_csv(metafile[0],delimiter=',',header=0,decimal=".",na_values=-999)
                yyyymmdd_start = meta['yyyymmdd_start'][0]
                yyyymmdd_end = meta['yyyymmdd_end'][0]
                print('keeping data from ' + L3dir + ' between ' + yyyymmdd_start + ' and ' + yyyymmdd_end)
                ds  = ds.sel(time=slice(yyyymmdd_start, yyyymmdd_end))
                # ds = ds.drop_duplicates(dim="time", keep="last")
                _, index = np.unique(ds['time'], return_index=True)
                ds = ds.isel(time=index)

        else:
            tmp = xr.open_mfdataset("*L3*.nc")
            metafile = glob.glob('*meta.csv*')
            if metafile:

                meta = pd.read_csv(metafile[0],delimiter=',',header=0,decimal=".",na_values=-999)
                yyyymmdd_start = meta['yyyymmdd_start'][0]
                yyyymmdd_end = meta['yyyymmdd_end'][0]
                print('keeping data from ' + L3dir + ' between ' + yyyymmdd_start + ' and ' + yyyymmdd_end)
                tmp  = tmp.sel(time=slice(yyyymmdd_start, yyyymmdd_end))
                # tmp = tmp.drop_duplicates(dim="time", keep="last")
                _, index = np.unique(tmp['time'], return_index=True)
                tmp = tmp.isel(time=index)
            ds = xr.merge([ds,tmp])

        fid = fid + 1
        os.chdir(base_dir)
        
    if not os.path.exists(merged_L3dir):
        os.makedirs(merged_L3dir) 
        
    os.chdir(merged_L3dir)
    
    file_out = merged_L3dir + LOC + '_' + ID + '_' + sensor + '_' + input_type + '_' + "L3_" + version +  '_all.nc'
    
    ds.to_netcdf(file_out, mode='w')
    
    
#_______________________________________________________
def load_iWS(file,nml):
    """
    Load single raw iWS datafile in a data frame
    
    Input
    ----------
    file: str
        filename
    nml: python f90nml namelist 
        namelist containing all parameters
        
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
    
    WDoffset      = nml['L0toL1A']['WDoffset']
    gillfactor    = nml['L0toL1A']['gillfactor']

    # Load data in a data frame
    column_names       = ['yyyy','doy','HH','MM','SS','T1','T2','T_intern','U','WD','w','x1','x2']
    df_raw             = pd.read_csv(file,delimiter = '\t',names = column_names)
    
    # Protect against abnormal years and doy in raw files
    df_raw.loc[abs(df_raw['yyyy']-df_raw['yyyy'][0])>=1, 'yyyy'] = np.nan 
    df_raw.loc[abs(df_raw['doy'] -df_raw['doy'][0]) >=1, 'doy']  = np.nan 
    
    # Make time index 
    df_raw.index       = compose_date(df_raw['yyyy'], days=df_raw['doy'], hours=df_raw['HH'], minutes=df_raw['MM'], seconds=df_raw['SS'])
    
    # Remove duplicated timestamps
    df_raw             = df_raw[~df_raw.index.duplicated()]
    
    # Add nans when there is no data
    df_raw             = df_raw.reindex(pd.date_range(min(df_raw.index), max(df_raw.index), freq=pd.infer_freq(df_raw.index[0:3])))
    
    # Rename index as 'time'
    df_raw.index.names = ['time']
    
    # Rotate wind direction in azimuth reference
    df_raw.WD = df_raw.WD + WDoffset - 180
    
    # Keep wind direction in [0;360[ range
    df_raw.WD[df_raw.WD < 0]   = df_raw.WD[df_raw.WD < 0]   + 360 
    df_raw.WD[df_raw.WD > 360] = df_raw.WD[df_raw.WD > 360] - 360 
    
    # Convert (U,dd) to (u,v)
    # (Note: meteorological wind direction convention (wind FROM the north <=> WD = 0) is assumed !!!)
    df_raw['u']  = - df_raw.U * np.sin(df_raw.WD*np.pi/180)  
    df_raw['v']  = - df_raw.U * np.cos(df_raw.WD*np.pi/180) 
    
    # Convert temperature to Kelvin
    df_raw['T1']       = df_raw.T1       + 273.15
    df_raw['T2']       = df_raw.T2       + 273.15
    df_raw['T_intern'] = df_raw.T_intern + 273.15
    
    # Calibration of Gill signal
    df_raw['w'] = gillfactor * df_raw.w

    # Remove useless columns
    df_raw.drop(['yyyy','doy','HH','MM','SS','x1','x2','U'] , axis=1, inplace=True)
    
    return df_raw


#_______________________________________________________
def load_TOA_CSAT(file,nml):
    """
    Load single raw CSAT datafile in a data frame
    
    Input
    ----------
    file: str
        filename
    nml: python f90nml namelist 
        namelist containing all parameters

    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """

    column_names = nml['L0toL1A']['headers']
    WDoffset     = nml['L0toL1A']['WDoffset']
    freq         = nml['L0toL1A']['frequency']
    qfactor      = nml['L0toL1A']['qfactor']
    qbias        = nml['L0toL1A']['qbias']
    lreplace_24h = nml['L0toL1A']['lreplace_24h']
    
    # Read file, skip lines nr 1, 3 and 4 and use line 2 as variable names
    df_raw                     = pd.read_csv(file,delimiter = ',', na_values = 'NAN',skiprows=[0,1,2,3],names=column_names)
    
    # Replace 24:00 to 00:00
    if lreplace_24h:
        for k in range(np.shape(df_raw.TIMESTAMP)[0]):
            if df_raw.TIMESTAMP[k][-8:-6] == '24':
                df_raw.TIMESTAMP[k] =  df_raw.TIMESTAMP[k][:-8] + df_raw.TIMESTAMP[k][-8:].replace('24', '00') 
                 
    # Use first columns as dataframe index
    df_raw                     = df_raw.set_index(df_raw.columns[0])
    
    # Convert index to datetime
    df_raw.index               = pd.to_datetime(df_raw.index,format="ISO8601")

    # Remove duplicated timestamps
    df_raw                     = df_raw[~df_raw.index.duplicated()]
    
    # Fill skipped timestamps with NaNs
    df_raw                     = df_raw.reindex(pd.date_range(min(df_raw.index), max(df_raw.index), freq=freq))
    
    # Rename variables
    df_raw.index.names         = ['time']

    # Calculate wind direction in degrees
    df_raw['WD']               = uvtoWD(df_raw.u,df_raw.v)
                
    # Rotate wind direction in azimuth reference
    df_raw.WD                  = df_raw.WD + WDoffset - 270
    
    # Keep wind direction in [0;360[ range
    df_raw.loc[df_raw['WD'] < 0, 'WD']   += 360
    df_raw.loc[df_raw['WD'] > 360, 'WD'] -= 360
    
    # Convert temperature to Kelvin
    df_raw['T_sonic']          = df_raw.T_sonic   + 273.15
    df_raw['T_couple']         = df_raw.T_couple  + 273.15
    
    # CSAT abolute wind speed
    UCSAT = (df_raw['u']**2 + df_raw['v']**2)**(0.5)
    
    # Convert CSAT wind angles from (Ux,Uy) to (uW,vN)
    # Note: meteorological wind direction convention (wind FROM the north <=> WD = 0) is assumed !!!
    df_raw['u']  = - UCSAT * np.sin(df_raw.WD*np.pi/180)  
    df_raw['v']  = - UCSAT * np.cos(df_raw.WD*np.pi/180) 
    
    # Convert molar density (mmol/m3) to mass density (g/m3)
    if 'H2O' in df_raw.columns:
        df_raw['H2O'] = (18/1000)*df_raw['H2O']
    if 'CO2' in df_raw.columns:
        df_raw['CO2'] = (44/1000)*df_raw['CO2']
    # Convert g/m3 to kg/m3
    if 'q' in df_raw.columns:
        df_raw['q'] = df_raw['q']*qfactor + qbias
        df_raw['q'] = (1/1000)*df_raw['q']

    # Remove useless columns
    df_raw = df_raw[df_raw.columns.drop(list(df_raw.filter(regex='tmp')))]
    
    return df_raw

#_______________________________________________________
def load_TOA_Young(file,nml):
    """
    Load single raw CSAT datafile in a data frame
    
    Input
    ----------
    file: str
        filename
    nml: python f90nml namelist 
        namelist containing all parameters
    
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
    
    column_names = nml['L0toL1A']['headers']
    WDoffset     = nml['L0toL1A']['WDoffset']
    gillfactor   = nml['L0toL1A']['gillfactor']
    youngfactor  = nml['L0toL1A']['youngfactor']
    freq         = nml['L0toL1A']['frequency']  

    # Read file, skip lines nr 1, 3 and 4 and use line 2 as variable names
    df_raw                     = pd.read_csv(file,delimiter = ',', na_values = 'NAN',skiprows=[0,1,2,3],names = column_names)
    
    # Use first columns as dataframe index
    df_raw                     = df_raw.set_index(df_raw.columns[0])
    
    # Convert index to datetime
    df_raw.index               = pd.to_datetime(df_raw.index)

    # Remove duplicated timestamps
    df_raw                     = df_raw[~df_raw.index.duplicated()]
    
    # Fill skipped timestamps with NaNs
    df_raw                     = df_raw.reindex(pd.date_range(min(df_raw.index), max(df_raw.index), freq=freq))
    
    # Rename variables
    df_raw.index.names         = ['time']
           
    # Rotate wind direction in azimuth reference
    df_raw.WD                  = df_raw.WD + WDoffset - 180
    
    # Keep wind direction in [0;360[ range
    df_raw.loc[df_raw['WD'] < 0, 'WD']   += 360
    df_raw.loc[df_raw['WD'] > 360, 'WD'] -= 360

    # Convert Gill voltage (mV) to wind velocity (m/s)
    df_raw['w'] = gillfactor * df_raw.w

    # Apply Young calibration
    df_raw['U'] = youngfactor * df_raw.U
    
    # Convert temperature to Kelvin
    df_raw['T1']         = df_raw.T1  + 273.15

    # Convert Young (U,dd) to (u,v)
    # Note: meteorological wind direction convention (wind FROM the north <=> WD = 0) is assumed !!!
    df_raw['u']  = - df_raw.U * np.sin(df_raw.WD*np.pi/180)  
    df_raw['v']  = - df_raw.U * np.cos(df_raw.WD*np.pi/180) 
       
    # Remove useless columns
    df_raw.drop(['U'] , axis=1, inplace=True)
    
    # Remove useless columns
    df_raw = df_raw[df_raw.columns.drop(list(df_raw.filter(regex='tmp')))]
    
    return df_raw

#_______________________________________________________
def load_CSAT_iWS(group,nml):
    """
    Load raw datafiles in the same group containing both CSAT and iWS data sampled at 10Hz in two separate a dataframes
    
    Input
    ----------
    group: list
        list of grouped files
    nml: python f90nml namelist 
        namelist containing all parameters
        
    Output
    ----------
    df_raw_iWS: dataframe
        dataframe containing raw time-indexed data from Gill and Young anemometers and thermocouples
    df_raw_CSAT
        dataframe containing raw time-indexed data from CSAT instrument
    """
    
     # Load data in a data frame    
    column_names  = nml['L0toL1A']['headers']
    WDoffset_CSAT = nml['L0toL1A']['WDoffset_CSAT']
    WDoffset_iWS  = nml['L0toL1A']['WDoffset_iWS']
    gillfactor    = nml['L0toL1A']['gillfactor']
    youngfactor   = nml['L0toL1A']['youngfactor']
    freq          = nml['L0toL1A']['frequency']
    
    # Read all individual files
    fid = 0
    for file in group:
        if fid == 0:
            print(file)
            df_raw = pd.read_csv(os.getcwd() + "/" + file,delimiter = ',',names = column_names,header = 3, na_values = 'NAN')
            fid = fid + 1
        else:
            print(file)
            tmp = pd.read_csv(os.getcwd() + "/" + file,delimiter = ',',names = column_names,header = 3, na_values = 'NAN')
            # Concatenate data frames
            df_raw = pd.concat([df_raw,tmp])
            fid = fid + 1
                    
    # Split dataframes
    df_raw_CSAT = df_raw[['time','u','v','w','T_sonic','T_couple']]
    df_raw_iWS  = df_raw[['time','T_sonic','U','WD','w_gill','T_couple']]
    
    df_raw_CSAT                  = df_raw_CSAT.set_index('time')
    df_raw_CSAT.index            = pd.to_datetime(df_raw_CSAT.index)
    df_raw_CSAT                  = df_raw_CSAT[~df_raw_CSAT.index.duplicated()]
    df_raw_CSAT                  = df_raw_CSAT.reindex(pd.date_range(min(df_raw_CSAT.index), max(df_raw_CSAT.index), freq=freq))
    df_raw_CSAT.index.names      = ['time']
    
    df_raw_iWS                  = df_raw_iWS.set_index('time')
    df_raw_iWS.index            = pd.to_datetime(df_raw_iWS.index)
    df_raw_iWS                  = df_raw_iWS[~df_raw_iWS.index.duplicated()]
    df_raw_iWS                  = df_raw_iWS.reindex(pd.date_range(min(df_raw_iWS.index), max(df_raw_iWS.index), freq=freq))
    df_raw_iWS.index.names      = ['time']
    
    # Rename variables
    df_raw_iWS.rename(columns={'T_sonic': 'T2', 'T_couple': 'T1','w_gill': 'w'}, inplace=True)
    
    # Convert Gill voltage (mV) to wind velocity (m/s)
    df_raw_iWS['w'] = gillfactor * df_raw_iWS.w
        
    # Save CSAT wind direction in degrees
    df_raw_CSAT['WD'] = uvtoWD(df_raw_CSAT.u,df_raw_CSAT.v)
                
    # Rotate wind direction in azimuth reference
    df_raw_CSAT.WD = df_raw_CSAT.WD + WDoffset_CSAT - 270
    df_raw_iWS.WD  = df_raw_iWS.WD + WDoffset_iWS - 180
    
    # Keep wind direction in [0;360[ range
    df_raw_CSAT.WD[df_raw_CSAT.WD < 0]   = df_raw_CSAT.WD[df_raw_CSAT.WD < 0]   + 360 
    df_raw_CSAT.WD[df_raw_CSAT.WD > 360] = df_raw_CSAT.WD[df_raw_CSAT.WD > 360] - 360 
    df_raw_iWS.WD[df_raw_iWS.WD < 0]   = df_raw_iWS.WD[df_raw_iWS.WD < 0]   + 360 
    df_raw_iWS.WD[df_raw_iWS.WD > 360] = df_raw_iWS.WD[df_raw_iWS.WD > 360] - 360 
        
    # Convert temperature to Kelvin
    df_raw_CSAT['T_sonic']   = df_raw_CSAT.T_sonic  + 273.15
    df_raw_CSAT['T_couple']  = df_raw_CSAT.T_couple + 273.15  
    df_raw_iWS['T1'] = df_raw_iWS.T1 + 273.15
    df_raw_iWS['T2'] = df_raw_iWS.T2 + 273.15   
    
    # Convert Young pulses (# per sample) to wind velocity (m/s)
    df_raw_iWS['U'] = youngfactor * df_raw_iWS.U

    # CSAT abolute wind speed
    UCSAT = (df_raw_CSAT['u']**2 + df_raw_CSAT['v']**2)**(0.5)
    
    # Convert CSAT wind angles from (Ux,Uy) to (uW,vN)
    # Note: meteorological wind direction convention (wind FROM the north <=> WD = 0) is assumed !!!
    df_raw_CSAT['u']  = - UCSAT * np.sin(df_raw_CSAT.WD*np.pi/180)  
    df_raw_CSAT['v']  = - UCSAT * np.cos(df_raw_CSAT.WD*np.pi/180) 
    
    # Convert Young (U,dd) to (u,v)
    # Note: meteorological wind direction convention (wind FROM the north <=> WD = 0) is assumed !!!
    df_raw_iWS['u']  = - df_raw_iWS.U * np.sin(df_raw_iWS.WD*np.pi/180)  
    df_raw_iWS['v']  = - df_raw_iWS.U * np.cos(df_raw_iWS.WD*np.pi/180) 
       
    # Remove useless columns
    df_raw_iWS.drop(['U'] , axis=1, inplace=True)
    
    return  df_raw_iWS, df_raw_CSAT

#_______________________________________________________
def load_CSAT_Young(file,nml):
    """
    Load raw datafiles in the same group containing both CSAT and Young data sampled at 10Hz in two separate a dataframes
    
    Input
    ----------
    group: list
        list of grouped files
    nml: python f90nml namelist 
        namelist containing all parameters
        
    Output
    ----------
    df_raw_Young: dataframe
        dataframe containing raw time-indexed data from Gill and Young anemometers and thermocouples
    df_raw_CSAT
        dataframe containing raw time-indexed data from CSAT instrument
    """
    
     # Load data in a data frame    
    column_names  = nml['L0toL1A']['headers']
    WDoffset_CSAT = nml['L0toL1A']['WDoffset_CSAT']
    WDoffset_iWS  = nml['L0toL1A']['WDoffset_iWS']
    gillfactor    = nml['L0toL1A']['gillfactor']
    youngfactor   = nml['L0toL1A']['youngfactor']
    freq          = nml['L0toL1A']['frequency']
    
            
    df_raw                  = pd.read_csv(file,delimiter = ',', na_values = 'NAN',skiprows=[0,1,2,3],names = column_names)
    df_raw                  = df_raw.set_index(df_raw.columns[0])
    df_raw.index            = pd.to_datetime(df_raw.index,format="ISO8601")
    df_raw                  = df_raw[~df_raw.index.duplicated()]
    df_raw                  = df_raw.reindex(pd.date_range(min(df_raw.index), max(df_raw.index), freq=freq))
    df_raw.index.names      = ['time']
    
    # Rename variables
    
    # Convert Gill voltage (mV) to wind velocity (m/s)
    df_raw['wg'] = gillfactor * df_raw.wg
        
    # Save CSAT wind direction in degrees
    df_raw['WD'] = uvtoWD(df_raw.u,df_raw.v)
        
    # Rotate wind direction in azimuth reference
    df_raw.WD = df_raw.WD + WDoffset_CSAT - 270
    df_raw.WDy  = df_raw.WDy + WDoffset_iWS - 180
    
    # Keep wind direction in [0;360[ range
    # Keep wind direction in [0;360[ range
    df_raw.loc[df_raw['WD'] < 0, 'WD']   += 360
    df_raw.loc[df_raw['WD'] > 360, 'WD'] -= 360
    df_raw.loc[df_raw['WDy'] < 0, 'WDy']   += 360
    df_raw.loc[df_raw['WDy'] > 360, 'WDy'] -= 360
    
    # Convert temperature to Kelvin
    df_raw['T_sonic']   = df_raw.T_sonic  + 273.15
    df_raw['T_couple']  = df_raw.T_couple + 273.15  
    df_raw['T_couple2']  = df_raw.T_couple2 + 273.15  
    
    # Convert Young pulses (# per sample) to wind velocity (m/s)
    df_raw['Uy'] = youngfactor * df_raw.Uy

    # CSAT abolute wind speed
    UCSAT = (df_raw['u']**2 + df_raw['v']**2)**(0.5)
    
    # Convert CSAT wind angles from (Ux,Uy) to (uW,vN)
    # Note: meteorological wind direction convention (wind FROM the north <=> WD = 0) is assumed !!!
    df_raw['u']  = - UCSAT * np.sin(df_raw.WD*np.pi/180)  
    df_raw['v']  = - UCSAT * np.cos(df_raw.WD*np.pi/180) 
    
    # Convert Young (U,dd) to (u,v)
    # Note: meteorological wind direction convention (wind FROM the north <=> WD = 0) is assumed !!!
    df_raw['uy']  = - df_raw.Uy * np.sin(df_raw.WDy*np.pi/180)  
    df_raw['vy']  = - df_raw.Uy * np.cos(df_raw.WDy*np.pi/180) 
       
    # Convert molar density (mmol/m3) to mass density (g/m3)
    # if 'H2O' in df_raw.columns:
    #     df_raw['H2O'] = (18/1000)*df_raw['H2O']
    # if 'CO2' in df_raw.columns:
    #     df_raw['CO2'] = (44/1000)*df_raw['CO2']
        
    
    # Remove useless columns
    df_raw = df_raw[df_raw.columns.drop(list(df_raw.filter(regex='tmp')))]
    
    return df_raw


#_______________________________________________________
def load_custom(file,nml):
    """
    Load single custom datafile in a data frame
    This function is used to test different sampling strategies of a datalogger. It is is modified for every different test
    
    Input
    ----------
    file: str
        filename
    nml: python f90nml namelist 
        namelist containing all parameters
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """

    
    column_names = nml['L0toL1A']['headers']
    WDoffset     = nml['L0toL1A']['WDoffset']
    gillfactor   = nml['L0toL1A']['gillfactor']
    youngfactor  = nml['L0toL1A']['youngfactor']
    sensorname   = nml['L0toL1A']['sensorname']
    
    # Read file, skip lines nr 1, 3 and 4 and use input headers as variable names
    df_raw                     = pd.read_csv(file,delimiter = ',',header = 0, names = column_names, na_values = 'NAN',skiprows=[0,2,3])
    
    # Use first columns as dataframe index
    df_raw                     = df_raw.set_index(df_raw.columns[0])
    
    # Convert index to datetime
    df_raw.index               = pd.to_datetime(df_raw.index)
    
    # Fill skipped timestamps with NaNs
    df_raw                     = df_raw.reindex(pd.date_range(min(df_raw.index), max(df_raw.index), freq=pd.infer_freq(df_raw.index[0:3])))
    
    # Rename variables
    df_raw.index.names         = ['time']
    
    # Sensor specific corrections
    if sensorname == 'CSAT':
        # Convert temperature to Kelvin
        df_raw['T_sonic']  = df_raw.T_sonic   + 273.15
        df_raw['T_couple'] = df_raw.T_couple  + 273.15
        
        # Save wind direction in degrees
        df_raw['WD'] = uvtoWD(df_raw.u,df_raw.v)
                    
        # Rotate wind direction in azimuth reference
        df_raw.WD = df_raw.WD + WDoffset - 270
        
        # Keep wind direction in [0;360[ range
        df_raw.WD[df_raw.WD < 0]   = df_raw.WD[df_raw.WD < 0]   + 360 
        df_raw.WD[df_raw.WD > 360] = df_raw.WD[df_raw.WD > 360] - 360 
        
        # Rotate wind speeds
        U = (df_raw.u**2 + df_raw.v**2)**0.5
        df_raw.u, df_raw.v = UWDtouv(U,df_raw.WD)
        
    elif sensorname == 'Young':
        # Convert temperature to Kelvin
        df_raw['T1'] = df_raw.T1 + 273.15
        
        # Convert Gill voltage (mV) to wind velocity (m/s)
        df_raw['w'] = gillfactor  * df_raw.w
        
        # Apply Young calibration factor
        df_raw['U']  =  df_raw.U
        df_raw['Up'] = youngfactor * df_raw.Up
        
        # Rotate wind direction in azimuth reference
        df_raw.WD = df_raw.WD + WDoffset - 180
        
        # Keep wind direction in [0;360[ range
        df_raw.WD[df_raw.WD < 0]   = df_raw.WD[df_raw.WD < 0]   + 360 
        df_raw.WD[df_raw.WD > 360] = df_raw.WD[df_raw.WD > 360] - 360 
        
        # Convert Young (U,dd) to (u,v)
        # Note: meteorological wind direction convention (wind FROM the north <=> WD = 0) is assumed !!!
        df_raw['u']  = - df_raw.U * np.sin(df_raw.WD*np.pi/180)  
        df_raw['v']  = - df_raw.U * np.cos(df_raw.WD*np.pi/180) 
        
        # Remove rdundant data 
        df_raw.drop(['U'] , axis=1, inplace=True)

    # Remove useless columns
    cols = [c for c in df_raw.columns if c.lower()[:4] != '*tmp*']
    df_raw=df_raw[cols]
    
    return df_raw

#_______________________________________________________
def read_TOA(file,nml):
    """
    Load single custom datafile in a data frame
    This function is a simple function used as a basis for making more complex I/O scripts
    
    Input
    ----------
    file: str
        filename
    nml: python f90nml namelist 
        namelist containing all parameters
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """    

    column_names = nml['L0toL1A']['headers']
    gillfactor   = nml['L0toL1A']['gillfactor']
    
    # Read file, skip lines nr 1, 3 and 4 and use line 2 as variable names
    df_raw                     = pd.read_csv(file,delimiter = ',', names = column_names,header = 0, na_values = 'NAN',skiprows=[0,2,3])
    
    # Use first columns as dataframe index
    df_raw                     = df_raw.set_index(df_raw.columns[0])
    
    # Convert index to datetime
    df_raw.index               = pd.to_datetime(df_raw.index)
    
    # Fill skipped timestamps with NaNs
    df_raw                     = df_raw.reindex(pd.date_range(min(df_raw.index), max(df_raw.index), freq=pd.infer_freq(df_raw.index[0:3])))
    
    # Rename variables
    df_raw.index.names         = ['time']
    
    # Sensor specific corrections

    # Convert temperature to Kelvin
    df_raw['T2'] = df_raw.T2 + 273.15
    
    # Convert Gill voltage (mV) to wind velocity (m/s)
    df_raw['w'] = gillfactor * df_raw.w 
    
    return df_raw

#_______________________________________________________
def read_cal_TOA_CSAT(file,nml):
    """
    Load single datafile containing averages in TOA format
    
    Input
    ----------
    file: str
        filename
    nml: python f90nml namelist 
        namelist containing all parameters
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
    
    
    column_names = nml['L0toL1A']['headers']
    WDoffset   = nml['L0toL1A']['WDoffset']
    sensorname   = nml['L0toL1A']['sensorname']
    lreplace_24h = nml['L0toL1A']['lreplace_24h']
    
    # Read file, skip lines nr 1, 3 and 4 and use line 2 as variable names
    df_raw                     = pd.read_csv(file,delimiter = ',', names = column_names, na_values = 'NAN',skiprows=[0,1,2,3])

    idx = []
    # Replace 24:00 to 00:00
    if lreplace_24h:
        for k in range(np.shape(df_raw.TIMESTAMP)[0]):
            if df_raw.TIMESTAMP[k][-8:-6] == '24':
                idx.append(k)
                df_raw.TIMESTAMP[k] =  df_raw.TIMESTAMP[k][:-8] + df_raw.TIMESTAMP[k][-8:].replace('24', '00') 
             
    # Use first columns as dataframe index
    df_raw                     = df_raw.set_index(df_raw.columns[0])
    
    # Convert index to datetime
    df_raw.index               = pd.to_datetime(df_raw.index,format='ISO8601')
    # df_raw.index               = pd.to_datetime(df_raw.index)

    # Convert index to datetime
    df_raw                     = df_raw[~df_raw.index.duplicated()]
    
    if lreplace_24h:
        as_list = df_raw.index.tolist()
        for k in range(len(idx)):
            as_list[idx[k]] = df_raw.index[idx[k]] + np.timedelta64(1,'D')
        df_raw.index = as_list
       
    # Rename variables
    df_raw.index.names         = ['time']
    
    df_raw['WD'] = uvtoWD(df_raw.u,df_raw.v)
    
    # Rotate wind direction in azimuth reference
    df_raw.WD                  = df_raw.WD + WDoffset - 270
    
    # Keep wind direction in [0;360[ range
    df_raw.loc[df_raw['WD'] < 0, 'WD']   += 360
    df_raw.loc[df_raw['WD'] > 360, 'WD'] -= 360

    if sensorname == 'CSAT':
        df_raw['T_sonic']         = df_raw.T_sonic  + 273.15
        df_raw['T_couple']         = df_raw.T_couple  + 273.15
    
    # Remove useless columns
    df_raw = df_raw[df_raw.columns.drop(list(df_raw.filter(regex='tmp')))]
    df_raw = df_raw[df_raw.columns.drop(list(df_raw.filter(regex='Cov')))]  

    # # Convert molar density (mmol/m3) to mass density (kg/m3)
    # if 'q' in df_raw.columns:
    #     df_raw['q'] = (18/1000**2)*df_raw['q']
    #     df_raw['uq'] = (18/1000**2)*df_raw['uq']
    #     df_raw['vq'] = (18/1000**2)*df_raw['vq']
    #     df_raw['wq'] = (18/1000**2)*df_raw['wq']
    #     df_raw['sigmaq'] = (18/1000**2)*df_raw['sigmaq']
        
    # if 'CO2' in df_raw.columns:
    #     df_raw['CO2'] = (44/1000**2)*df_raw['CO2']
    #     df_raw['uCO2'] = (44/1000**2)*df_raw['uCO2']
    #     df_raw['vCO2'] = (44/1000**2)*df_raw['vCO2']
    #     df_raw['wCO2'] = (44/1000**2)*df_raw['wCO2']
    #     df_raw['sigmaCO2'] = (44/1000**2)*df_raw['sigmaCO2']

    if 'sigmaTs' in df_raw.columns:
        df_raw['TsTs'] = df_raw['sigmaTs']**2
        df_raw['TcTc'] = df_raw['sigmaTc']**2
        if 'q' in df_raw.columns:
            # df_raw['CO2CO2'] = df_raw['sigmaCO2']**2
            df_raw['qq'] = df_raw['sigmaq']**2
    # Convert mass density (g/m3) to mass density (kg/m3)
    if 'q' in df_raw.columns:
        df_raw['q'] = (1/1000)*df_raw['q']
        df_raw['uq'] = (1/1000)*df_raw['uq']
        df_raw['vq'] = (1/1000)*df_raw['vq']
        df_raw['wq'] = (1/1000)*df_raw['wq']
        df_raw['qq'] = (1/1000)**2*df_raw['qq']
    # Convert mass density (mg/m3) to mass density (kg/m3)
    if 'CO2' in df_raw.columns:
        df_raw['CO2'] = (1/1000**2)*df_raw['CO2']
        df_raw['uCO2'] = (1/1000**2)*df_raw['uCO2']
        df_raw['vCO2'] = (1/1000**2)*df_raw['vCO2']
        df_raw['wCO2'] = (1/1000**2)*df_raw['wCO2']
        df_raw['CO2CO2'] = (1/1000**2)**2*df_raw['CO2CO2']
    return df_raw
    
#_______________________________________________________
def read_cal_TOA_Young(file,nml):
    """
    Load single datafile containing averages in TOA format
    
    Input
    ----------
    file: str
        filename
    nml: python f90nml namelist 
        namelist containing all parameters
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
    
    
    column_names = nml['L0toL1A']['headers']
    WDoffset   = nml['L0toL1A']['WDoffset']
    sensorname   = nml['L0toL1A']['sensorname']   
    gillfactor   = nml['L0toL1A']['gillfactor']
    youngfactor  = nml['L0toL1A']['youngfactor']
    
    # Read file, skip lines nr 1, 3 and 4 and use line 2 as variable names
    df_raw                     = pd.read_csv(file,delimiter = ',', na_values = 'NAN',skiprows=[0,1,2,3],names = column_names)
    
    # Use first columns as dataframe index
    df_raw                     = df_raw.set_index(df_raw.columns[0])
    
    # Convert index to datetime
    df_raw.index               = pd.to_datetime(df_raw.index)

    # Remove duplicated timestamps
    df_raw                     = df_raw[~df_raw.index.duplicated()]
    
    # Fill skipped timestamps with NaNs
    df_raw                     = df_raw.reindex(pd.date_range(min(df_raw.index), max(df_raw.index), freq=freq))
    
    # Rename variables
    df_raw.index.names         = ['time']
           
    # Rotate wind direction in azimuth reference
    df_raw.WD                  = df_raw.WD + WDoffset - 180
    
    # Keep wind direction in [0;360[ range
    df_raw.loc[df_raw['WD'] < 0, 'WD']   += 360
    df_raw.loc[df_raw['WD'] > 360, 'WD'] -= 360

    # Convert Gill voltage (mV) to wind velocity (m/s)
    df_raw['w'] = gillfactor * df_raw.w

    # Apply Young calibration
    df_raw['U'] = youngfactor * df_raw.U
    
    # Convert temperature to Kelvin
    df_raw['T1']         = df_raw.T1  + 273.15

    # Convert Young (U,dd) to (u,v)
    # Note: meteorological wind direction convention (wind FROM the north <=> WD = 0) is assumed !!!
    df_raw['u']  = - df_raw.U * np.sin(df_raw.WD*np.pi/180)  
    df_raw['v']  = - df_raw.U * np.cos(df_raw.WD*np.pi/180) 
       
    # Remove useless columns
    df_raw.drop(['U'] , axis=1, inplace=True)
    
    # Remove useless columns
    df_raw = df_raw[df_raw.columns.drop(list(df_raw.filter(regex='tmp')))]

#_______________________________________________________
def read_cal_txt(file,nml):
    """
    Load single datafile containing averages in txt format
    
    Input
    ----------
    file: str
        filename
    nml: python f90nml namelist 
        namelist containing all parameters
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
    
    
    column_names = nml['L0toL1A']['headers']
    gillfactor   = nml['L0toL1A']['gillfactor']
    youngfactor = nml['L0toL1A']['youngfactor'] 
    WDoffset   = nml['L0toL1A']['WDoffset']
    sensorname   = nml['L0toL1A']['sensorname']
    yearstart = nml['L0toL1A']['yearstart']
    doystart = nml['L0toL1A']['doystart']
    
    # Read file, skip lines nr 1, 3 and 4 and use line 2 as variable names
    df_raw                     = pd.read_csv(file,delimiter = ',', names = column_names, na_values = [-99999,99999])

     # Remove wrong lines
    df_raw = df_raw.dropna(thresh = 18)
    
    # Remove useless columns 
    df_raw = df_raw[df_raw.columns.drop(list(df_raw.filter(regex='tmp')))]
        
    # Convert index to datetime
    if doystart == 0:
        df_raw.index       = compose_date(yearstart, days=df_raw['doy'], hours=(df_raw['hhmm']/100).astype(int),minutes = (df_raw['hhmm']%100).astype(int))
    else:
        df_raw['year'] = df_raw['doy']*0 + yearstart 
        df_raw['year'][df_raw['doy']<doystart] = yearstart+1
        df_raw.index       = compose_date(df_raw['year'], days=df_raw['doy'], hours=(df_raw['hhmm']/100).astype(int),minutes = (df_raw['hhmm']%100).astype(int))
    
    # Remove duplicated timestamps
    df_raw                     = df_raw[~df_raw.index.duplicated()]
    

    
    # Fill skipped timestamps with NaNs
    df_raw                     = df_raw.reindex(pd.date_range(min(df_raw.index), max(df_raw.index), freq=pd.infer_freq(df_raw.index[0:3])))
    
    # Rename variables
    df_raw.index.names         = ['time']
    
    df_raw['WD'] = uvtoWD(df_raw.u,df_raw.v)
    
    # Rotate wind direction in azimuth reference
    df_raw.WD                  = df_raw.WD + WDoffset - 270
    
    # Keep wind direction in [0;360[ range
    df_raw.loc[df_raw['WD'] < 0, 'WD']   += 360
    df_raw.loc[df_raw['WD'] > 360, 'WD'] -= 360

    if sensorname == 'CSAT':
        df_raw['T_sonic']         = df_raw.T_sonic  + 273.15
        df_raw['T_couple']         = df_raw.T_couple  + 273.15
    elif sensorname == 'Young':
        # Convert Gill voltage (mV) to wind velocity (m/s)
        df_raw['w'] = gillfactor * df_raw.w
    
        # Apply Young calibration
        df_raw['U'] = youngfactor * df_raw.U
    
        # Convert temperature to Kelvin
        df_raw['T1']         = df_raw.T1  + 273.15
    
        # Convert Young (U,dd) to (u,v)
        # Note: meteorological wind direction convention (wind FROM the north <=> WD = 0) is assumed !!!
        df_raw['u']  = - df_raw.U * np.sin(df_raw.WD*np.pi/180)  
        df_raw['v']  = - df_raw.U * np.cos(df_raw.WD*np.pi/180) 
           
        # Remove useless columns
        df_raw.drop(['U'] , axis=1, inplace=True)
        
    # Convert molar density (mmol/m3) to mass density (kg/m3)
    if 'q' in df_raw.columns:
        df_raw['q'] = (18/1000**2)*df_raw['q']
        df_raw['uq'] = (18/1000**2)*df_raw['uq']
        df_raw['vq'] = (18/1000**2)*df_raw['vq']
        df_raw['wq'] = (18/1000**2)*df_raw['wq']
        df_raw['qq'] = (18/1000**2)**2*df_raw['qq']
        df_raw['Tsq'] = (18/1000**2)*df_raw['Tsq']
        df_raw['Tcq'] = (18/1000**2)*df_raw['Tcq']
        
    if 'CO2' in df_raw.columns:
        df_raw['CO2'] = (44/1000**2)*df_raw['CO2']
    # Remove useless columns
    df_raw.drop(['doy','hhmm'] , axis=1, inplace=True)
    df_raw = df_raw[df_raw.columns.drop(list(df_raw.filter(regex='Cov')))]  
    
    return df_raw
 
#_______________________________________________________
def read_cal_dat_PKM(file,nml):
    """
    Load single datafile containing averages in txt format
    
    Input
    ----------
    file: str
        filename
    nml: python f90nml namelist 
        namelist containing all parameters
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
    
    
    column_names = nml['L0toL1A']['headers']
    gillfactor   = nml['L0toL1A']['gillfactor']
    youngfactor = nml['L0toL1A']['youngfactor'] 
    WDoffset   = nml['L0toL1A']['WDoffset']
    sensorname   = nml['L0toL1A']['sensorname']
    yearstart = nml['L0toL1A']['yearstart']
    doystart = nml['L0toL1A']['doystart']
    
    # Read file, skip lines nr 1, 3 and 4 and use line 2 as variable names
    df_raw                     = pd.read_csv(file,delim_whitespace=True, names = column_names, na_values = [-99999,99999],skiprows=[0])

     # Remove wrong lines
    df_raw = df_raw.dropna(thresh = 18)

    
    # Remove useless columns 
    df_raw = df_raw[df_raw.columns.drop(list(df_raw.filter(regex='tmp')))]
        
    # Convert index to datetime
    if doystart == 0:
        df_raw.index       = compose_date(yearstart, days=df_raw['doy'], hours=(df_raw['hhmm']/100).astype(int),minutes = (df_raw['hhmm']%100).astype(int))
    else:
        df_raw['year'] = df_raw['doy']*0 + yearstart 
        df_raw['year'][df_raw['doy']<doystart] = yearstart+1
        df_raw.index       = compose_date(df_raw['year'], days=df_raw['doy'], hours=(df_raw['hhmm']/100).astype(int),minutes = (df_raw['hhmm']%100).astype(int))
    
    # Remove duplicated timestamps
    df_raw                     = df_raw[~df_raw.index.duplicated()]
    

    
    # Fill skipped timestamps with NaNs
    df_raw                     = df_raw.reindex(pd.date_range(min(df_raw.index), max(df_raw.index), freq=pd.infer_freq(df_raw.index[15:20])))
    
    # Rename variables
    df_raw.index.names         = ['time']
    
    df_raw['WD'] = uvtoWD(df_raw.u,df_raw.v)
    
    # Rotate wind direction in azimuth reference
    df_raw.WD                  = df_raw.WD + WDoffset - 270
    
    # Keep wind direction in [0;360[ range
    df_raw.loc[df_raw['WD'] < 0, 'WD']   += 360
    df_raw.loc[df_raw['WD'] > 360, 'WD'] -= 360

    if sensorname == 'CSAT':
        df_raw['T_sonic']         = df_raw.T_sonic  + 273.15
        df_raw['T_couple']         = df_raw.T_couple  + 273.15
    elif sensorname == 'Young':
        # Convert Gill voltage (mV) to wind velocity (m/s)
        df_raw['w'] = gillfactor * df_raw.w
    
        # Apply Young calibration
        df_raw['U'] = youngfactor * df_raw.U
    
        # Convert temperature to Kelvin
        df_raw['T1']         = df_raw.T1  + 273.15
    
        # Convert Young (U,dd) to (u,v)
        # Note: meteorological wind direction convention (wind FROM the north <=> WD = 0) is assumed !!!
        df_raw['u']  = - df_raw.U * np.sin(df_raw.WD*np.pi/180)  
        df_raw['v']  = - df_raw.U * np.cos(df_raw.WD*np.pi/180) 
           
        # Remove useless columns
        df_raw.drop(['U'] , axis=1, inplace=True)
        
    # Convert molar density (mmol/m3) to mass density (kg/m3)
    if 'q' in df_raw.columns:
        df_raw['q'] = (18/1000**2)*df_raw['q']
        df_raw['uq'] = (18/1000**2)*df_raw['uq']
        df_raw['vq'] = (18/1000**2)*df_raw['vq']
        df_raw['wq'] = (18/1000**2)*df_raw['wq']
        df_raw['qq'] = (18/1000**2)**2*df_raw['qq']
        df_raw['Tsq'] = (18/1000**2)*df_raw['Tsq']
        df_raw['Tcq'] = (18/1000**2)*df_raw['Tcq']
        
    if 'CO2' in df_raw.columns:
        df_raw['CO2'] = (44/1000**2)*df_raw['CO2']
    # Remove useless columns
    df_raw.drop(['doy','hhmm'] , axis=1, inplace=True)
    df_raw = df_raw[df_raw.columns.drop(list(df_raw.filter(regex='Cov')))]  
    print(df_raw)
    
    return df_raw

#_______________________________________________________
def read_cal_txt_v02(file,nml):
    """
    Load single datafile containing averages in txt format
    
    Input
    ----------
    file: str
        filename
    nml: python f90nml namelist 
        namelist containing all parameters
    Output
    ----------
    df_raw: dataframe
        dataframe containing raw time-indexed data
    """
    
    
    WDoffset   = nml['L0toL1A']['WDoffset']
    sensorname   = nml['L0toL1A']['sensorname']
    
    # Read file, skip lines nr 1, 3 and 4 and use line 2 as variable names
    df_raw                     = pd.read_csv(file,delimiter = ',',header = 0, na_values = 'NaN')

             
    # Use first columns as dataframe index
    df_raw                     = df_raw.set_index(df_raw.columns[0])
    
    # Convert index to datetime
    df_raw.index               = pd.to_datetime(df_raw.index)
    
    
    # Rename variables
    df_raw.index.names         = ['time']
    
    df_raw['WD'] = uvtoWD(df_raw.u,df_raw.v)
    
    # Rotate wind direction in azimuth reference
    df_raw.WD                  = df_raw.WD + WDoffset - 270
    
    # Keep wind direction in [0;360[ range
    df_raw.loc[df_raw['WD'] < 0, 'WD']   += 360
    df_raw.loc[df_raw['WD'] > 360, 'WD'] -= 360

    if sensorname == 'CSAT':
        df_raw['T_sonic']         = df_raw.T_sonic  + 273.15
        df_raw['T_couple']         = df_raw.T_couple  + 273.15
    
    return df_raw

#_______________________________________________________
def cospectrum(x,y,N, frequency = 10):
    """
    Computes the cospectrum between time series x and y
    
    Input
    ----------
    x: numpy 1D array
        x time series
    y: numpy 1D array
        y time series
    N: int
        Amount of samples of x and y used for fft
    frequency: float
        sampling frequency (Hz)
    
    Output
    ----------
    Cospectrum of xy, defined as the real part of the cross spectrum. The zero-frequency component is removed.
    Note: The spectra are normalised such that sum(Sxy x df) = var(xy)
    
    Example
    ----------
    grouped.apply(lambda x: cospectrum(x['T_sonic'],x['w'],18000))
    
    Required packages
    ----------
    numpy
    
    Required functions
    ----------
    none
    
    """
    # Check if there are enough samples
    if len(x) < N:
        Co = np.full(int(N/2), np.nan)
        
    else:
        # check if there are enough non-nan samples
        nanfracx = np.count_nonzero(np.isnan(x))/len(x)
        nanfracy = np.count_nonzero(np.isnan(y))/len(y)
        
        if nanfracx > 0.05:
            Co = np.full(int(N/2), np.nan)
        elif nanfracy > 0.05:
            Co = np.full(int(N/2), np.nan)                       
        else:
            # Calculate mean
            xm = np.nanmean(x)
            ym = np.nanmean(y)
        
            # Calculate deviations from the mean
            x = x - xm
            y = y - ym
    
            # Replace NaNs by 0       
            x[np.isnan(x)] = 0 
            y[np.isnan(y)] = 0 
    
            # Calculate cross spectrum
            X     = np.fft.rfft(x[:N])
            Y     = np.fft.rfft(y[:N])
            Cross = np.conj(X) * Y
            
            # Calculate normalized cospectrum
            Co    = np.real(Cross)
            Co    *= 2./(frequency*N)
        
            # Remove zero frequency
            Co = np.delete(Co,0)

    return Co

#_______________________________________________________
def meanWindDir(U,WD):
    """
    Computes the average wind direction as the direction of the mean wind vector
    
    Input
    ----------
    U: numpy 1D array
        Wind speed time series in m/s
    WD: numpy 1D array
        Meteorological wind direction time series in degrees (0 - 360)

    
    Output
    ----------
    Mean meteorological wind direction in degrees (0 - 360)
    
    Example
    ----------
    
    
    Required packages
    ----------
    numpy
    
    Required functions
    ----------
    none
    
    """
    # Convert angles to radians
    WD = (np.pi/180)*WD
    
    # Sum wind vectors in cartesian coordinates
    xtot = np.nansum(np.multiply(U,-np.sin(WD)))
    ytot = np.nansum(np.multiply(U,-np.cos(WD)))
    
    # Check if there is data
    if np.all(np.isnan(U)):
        return np.nan
    
    # Wind direction of sum vector
    WD_mean = (180/np.pi)* math.atan2(-xtot,-ytot)
    
    # Change output interval from [-180:180[ to [0:360[
    if (WD_mean < 0):
        return WD_mean + 360
    else:
        return WD_mean
    
#_______________________________________________________
def uvtoWD(u,v):
    """
    Calculates wind direction from horizontal wind components
    
    Input
    ----------
    u: numpy 1D array
        Zonal wind velocity
    v: numpy 1D array
        Meridional wind velocity 

    
    Output
    ----------
    Meteorological wind direction in degrees (0 - 360)
    
    Example
    ----------
    
    
    Required packages
    ----------
    numpy
    
    Required functions
    ----------
    none
    
    """
    # Check if there is data
    if np.all(np.isnan(u)):
        return u         
       
    else:       
        # Wind direction of sum vector
        WD = (180/np.pi)* np.arctan2(-u,-v)
        
        # Set WD=0 when there is no wind
        WD[(u==0) & (v==0)] = np.nan
        
        # Change output interval from [-pi:pi] to [0:2pi]
        WD[WD < 0] = WD[WD < 0] + 360 
          
        return WD
 
#_______________________________________________________
def UWDtouv(U,WD):
    """
    Calculates horizontal wind components from wind direction and wind speed
    
    Input
    ----------
    U: numpy 1D array
        Horizontal wind velocity
    WD: numpy 1D array
        Meterological wind direction in degrees (0,360)

    
    Output
    ----------
    u : zonal wind speed
    v: meridional wind speed
    
    Example
    ----------
    
    
    Required packages
    ----------
    numpy
    
    Required functions
    ----------
    none
    
    """
    
    u  = - U * np.sin(WD*np.pi/180)  
    v  = - U * np.cos(WD*np.pi/180) 
      
    return u, v
    
    
    
#_______________________________________________________
def despike(da,q=7,T=150,loops=1):
    """
    Removes spikes in the data array using the median absolute difference method
    The medians are typically calculated in a period of 10 minutes (T = 150 for iWS, T = 6000 for CSAT)
    
    Input
    ----------
    da: xarray data array
        Data to despike (e.g temperature)
    q: integer
        Tolerance factor (=7 by default)
    T: integer
        Length of rolling median filter (= 150 by defualt)
    loops: integer
        Amount of despiking iterations to perform
  
    Output
    ----------
    xarray data array
        data with spikes replaced by NaNs
    
    Example
    ----------
    
    
    Required packages
    ----------
    numpy, pandas
    
    Required functions
    ----------
    none
    
    """   
    loop = 1
    
    while loop <= loops:
        # Check if time window is of adequate size
        if T > len(da):
            T = len(da)
        # Absolute difference
        dif = abs(da - da.rolling(time = T,min_periods=1).median())
        
        # Median absolute difference 
        MAD = abs(da - da.rolling(time = T,min_periods=1).median()).rolling(time = T,min_periods=1).median()
        
        # Remove spikes
        da[dif > (q/0.6745)*MAD] = np.nan
        
        loop += 1
        
    return da

#_______________________________________________________
def nandetrend(x):
    """
    Detrends the timeseries x(t) by removing least squares linear fit to original data
    
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
        return x
    
    # Perform least-sques regression
    m, b, r_val, p_val, std_err = stats.linregress(t[not_nan_ind],x[not_nan_ind])
    
    # Remove regression line from data
    return  x - (m*t + b)


#_______________________________________________________
def PlanarFit_matrix(df):
    """
    Computes the planar fit (pitch + roll rotation) correction matrix using block averaged wind velocities
    This function discards the measurements that are flagged as distorted
    
    Input
    ----------
    df: pandas dataframe
        Time-indexed dataframe that contains the three block-averaged wind velocities
  
    Output
    ----------
    A_pf: numpy (3,3) array
        Planar fit (pitch + roll rotation) correction matrix
    
    Example
    ----------
    A_pf = PlanarFit_matrix(df)
    
    Required packages
    ----------
    numpy, pandas
    
    Required functions
    ----------
    none
    
    """   

    # Horizontal wind speed
    U = (df['u']**2 + df['v']**2)**(0.5)
    
    # Reject low and high wind speeds to avoid clustering 
    idx = (U < 2) | (U > 15 ) | (abs(df['w']) < 0.03)
    
    df['u'][idx]      = np.nan
    df['v'][idx]      = np.nan
    df['w'][idx]      = np.nan

    # Reject flagged measurements
    df['u'][df.flagPF] = np.nan
    df['v'][df.flagPF] = np.nan
    df['w'][df.flagPF] = np.nan
    
    # Calculate long-term time averages
    u_av  = np.nanmean(df['u'])
    v_av  = np.nanmean(df['v'])
    w_av  = np.nanmean(df['w'])
    u2_av = np.nanmean(np.multiply(df['u'],df['u']))
    v2_av = np.nanmean(np.multiply(df['v'],df['v']))
    uv_av = np.nanmean(np.multiply(df['u'],df['v']))
    uw_av = np.nanmean(np.multiply(df['u'],df['w']))
    vw_av = np.nanmean(np.multiply(df['v'],df['w']))   
    
    # check if there is data remaining
    if np.isnan(u_av*v_av*w_av*u2_av*v2_av*uv_av*uw_av*vw_av):
        return np.array([[1, 0 ,0],[0, 1, 0],[0, 0, 1]])
    
    # Calculate least squares coefficients
    B = np.dot(np.matrix([[1, u_av, v_av],[u_av, u2_av, uv_av],[v_av, uv_av, v2_av]]).I,np.matrix([[w_av],[uw_av],[vw_av]]))
    B = np.asarray(B)
    
    # Convert to pitch and roll angles
    cosa = np.sqrt(float(B[2])**2 + 1) / np.sqrt(float(B[1])**2+float(B[2])**2+1)
    sina = - float(B[1])        / np.sqrt(float(B[1])**2+float(B[2])**2+1)
    cosb = 1                    / np.sqrt(float(B[2])**2+1)
    sinb = float(B[2])          / np.sqrt(float(B[2])**2+1)
    
    # Calculate angles in degrees
    a    = np.arccos(cosa)*(180/np.pi)
    b    = np.arccos(cosb)*(180/np.pi)
    
    # Write un-tilt tensor (pitch rotation matrix x roll rotation matrix)
    A_pitch  = np.array([[cosa, 0 ,-sina],[0, 1, 0],[sina, 0, cosa]])
    A_roll = np.array([[1, 0, 0],[0, cosb, sinb], [0, sinb, cosb]])
    
    A_pf    = np.dot(A_pitch,A_roll)
    
    return A_pf, a, b

#_______________________________________________________
def PlanarFit_correction(df,A_pf):
    '''
    Applies the planar fit + yaw correction to input data. 
    The vertical velocity bias is removed from the original data
    
    Input
    ----------
    df: pandas dataframe
        Time-indexed dataframe that contains the wind velocities in the three directions
    A_pf: numpy (3x3) array
        Planar fit (roll + pitch angle) rotation matrix
  
    Output
    ----------
    A_final : numpy (3,3) array
        Planar fit (pitch + roll rotation) correction matrix
    
    Example
    ----------
    PlanarFit_correction(df,A_pf)
    
    Required packages
    ----------
    numpy, pandas
    
    Required functions
    ----------
    none
    
    '''

    
    # Make input data matrix
    x_in = np.stack((df['u'].values, df['v'].values, df['w'].values))
    
    # Calculate time averages
    u_av  = np.nanmean(x_in[0],axis=0)
    v_av  = np.nanmean(x_in[1],axis=0)
    w_av  = np.nanmean(x_in[2],axis=0)
    
    # Apply roll and pitch rotation to average wind velocities
    [u_av, v_av, w_av] =  np.dot(A_pf,np.array([u_av,v_av,w_av]))
    
    # Total wind speed
    U_av  = np.sqrt(u_av**2 + v_av**2)
    
    # Check if there is data
    if np.isnan(U_av):
        df['u'] = df['u']
        df['v'] = df['v']
        df['w'] = df['w'] 

    else:
        # Write yaw rotation matrix
        A_yaw = np.array([[u_av/U_av, v_av/U_av, 0],[-v_av/U_av,u_av/U_av, 0],[0, 0, 1]])
        
        # Write planar fit + yaw rotation matrix
        A_final = np.dot(A_yaw,A_pf)
     
        # Apply planar fit + yaw rotation correction to input data
        [u,v,w] =  np.dot(A_final,x_in)
        
        df['u']   = u
        df['v']   = v
        df['w']   = w
        df['yaw'] = np.arctan2(v_av, u_av)
    
    return df

#_______________________________________________________
def Double_rotation(df,itype=0):
    '''
    Applies the double-rotation (DR) correction on the data
    
    Input
    ----------
    df: pandas dataframe
        Time-indexed dataframe that contains the raw wind velocities in the three directions
  
    Output
    ----------
    df: pandas dataframe
        Time-indexed dataframe that contains the rotated wind velocities in the three directions
    
    Example
    ----------
    Double_rotation(df)
    
    Required packages
    ----------
    numpy, pandas
    
    Required functions
    ----------
    none
    
    '''

    if itype == 0:
        if df.empty: return 
        else: 
            # Make input data matrix
            x_in = np.stack((df['u'].values, df['v'].values, df['w'].values))
            
            # Calculate time averages
            u_av  = np.nanmean(x_in[0],axis=0)
            v_av  = np.nanmean(x_in[1],axis=0)
            w_av  = np.nanmean(x_in[2],axis=0)
            
            # Total wind speed
            U_av  = np.sqrt(u_av**2 + v_av**2)
            
            # Yaw angle 
            gamma = np.arctan(v_av / u_av)*(180/np.pi)
            
            # Yaw rotation matrix (around z-axis)
            A_yaw = np.array([[u_av/U_av, v_av/U_av, 0],[-v_av/U_av,u_av/U_av, 0],[0, 0, 1]])
            
            # Apply yaw rotation to wind velocities
            [u1, v1, w1] =  np.dot(A_yaw,np.array([df['u'].values,df['v'].values,df['w'].values]))
            
            # Calculate new time averages
            u_av  = np.nanmean(u1,axis=0)
            v_av  = np.nanmean(v1,axis=0)
            w_av  = np.nanmean(w1,axis=0)
            
            # Denomniator in pitch matrix
            W_av  = np.sqrt(u_av**2 + w_av**2)
            
            # Pitch angle
            alpha =  np.arctan2(w_av,u_av)*(180/np.pi)
            
            # Pitch rotation  matrix (around y-axis)
            A_pitch  = np.array([[u_av/W_av, 0, w_av/W_av],[0, 1, 0],[-w_av/W_av, 0, u_av/W_av]])
        
            # Apply pitch rotation to wind velocities
            [u2, v2, w2] =  np.dot(A_pitch,np.array([u1,v1,w1]))
            
            df['yaw']   = gamma
            df['pitch'] = alpha
                
            df['u'] = u2
            df['v'] = v2
            df['w'] = w2
    elif itype == 1: # use uy,vy and wgill input
        if df.empty: return 
        else: 
            # Make input data matrix
            x_in = np.stack((df['uy'].values, df['vy'].values, df['wg'].values))
            
            # Calculate time averages
            u_av  = np.nanmean(x_in[0],axis=0)
            v_av  = np.nanmean(x_in[1],axis=0)
            w_av  = np.nanmean(x_in[2],axis=0)
            
            # Total wind speed
            U_av  = np.sqrt(u_av**2 + v_av**2)
            
            # Yaw angle 
            gamma = np.arctan(v_av / u_av)*(180/np.pi)
            
            # Yaw rotation matrix (around z-axis)
            A_yaw = np.array([[u_av/U_av, v_av/U_av, 0],[-v_av/U_av,u_av/U_av, 0],[0, 0, 1]])
            
            # Apply yaw rotation to wind velocities
            [u1, v1, w1] =  np.dot(A_yaw,np.array([df['uy'].values,df['vy'].values,df['wg'].values]))
            
            # Calculate new time averages
            u_av  = np.nanmean(u1,axis=0)
            v_av  = np.nanmean(v1,axis=0)
            w_av  = np.nanmean(w1,axis=0)
            
            # Denomniator in pitch matrix
            W_av  = np.sqrt(u_av**2 + w_av**2)
            
            # Pitch angle
            alpha =  np.arctan2(w_av,u_av)*(180/np.pi)
            
            # Pitch rotation  matrix (around y-axis)
            A_pitch  = np.array([[u_av/W_av, 0, w_av/W_av],[0, 1, 0],[-w_av/W_av, 0, u_av/W_av]])
        
            # Apply pitch rotation to wind velocities
            [u2, v2, w2] =  np.dot(A_pitch,np.array([u1,v1,w1]))
            
            df['yaw']   = gamma
            df['pitch'] = alpha
                
            df['uy'] = u2
            df['vy'] = v2
            df['wg'] = w2

    return df
#_______________________________________________________
def Double_rotation_cal(df):
    '''
    Applies the double-rotation (DR) correction on the already averaged wind speed and covariances
    
    Input
    ----------
    df: pandas dataframe
        Time-indexed dataframe that contains the calculated wind velocities in the three directions
  
    Output
    ----------
    df: pandas dataframe
        Time-indexed dataframe that contains the rotated wind velocities in the three directions
    
    Example
    ----------
    Double_rotation(df)
    
    Required packages
    ----------
    numpy, pandas
    
    Required functions
    ----------
    none
    
    '''

    
    if df.empty: return 
    else: 
    
        for i in range(0,len(df)):
            if np.isnan(df['u'])[i]: continue
            # Calculate time averages
            u_av  = df['u'][i]
            v_av  = df['v'][i]
            w_av  = df['w'][i]
            
            # Total wind speed
            U_av  = np.sqrt(u_av**2 + v_av**2)
            
            # Yaw angle 
            gamma = np.arctan2(v_av,u_av)*(180/np.pi)
            
            # Yaw rotation matrix (around z-axis)
            A_yaw = np.array([[u_av/U_av, v_av/U_av, 0],[-v_av/U_av,u_av/U_av, 0],[0, 0, 1]])
            
            # Apply yaw rotation to wind velocities
            [u_av_2, v_av_2, w_av_2] =  np.dot(A_yaw,np.array([u_av,v_av,w_av]))
                        
            # Denomniator in pitch matrix
            W_av_2  = np.sqrt(u_av_2**2 + w_av_2**2)
            
            # Pitch angle
            alpha =  np.arctan(w_av_2 / u_av_2)*(180/np.pi)
            
            # Pitch rotation  matrix (around y-axis)
            A_pitch  = np.array([[u_av_2/W_av_2, 0, w_av_2/W_av_2],[0, 1, 0],[-w_av_2/W_av_2, 0, u_av_2/W_av_2]])
        

            
            # Final rotation matrix
            A_tot = np.dot(A_pitch,A_yaw)
            
            # Apply both rotation to wind velocities
            [u_av, v_av, w_av] =  np.dot(A_tot,np.array([u_av, v_av, w_av]))
            
            # Apply both rotations to v(co)variances 
            wTs = np.array([df.uTs[i],df.vTs[i],df.wTs[i]])
            wTc = np.array([df.uTc[i],df.vTc[i],df.wTc[i]])

            uu  = np.array([[df.uu[i],df.uv[i],df.uw[i]],[df.uv[i],df.vv[i],df.vw[i]],[df.uw[i],df.vw[i],df.ww[i]]])
            
            wTs = np.dot(A_tot,wTs)
            wTc = np.dot(A_tot,wTc)
            uu = np.dot(np.dot(A_tot,uu),A_tot.transpose())
            
            df['DRyaw'][i]   = gamma
            df['DRpitch'][i] = alpha
                
            
            df['u'][i] = u_av
            df['v'][i] = v_av
            df['w'][i] = w_av
            
            df['uu'][i] = uu[0,0]
            df['uv'][i] = uu[0,1]
            df['uw'][i] = uu[0,2]
            df['vv'][i] = uu[1,1]
            df['vw'][i] = uu[2,1]
            df['ww'][i] = uu[2,2]
            
            df['uTs'][i] = wTs[0]
            df['vTs'][i] = wTs[1]
            df['wTs'][i] = wTs[2]
            
            df['uTc'][i] = wTc[0]
            df['vTc'][i] = wTc[1]
            df['wTc'][i] = wTc[2]
            
            if 'wq' in list(df.columns):
                wq = np.array([df.uq[i],df.vq[i],df.wq[i]])
                wq = np.dot(A_tot,wq)
                df['uq'][i] = wq[0]
                df['vq'][i] = wq[1]
                df['wq'][i] = wq[2]       
            if 'wCO2' in list(df.columns):
                wCO2 = np.array([df.uCO2[i],df.vCO2[i],df.wCO2[i]])
                wCO2 = np.dot(A_tot,wCO2)
                df['uCO2'][i] = wCO2[0]
                df['vCO2'][i] = wCO2[1]
                df['wCO2'][i] = wCO2[2]
    return df

#_______________________________________________________
def PlanarFit_cal(df):
    """
    Computes the planar fit (pitch + roll rotation) correction matrix using block averaged wind velocities
    This function discards the measurements that are flagged as distorted
    
    Input
    ----------
    df: pandas dataframe
        Time-indexed dataframe that contains the three block-averaged wind velocities
  
    Output
    ----------
    A_pf: numpy (3,3) array
        Planar fit (pitch + roll rotation) correction matrix
    
    Example
    ----------
    A_pf = PlanarFit_matrix(df)
    
    Required packages
    ----------
    numpy, pandas
    
    Required functions
    ----------
    none
    
    """   
    

    if df.empty: return 
    else: 
    
        # Horizontal wind speed
        U = (df['u']**2 + df['v']**2)**(0.5)
        
        # Reject low and high wind speeds to avoid clustering 
        idx = (U < 2) | (U > 15 ) | (abs(df['w']) < 0.03)
        
    
        df['u'][idx]      = np.nan
        df['v'][idx]      = np.nan
        df['w'][idx]      = np.nan

        # Reject flagged measurements
        df['u'][df.flagPF] = np.nan
        df['v'][df.flagPF] = np.nan
        df['w'][df.flagPF] = np.nan
    
       # Calculate long-term time averages
        u_av  = np.nanmean(df['u'])
        v_av  = np.nanmean(df['v'])
        w_av_long  = np.nanmean(df['w'])
        u2_av = np.nanmean(np.multiply(df['u'],df['u']))
        v2_av = np.nanmean(np.multiply(df['v'],df['v']))
        uv_av = np.nanmean(np.multiply(df['u'],df['v']))
        uw_av = np.nanmean(np.multiply(df['u'],df['w']))
        vw_av = np.nanmean(np.multiply(df['v'],df['w']))   
    
        # check if there is data remaining
        if np.isnan(u_av*v_av*w_av_long*u2_av*v2_av*uv_av*uw_av*vw_av):
            A_pf =  np.array([[1, 0 ,0],[0, 1, 0],[0, 0, 1]])
        else:
        
            # Calculate least squares coefficients
            B = np.dot(np.matrix([[1, u_av, v_av],[u_av, u2_av, uv_av],[v_av, uv_av, v2_av]]).I,np.matrix([[w_av_long],[uw_av],[vw_av]]))
            B = np.asarray(B)
            
            # Convert to pitch and roll angles
            cosa = np.sqrt(float(B[2])**2 + 1) / np.sqrt(float(B[1])**2+float(B[2])**2+1)
            sina = - float(B[1])               / np.sqrt(float(B[1])**2+float(B[2])**2+1)
            cosb = 1                    / np.sqrt(float(B[2])**2+1)
            sinb = float(B[2])          / np.sqrt(float(B[2])**2+1)
            
            # Calculate angles in degrees
            pitch   = np.arccos(cosa)*(180/np.pi)
            roll    = np.arccos(cosb)*(180/np.pi)
            
            # Write un-tilt tensor (pitch rotation matrix x roll rotation matrix)
            A_pitch  = np.array([[cosa, 0 ,-sina],[0, 1, 0],[sina, 0, cosa]])
            A_roll = np.array([[1, 0, 0],[0, cosb, sinb], [0, sinb, cosb]])
            
            # A_pf    = np.dot(A_roll,A_pitch)
            A_pf    = np.dot(A_pitch,A_roll)
       
        for i in range(0,len(df)):
            if np.isnan(df['u'])[i]: continue
            # Calculate time averages
            u_av  = df['u'][i]
            v_av  = df['v'][i]
            w_av  = df['w'][i]
            # Apply roll and pitch rotation to average wind velocities
            [u_av_2, v_av_2, w_av_2] =  np.dot(A_pf,np.array([u_av,v_av,w_av]))
            
            # Total wind speed
            U_av_2  = np.sqrt(u_av_2**2 + v_av_2**2)
            
            # Check if there is data
            if np.isnan(U_av_2):
                df['u'] = df['u']
                df['v'] = df['v']
                df['w'] = df['w'] 
        
            else:
                # Write yaw rotation matrix
                A_yaw = np.array([[u_av_2/U_av_2, v_av_2/U_av_2, 0],[-v_av_2/U_av_2,u_av_2/U_av_2, 0],[0, 0, 1]])
                yaw = np.arctan2(v_av_2,u_av_2)*(180/np.pi)
                # Write planar fit + yaw rotation matrix
                A_final = np.dot(A_yaw,A_pf)
             
                # Apply planar fit + yaw rotation correction to input data
                [u,v,w] =  np.dot(A_final,np.array([u_av,v_av,w_av]))
                
                df['u'][i]   = u
                df['v'][i]   = v
                df['w'][i]   = w
                # df['yaw'][i] = np.arctan2(v_av, u_av)
    
                wTs = np.array([df.uTs[i],df.vTs[i],df.wTs[i]])
                wTc = np.array([df.uTc[i],df.vTc[i],df.wTc[i]])
                uu  = np.array([[df.uu[i],df.uv[i],df.uw[i]],[df.uv[i],df.vv[i],df.vw[i]],[df.uw[i],df.vw[i],df.ww[i]]])
                
                wTs = np.dot(A_final,wTs)
                wTc = np.dot(A_final,wTc)
                uu = np.dot(np.dot(A_final,uu),A_final.transpose())
                
                # df['yaw'][i]   = gamma
                # df['pitch'][i] = alpha
                    
                
                df['u'][i] = u_av
                df['v'][i] = v_av
                df['w'][i] = w_av
                
                df['uu'][i] = uu[0,0]
                df['uv'][i] = uu[0,1]
                df['uw'][i] = uu[0,2]
                df['vv'][i] = uu[1,1]
                df['vw'][i] = uu[2,1]
                df['ww'][i] = uu[2,2]
                
                df['uTs'][i] = wTs[0]
                df['vTs'][i] = wTs[1]
                df['wTs'][i] = wTs[2]
                
                df['uTc'][i] = wTc[0]
                df['vTc'][i] = wTc[1]
                df['wTc'][i] = wTc[2]
                
                df['PFroll'][i] = roll
                df['PFpitch'][i] = pitch
                df['PFyaw'][i] = yaw
                
                if 'wq' in list(df.columns):
                    wq = np.array([df.uq[i],df.vq[i],df.wq[i]])
                    wq = np.dot(A_final,wq)
                    df['uq'][i] = wq[0]
                    df['vq'][i] = wq[1]
                    df['wq'][i] = wq[2]       
                if 'wCO2' in list(df.columns):
                    wCO2 = np.array([df.uCO2[i],df.vCO2[i],df.wCO2[i]])
                    wCO2 = np.dot(A_final,wCO2)
                    df['uCO2'][i] = wCO2[0]
                    df['vCO2'][i] = wCO2[1]
                    df['wCO2'][i] = wCO2[2]
                
        return df
            

#_______________________________________________________
def mergeL1A(nml):
    """
    Merges all L1A files from differen sensors in one folder. Removes duplicates.
    
    Input
    ----------
    nml: python f90nml namelist 
        namelist containing all parameters
    
    Output
    ----------
    monthly nc files containing raw corrected data L1B
    
    Example
    ----------
    L1AtoL1B()
    
    Required packages
    ----------
    glob, os, itertools, pandas
    
    Required functions
    ----------
    
    
    """


    base_dir = nml['global']['DIR']
    merged_L1Adir = nml['global']['DIR'] + 'L1A/'
    ID			= nml['global']['ID']    
    LOC			= nml['global']['LOC']
    sensor     = nml['global']['sensor']
    version		= nml['global']['version']
    input_type	= nml['global']['input_type']
    
    os.chdir(base_dir)
    L1Adirs = sorted(glob.glob("*L1A_*"))
    fid = 0
    for L1Adir in  L1Adirs:
        os.chdir(L1Adir)
        if fid == 0:
            ds = xr.open_mfdataset("*L1A*.nc")
            _, index = np.unique(ds['time'], return_index=True)
            ds = ds.isel(time=index)
        else:
            tmp = xr.open_mfdataset("*L1A*.nc")
            _, index = np.unique(tmp['time'], return_index=True)
            tmp = tmp.isel(time=index)
            ds = ds.combine_first(tmp)
        fid = fid + 1
        os.chdir(base_dir)
        
    if not os.path.exists(merged_L1Adir):
        os.makedirs(merged_L1Adir) 
        
    os.chdir(merged_L1Adir)
    
    for monthid,out_monthly in ds.resample(time='1M'):
         file_out = merged_L1Adir + LOC + '_' + ID + '_' + sensor + '_' + input_type + '_' + "L1A_" + version +  '_' +\
            str(pd.to_datetime((monthid)).year) + '{:02d}'.format(pd.to_datetime((monthid)).month) + ".nc"
         out_monthly.to_netcdf(file_out, mode='w',encoding={'time': {'units':'minutes since ' + str(pd.to_datetime((monthid)).year) + \
                                                           '-' +  str(pd.to_datetime((monthid)).month) + '-01'}})    
         
                     
#_______________________________________________________
def L0toL1A(nml):
    """
    Reads all the selected raw data files, structures the data, fills the gaps, rotates the wind veolicties in the average wind vector reference frame,
    flags the undistorted data and writes the data in a single monthly nc file.
    This function needs the input parameters from the namoptions file 
    """
    

    base_dir        = nml['global']['dir']
    L0dir           = base_dir + nml['global']['L0dir']
    L1Adir          = base_dir + 'L1A/'
    sensor          = nml['global']['sensor']
        
    ID				= nml['global']['ID']    
    LOC				= nml['global']['LOC']
    version			= nml['global']['version']
    input_type		= nml['global']['input_type']
    signature		= nml['global']['signature']
    institute		= nml['global']['institute']
    LOC				= nml['global']['LOC']
    
    toffset_min		= nml['L0toL1A']['toffset']


    
    if not os.path.exists(L1Adir):
        os.makedirs(L1Adir) 
        
    # Move to input data directory
    if not L0dir:
        os.chdir(os.getcwd())  
    else:
        os.chdir(L0dir)  
    
    # Switch on sensor type
    if sensor == 'CSAT' :
        
        # Group files per {IDYYMM}
        groups =  [list(g) for _, g in itertools.groupby(sorted(glob.glob("*TOA5*.dat"),key=lambda x: x[-19:]), lambda x: x[-19:-12])]
 
        # Loop over {IDYYMM} files
        for group in groups:
            # Group files per {DD}
            subgroups = [list(g) for _, g in itertools.groupby(sorted(group,key=lambda x: x[-19:]), lambda x: x[-11:-9])]
            
            i = 0
            for subgroup in subgroups:
                fid = 0
                # Load raw file in a dataframe
                for file in subgroup:
                    if fid == 0:
                        print(file,file[-19:-12],file[-11:-9])

                        out = load_TOA_CSAT(os.getcwd() + "/" + file,nml)
                        fid = fid + 1
                    else:
                        print(file)
                        tmp = load_TOA_CSAT(os.getcwd() + "/" + file,nml)
                        out = pd.concat([out,tmp])
                        fid = fid + 1
                # Convert to xarray
                x = out.to_xarray()
                i = i + 1
                x = x.sortby(x.time)
                # Export to temporary file
                x.to_netcdf('tmp-%04d.nc' % i)  
            # Combine temporary netCDF files
            final = xr.open_mfdataset('tmp-*.nc',combine='by_coords')
            
            # Make sure data is sorted in increasing time
            final = final.sortby(final.time)
            
            # Correct time to UTC
            final['time'] = final.time.values + np.timedelta64(toffset_min,'m')
         
            # Export dataframe to net CDF       
            file_out = L1Adir + LOC + '_' + ID + '_' + sensor + '_' + input_type + '_' + "L1A_" + version + '_' + group[0][-19:-15] + group[0][-14:-12] + ".nc"
            
            # File attributes
            final                                      = utils.Add_dataset_attributes(final,'CSAT_EC_L1A.JSON')
            final.attrs['location']                    = LOC + '_' + ID
            final.attrs['file_creation_date_time']     = str(datetime.datetime.now())
            final.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
            final.attrs['IceEddie_namelist_L0toL1A']   = str(nml['L0toL1A'])
            
            # Export to net CDF
            final.to_netcdf(file_out,mode = 'w',encoding={'time': \
                         {'units':'seconds since ' + group[0][-19:-15] + '-' +  group[0][-14:-12] + '-01'}})
    
            # Delete temporary files
            del final
            for tmpfiles in glob.glob("tmp*"):
                os.remove(tmpfiles)
                
                
    elif sensor == 'Young' :
        
        # Group files per {YYMM}
        groups =  [list(g) for _, g in itertools.groupby(sorted(glob.glob("*TOA5*raw*.dat"),key=lambda x: x[-19:]), lambda x: x[-19:-12])]
        
        
        # Loop over {IDYYMM} files
        for group in groups:
            
            # Group files per {DD}
            subgroups = [list(g) for _, g in itertools.groupby(sorted(group,key=lambda x: x[-19:]), lambda x: x[-11:-9])]
            
            i = 0
            for subgroup in subgroups:
                
                fid = 0
                # Load raw file in a dataframe
                for file in subgroup:
                    if fid == 0:
                        print(file)
                        out = load_TOA_Young(os.getcwd() + "/" + file,nml)
                        fid = fid + 1
                    else:
                        print(file)
                        tmp = load_TOA_Young(os.getcwd() + "/" + file,nml)
                        out = pd.concat([out,tmp])
                        fid = fid + 1

                
                # Convert to xarray
                x = out.to_xarray()
                
                i = i + 1
                # Export to temporary file
                x.to_netcdf('tmp-%04d.nc' % i)
                
            # Combine temporary netCDF files
            final = xr.open_mfdataset('tmp-*.nc',combine='by_coords')
            
            # Make sure data is sorted in increasing time
            final = final.sortby(final.time)
            
            # Correct time to UTC
            final['time'] = final.time.values + np.timedelta64(toffset_min,'m')
            
            # Export dataframe to net CDF   
            file_out = L1Adir + LOC + '_' + ID + '_' + sensor + '_' + input_type + '_' + "L1A_" + version + '_' + group[0][-19:-15] + group[0][-14:-12] + ".nc"
            
            # File attributes
            final                                      = utils.Add_dataset_attributes(final,'YOUNG_EC_L1A.JSON')
            final.attrs['location']                    = LOC + '_' + ID
            final.attrs['file_creation_date_time']     = str(datetime.datetime.now())
            final.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
            final.attrs['IceEddie_namelist_L0toL1A']   = str(nml['L0toL1A'])
            
            # Export to net CDF
            final.to_netcdf(file_out,mode = 'w',encoding={'time': \
                         {'units':'seconds since ' + group[0][-19:-15] + '-' +  group[0][-14:-12] + '-01'}})
    
            # Delete temporary files
            del final
            for tmpfiles in glob.glob("tmp*"):
                os.remove(tmpfiles)
                
    elif sensor == 'iWS':

        # Group files per {IDYYMM}
        groups =  [list(g) for _, g in itertools.groupby(sorted(glob.glob("*SECOND.txt")), lambda x: x[0:6])]
        # Loop over {IDYYMM} files
        for group in groups:
            fid = 0
            # Load raw file in a dataframe
            for file in group:
                if fid == 0:
                    print(file)
                    out = load_iWS(os.getcwd() + "/" + file,nml)
                    fid = fid + 1
                else:
                    print(file)
                    tmp = load_iWS(os.getcwd() + "/" + file,nml)
                    # Load raw file in a dataframe
                    out = pd.concat([out,tmp])
                    fid = fid + 1                    
            
            # Export dataframe to net CDF     
            file_out = L1Adir + LOC + '_' + ID + '_' + sensor + '_' + input_type + '_' + "L1A_" + version + '_20' + group[0][2:6] + ".nc"
            
            # Convert to xarray
            x = out.to_xarray()

            # File attributes
            x                                      = utils.Add_dataset_attributes(x,'iWS_EC_L1A.JSON')
            x.attrs['location']                    = LOC + '_' + ID
            x.attrs['file_creation_date_time']     = str(datetime.datetime.now())
            x.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
            x.attrs['IceEddie_namelist_L0toL1A']   = str(nml['L0toL1A'])
            
            # Export to net CDF
            x.to_netcdf(file_out,mode = 'w',encoding={'time': \
                         {'units':'seconds since  20' + group[0][2:4] + '-' +  group[0][4:6] + '-01'}}) 
    

    elif sensor == 'iWS+CSAT':

        # Group files per {IDYYMM}
        groups =  [list(g) for _, g in itertools.groupby(sorted(glob.glob("*.dat")), lambda x: x[-19:-12])]
        # Loop over {IDYYMM} files
        for group in groups:    
            # Group files per {DD}
            subgroups = [list(g) for _, g in itertools.groupby(sorted(group), lambda x: x[-11:-9])]
            # Process files in chunks of 1 day
            i = 0
            for subgroup in subgroups:
                
                # Load raw file in a dataframe
                df_raw_iWS, df_raw_CSAT = load_CSAT_iWS(subgroup,nml)
                
                # Convert to xarray
                x_CSAT = df_raw_CSAT.to_xarray()
                x_iWS  = df_raw_iWS.to_xarray()
                
                i = i + 1
                # Write to temporary files
                x_CSAT.to_netcdf('tmpCSAT-%04d.nc' % i)
                x_iWS.to_netcdf('tmpiWS-%04d.nc' % i)
                     
            # Combine temporary netCDF files
            out_CSAT = xr.open_mfdataset(glob.glob("tmpCSAT*"),combine='by_coords')
            out_iWS  = xr.open_mfdataset(glob.glob("tmpiWS*"),combine='by_coords')
                        
            # Make sure data is sorted in increasing time
            out_CSAT = out_CSAT.sortby(out_CSAT.time)
            out_iWS = out_iWS.sortby(out_iWS.time)
            
            # Make output files
            file_out_CSAT = L1Adir + LOC + '_' + ID + '_' + 'CSAT' + '_' + input_type + '_' + "L1A_" + version + '_' + group[0][-19:-15] + group[0][-14:-12] + ".nc"
            file_out_iWS = L1Adir + LOC + '_' + ID + '_' + 'iWS' + '_' + input_type + '_' + "L1A_" + version + '_' + group[0][-19:-15] + group[0][-14:-12] + ".nc"

            
            # File attributes
            out_CSAT                                      = utils.Add_dataset_attributes(out_CSAT,'CSAT_EC_L1A.JSON')
            out_CSAT.attrs['location']                    = LOC + '_' + ID
            out_CSAT.attrs['file_creation_date_time']     = str(datetime.datetime.now())
            out_CSAT.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
            out_CSAT.attrs['IceEddie_namelist_L0toL1A']   = str(nml['L0toL1A'])
            
            out_iWS                                      = utils.Add_dataset_attributes(out_iWS,'iWS_EC_L1A.JSON')
            out_iWS.attrs['location']                    = LOC + '_' + ID
            out_iWS.attrs['file_creation_date_time']     = str(datetime.datetime.now())
            out_iWS.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
            out_iWS.attrs['IceEddie_namelist_L0toL1A']   = str(nml['L0toL1A'])

            
            # Export to net CDF
            out_CSAT.to_netcdf(file_out_CSAT,mode = 'w',encoding={'time': \
                         {'units':'seconds since ' + group[0][-19:-15] + '-' +  group[0][-14:-12] + '-01' }}) 
            out_iWS.to_netcdf(file_out_iWS,mode = 'w',encoding={'time': \
                         {'units':'seconds since ' + group[0][-19:-15] + '-' +  group[0][-14:-12] + '-01' }}) 
    
            # Delete temporary files
            del out_iWS, out_CSAT
            for tmpfiles in glob.glob(L0dir+ "/*tmp*"):
                os.remove(tmpfiles)
    
    elif sensor == 'Custom':  
        
        fnameprefix     = nml['L0toL1A']['fnameprefix']
        sensorname      = nml['L0toL1A']['sensorname']
        # Group files per {IDYYMM}
        groups =  [list(g) for _, g in itertools.groupby(sorted(glob.glob("*" + fnameprefix + "*.dat"),key=lambda x: x[-19:]), lambda x: x[-19:-12])]

            
        # Loop over {IDYYMM} files
        for group in groups:
                   
            # Group files per {DD}
            subgroups = [list(g) for _, g in itertools.groupby(sorted(group,key=lambda x: x[-19:]), lambda x: x[-11:-9])]
            
            i = 0
            for subgroup in subgroups:
                
                fid = 0
                # Load raw file in a dataframe
                for file in subgroup:
                    if fid == 0:
                        print(file)
                        out = load_custom(os.getcwd() + "/" + file,nml)
                        fid = fid + 1
                    else:
                        print(file)
                        tmp = load_custom(os.getcwd() + "/" + file,nml)
                        # Load raw file in a dataframe
                        out = pd.concat([out,tmp])
                        fid = fid + 1
                        
                    
                # Export dataframe to net CDF 
                file_out = L1Adir + LOC + '_' + ID + '_' + sensorname + '_' + input_type + '_' + "L1A_" + version + '_' + group[0][-19:-15] + group[0][-14:-12] + ".nc"
                    
                # Convert to xarray
                x = out.to_xarray()
                
                i = i + 1
                # Export to temporary file
                x.to_netcdf('tmp-%04d.nc' % i)
                
            # Combine temporary netCDF files
            final = xr.open_mfdataset('tmp-*.nc',combine='by_coords')
            
            # Make sure data is sorted in increasing time
            final = final.sortby(final.time)
            
            # Correct time to UTC
            final['time'] = final.time.values + np.timedelta64(toffset_min,'m')
            
            # Variable attributes
            if sensorname == 'CSAT':
                final                                      = utils.Add_dataset_attributes(final,'CSAT_EC_L1A.JSON')
                final.attrs['location']                    = LOC + '_' + ID
                final.attrs['file_creation_date_time']     = str(datetime.datetime.now())
                final.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
                final.attrs['IceEddie_namelist_L0toL1A']   = str(nml['L0toL1A'])
                
            elif sensorname == 'Young':
                final                                      = utils.Add_dataset_attributes(final,'YOUNG_EC_L1A.JSON')
                final.attrs['location']                    = LOC + '_' + ID
                final.attrs['file_creation_date_time']     = str(datetime.datetime.now())
                final.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
                final.attrs['IceEddie_namelist_L0toL1A']   = str(nml['L0toL1A'])
                
            # Export to net CDF
            final.to_netcdf(file_out,mode = 'w',encoding={'time': \
                         {'units':'seconds since ' + group[0][-19:-15] + '-' +  group[0][-14:-12] + '-01' }})   
    
            # Delete temporary files
            del final
            for tmpfiles in glob.glob("tmp*"):
                os.remove(tmpfiles)
                
    elif sensor == 'Gill':  
        
        # Group files per {IDYYMM}
        groups =  [list(g) for _, g in itertools.groupby(sorted(glob.glob("*raw_1sec*.dat"),key=lambda x: x[-19:]), lambda x: x[-19:-12])]
            
        # Loop over {IDYYMM} files
        for group in groups:
                   
            # Group files per {DD}
            subgroups = [list(g) for _, g in itertools.groupby(sorted(group,key=lambda x: x[-19:]), lambda x: x[-11:-9])]
            
            i = 0
            for subgroup in subgroups:
                
                fid = 0
                # Load raw file in a dataframe
                for file in subgroup:
                    if fid == 0:
                        print(file)
                        out = read_TOA(os.getcwd() + "/" + file)
                        fid = fid + 1
                    else:
                        print(file)
                        tmp = read_TOA(os.getcwd() + "/" + file)
                        # Load raw file in a dataframe
                        out = pd.concat([out,tmp])
                        fid = fid + 1
                        
                    
                # Export dataframe to net CDF 
                file_out = L1Adir + LOC + '_' + ID + '_' + fnameprefix + '_' + input_type + '_' + "L1A_" + version + '_' + group[0][-19:-15] + group[0][-14:-12] + ".nc"
                    
                # Convert to xarray
                x = out.to_xarray()
                
                i = i + 1
                # Export to temporary file
                x.to_netcdf('tmp-%04d.nc' % i,encoding={'time': \
                         {'units':'seconds since ' + group[0][-19:-15] + '-' +  group[0][-14:-12] + '-01' }})  
                
            # Combine and remove temporary netCDF files   
            fid = 0
            for tmpfile in sorted(glob.glob("tmp*")):
                if fid == 0:
                    final = xr.open_dataset(os.getcwd() + "/" + tmpfile)
                else:
                    final = xr.concat([final, xr.open_dataset(os.getcwd() + "/" + tmpfile)], 'time') 
                fid = fid + 1
                os.remove(tmpfile)  

            

            # Add new attributes
            final                                      = utils.Add_dataset_attributes(final,'GILL_EC_L1A.JSON')
            final.attrs['location']                    = LOC + '_' + ID
            final.attrs['file_creation_date_time']     = str(datetime.datetime.now())
            final.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
            final.attrs['IceEddie_namelist_L0toL1A']   = str(nml['L0toL1A'])

            # Export to net CDF
            final.to_netcdf(file_out,mode = 'w',encoding={'time': \
                         {'units':'seconds since ' + group[0][-19:-15] + '-' +  group[0][-14:-12] + '-01' }})   
                 
    elif sensor == 'CSAT_Young':
        
        # Group files per {IDYYMM}
        groups =  [list(g) for _, g in itertools.groupby(sorted(glob.glob("*TOA5*.dat"),key=lambda x: x[-19:]), lambda x: x[-19:-12])]
 
        # Loop over {IDYYMM} files
        for group in groups:
            # Group files per {DD}
            subgroups = [list(g) for _, g in itertools.groupby(sorted(group,key=lambda x: x[-19:]), lambda x: x[-11:-9])]
            
            i = 0
            for subgroup in subgroups:
                fid = 0
                # print(subgroup)
                # Load raw file in a dataframe
                for file in subgroup:
                    if fid == 0:
                        print(file)
                        # print(fid)
                        out = load_CSAT_Young(file,nml)
                        fid = fid + 1
                    else:
                        print(file)
                        # print(fid)
                        tmp = load_CSAT_Young(file,nml)
                        out = pd.concat([out,tmp])
                        fid = fid + 1
                # Convert to xarray
                # print(list(out.keys()))
                x = out.to_xarray()
                i = i + 1
                # Export to temporary file
                x.to_netcdf('tmp-%04d.nc' % i)  
            # Combine temporary netCDF files
            final = xr.open_mfdataset('tmp-*.nc',combine='by_coords')
            
            # Make sure data is sorted in increasing time
            final = final.sortby(final.time)
            
            # Correct time to UTC
            final['time'] = final.time.values + np.timedelta64(toffset_min,'m')
         
            # Export dataframe to net CDF       
            file_out = L1Adir + LOC + '_' + ID + '_' + sensor + '_' + input_type + '_' + "L1A_" + version + '_' + group[0][-19:-15] + group[0][-14:-12] + ".nc"
            
            # File attributes
            final                                      = utils.Add_dataset_attributes(final,'CSAT_EC_L1A.JSON')
            final.attrs['location']                    = LOC + '_' + ID
            final.attrs['file_creation_date_time']     = str(datetime.datetime.now())
            final.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
            final.attrs['IceEddie_namelist_L0toL1A']   = str(nml['L0toL1A'])
            
            # Export to net CDF
            final.to_netcdf(file_out,mode = 'w',encoding={'time': \
                         {'units':'seconds since ' + group[0][-19:-15] + '-' +  group[0][-14:-12] + '-01'}})
    
            # Delete temporary files
            final.close()
            del final
            for tmpfiles in glob.glob("tmp*"):
                os.remove(tmpfiles)
    else:
        print('Sensor not supported')
        
        
#_______________________________________________________
def L1AtoL1B(nml):
    """
    Reads all the monthly raw nc data files, removes outliers, removes spikes , applies coordinate rotation
    If no paths are provided, this function reads all the files in the current working directory.
    
    Input
    ----------
    nml: python f90nml namelist 
        namelist containing all parameters
    
    Output
    ----------
    monthly nc files containing raw corrected data L1B
    
    Example
    ----------
    L1AtoL1B()
    
    Required packages
    ----------
    glob, os, itertools, pandas
    
    Required functions
    ----------
    
    
    """
    
    # np.warnings.filterwarnings('ignore')
    

    base_dir   = nml['global']['dir']
    L1Adir     = base_dir + 'L1A/'
    L1Bdir     = base_dir + 'L1B/'
    sensor     = nml['global']['sensor']
    WDvalid    = nml['L1AtoL1B']['WDvalid']
    frequency  = nml['L1AtoL1B']['frequency']
    downsample = nml['L1AtoL1B']['ldownsample']
    rolling    = nml['L1AtoL1B']['lrolling']
    average    = nml['L1AtoL1B']['laverage']
    shift_w_sample  = nml['L1AtoL1B']['shift_w_sample']
    dT		   = nml['L1AtoL1B']['dT']
    Fd		   = nml['L1AtoL1B']['Fd']
    maxnanp    = nml['L1AtoL1B']['maxnanp'] / 100
    Rotation	   = nml['L1AtoL1B']['Rotation']
    q          = nml['L1AtoL1B']['qMAD']
    loops      = nml['L1AtoL1B']['LoopsMAD']
    maxdiag = nml['L1AtoL1B']['maxdiag']
    maxdiagCSAT = nml['L1AtoL1B']['maxdiag']
    # maxdiagH2O = nml['L1AtoL1B']['maxdiagH2O']
    dT_PF		      = nml['L1AtoL1B']['dT_PF']
    
    LOC        = nml['global']['LOC']
    ID         = nml['global']['ID']


    
    if not os.path.exists(L1Bdir):
        os.makedirs(L1Bdir) 
        
    # Move to input data directory
    if not L1Adir:
        os.chdir(os.getcwd())  
    else:
        os.chdir(L1Adir)  
        
    # Time window for spike removal
    T = int(frequency * 600) # 10min
    
    # Number of maximum allowed consecutive interpolations
    N_int = int(frequency * 1) #10s
    
    # Max number of samples per time block
    N = int(pd.to_timedelta(dT).total_seconds() * frequency)
    
    
    if sensor == 'CSAT':
        # Process files sequentially
        for file in sorted(glob.glob('*' + sensor + '_' + "*nc")):
            print(file)
            ds = xr.open_dataset(file)
            
            print('Correcting raw data')
            
            # Consistency check
            ds['u'][(abs(ds['u']) > 50)]                          = np.nan
            ds['v'][(abs(ds['v']) > 50)]                          = np.nan
            ds['w'][(abs(ds['w']) > 10)]                          = np.nan
            ds['T_sonic'][(abs(ds['T_sonic'] - 273.15)   > 50)]   = np.nan
            ds['T_couple'][(abs(ds['T_couple'] - 273.15) > 50)]   = np.nan
            
            # Spike removal
            ds['u']         = despike(ds['u'],T=T,q=q,loops = loops)
            ds['v']         = despike(ds['v'],T=T,q=q,loops = loops)
            ds['w']         = despike(ds['w'],T=T,q=q,loops = loops)
            ds['T_sonic']   = despike(ds['T_sonic'],T=T,q=q,loops = loops)
            ds['T_couple']  = despike(ds['T_couple'],T=T,q=q,loops = loops)
            
            # Flag non-nans
            if 'diag' in list(ds.variables):
                flagU    = (np.isnan(ds['u'])) | (ds['diag'] > maxdiag)
                flagw    = (np.isnan(ds['w']))  | (ds['diag'] > maxdiag)
                flagTs   = (np.isnan(ds['T_sonic']))  | (ds['diag'] > maxdiag)
                flagTc   = np.isnan(ds['T_couple'])

            else:
                flagU    = np.isnan(ds['u'])
                flagw    = np.isnan(ds['w'])
                flagTs   = np.isnan(ds['T_sonic'])
                flagTc   = np.isnan(ds['T_couple'])
            
            # do the same if there is Licor data
            if 'H2O' in list(ds.variables):
                ds['H2O'][(abs(ds['H2O']) < 0)] = np.nan
                # ds['CO2'][(abs(ds['CO2']) < 0)] = np.nan
                if 'diag' in list(ds.variables): 
                    flagH2O = (np.isnan(ds['H2O'])) | (ds['diag'] > maxdiag)
                    # flagCO2 = (np.isnan(ds['CO2'])) | (ds['diag'] > maxdiag)
                else:
                    flagH2O = np.isnan(ds['H2O'])
                    # flagCO2 = np.isnan(ds['CO2'])
    #                ds['H2O']                       = despike(ds['H2O'],T=T,q=q,loops = loops)
    #                ds['CO2']                       = despike(ds['CO2'],T=T,q=q,loops = loops)

            # Interpolate missing data up to 10s
            ds              = ds.interpolate_na(dim='time',limit = N_int)
            
            # Flag inteprolated data
            ds['flagU']     = flagU
            ds['flagw']     = flagw
            ds['flagTs']    = flagTs
            ds['flagTc']    = flagTc
            if 'H2O' in list(ds.variables):
                ds['flagH2O']    = flagH2O
                # ds['flagCO2']    = flagCO2
                
            
            # Convert to dataframe
            df = ds.to_dataframe()

            # Calculate time averages
            grouped = df.groupby(pd.Grouper(freq=dT,label='right'), group_keys=False)
            df_av   = grouped.mean()
            
            # Mean wind vector
            df_av['WD']      =  grouped.apply(lambda x: meanWindDir((x['u']**2+x['v']**2)**(0.5),x['WD']))
            
            # Count number of missing data in runs
            df_av['pnanU']   =  grouped.apply(lambda x: np.sum(x['flagU'])/N)
            df_av['pnanw']   =  grouped.apply(lambda x: np.sum(x['flagw'])/N)
            
            # Flag runs with distorted measurements for Planar Fit rotation
            if WDvalid[0]>WDvalid[1]:
                df_av['flagPF'] = ~((df_av.WD>=WDvalid[0]) | (df_av.WD<=WDvalid[1]))
            else:
                df_av['flagPF'] = ~((df_av.WD>=WDvalid[0]) & (df_av.WD<=WDvalid[1]))  
                
            # Flag runs with not enough data 
            df_av['flagPF'] = (df_av['flagPF'] & ((df_av['pnanU']>maxnanp) | (df_av['pnanw']>maxnanp)))
                
            print(Rotation + ' rotation')
            
            if Rotation == 'PF':
                
                print('Planar Fit rotation')
                
                # Calculate planar fit correction matrix using all averages
                A_pf, a, b = PlanarFit_matrix(df_av)
                
                # Apply planar fit + yaw rotation matrix per time block
                df_PF    = grouped.apply(lambda x: PlanarFit_correction(x,A_pf))
                
                # Store output in dataset
                ds['u']  = df_PF['u']
                ds['v']  = df_PF['v']
                ds['w']  = df_PF['w'] - np.nanmean(df_PF['w'])
                
                ds['roll']   = a
                ds['pitch']  = b
                ds['yaw']    = df_PF['yaw']
                
            elif Rotation == 'DR':

                # Apply planar fit + yaw rotation matrix per time block
                df_DR    = grouped.apply(lambda x: Double_rotation(x))
                
                # Store output in dataset
                ds['u']  = df_DR['u']
                ds['v']  = df_DR['v']
                ds['w']  = df_DR['w']
                
                ds['yaw']   = df_DR['yaw']
                ds['pitch'] = df_DR['pitch']
                
    
            else: print('No rotation applied')
                
            # Store old attribute
            att1                                        = ds.attrs['IceEddie_namelist_L0toL1A']

            # Add new attributes
            ds                                      = utils.Add_dataset_attributes(ds,'CSAT_EC_L1B.JSON')
            ds.attrs['location']                    = LOC + '_' + ID
            ds.attrs['file_creation_date_time']     = str(datetime.datetime.now())
            ds.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
            ds.attrs['IceEddie_namelist_L0toL1A']   = att1
            ds.attrs['IceEddie_namelist_L1AtoL1B']  = str(nml['L1AtoL1B'])
            ds.attrs['Rotation_period'] = dT
 

            file_out = L1Bdir + file.replace('L1A','L1B')
                
            if os.path.isfile(file_out):
                os.remove(file_out)
                
            ds.to_netcdf(file_out,mode = 'w',encoding={'time': \
                         {'units':'seconds since  ' + str(min(df.index).year) + '-' +  str(min(df.index).month) + '-01'}}) 

      
    elif sensor == 'iWS':
    
        for file in sorted(glob.glob('*' + sensor + '*L1A*' + "*nc")):
            print(file)
            ds = xr.open_dataset(file)
            
            print('Correcting raw data')
            
            #  Consistency check
            ds['u'][(abs(ds['u']) > 30)]                         = np.nan
            ds['v'][(abs(ds['v']) > 30)]                         = np.nan
            ds['w'][(abs(ds['w']) > 7)]                          = np.nan
            ds['T1'][(abs(ds['T1'] - 273.15 ) > 50)]             = np.nan
            ds['T2'][(abs(ds['T2'] - 273.15 ) > 50)]             = np.nan
            
            # Spike removal
            ds['u']         = despike(ds['u'],T=T,q=q,loops = loops)
            ds['v']         = despike(ds['v'],T=T,q=q,loops = loops)
            ds['w']         = despike(ds['w'],T=T,q=q,loops = loops)
            ds['T1']        = despike(ds['T1'],T=T,q=q,loops = loops)
            ds['T2']        = despike(ds['T2'],T=T,q=q,loops = loops)

            # Flag non-nans
            flagU    = np.isnan(ds['u'])
            flagw    = np.isnan(ds['w'])
            flagT1   = np.isnan(ds['T1'])
            flagT2   = np.isnan(ds['T2'])
            
            # Interpolate missing data
            ds       = ds.interpolate_na(dim='time',limit = N_int)
            
            # Flag interpolated data
            ds['flagU']     = flagU
            ds['flagw']     = flagw
            ds['flagT1']    = flagT1
            ds['flagT2']    = flagT2
                        
            # Convert to dataframe
            df = ds.to_dataframe()
            
            if downsample:
                print('Downsampling')
                df = df.resample(Fd).first()
                
            if rolling:
                print('Rolling average')
                df['u'] = df['u'].rolling(Fd).mean()  
                df['v'] = df['v'].rolling(Fd).mean() 
                
            # Calculate time averages
            grouped = df.groupby(pd.Grouper(freq=dT,label='right'), group_keys=False)
            df_av   = grouped.mean()
            
            # Mean wind vector
            df_av['WD']      =  grouped.apply(lambda x: meanWindDir((x['u']**2+x['v']**2)**(0.5),x['WD']))
            
            # Count number of missing data in runs
            df_av['pnanU']   =  grouped.apply(lambda x: np.sum(x['flagU'])/N)
            df_av['pnanw']   =  grouped.apply(lambda x: np.sum(x['flagw'])/N)
            
            # Flag runs with distorted measurements for Planar Fit rotation
            if WDvalid[0]>WDvalid[1]:
                df_av['flagPF'] = ~((df_av.WD>=WDvalid[0]) | (df_av.WD<=WDvalid[1]))
            else:
                df_av['flagPF'] = ~((df_av.WD>=WDvalid[0]) & (df_av.WD<=WDvalid[1]))  
                
            # Flag runs with not enough data 
            df_av['flagPF'] = (df_av['flagPF'] & ((df_av['pnanU']>maxnanp) | (df_av['pnanw']>maxnanp)))
                
            print(Rotation + ' rotation')
            
            if Rotation == 'PF':
                
                print('Planar Fit rotation')
                # Calculate planar fit correction matrix using all averages
                A_pf, a, b = PlanarFit_matrix(df_av)
                
                # Apply planar fit + yaw rotation matrix per time block
                df_PF    = grouped.apply(lambda x: PlanarFit_correction(x,A_pf))
                
                # Store output in dataset
                ds['u']  = df_PF['u']
                ds['v']  = df_PF['v']
                ds['w']  = df_PF['w'] - np.nanmean(df_PF['w'])
                
                ds['roll']   = a
                ds['pitch']  = b
                ds['yaw']    = df_PF['yaw']
                
            elif Rotation == 'DR':

                # Apply planar fit + yaw rotation matrix per time block
                df_DR    = grouped.apply(lambda x: Double_rotation(x))
                
                # Store output in dataset
                ds['u']  = df_DR['u']
                ds['v']  = df_DR['v']
                ds['w']  = df_DR['w']
                
                ds['yaw']   = df_DR['yaw']
                ds['pitch'] = df_DR['pitch']
                
                
            else: print('No rotation applied')
            
            # Store old attribute
            att1                                        = ds.attrs['IceEddie_namelist_L0toL1A']

            # Add new attributes
            ds                                      = utils.Add_dataset_attributes(ds,'iWS_EC_L1B.JSON')
            ds.attrs['location']                    = LOC + '_' + ID
            ds.attrs['file_creation_date_time']     = str(datetime.datetime.now())
            ds.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
            ds.attrs['IceEddie_namelist_L0toL1A']   = att1
            ds.attrs['IceEddie_namelist_L1AtoL1B']  = str(nml['L1AtoL1B'])
            ds.attrs['Rotation_period'] = dT
            
 
            # Export no net CDF    
            if downsample:
                file_out = L1Bdir + file.replace('L1A','L1B_' + Fd)
            elif rolling:
                file_out = L1Bdir + file.replace('L1A','L1B_av' + Fd)
            else:
                file_out = L1Bdir + file.replace('L1A','L1B')
                
            if os.path.isfile(file_out):
                os.remove(file_out)
                
            ds.to_netcdf(file_out,mode = 'w',encoding={'time': \
                         {'units':'seconds since  ' + str(min(df.index).year) + '-' +  str(min(df.index).month) + '-01'}}) 
    
    
    elif sensor == 'Young':
    
        for file in sorted(glob.glob('*' + sensor + '*L1A*' + "*nc")):
            print(file)
            ds = xr.open_dataset(file)
            
            print('Correcting raw data')
            
            #  Consistency check
            ds['u'][(abs(ds['u']) > 30)]                         = np.nan
            ds['v'][(abs(ds['v']) > 30)]                         = np.nan
            ds['w'][(abs(ds['w']) > 7)]                          = np.nan
            ds['T1'][(abs(ds['T1'] - 273.15 ) > 50)]             = np.nan
            
            # Spike removal
            ds['u']         = despike(ds['u'],T=T,q=q,loops = loops)
            ds['v']         = despike(ds['v'],T=T,q=q,loops = loops)
            ds['w']         = despike(ds['w'],T=T,q=q,loops = loops)
            ds['T1']        = despike(ds['T1'],T=T,q=q,loops = loops)
            
            if shift_w_sample !=0:
                ds['w'].values = np.roll(ds['w'], shift_w_sample)
                if shift_w_sample > 0:
                    ds['w'].values[0:shift_w_sample] = np.nan
                elif shift_w_sample < 0:
                    ds['w'].values[-1:-shift_w_sample] = np.nan

            # Flag non-nans
            flagU    = np.isnan(ds['u'])
            flagw    = np.isnan(ds['w'])
            flagT1   = np.isnan(ds['T1'])
            
            # Interpolate missing data
            ds       = ds.interpolate_na(dim='time',limit = N_int)
            
            # Flag inteprolated data
            ds['flagU']     = flagU
            ds['flagw']     = flagw
            ds['flagT1']    = flagT1
            
            # Convert to dataframe
            df = ds.to_dataframe()
            

                
            if rolling:
                print('Rolling average')
                df['u'] = df['u'].rolling(Fd).mean()  
                df['v'] = df['v'].rolling(Fd).mean() 
                
            if downsample:
                print('Downsampling')
                df = df.resample(Fd).first()

            if average:
                print('Averging')
                df = df.resample(Fd).mean()
                
            # Calculate time averages
            grouped = df.groupby(pd.Grouper(freq=dT,label='right'), group_keys=False)
            df_av   = grouped.mean()
            
            # Mean wind vector
            df_av['WD']      =  grouped.apply(lambda x: meanWindDir((x['u']**2+x['v']**2)**(0.5),x['WD']))
            
            # Count number of missing data in runs
            df_av['pnanU']   =  grouped.apply(lambda x: np.sum(x['flagU'])/N)
            df_av['pnanw']   =  grouped.apply(lambda x: np.sum(x['flagw'])/N)
            
            # Flag runs with distorted measurements for Planar Fit rotation
            if WDvalid[0]>WDvalid[1]:
                df_av['flagPF'] = ~((df_av.WD>=WDvalid[0]) | (df_av.WD<=WDvalid[1]))
            else:
                df_av['flagPF'] = ~((df_av.WD>=WDvalid[0]) & (df_av.WD<=WDvalid[1]))  
                
            # Flag runs with not enough data 
            df_av['flagPF'] = (df_av['flagPF'] & ((df_av['pnanU']>maxnanp) | (df_av['pnanw']>maxnanp)))
                
            print(Rotation + ' rotation')
            
            if Rotation == 'PF':
                
                print('Planar Fit rotation')
                # Calculate planar fit correction matrix using all averages
                A_pf, a, b = PlanarFit_matrix(df_av)
                
                # Apply planar fit + yaw rotation matrix per time block
                df_PF    = grouped.apply(lambda x: PlanarFit_correction(x,A_pf))
                
                # Store output in dataset
                df['u']  = df_PF['u']
                df['v']  = df_PF['v']
                df['w']  = df_PF['w'] - np.nanmean(df_PF['w'])
                
                df['roll']   = a
                df['pitch']  = b
                df['yaw']    = df_PF['yaw']
                
            elif Rotation == 'DR':

                # Apply planar fit + yaw rotation matrix per time block
                df_DR    = grouped.apply(lambda x: Double_rotation(x))
                
                # Store output in dataset
                df['u']  = df_DR['u']
                df['v']  = df_DR['v']
                df['w']  = df_DR['w']
                
                df['yaw']   = df_DR['yaw']
                df['pitch'] = df_DR['pitch']
                
                
                
            else: print('No rotation applied')
            
            # Convert to Dataset
            x = df.to_xarray()
            
            # Store old attribute
            att1                                        = ds.attrs['IceEddie_namelist_L0toL1A']

            # Add new attributes
            x                                      = utils.Add_dataset_attributes(x,'YOUNG_EC_L1B.JSON')
            x.attrs['location']                    = LOC + '_' + ID
            x.attrs['file_creation_date_time']     = str(datetime.datetime.now())
            x.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
            x.attrs['IceEddie_namelist_L0toL1A']   = att1
            x.attrs['IceEddie_namelist_L1AtoL1B']  = str(nml['L1AtoL1B'])

            # Export no net CDF    
            if downsample:
                file_out = L1Bdir + file.replace('L1A','L1B_' + Fd)
            elif rolling:
                file_out = L1Bdir + file.replace('L1A','L1B_av' + Fd)
            else:
                file_out = L1Bdir + file.replace('L1A','L1B')
                
            if os.path.isfile(file_out):
                os.remove(file_out)
                
            x.to_netcdf(file_out,mode = 'w',encoding={'time': \
                         {'units':'seconds since  ' + str(min(df.index).year) + '-' +  str(min(df.index).month) + '-01'}}) 
            
    elif sensor == 'CSAT_Young':
    
        for file in sorted(glob.glob('*' + sensor + '*L1A*' + "*nc")):
            print(file)
            ds = xr.open_dataset(file)
            
            print('Correcting raw data')
            
            #  Consistency check
            ds['uy'][(abs(ds['uy']) > 30)]                         = np.nan
            ds['vy'][(abs(ds['vy']) > 30)]                         = np.nan
            ds['wg'][(abs(ds['wg']) > 7)]                          = np.nan
            ds['T_couple2'][(abs(ds['T_couple2'] - 273.15) > 80)]   = np.nan

            ds['u'][(abs(ds['u']) > 50)]                          = np.nan
            ds['v'][(abs(ds['v']) > 50)]                          = np.nan
            ds['w'][(abs(ds['w']) > 10)]                          = np.nan
            ds['T_sonic'][(abs(ds['T_sonic'] - 273.15)   > 50)]   = np.nan
            ds['T_couple'][(abs(ds['T_couple'] - 273.15) > 80)]   = np.nan

            if 'diagCSAT' in list(ds.variables):
                ds['u'][ds['diagCSAT'] > maxdiagCSAT]             = np.nan 
                ds['v'][ds['diagCSAT'] > maxdiagCSAT]             = np.nan 
                ds['w'][ds['diagCSAT'] > maxdiagCSAT]             = np.nan 
                ds['T_sonic'][ds['diagCSAT'] > maxdiagCSAT]       = np.nan 
                if 'q' in list(ds.variables):
                    ds['q'][(ds['q'] < 0)] = np.nan
                    ds['q'][(ds['q'] > 50)] = np.nan
                    ds['q'][ds['diagH2O'] != 0] = np.nan
                    ds['CO2'][(ds['CO2']) < 0] = np.nan
                    ds['CO2'][ds['diagH2O'] != 0] = np.nan        

            # Spike removal
            ds['uy']         = despike(ds['uy'],T=T,q=q,loops = loops)
            ds['vy']         = despike(ds['vy'],T=T,q=q,loops = loops)
            ds['wg']         = despike(ds['wg'],T=T,q=q,loops = loops)
            ds['T_couple2']  = despike(ds['T_couple2'],T=T,q=q,loops = loops)

            ds['u']         = despike(ds['u'],T=T,q=q,loops = loops)
            ds['v']         = despike(ds['v'],T=T,q=q,loops = loops)
            ds['w']         = despike(ds['w'],T=T,q=q,loops = loops)
            ds['T_sonic']   = despike(ds['T_sonic'],T=T,q=q,loops = loops)
            ds['T_couple']  = despike(ds['T_couple'],T=T,q=q,loops = loops)

            if 'q' in list(ds.variables):
                ds['q'] = despike(ds['q'],T=T,q=q,loops = loops)

            if shift_w_sample !=0:
                ds['wg'].values = np.roll(ds['wg'], shift_w_sample)
                if shift_w_sample > 0:
                    ds['wg'].values[0:shift_w_sample] = np.nan
                elif shift_w_sample < 0:
                    ds['wg'].values[-1:-shift_w_sample] = np.nan

            # Flag non-nans
            flagUy    = np.isnan(ds['uy'])
            flagwg    = np.isnan(ds['wg'])
            
            if 'diagCSAT' in list(ds.variables):
                flagU    = (np.isnan(ds['u'])) | (ds['diagCSAT'] > maxdiagCSAT)
                flagw    = (np.isnan(ds['w']))  | (ds['diagCSAT'] > maxdiagCSAT)
                flagTs   = (np.isnan(ds['T_sonic']))  | (ds['diagCSAT'] > maxdiagCSAT)
                flagTc   = np.isnan(ds['T_couple'])
            else:
                flagU    = np.isnan(ds['u'])
                flagw    = np.isnan(ds['w'])
                flagTs   = np.isnan(ds['T_sonic'])
                flagTc   = np.isnan(ds['T_couple'])
            if 'q' in list(ds.variables):
                flagH2O = np.isnan(ds['q'])
                    
            # Interpolate missing data
            ds       = ds.interpolate_na(dim='time',limit = N_int)
            
            # Flag inteprolated data
            ds['flagUy']     = flagUy
            ds['flagwg']     = flagwg
            
            ds['flagU']     = flagU
            ds['flagw']     = flagw
            ds['flagTs']    = flagTs
            ds['flagTc']    = flagTc
            
            if 'q' in list(ds.variables):
                ds['flagH2O']    = flagH2O  

            # Convert to dataframe
            df = ds.to_dataframe()
            
            # Calculate time averages
            grouped = df.groupby(pd.Grouper(freq=dT,label='right'), group_keys=False)
            df_av   = grouped.mean()
            
            # Mean wind vector
            df_av['WDy']     =  grouped.apply(lambda x: meanWindDir((x['uy']**2+x['vy']**2)**(0.5),x['WDy']))
            df_av['WD']      =  grouped.apply(lambda x: meanWindDir((x['u']**2+x['v']**2)**(0.5),x['WD']))
            
            # Count number of missing data in runs
            df_av['pnanUy']   =  grouped.apply(lambda x: np.sum(x['flagUy'])/N)
            df_av['pnanwg']   =  grouped.apply(lambda x: np.sum(x['flagwg'])/N)
            df_av['pnanU']   =  grouped.apply(lambda x: np.sum(x['flagU'])/N)
            df_av['pnanw']   =  grouped.apply(lambda x: np.sum(x['flagw'])/N)
            
            # Flag runs with distorted measurements for Planar Fit rotation
            if WDvalid[0]>WDvalid[1]:
                df_av['flagPF'] = ~((df_av.WD>=WDvalid[0]) | (df_av.WD<=WDvalid[1]))
            else:
                df_av['flagPF'] = ~((df_av.WD>=WDvalid[0]) & (df_av.WD<=WDvalid[1]))  
                
            # Flag runs with not enough data 
            df_av['flagPF'] = (df_av['flagPF'] & ((df_av['pnanU']>maxnanp) | (df_av['pnanw']>maxnanp)))
                
            print(Rotation + ' rotation')
            
            if Rotation == 'PF':
                
                print('Planar Fit rotation not implemented')
                
            elif Rotation == 'DR':

                # Apply planar fit + yaw rotation matrix per time block
                df_DR    = grouped.apply(lambda x: Double_rotation(x))
                df_DRy   = grouped.apply(lambda x: Double_rotation(x,1))
                
                # Store output in dataset
                df['u']  = df_DR['u']
                df['v']  = df_DR['v']
                df['w']  = df_DR['w']
                
                df['yaw']   = df_DR['yaw']
                df['pitch'] = df_DR['pitch']
            
                df['uy']  = df_DRy['uy']
                df['vy']  = df_DRy['vy']
                df['wg']  = df_DRy['wg']
                
                df['yawy']   = df_DRy['yaw']
                df['pitchy'] = df_DRy['pitch']
                
            else: print('No rotation applied')
            
            # Convert to Dataset
            x = df.to_xarray()
            
            # Store old attribute
            att1                                        = ds.attrs['IceEddie_namelist_L0toL1A']

            # Add new attributes
            x                                      = utils.Add_dataset_attributes(x,'YOUNG_EC_L1B.JSON')
            x.attrs['location']                    = LOC + '_' + ID
            x.attrs['file_creation_date_time']     = str(datetime.datetime.now())
            x.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
            x.attrs['IceEddie_namelist_L0toL1A']   = att1
            x.attrs['IceEddie_namelist_L1AtoL1B']  = str(nml['L1AtoL1B'])

            # Export no net CDF    
            if downsample:
                file_out = L1Bdir + file.replace('L1A','L1B_' + Fd)
            elif rolling:
                file_out = L1Bdir + file.replace('L1A','L1B_av' + Fd)
            else:
                file_out = L1Bdir + file.replace('L1A','L1B')
                
            if os.path.isfile(file_out):
                os.remove(file_out)

            if 'CO2' in list(ds.variables):
                x = x.drop({'CO2'})
            if 'diagCSAT' in list(ds.variables):
                x = x.drop({'diagCSAT'})
            if 'diagH2O' in list(ds.variables):
                x = x.drop({'diagH2O'})
            if 'rho_w' in list(ds.variables):
                x = x.drop({'rho_w'})
            # enc = {}

            # for k in x.data_vars:
            #     if x[k].ndim < 2:
            #         continue

            #     enc[k] = {
            #         "zlib": True,
            #         "complevel": 5,
            #         "fletcher32": True,
            #         "chunksizes": tuple(map(lambda x: x//2, ds[k].shape))
            #     }

            # enc['time'] = {'units':'seconds since  ' + str(min(df.index).year) + '-' +  str(min(df.index).month) + '-01'}
            print(x)
            print(x.nbytes*9.313225746E-10)

            x.to_netcdf(file_out,mode = 'w', format="NETCDF4", engine="h5netcdf", encoding={'time': {'units':'seconds since  ' + str(min(df.index).year) + '-' +  str(min(df.index).month) + '-01'}})
            # x.to_netcdf(file_out,mode = 'w' ,encoding={'time': \
                                                    #    {'units':'seconds since  ' + str(min(df.index).year) + '-' +  str(min(df.index).month) + '-01'}})

            x.close()
            del x 
#_______________________________________________________
def L0toL2(nml):
    """
    Reads all the files containing calculated 5-min data, applies rotation, averages to 30-min and saves to monthly file
    If no paths are provided, this function reads all the files in the current working directory.
    
    Input
    ----------
    nml: python f90nml namelist 
        namelist containing all parameters    
    Output
    ----------
    monthly nc files containing uncorrected averaged data L2
    
    Example
    ----------
    L1BtoL2('iWS')
    
    Required packages
    ----------
    glob, os, itertools, pandas
    
    Required functions
    ----------
    
    
    """
    
    
    base_dir        = nml['global']['dir']
    L0dir           = base_dir + nml['global']['L0dir']
    L2dir         = base_dir + 'L2/'
    sensor          = nml['global']['sensor']
    L1B_30min_dir = nml['global']['l1b_30min_dir']
    
    ID				= nml['global']['ID']    
    LOC				= nml['global']['LOC']
    version			= nml['global']['version']
    input_type		= nml['global']['input_type']
    signature		= nml['global']['signature']
    institute		= nml['global']['institute']
    LOC				= nml['global']['LOC']
    sensor        = nml['global']['sensor']
    base_dir      = nml['global']['dir']
    L1Bdir        = base_dir + 'L1B/'
    
    toffset		= nml['L0toL1A']['toffset']
    toffset_method		= nml['L0toL1A']['toffset_method']
    ifiletype		= nml['L0toL1A']['ifiletype']
    
    Rotation	   = nml['L1AtoL1B']['Rotation']
    WDvalid    = nml['L1AtoL1B']['WDvalid']
    dT_PF		      = nml['L1AtoL1B']['dT_PF']

    dT		      = nml['L1BtoL2']['dT']
    
    store_spectra = nml['L1BtoL2']['lwritespectra']
    lfixedheight  = nml['L1BtoL2']['lfixedheight']
    lheightinfile = nml['L1BtoL2']['lheightinfile']
    WDvalid       = nml['L1BtoL2']['wdvalid']
    zp            = nml['L1BtoL2']['z']
    EC_hgt_offset = nml['L1BtoL2']['EC_hgt_offset']    
    d           = nml['L1BtoL2']['d']
    H         = nml['L1BtoL2']['H']
    lcorrH         = nml['L1BtoL2']['lcorrH']

    
    if not os.path.exists(L2dir):
        os.makedirs(L2dir) 
        
    # Move to input data directory
    if not L0dir:
        os.chdir(os.getcwd())  
    else:
        os.chdir(L0dir)  
        
    
    if ifiletype      == 'TOA':
        files =  sorted(glob.glob("TOA5*"), key=lambda x: x[1:])     
        fid = 0
        for file in files:
            if fid == 0:
                print(file)
                if sensor == 'CSAT':
                    df = read_cal_TOA_CSAT(os.getcwd() + "/" + file,nml)
                elif sensor == 'Young':
                    df = read_cal_TOA_Young(os.getcwd() + "/" + file,nml)
                fid = fid + 1
            else:
                print(file)
                if sensor == 'CSAT':
                    tmp = read_cal_TOA_CSAT(os.getcwd() + "/" + file,nml)
                elif sensor == 'Young':
                    tmp = read_cal_TOA_Young(os.getcwd() + "/" + file,nml)
                # Load raw file in a dataframe
                df = pd.concat([df,tmp])
                fid = fid + 1     
    elif ifiletype      == 'rawtxt':
        files =  sorted(glob.glob("*txt"), key=lambda x: x[1:])     
        fid = 0
        for file in files:
            if fid == 0:
                print(file)
                df = read_cal_txt(os.getcwd() + "/" + file,nml)
                fid = fid + 1
            else:
                print(file)
                tmp = read_cal_txt(os.getcwd() + "/" + file,nml)
                # Load raw file in a dataframe
                df = pd.concat([df,tmp])
                fid = fid + 1    
    elif ifiletype      == 'dat_PKM':
        files =  sorted(glob.glob("*dat"), key=lambda x: x[1:])     
        fid = 0
        for file in files:
            if fid == 0:
                print(file)
                df = read_cal_dat_PKM(os.getcwd() + "/" + file,nml)
                fid = fid + 1
            else:
                print(file)
                tmp = read_cal_dat_PKM(os.getcwd() + "/" + file,nml)
                # Load raw file in a dataframe
                df = pd.concat([df,tmp])
                fid = fid + 1  
    elif ifiletype      == 'headertxt':
        files =  sorted(glob.glob("*txt"), key=lambda x: x[1:])     
        fid = 0
        for file in files:
            if fid == 0:
                print(file)
                df = read_cal_txt_v02(os.getcwd() + "/" + file,nml)
                fid = fid + 1
            else:
                print(file)
                tmp = read_cal_txt_v02(os.getcwd() + "/" + file,nml)
                # Load raw file in a dataframe
                df = pd.concat([df,tmp])
                fid = fid + 1    
            
            
            

    # Flag runs with distorted measurements for Planar Fit rotation
    if WDvalid[0]>WDvalid[1]:
        df['flagPF'] = ~((df.WD>=WDvalid[0]) | (df.WD<=WDvalid[1]))
    else:
        df['flagPF'] = ~((df.WD>=WDvalid[0]) & (df.WD<=WDvalid[1]))  
        
            
    if Rotation == 'PF':           
        print('Planar fit rotation')
        df['PFroll'] = df['u']*np.nan
        df['PFpitch'] = df['u']*np.nan
        df['PFyaw'] = df['u']*np.nan
        grouped = df.groupby(pd.Grouper(freq=dT_PF,label='right'), group_keys=False)
        df_PF = grouped.apply(lambda x: PlanarFit_cal(x))
        df['u']  = df_PF['u']
        df['v']  = df_PF['v']
        df['w']  = df_PF['w']
        df['uu']  = df_PF['uu']
        df['uv']  = df_PF['uv']
        df['uw']  = df_PF['uw']
        df['vv']  = df_PF['vv']
        df['vw']  = df_PF['vw']
        df['ww']  = df_PF['ww']
        df['uTs']  = df_PF['uTs']
        df['vTs']  = df_PF['vTs']
        df['wTs']  = df_PF['wTs']
        df['uTc']  = df_PF['uTc']
        df['vTc']  = df_PF['vTc']
        df['wTc']  = df_PF['wTc']
        df['PFroll']  = df_PF['PFroll']
        df['PFpitch']  = df_PF['PFpitch']
        df['PFyaw']  = df_PF['PFyaw']
        
        if 'wq' in list(df.columns):
            df['uq']  = df_PF['uq']
            df['vq']  = df_PF['vq']
            df['wq']  = df_PF['wq']   
        if 'wCO2' in list(df.columns):
            df['uCO2']  = df_PF['uCO2']
            df['vCO2']  = df_PF['vCO2']
            df['wCO2']  = df_PF['wCO2'] 
            
    elif Rotation == 'DR':     
        print('Double rotation')
        df['DRyaw']   = df['u']*np.nan
        df['DRpitch'] = df['u']*np.nan
        df = Double_rotation_cal(df)
    else:
        print('No rotation')

        # Compute friction velocity
    df['ustar'] = (df['uw']**2 + df['vw']**2)**(1/4)
    
    # Compute temperature scale
    df['tstar'] = -np.divide(df['wTs'],df['ustar'])
    
    # Compute Obukhov length
    df['obh']   = -(df['ustar']**3)*df['T_sonic']/(9.81*0.4*df['wTs'])
    
    if 'wq' in list(df.columns):
        df['qstar'] = -np.divide(df['wq'],df['ustar'])
                
    ds_out = df.to_xarray()
    ds_out = ds_out.sortby('time')

    # Correct time to UTC
    if toffset_method == 'min':
        ds_out['time'] = ds_out.time.values + np.timedelta64(toffset,'m')
    elif toffset_method == 'samples':
        ds_out['time'] = np.roll(ds_out.time.values,-toffset)
    ds_out = ds_out.sortby('time')
                
    # if dT != '30min':
    #     ds_out = ds_out.resample(time="30min").mean()
        
    # Go to folder where 30-min height data is stored
    if not lheightinfile:
        if not lfixedheight:
            os.chdir(L1B_30min_dir) 
            files_30min = glob.glob("*nc") 
            print(files_30min)
            ds_30min = xr.open_mfdataset(files_30min)
            # Add height to dataset
            ds_out['zm']   = ds_30min['zm']
            ds_out['d']    = ds_30min['d']
            ds_out['H']    = ds_30min['H']
            if lcorrH:
                # ds_out['z']    = ds_30min['zm']+ds_30min['H']-ds_30min['snowheight']-ds_30min['d']+EC_hgt_offset
                ds_out['z']    = -ds_30min['snowheight']+zp
            else:
                ds_out['z']    = ds_30min['zm']+ds_30min['H']-ds_30min['d']+EC_hgt_offset
            
            # Go back to L1B dir
            os.chdir(L0dir)  
        else:
            z      = xr.DataArray(np.full(len(ds_out.time),float(zp + H - d)),dims=['time'])
            ds_out = ds_out.assign({'z': z})
        
    if store_spectra == True:
        # Calculate Kaimal spectra
        ds_out = Kaimal(ds_out,nml,level = 'L2')

        # Calculate transfer functions
        ds_out = transfer_functions(ds_out,nml,level = 'L2')
            
    # Variable attributes
    if sensor == 'CSAT':                          
        ds_out = utils.Add_dataset_attributes(ds_out,'CSAT_EC_L2.JSON')
        ds_out.attrs['location']                    = LOC + '_' + ID
        ds_out.attrs['file_creation_date_time']     = str(datetime.datetime.now())
        ds_out.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
        ds_out.attrs['IceEddie_namelist_L0toL1A']   = str(nml['L0toL1A'])
        ds_out.attrs['IceEddie_namelist_L1AtoL1B']  = '-'
        ds_out.attrs['IceEddie_namelist_L1BtoL2']   = str(nml['L1BtoL2'])
        
    elif sensor == 'Young':
        ds_out                                      = utils.Add_dataset_attributes(ds_out,'YOUNG_EC_L2.JSON')
        ds_out.attrs['location']                    = LOC + '_' + ID
        ds_out.attrs['file_creation_date_time']     = str(datetime.datetime.now())
        ds_out.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
        ds_out.attrs['IceEddie_namelist_L0toL1A']   = str(nml['L0toL1A'])
        ds_out.attrs['IceEddie_namelist_L1AtoL1B']   = '-'
        ds_out.attrs['IceEddie_namelist_L1BtoL2']   = str(nml['L1BtoL2'])




    for monthid,out_monthly in ds_out.resample(time='1M'):
        file_out = L2dir + LOC + '_' + ID + '_' + sensor + '_' + input_type + '_' + "L2_" + version + '_' + str(pd.to_datetime((monthid)).year) + '{:02d}'.format(pd.to_datetime((monthid)).month)  + ".nc"
        if np.all(np.isnan(out_monthly.u)): continue
        print(list(out_monthly.keys()))
        print(file_out)
        print(out_monthly)
        out_monthly.to_netcdf(file_out, mode='w',encoding={'time': {'units':'minutes since ' + str(pd.to_datetime((monthid)).year) + \
                                                           '-' +  str(pd.to_datetime((monthid)).month) + '-01'}})    

            
    
#_______________________________________________________
def L1BtoL2(nml):
    """
    Reads all the corrected raw net CDF data files, applies detrending, applies block averaging, 
    calculates covariances and the raw cospectra.
    If no paths are provided, this function reads all the files in the current working directory.
    
    Input
    ----------
    nml: python f90nml namelist 
        namelist containing all parameters
    
    Output
    ----------
    monthly nc files containing uncorrected averaged data L2
    
    Example
    ----------
    L1BtoL2('iWS')
    
    Required packages
    ----------
    glob, os, itertools, pandas
    
    Required functions
    ----------
    
    
    """
    # np.warnings.filterwarnings('ignore')
    

    base_dir      = nml['global']['dir']
    L1Bdir        = base_dir + 'L1B/'
    L2dir         = base_dir + 'L2/'
    L1B_30min_dir = nml['global']['l1b_30min_dir']
    sensor        = nml['global']['sensor']
    LOC           = nml['global']['LOC']
    ID            = nml['global']['ID']
    
    frequency     = nml['L1AtoL1B']['frequency']
    maxnanp       = nml['L1AtoL1B']['maxnanp'] / 100
    
    inputdata     = nml['L1BtoL2']['inputdata']
    dT		      = nml['L1BtoL2']['dt']
    store_spectra = nml['L1BtoL2']['lwritespectra']
    lfixedheight  = nml['L1BtoL2']['lfixedheight']
    WDvalid       = nml['L1BtoL2']['wdvalid']
    zp            = nml['L1BtoL2']['z']# - nml['L1BtoL2']['d']
    ignore_samples_rhs   = nml['L1BtoL2']['ignore_samples_rhs']
    ldetrend       = nml['L1BtoL2']['ldetrend']
    EC_hgt_offset = nml['L1BtoL2']['EC_hgt_offset']    
    Tsensor        = nml['L1BtoL2']['Tsensor']
    d           = nml['L1BtoL2']['d']
    H         = nml['L1BtoL2']['H']
    lcorrH  = nml['L1BtoL2']['lcorrH']
    zp            = nml['L1BtoL2']['z']    
    zpy           = nml['L1BtoL2']['zy']    
    
    tauT          = nml['L2toL3']['tauT']
    Aw            = nml['L2toL3']['Aw']
    Lu            = nml['L2toL3']['Lu']

    


                
    if not os.path.exists(L2dir):
       os.makedirs(L2dir) 
        
        
    if not L1Bdir:
        os.chdir(os.getcwd())  
    else:
        os.chdir(L1Bdir)  
        
    if sensor == 'CSAT':

        # Process files sequentially
        for file in sorted(glob.glob('*' + sensor + "*" + inputdata + "*nc")):
            print(file)
            # Open net CDF
            ds = xr.open_dataset(file)

            # Convert to dataframe
            df = ds.to_dataframe()
            
            # Find sampling frequency of input data
            frequency = 1 / (df.index[1]-df.index[0]).total_seconds()
            
            # Max number of samples per time block
            N = int(pd.to_timedelta(dT).total_seconds() * frequency)
                  
            # Group original data in blocks
            grouped = df.groupby(pd.Grouper(freq=dT,label='right'))
            
            # Compute mean per block
            df_av   = grouped.mean()
 
            # Change the way mean wind direction is calculated
            df_av['WD']    =  grouped.apply(lambda x: meanWindDir((x['u']**2+x['v']**2)**(0.5),x['WD']))

            # Count number of missing data in runs
            df_av['pnanU']   =  grouped.apply(lambda x: np.sum(x['flagU'])/N)
            df_av['pnanw']   =  grouped.apply(lambda x: np.sum(x['flagw'])/N)
            df_av['pnanTs']  =  grouped.apply(lambda x: np.sum(x['flagTs'])/N)
            df_av['pnanTc']  =  grouped.apply(lambda x: np.sum(x['flagTc'])/N)
            
            # Flag runs with distorted measurements 
            if WDvalid[0]>WDvalid[1]:
                df_av['flagUW'] = ~((df_av.WD>=WDvalid[0]) | (df_av.WD<=WDvalid[1]))
                df_av['flagHs'] = ~((df_av.WD>=WDvalid[0]) | (df_av.WD<=WDvalid[1]))
                df_av['flagHc'] = ~((df_av.WD>=WDvalid[0]) | (df_av.WD<=WDvalid[1]))
            else:
                df_av['flagUW'] = ~((df_av.WD>=WDvalid[0]) & (df_av.WD<=WDvalid[1]))  
                df_av['flagHs'] = ~((df_av.WD>=WDvalid[0]) & (df_av.WD<=WDvalid[1]))
                df_av['flagHc'] = ~((df_av.WD>=WDvalid[0]) & (df_av.WD<=WDvalid[1]))
                
            # Flag runs with not enough data 
            df_av['flagUW'] = (df_av['flagUW'] | ((df_av['pnanU']>maxnanp) | (df_av['pnanw']>maxnanp)))
            df_av['flagHs'] = (df_av['flagHs'] | ((df_av['pnanw']>maxnanp) | (df_av['pnanTs']>maxnanp)))
            df_av['flagHc'] = (df_av['flagHc'] | ((df_av['pnanw']>maxnanp) | (df_av['pnanTc']>maxnanp)))
                 
            # Delete column with fraction of missing data
            df_av.drop(['flagU','flagw','flagTs','flagTc','pnanU','pnanw','pnanTs','pnanTc'] , axis=1, inplace=True)
            
            # Detrending
            if ldetrend:
                print('Detrending...')
                df['u']        = grouped.apply(lambda x: nandetrend(x['u'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['v']        = grouped.apply(lambda x: nandetrend(x['v'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['w']        = grouped.apply(lambda x: nandetrend(x['w'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['T_sonic']  = grouped.apply(lambda x: nandetrend(x['T_sonic'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['T_couple'] = grouped.apply(lambda x: nandetrend(x['T_couple'])).reset_index(1).reset_index(drop=True).set_index('time')
                if 'q' in list(ds.variables):
                    df['q'] = grouped.apply(lambda x: nandetrend(x['q'])).reset_index(1).reset_index(drop=True).set_index('time')
                    df['CO2'] = grouped.apply(lambda x: nandetrend(x['CO2'])).reset_index(1).reset_index(drop=True).set_index('time')
            
            # Group detrended data in blocks
            print('Calculating covariances & spectra ...')
            grouped = df.groupby(pd.Grouper(freq=dT,label='right'))

            # Compute covariance per block
            df_av['wTs']   = grouped.apply(lambda x: x['T_sonic'].cov(x['w']))
            df_av['wTc']   = grouped.apply(lambda x: x['T_couple'].cov(x['w']))
            df_av['uw']    = grouped.apply(lambda x: x['u'].cov(x['w']))
            df_av['vw']    = grouped.apply(lambda x: x['v'].cov(x['w']))
            df_av['ww']    = grouped.apply(lambda x: x['w'].cov(x['w']))
            df_av['uv']    = grouped.apply(lambda x: x['u'].cov(x['v']))
            df_av['uTs']   = grouped.apply(lambda x: x['u'].cov(x['T_sonic']))
            df_av['uTc']   = grouped.apply(lambda x: x['u'].cov(x['T_couple']))  
            df_av['uu']    = grouped.apply(lambda x: x['u'].cov(x['u']))
            df_av['vv']    = grouped.apply(lambda x: x['v'].cov(x['v']))
            df_av['TsTs']  = grouped.apply(lambda x: x['T_sonic'].cov(x['T_sonic']))
            df_av['TcTc']  = grouped.apply(lambda x: x['T_couple'].cov(x['T_couple']))
            # do the same if there is Licor data
            if 'q' in list(ds.variables):
                df_av['wq']  = grouped.apply(lambda x: x['w'].cov(x['q']))
                df_av['qq']  = grouped.apply(lambda x: x['q'].cov(x['q']))
            if 'CO2' in list(ds.variables):
                df_av['wCO2']  = grouped.apply(lambda x: x['w'].cov(x['CO2']))
                df_av['CO2CO2']  = grouped.apply(lambda x: x['CO2'].cov(x['CO2']))
        
            # Compute friction velocity
            df_av['ustar'] = (df_av['uw']**2 + df_av['vw']**2)**(1/4)
            
            # Compute temperature scale
            df_av['tstar'] = -np.divide(df_av['wTs'],df_av['ustar'])
            
            # Compute Obukhov length
            df_av['obh']   = -(df_av['ustar']**3)*df_av['T_sonic']/(9.81*0.4*df_av['wTs'])
            
            if 'q' in list(ds.variables):
                df_av['qstar'] = -np.divide(df_av['wq'],df_av['ustar'])
            
            # Make dataset
            ds_out = xr.Dataset(df_av)
                       
            # Calculate spectra only if rotated input data is used
            if inputdata=='L1B':
                # Linear detrending of raw data           
                df['u']        = grouped.apply(lambda x: nandetrend(x['u'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['v']        = grouped.apply(lambda x: nandetrend(x['v'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['w']        = grouped.apply(lambda x: nandetrend(x['w'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['T_sonic']  = grouped.apply(lambda x: nandetrend(x['T_sonic'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['T_couple'] = grouped.apply(lambda x: nandetrend(x['T_couple'])).reset_index(1).reset_index(drop=True).set_index('time')
                
                # Group again
                grouped = df.groupby(pd.Grouper(freq=dT,label='right')) 
                
                # Calculate cospectra
                SwTs        = grouped.apply(lambda x: cospectrum(x['T_sonic'],x['w'],N,frequency))
                SwTc        = grouped.apply(lambda x: cospectrum(x['T_couple'],x['w'],N,frequency))
                Suw         = grouped.apply(lambda x: cospectrum(x['w'],x['u'],N,frequency))
                Svw         = grouped.apply(lambda x: cospectrum(x['w'],x['v'],N,frequency))
                Sww         = grouped.apply(lambda x: cospectrum(x['w'],x['w'],N,frequency))
                Suu         = grouped.apply(lambda x: cospectrum(x['u'],x['u'],N,frequency))
                Svv         = grouped.apply(lambda x: cospectrum(x['v'],x['v'],N,frequency))
                STsTs       = grouped.apply(lambda x: cospectrum(x['T_sonic'],x['T_sonic'],N,frequency))
                STcTc       = grouped.apply(lambda x: cospectrum(x['T_couple'],x['T_couple'],N,frequency))
                
                # Convert dataframe of lists to numpy arrays
                SwTs  = np.stack(np.asarray(SwTs.to_xarray()), axis=0)
                SwTc  = np.stack(np.asarray(SwTc.to_xarray()), axis=0)
                Suw   = np.stack(np.asarray(Suw.to_xarray()), axis=0)
                Svw   = np.stack(np.asarray(Svw.to_xarray()), axis=0)
                Sww   = np.stack(np.asarray(Sww.to_xarray()), axis=0)
                Suu   = np.stack(np.asarray(Suu.to_xarray()), axis=0)
                Svv   = np.stack(np.asarray(Svv.to_xarray()), axis=0)
                STsTs = np.stack(np.asarray(STsTs.to_xarray()), axis=0)
                STcTc = np.stack(np.asarray(STcTc.to_xarray()), axis=0)
    
                # Define dataset output coordinates
                index = df_av.index
                freq  = np.fft.rfftfreq(N, 1/frequency)
                
                # Remove zero frequency
                freq = np.delete(freq,0)
    
                # Convert to DataArrays
                SwTs  = xr.DataArray(SwTs, coords=[index,freq], dims=['time','freq'])
                SwTc  = xr.DataArray(SwTc, coords=[index,freq], dims=['time','freq'])
                Suw   = xr.DataArray(Suw, coords=[index,freq], dims=['time','freq'])
                Svw   = xr.DataArray(Svw, coords=[index,freq], dims=['time','freq'])
                Sww   = xr.DataArray(Sww, coords=[index,freq], dims=['time','freq'])
                Suu   = xr.DataArray(Suu, coords=[index,freq], dims=['time','freq'])
                Svv   = xr.DataArray(Svv, coords=[index,freq], dims=['time','freq'])
                STsTs = xr.DataArray(STsTs, coords=[index,freq], dims=['time','freq'])
                STcTc = xr.DataArray(STcTc, coords=[index,freq], dims=['time','freq'])
                
                if 'q' in list(ds.variables):
                    Swq        = grouped.apply(lambda x: cospectrum(x['q'],x['w'],N,frequency))
                    Sqq      = grouped.apply(lambda x: cospectrum(x['q'],x['q'],N,frequency))
                    if 'CO2' in list(ds.variables):
                        SwCO2      = grouped.apply(lambda x: cospectrum(x['CO2'],x['w'],N,frequency))
                        SCO2O2      = grouped.apply(lambda x: cospectrum(x['CO2'],x['CO2'],N,frequency))
                    
                    Swq  = np.stack(np.asarray(Swq.to_xarray()), axis=0)
                    Sqq  = np.stack(np.asarray(Sqq.to_xarray()), axis=0)
                    Swq  = xr.DataArray(Swq, coords=[index,freq], dims=['time','freq'])
                    Sqq  = xr.DataArray(Sqq, coords=[index,freq], dims=['time','freq'])

                    if 'CO2' in list(ds.variables):
                        SwCO2  = np.stack(np.asarray(SwCO2.to_xarray()), axis=0)
                        SCO2O2  = np.stack(np.asarray(SCO2O2.to_xarray()), axis=0)
                        SwCO2  = xr.DataArray(SwCO2, coords=[index,freq], dims=['time','freq'])
                        SCO2O2  = xr.DataArray(SCO2O2, coords=[index,freq], dims=['time','freq'])

                    
                
                # Add spectra to dataset          
                ds_out = ds_out.assign({'SwTs': SwTs, 'SwTc': SwTc, 'Suw': Suw, \
                                    'Svw': Svw, 'Sww': Sww, 'Suu': Suu, \
                                    'Svv': Svv, 'STsTs': STsTs, 'STcTc': STcTc})
    
                if 'q' in list(ds.variables):
                    # ds_out = ds_out.assign({'SwTs': SwTs, 'SwTc': SwTc, 'Suw': Suw, \
                    # 'Svw': Svw, 'Sww': Sww, 'Suu': Suu, \
                    # 'Svv': Svv, 'STsTs': STsTs, 'STcTc': STcTc, 'Swq': Swq, 'SwCO2': SwCO2, 'Sqq': Sqq, 'SCO2O2': SCO2O2})
                    ds_out = ds_out.assign({'SwTs': SwTs, 'SwTc': SwTc, 'Suw': Suw, \
                    'Svw': Svw, 'Sww': Sww, 'Suu': Suu, \
                    'Svv': Svv, 'STsTs': STsTs, 'STcTc': STcTc, 'Swq': Swq, 'Sqq': Sqq})
                    if 'CO2' in list(ds.variables):
                        ds_out = ds_out.assign({'SwCO2': SwCO2, 'SCO2O2': SCO2O2})
            if dT != '30min':
                ds_out = ds_out.resample(time="30min").mean()
                
            # Go to folder where 30-min height data is stored
            if not lfixedheight:
                os.chdir(L1B_30min_dir) 
                file_30min = glob.glob("*WS*nc")
                # file_30min = glob.glob("*WS*" + file[ -9:-3] + "*nc") 
                ds_30min = xr.open_dataset(file_30min[0])
                ds_30min = ds_30min.resample(time="30min").interpolate("linear")
                
                # Add height to dataset
                ds_out['zm']   = ds_30min['zm']
                ds_out['d']    = ds_30min['d']
                ds_out['H']    = ds_30min['H']
                if lcorrH:
                    # ds_out['z']    = ds_30min['zm']+ds_30min['H']-ds_30min['snowheight']-ds_30min['d']+EC_hgt_offset
                    ds_out['z'] = - ds_30min['snowheight'] + zp
                else:
                    ds_out['z']    = ds_30min['zm']+ds_30min['H']-ds_30min['d']+EC_hgt_offset
                
                # Go back to L1B dir
                os.chdir(L1Bdir)  
            else:
                z      = xr.DataArray(np.full(len(ds_out.time),float(zp + H - d)),dims=['time'])
                ds_out = ds_out.assign({'z': z})
                
            if store_spectra == True:
                # Calculate Kaimal spectra
                ds_out = Kaimal(ds_out,nml,level = 'L2')
        
                # Calculate transfer functions
                ds_out = transfer_functions(ds_out,nml,level = 'L2')
                    


            
            # Store old attribute
            att1                                    = ds.attrs['IceEddie_namelist_L0toL1A']
            att2                                    = ds.attrs['IceEddie_namelist_L1AtoL1B']

            # Add new attributes
            ds_out                                      = utils.Add_dataset_attributes(ds_out,'CSAT_EC_L2.JSON')
            ds_out.attrs['location']                    = LOC + '_' + ID
            ds_out.attrs['file_creation_date_time']     = str(datetime.datetime.now())
            ds_out.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
            ds_out.attrs['IceEddie_namelist_L0toL1A']   = att1
            ds_out.attrs['IceEddie_namelist_L1AtoL1B']  = att2
            ds_out.attrs['IceEddie_namelist_L1BtoL2']   = str(nml['L1BtoL2'])
            ds_out.attrs['Averaging_time']              = dT
                        
            # Export dataset to net CDF
            if inputdata=='L1B':
                ds_out.to_netcdf(L2dir + file.replace('L1B','L2_t' + dT),mode = 'w',encoding={'time': \
                             {'units':'minutes since  '  + str(min(df_av.index).year) + '-' +  str(min(df_av.index).month) +  '-01'}}) 
            else:
                ds_out.u.attrs['long_name']       = 'zonal wind velocity (m/s)'
                ds_out.v.attrs['long_name']       = 'meridional wind velocity (m/s)'
                ds_out.to_netcdf(L2dir + file.replace('L1A','L2_t' + dT),mode = 'w',encoding={'time': \
                             {'units':'minutes since  '  + str(min(df_av.index).year) + '-' +  str(min(df_av.index).month) + '-01'}}) 
            
    
    if sensor == 'CSAT_Young':

        # Process files sequentially
        for file in sorted(glob.glob('*' + sensor + "*" + inputdata + "*nc")):
            print(file)
            # Open net CDF
            ds = xr.open_dataset(file)

            # Convert to dataframe
            df = ds.to_dataframe()
            
            # Find sampling frequency of input data
            frequency = 1 / (df.index[1]-df.index[0]).total_seconds()
            
            # Max number of samples per time block
            N = int(pd.to_timedelta(dT).total_seconds() * frequency)
                  
            # Group original data in blocks
            grouped = df.groupby(pd.Grouper(freq=dT,label='right'))
            
            # Compute mean per block
            df_av   = grouped.mean()
 
            # Change the way mean wind direction is calculated
            df_av['WD']    =  grouped.apply(lambda x: meanWindDir((x['u']**2+x['v']**2)**(0.5),x['WD']))
            df_av['WDy']    =  grouped.apply(lambda x: meanWindDir((x['uy']**2+x['vy']**2)**(0.5),x['WDy']))

            # Count number of missing data in runs
            df_av['pnanU']   =  grouped.apply(lambda x: np.sum(x['flagU'])/N)
            df_av['pnanw']   =  grouped.apply(lambda x: np.sum(x['flagw'])/N)
            df_av['pnanTs']  =  grouped.apply(lambda x: np.sum(x['flagTs'])/N)
            df_av['pnanTc']  =  grouped.apply(lambda x: np.sum(x['flagTc'])/N)
            
            # Flag runs with distorted measurements 
            if WDvalid[0]>WDvalid[1]:
                df_av['flagUW'] = ~((df_av.WD>=WDvalid[0]) | (df_av.WD<=WDvalid[1]))
                df_av['flagHs'] = ~((df_av.WD>=WDvalid[0]) | (df_av.WD<=WDvalid[1]))
                df_av['flagHc'] = ~((df_av.WD>=WDvalid[0]) | (df_av.WD<=WDvalid[1]))
            else:
                df_av['flagUW'] = ~((df_av.WD>=WDvalid[0]) & (df_av.WD<=WDvalid[1]))  
                df_av['flagHs'] = ~((df_av.WD>=WDvalid[0]) & (df_av.WD<=WDvalid[1]))
                df_av['flagHc'] = ~((df_av.WD>=WDvalid[0]) & (df_av.WD<=WDvalid[1]))
                
            # Flag runs with not enough data 
            df_av['flagUW'] = (df_av['flagUW'] | ((df_av['pnanU']>maxnanp) | (df_av['pnanw']>maxnanp)))
            df_av['flagHs'] = (df_av['flagHs'] | ((df_av['pnanw']>maxnanp) | (df_av['pnanTs']>maxnanp)))
            df_av['flagHc'] = (df_av['flagHc'] | ((df_av['pnanw']>maxnanp) | (df_av['pnanTc']>maxnanp)))
                 
            # Delete column with fraction of missing data
            df_av.drop(['flagU','flagw','flagTs','flagTc','pnanU','pnanw','pnanTs','pnanTc'] , axis=1, inplace=True)
            
            # Detrending
            if ldetrend:
                print('Detrending...')
                df['u']        = grouped.apply(lambda x: nandetrend(x['u'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['v']        = grouped.apply(lambda x: nandetrend(x['v'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['w']        = grouped.apply(lambda x: nandetrend(x['w'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['T_sonic']  = grouped.apply(lambda x: nandetrend(x['T_sonic'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['T_couple'] = grouped.apply(lambda x: nandetrend(x['T_couple'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['uy']        = grouped.apply(lambda x: nandetrend(x['uy'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['vy']        = grouped.apply(lambda x: nandetrend(x['vy'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['wg']        = grouped.apply(lambda x: nandetrend(x['wg'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['T2']       = grouped.apply(lambda x: nandetrend(x['T_couple2'])).reset_index(1).reset_index(drop=True).set_index('time')
                if 'q' in list(ds.variables):
                    df['q']       = grouped.apply(lambda x: nandetrend(x['q'])).reset_index(1).reset_index(drop=True).set_index('time')
            
            # Group detrended data in blocks
            print('Calculating covariances & spectra ...')
            grouped = df.groupby(pd.Grouper(freq=dT,label='right'))

            # Compute covariance per block
            df_av['wTs']   = grouped.apply(lambda x: x['T_sonic'].cov(x['w']))
            df_av['wTc']   = grouped.apply(lambda x: x['T_couple'].cov(x['w']))
            df_av['uw']    = grouped.apply(lambda x: x['u'].cov(x['w']))
            df_av['vw']    = grouped.apply(lambda x: x['v'].cov(x['w']))
            df_av['ww']    = grouped.apply(lambda x: x['w'].cov(x['w']))
            df_av['uv']    = grouped.apply(lambda x: x['u'].cov(x['v']))
            df_av['uTs']   = grouped.apply(lambda x: x['u'].cov(x['T_sonic']))
            df_av['uTc']   = grouped.apply(lambda x: x['u'].cov(x['T_couple']))  
            df_av['uu']    = grouped.apply(lambda x: x['u'].cov(x['u']))
            df_av['vv']    = grouped.apply(lambda x: x['v'].cov(x['v']))
            df_av['TsTs']  = grouped.apply(lambda x: x['T_sonic'].cov(x['T_sonic']))
            df_av['TcTc']  = grouped.apply(lambda x: x['T_couple'].cov(x['T_couple']))
            df_av['wgT2']  = grouped.apply(lambda x: x['T_couple2'].cov(x['wg']))
            df_av['uywg']  = grouped.apply(lambda x: x['uy'].cov(x['wg']))
            df_av['vywg']  = grouped.apply(lambda x: x['vy'].cov(x['wg']))
            df_av['wgwg']  = grouped.apply(lambda x: x['wg'].cov(x['wg']))
            df_av['uyvy']  = grouped.apply(lambda x: x['uy'].cov(x['vy']))
            df_av['uyT2']  = grouped.apply(lambda x: x['uy'].cov(x['T_couple2']))
            df_av['uyuy']  = grouped.apply(lambda x: x['uy'].cov(x['uy']))
            df_av['vyvy']  = grouped.apply(lambda x: x['vy'].cov(x['vy']))
            df_av['T2T2']  = grouped.apply(lambda x: x['T_couple2'].cov(x['T_couple2']))
            if 'q' in list(ds.variables):
                df_av['wq']    = grouped.apply(lambda x: x['w'].cov(x['q']))
                df_av['qq']    = grouped.apply(lambda x: x['q'].cov(x['q']))
                df_av['wgq']   = grouped.apply(lambda x: x['wg'].cov(x['q']))


            # Compute friction velocity
            df_av['ustar'] = (df_av['uw']**2 + df_av['vw']**2)**(1/4)
            df_av['ustary'] = (df_av['uywg']**2 + df_av['vywg']**2)**(1/4)
            
            # Compute temperature scale
            df_av['tstar'] = -np.divide(df_av['wTs'],df_av['ustar'])
            df_av['tstary'] = -np.divide(df_av['wgT2'],df_av['ustary'])
            
            # Compute Obukhov length
            df_av['obh']   = -(df_av['ustar']**3)*df_av['T_sonic']/(9.81*0.4*df_av['wTs'])
            df_av['obhy']  = -(df_av['ustary']**3)*df_av['T_couple2']/(9.81*0.4*df_av['wgT2'])
            
            if 'q' in list(ds.variables):
                df_av['qstar']  = -np.divide(df_av['wq'],df_av['ustar'])
                df_av['qstary'] = -np.divide(df_av['wgq'],df_av['ustary'])


            # Make dataset
            ds_out = xr.Dataset(df_av)
                       
            # Calculate spectra only if rotated input data is used
            if inputdata=='L1B':
                # Linear detrending of raw data           
                df['u']        = grouped.apply(lambda x: nandetrend(x['u'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['v']        = grouped.apply(lambda x: nandetrend(x['v'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['w']        = grouped.apply(lambda x: nandetrend(x['w'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['T_sonic']  = grouped.apply(lambda x: nandetrend(x['T_sonic'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['T_couple'] = grouped.apply(lambda x: nandetrend(x['T_couple'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['uy']        = grouped.apply(lambda x: nandetrend(x['uy'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['vy']        = grouped.apply(lambda x: nandetrend(x['vy'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['wg']        = grouped.apply(lambda x: nandetrend(x['wg'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['T2'] = grouped.apply(lambda x: nandetrend(x['T_couple2'])).reset_index(1).reset_index(drop=True).set_index('time')
                if 'q' in list(ds.variables):
                    df['q'] = grouped.apply(lambda x: nandetrend(x['q'])).reset_index(1).reset_index(drop=True).set_index('time')

                # Group again
                grouped = df.groupby(pd.Grouper(freq=dT,label='right')) 
                
                # Calculate cospectra
                SwTs        = grouped.apply(lambda x: cospectrum(x['T_sonic'],x['w'],N,frequency))
                SwTc        = grouped.apply(lambda x: cospectrum(x['T_couple'],x['w'],N,frequency))
                Suw         = grouped.apply(lambda x: cospectrum(x['w'],x['u'],N,frequency))
                Svw         = grouped.apply(lambda x: cospectrum(x['w'],x['v'],N,frequency))
                Sww         = grouped.apply(lambda x: cospectrum(x['w'],x['w'],N,frequency))
                Suu         = grouped.apply(lambda x: cospectrum(x['u'],x['u'],N,frequency))
                Svv         = grouped.apply(lambda x: cospectrum(x['v'],x['v'],N,frequency))
                STsTs       = grouped.apply(lambda x: cospectrum(x['T_sonic'],x['T_sonic'],N,frequency))
                STcTc       = grouped.apply(lambda x: cospectrum(x['T_couple'],x['T_couple'],N,frequency))
                SwgT2       = grouped.apply(lambda x: cospectrum(x['T_couple2'],x['wg'],N,frequency))
                Suywg       = grouped.apply(lambda x: cospectrum(x['wg'],x['uy'],N,frequency))
                Svywg       = grouped.apply(lambda x: cospectrum(x['wg'],x['vy'],N,frequency))
                Swgwg       = grouped.apply(lambda x: cospectrum(x['wg'],x['wg'],N,frequency))
                Suyuy       = grouped.apply(lambda x: cospectrum(x['uy'],x['uy'],N,frequency))
                Svyvy       = grouped.apply(lambda x: cospectrum(x['vy'],x['vy'],N,frequency))
                ST2T2     = grouped.apply(lambda x: cospectrum(x['T_couple2'],x['T_couple2'],N,frequency))
                if 'q' in list(ds.variables):
                    Swq     = grouped.apply(lambda x: cospectrum(x['w'],x['q'],N,frequency))
                    Sqq     = grouped.apply(lambda x: cospectrum(x['q'],x['q'],N,frequency))
                    Swgq     = grouped.apply(lambda x: cospectrum(x['wg'],x['q'],N,frequency))

                # Convert dataframe of lists to numpy arrays
                SwTs  = np.stack(np.asarray(SwTs.to_xarray()), axis=0)
                SwTc  = np.stack(np.asarray(SwTc.to_xarray()), axis=0)
                Suw   = np.stack(np.asarray(Suw.to_xarray()), axis=0)
                Svw   = np.stack(np.asarray(Svw.to_xarray()), axis=0)
                Sww   = np.stack(np.asarray(Sww.to_xarray()), axis=0)
                Suu   = np.stack(np.asarray(Suu.to_xarray()), axis=0)
                Svv   = np.stack(np.asarray(Svv.to_xarray()), axis=0)
                STsTs = np.stack(np.asarray(STsTs.to_xarray()), axis=0)
                STcTc = np.stack(np.asarray(STcTc.to_xarray()), axis=0)

                SwgT2  = np.stack(np.asarray(SwgT2.to_xarray()), axis=0)
                Suywg  = np.stack(np.asarray(Suywg.to_xarray()), axis=0)
                Svywg   = np.stack(np.asarray(Svywg.to_xarray()), axis=0)
                Swgwg   = np.stack(np.asarray(Swgwg.to_xarray()), axis=0)
                Suyuy   = np.stack(np.asarray(Suyuy.to_xarray()), axis=0)
                Svyvy   = np.stack(np.asarray(Svyvy.to_xarray()), axis=0)
                ST2T2 = np.stack(np.asarray(ST2T2.to_xarray()), axis=0)

                if 'q' in list(ds.variables):
                    Swq  = np.stack(np.asarray(Swq.to_xarray()), axis=0)
                    Sqq  = np.stack(np.asarray(Sqq.to_xarray()), axis=0)
                    Swgq = np.stack(np.asarray(Swgq.to_xarray()), axis=0)

                # Define dataset output coordinates
                index = df_av.index
                freq  = np.fft.rfftfreq(N, 1/frequency)
                
                # Remove zero frequency
                freq = np.delete(freq,0)
    
                # Convert to DataArrays
                SwTs  = xr.DataArray(SwTs, coords=[index,freq], dims=['time','freq'])
                SwTc  = xr.DataArray(SwTc, coords=[index,freq], dims=['time','freq'])
                Suw   = xr.DataArray(Suw, coords=[index,freq], dims=['time','freq'])
                Svw   = xr.DataArray(Svw, coords=[index,freq], dims=['time','freq'])
                Sww   = xr.DataArray(Sww, coords=[index,freq], dims=['time','freq'])
                Suu   = xr.DataArray(Suu, coords=[index,freq], dims=['time','freq'])
                Svv   = xr.DataArray(Svv, coords=[index,freq], dims=['time','freq'])
                STsTs = xr.DataArray(STsTs, coords=[index,freq], dims=['time','freq'])
                STcTc = xr.DataArray(STcTc, coords=[index,freq], dims=['time','freq'])

                SwgT2  = xr.DataArray(SwgT2, coords=[index,freq], dims=['time','freq'])
                Suywg  = xr.DataArray(Suywg, coords=[index,freq], dims=['time','freq'])
                Svywg   = xr.DataArray(Svywg, coords=[index,freq], dims=['time','freq'])
                Swgwg   = xr.DataArray(Swgwg, coords=[index,freq], dims=['time','freq'])
                Suyuy   = xr.DataArray(Suyuy, coords=[index,freq], dims=['time','freq'])
                Svyvy   = xr.DataArray(Svyvy, coords=[index,freq], dims=['time','freq'])
                ST2T2   = xr.DataArray(ST2T2, coords=[index,freq], dims=['time','freq'])

                if 'q' in list(ds.variables):
                    Swq  = xr.DataArray(Swq, coords=[index,freq], dims=['time','freq'])
                    Sqq  = xr.DataArray(Sqq, coords=[index,freq], dims=['time','freq'])
                    Swgq = xr.DataArray(Swgq, coords=[index,freq], dims=['time','freq'])


                # Add spectra to dataset
                if 'q' in list(ds.variables):   
                    ds_out = ds_out.assign({'SwTs': SwTs, 'SwTc': SwTc, 'Suw': Suw, \
                                        'Svw': Svw, 'Sww': Sww, 'Suu': Suu, \
                                        'Svv': Svv, 'STsTs': STsTs, 'STcTc': STcTc, \
                                        'SwgT2':SwgT2,    'Suywg':Suywg, 'Svywg': Svywg, \
                                        'Swgwg':Swgwg,'Suyuy':Suyuy, 'Svyvy':Svyvy,'ST2T2':ST2T2 \
                                        ,'Swq':Swq ,'Sqq':Sqq  ,'Swgq':Swgq})
                else:                  
                    ds_out = ds_out.assign({'SwTs': SwTs, 'SwTc': SwTc, 'Suw': Suw, \
                                        'Svw': Svw, 'Sww': Sww, 'Suu': Suu, \
                                        'Svv': Svv, 'STsTs': STsTs, 'STcTc': STcTc, \
                                        'SwgT2':SwgT2,    'Suywg':Suywg, 'Svywg': Svywg, \
                                        'Swgwg':Swgwg,'Suyuy':Suyuy, 'Svyvy':Svyvy,'ST2T2':ST2T2 })
    
            # if dT != '30min':
            #     ds_out = ds_out.resample(time="30min").mean()
                
            # Go to folder where 30-min height data is stored
            if not lfixedheight:
                os.chdir(L1B_30min_dir) 
                file_30min = glob.glob("*WS*nc")
                # file_30min = glob.glob("*WS*" + file[ -9:-3] + "*nc") 
                ds_30min = xr.open_dataset(file_30min[0])
                ds_30min = ds_30min.resample(time=dT).interpolate("linear")
                
                # Add height to dataset
                ds_out['zm']   = ds_30min['zm']
                ds_out['d']    = ds_30min['d']
                ds_out['H']    = ds_30min['H']
                if lcorrH:
                    ds_out['z']    = -ds_30min['snowheight']+zp
                    ds_out['zy']   = -ds_30min['snowheight']+zpy
                else:
                    ds_out['z']    = ds_30min['zm']+ds_30min['H']-ds_30min['d']+EC_hgt_offset
                
                # Go back to L1B dir
                os.chdir(L1Bdir)  
            else:
                z      = xr.DataArray(np.full(len(ds_out.time),float(zp + H - d)),dims=['time'])
                ds_out = ds_out.assign({'z': z})
                
            if store_spectra == True:
                # Calculate Kaimal spectra
                ds_out = Kaimal(ds_out,nml,level = 'L2',height='zy')
        
                # Calculate transfer functions
                ds_out = transfer_functions(ds_out,nml, tauT = tauT, Aw = Aw, Lu = Lu,level = 'L2')
            
            # Store old attribute
            att1                                    = ds.attrs['IceEddie_namelist_L0toL1A']
            att2                                    = ds.attrs['IceEddie_namelist_L1AtoL1B']

            # Add new attributes
            # ds_out                                      = utils.Add_dataset_attributes(ds_out,'CSAT_EC_L2.JSON')
            ds_out.attrs['location']                    = LOC + '_' + ID
            ds_out.attrs['file_creation_date_time']     = str(datetime.datetime.now())
            ds_out.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
            ds_out.attrs['IceEddie_namelist_L0toL1A']   = att1
            ds_out.attrs['IceEddie_namelist_L1AtoL1B']  = att2
            ds_out.attrs['IceEddie_namelist_L1BtoL2']   = str(nml['L1BtoL2'])
            ds_out.attrs['Averaging_time']              = dT
                        
            # Export dataset to net CDF
            if inputdata=='L1B':
                ds_out.to_netcdf(L2dir + file.replace('L1B','L2_t' + dT),mode = 'w', format="NETCDF4", engine="h5netcdf", encoding={'time': \
                                {'units':'minutes since  '  + str(min(df_av.index).year) + '-' +  str(min(df_av.index).month) +  '-01'}}) 
                ds_out.close()
                del ds_out
                # ds_out.to_netcdf(L2dir + file.replace('L1B','L2_t' + dT),mode = 'w',encoding={'time': \
                #              {'units':'minutes since  '  + str(min(df_av.index).year) + '-' +  str(min(df_av.index).month) +  '-01'}}) 
            else:
                ds_out.u.attrs['long_name']       = 'zonal wind velocity (m/s)'
                ds_out.v.attrs['long_name']       = 'meridional wind velocity (m/s)'
                ds_out.to_netcdf(L2dir + file.replace('L1A','L2_t' + dT),mode = 'w',encoding={'time': \
                             {'units':'minutes since  '  + str(min(df_av.index).year) + '-' +  str(min(df_av.index).month) + '-01'}}) 
            
    
                   
    if sensor == 'iWS':
        
        
        # Process files sequentially
        for file in sorted(glob.glob('*' + sensor + "*" + inputdata + "*nc")):
            print(file)
            # Open net CDF
            ds = xr.open_dataset(file)
            
            # Convert to dataframe
            df = ds.to_dataframe()
            
            # Find sampling frequency of input data
            frequency = 1 / (df.index[1]-df.index[0]).total_seconds()
            
            # Max number of samples per time block
            N = int(pd.to_timedelta(dT).total_seconds() * frequency)
                  
            # Group original data in blocks
            grouped = df.groupby(pd.Grouper(freq=dT,label='right'))
            
            # Compute mean per block
            df_av   = grouped.mean()
 
            # Change the way mean wind direction is calculated
            df_av['WD']    =  grouped.apply(lambda x: meanWindDir((x['u']**2+x['v']**2)**(0.5),x['WD']))

            # Count number of missing data in runs
            df_av['pnanU']   =  grouped.apply(lambda x: np.sum(x['flagU'])/N)
            df_av['pnanw']   =  grouped.apply(lambda x: np.sum(x['flagw'])/N)
            df_av['pnanT1']  =  grouped.apply(lambda x: np.sum(x['flagT1'])/N)
            df_av['pnanT2']  =  grouped.apply(lambda x: np.sum(x['flagT2'])/N)
            
            # Flag runs with distorted measurements 
            if WDvalid[0]>WDvalid[1]:
                df_av['flagUW'] = ~((df_av.WD>=WDvalid[0]) | (df_av.WD<=WDvalid[1]))
                df_av['flagH1'] = ~((df_av.WD>=WDvalid[0]) | (df_av.WD<=WDvalid[1]))
                df_av['flagH2'] = ~((df_av.WD>=WDvalid[0]) | (df_av.WD<=WDvalid[1]))
            else:
                df_av['flagUW'] = ~((df_av.WD>=WDvalid[0]) & (df_av.WD<=WDvalid[1]))  
                df_av['flagH1'] = ~((df_av.WD>=WDvalid[0]) & (df_av.WD<=WDvalid[1]))
                df_av['flagH2'] = ~((df_av.WD>=WDvalid[0]) & (df_av.WD<=WDvalid[1]))
                
            # Flag runs with not enough data 
            df_av['flagUW'] = (df_av['flagUW'] | ((df_av['pnanU']>maxnanp) | (df_av['pnanw']>maxnanp)))
            df_av['flagH1'] = (df_av['flagH1'] | ((df_av['pnanw']>maxnanp) | (df_av['pnanT1']>maxnanp)))
            df_av['flagH2'] = (df_av['flagH2'] | ((df_av['pnanw']>maxnanp) | (df_av['pnanT2']>maxnanp)))
                
            # Delete column with fraction of missing data
            df_av.drop(['flagU','flagw','flagT1','flagT2','pnanU','pnanw','pnanT1','pnanT2'] , axis=1, inplace=True)
                        
            # Detrending
            if ldetrend:
                print('Detrending...')
                df['u']        = grouped.apply(lambda x: nandetrend(x['u'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['v']        = grouped.apply(lambda x: nandetrend(x['v'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['w']        = grouped.apply(lambda x: nandetrend(x['w'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['T1']       = grouped.apply(lambda x: nandetrend(x['T1'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['T2']       = grouped.apply(lambda x: nandetrend(x['T2'])).reset_index(1).reset_index(drop=True).set_index('time')
            
            # Group detrended data in blocks
            print('Calculating covariances & spectra ...')
            grouped = df.groupby(pd.Grouper(freq=dT,label='right'))
    
            # Compute covariance per block
            if (ignore_samples_rhs==0):
                df_av['wT1']   = grouped.apply(lambda x: x['T1'].cov(x['w']))
                df_av['wT2']   = grouped.apply(lambda x: x['T2'].cov(x['w']))
                df_av['uw']    = grouped.apply(lambda x: x['u'].cov(x['w']))
                df_av['vw']    = grouped.apply(lambda x: x['v'].cov(x['w']))
                df_av['ww']    = grouped.apply(lambda x: x['w'].cov(x['w']))
                df_av['uv']    = grouped.apply(lambda x: x['u'].cov(x['v']))
                df_av['uT1']   = grouped.apply(lambda x: x['u'].cov(x['T1']))
                df_av['uT2']   = grouped.apply(lambda x: x['u'].cov(x['T2']))
                df_av['uu']    = grouped.apply(lambda x: x['u'].cov(x['u']))
                df_av['vv']    = grouped.apply(lambda x: x['v'].cov(x['v']))
                df_av['T1T1']  = grouped.apply(lambda x: x['T1'].cov(x['T1']))
                df_av['T2T2']  = grouped.apply(lambda x: x['T2'].cov(x['T2']))
            else:
                df_av['wT1']   = grouped.apply(lambda x: x['T1'][:-ignore_samples_rhs].cov(x['w'][:-ignore_samples_rhs]))
                df_av['wT2']   = grouped.apply(lambda x: x['T2'][:-ignore_samples_rhs].cov(x['w'][:-ignore_samples_rhs]))
                df_av['uw']    = grouped.apply(lambda x: x['u'][:-ignore_samples_rhs].cov(x['w'][:-ignore_samples_rhs]))
                df_av['vw']    = grouped.apply(lambda x: x['v'][:-ignore_samples_rhs].cov(x['w'][:-ignore_samples_rhs]))
                df_av['ww']    = grouped.apply(lambda x: x['w'][:-ignore_samples_rhs].cov(x['w'][:-ignore_samples_rhs]))
                df_av['uv']    = grouped.apply(lambda x: x['u'][:-ignore_samples_rhs].cov(x['v'][:-ignore_samples_rhs]))
                df_av['uT1']   = grouped.apply(lambda x: x['u'][:-ignore_samples_rhs].cov(x['T1'][:-ignore_samples_rhs]))
                df_av['uT2']   = grouped.apply(lambda x: x['u'][:-ignore_samples_rhs].cov(x['T2'][:-ignore_samples_rhs]))
                df_av['uu']    = grouped.apply(lambda x: x['u'][:-ignore_samples_rhs].cov(x['u'][:-ignore_samples_rhs]))
                df_av['vv']    = grouped.apply(lambda x: x['v'][:-ignore_samples_rhs].cov(x['v'][:-ignore_samples_rhs]))
                df_av['T1T1']  = grouped.apply(lambda x: x['T1'][:-ignore_samples_rhs].cov(x['T1'][:-ignore_samples_rhs]))
                df_av['T2T2']  = grouped.apply(lambda x: x['T2'][:-ignore_samples_rhs].cov(x['T2'][:-ignore_samples_rhs]))

            # Compute friction velocity
            df_av['ustar'] = (df_av['uw']**2 + df_av['vw']**2)**(1/4)
            
            # Compute temperature scale and Obukhov length
            if Tsensor == 'T1':
                df_av['tstar'] = -np.divide(df_av['wT1'],df_av['ustar'])
                df_av['obh']   = -(df_av['ustar']**3)*df_av['T1']/(9.81*utils.kappa*df_av['wT1'])
            elif Tsensor == 'T2':
                df_av['tstar'] = -np.divide(df_av['wT2'],df_av['ustar'])
                df_av['obh']   = -(df_av['ustar']**3)*df_av['T2']/(9.81*utils.kappa*df_av['wT2'])            
            
            # Make dataset
            ds_out = xr.Dataset(df_av)
            

                    
            # Calculate spectra only if rotated input data is used
            if inputdata=='L1B':
                # Linear detrending of raw data           
                df['u']        = grouped.apply(lambda x: nandetrend(x['u'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['v']        = grouped.apply(lambda x: nandetrend(x['v'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['w']        = grouped.apply(lambda x: nandetrend(x['w'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['T1']       = grouped.apply(lambda x: nandetrend(x['T1'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['T2']       = grouped.apply(lambda x: nandetrend(x['T2'])).reset_index(1).reset_index(drop=True).set_index('time')
                
                # Group again
                grouped = df.groupby(pd.Grouper(freq=dT,label='right')) 
                
                # Calculate cospectra
                SwT1      = grouped.apply(lambda x: cospectrum(x['T1'],x['w'],N=N,frequency=frequency))
                SwT2      = grouped.apply(lambda x: cospectrum(x['T2'],x['w'],N=N,frequency=frequency))
                Suw       = grouped.apply(lambda x: cospectrum(x['w'],x['u'],N=N,frequency=frequency))
                Svw       = grouped.apply(lambda x: cospectrum(x['w'],x['v'],N=N,frequency=frequency))
                Sww       = grouped.apply(lambda x: cospectrum(x['w'],x['w'],N=N,frequency=frequency))
                Suu       = grouped.apply(lambda x: cospectrum(x['u'],x['u'],N=N,frequency=frequency))
                Svv       = grouped.apply(lambda x: cospectrum(x['v'],x['v'],N=N,frequency=frequency))
                ST1T1     = grouped.apply(lambda x: cospectrum(x['T1'],x['T1'],N=N,frequency=frequency))
                ST2T2     = grouped.apply(lambda x: cospectrum(x['T2'],x['T2'],N=N,frequency=frequency))
                
                # Convert dataframe of lists to numpy arrays
                SwT1  = np.stack(np.asarray(SwT1.to_xarray()), axis=0)
                SwT2  = np.stack(np.asarray(SwT2.to_xarray()), axis=0)
                Suw   = np.stack(np.asarray(Suw.to_xarray()), axis=0)
                Svw   = np.stack(np.asarray(Svw.to_xarray()), axis=0)
                Sww   = np.stack(np.asarray(Sww.to_xarray()), axis=0)
                Suu   = np.stack(np.asarray(Suu.to_xarray()), axis=0)
                Svv   = np.stack(np.asarray(Svv.to_xarray()), axis=0)
                ST1T1 = np.stack(np.asarray(ST1T1.to_xarray()), axis=0)
                ST2T2 = np.stack(np.asarray(ST2T2.to_xarray()), axis=0)
    
                # Define dataset output coordinates
                index = df_av.index
                freq  = np.fft.rfftfreq(N, 1/frequency)
                
                # Remove zero frequency
                freq = np.delete(freq,0)
    
                # Convert to DataArrays
                SwT1  = xr.DataArray(SwT1, coords=[index,freq], dims=['time','freq'])
                SwT2  = xr.DataArray(SwT2, coords=[index,freq], dims=['time','freq'])
                Suw   = xr.DataArray(Suw, coords=[index,freq], dims=['time','freq'])
                Svw   = xr.DataArray(Svw, coords=[index,freq], dims=['time','freq'])
                Sww   = xr.DataArray(Sww, coords=[index,freq], dims=['time','freq'])
                Suu   = xr.DataArray(Suu, coords=[index,freq], dims=['time','freq'])
                Svv   = xr.DataArray(Svv, coords=[index,freq], dims=['time','freq'])
                ST1T1 = xr.DataArray(ST1T1, coords=[index,freq], dims=['time','freq'])
                ST2T2 = xr.DataArray(ST2T2, coords=[index,freq], dims=['time','freq'])
                
                # Add spectra to dataset          
                ds_out = ds_out.assign({'SwT1': SwT1, 'SwT2': SwT2, 'Suw': Suw, \
                                        'Svw': Svw, 'Sww': Sww, 'Suu': Suu, \
                                        'Svv': Svv,'ST1T1': ST1T1, 'ST2T2': ST2T2})

            if dT != '30min':
                ds_out = ds_out.resample(time="30min").mean()
                
            # Go to folder where 30-min height data is stored
            if not lfixedheight:
                os.chdir(L1B_30min_dir) 
                file_30min = glob.glob("*WS*" + file[ -9:-3] + "*nc") 
                if not file_30min:
                    os.chdir(L1Bdir)
                    continue
                ds_30min = xr.open_dataset(file_30min[0])
                
                # Add height to dataset
                ds_out['zm']   = ds_30min['zm']
                ds_out['zm'].fillna(zp)
                ds_out['d']    = ds_30min['d']
                ds_out['H']    = ds_30min['H']
                ds_out['z']    = ds_30min['zm']+ds_30min['H']-ds_30min['d']+EC_hgt_offset
                
                
                
                # Go back to L1B dir
                os.chdir(L1Bdir)  
            else:
                z      = xr.DataArray(np.full(len(ds_out.time),float(zp + H - d)),dims=['time'])
                ds_out = ds_out.assign({'z': z}) 
                
            if store_spectra == True:
                # Calculate Kaimal spectra
                ds_out = Kaimal(ds_out,nml,level = 'L2')
        
                # Calculate transfer functions
                ds_out = transfer_functions(ds_out,nml, tauT = tauT, Aw = Aw, Lu = Lu,level = 'L2')
                        
            # Store old attribute
            att1                                    = ds.attrs['IceEddie_namelist_L0toL1A']
            att2                                    = ds.attrs['IceEddie_namelist_L1AtoL1B']

            # Add new attributes
            ds_out                                      = utils.Add_dataset_attributes(ds_out,'iWS_EC_L2.JSON')
            ds_out.attrs['location']                    = LOC + '_' + ID
            ds_out.attrs['file_creation_date_time']     = str(datetime.datetime.now())
            ds_out.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
            ds_out.attrs['IceEddie_namelist_L0toL1A']   = att1
            ds_out.attrs['IceEddie_namelist_L1AtoL1B']  = att2
            ds_out.attrs['IceEddie_namelist_L1BtoL2']   = str(nml['L1BtoL2'])
            ds_out.attrs['Averaging_time']              = dT
            
            # Export to net CDF
            if inputdata=='L1B':
                ds_out.to_netcdf(L2dir + file.replace('L1B','L2_t' + dT),mode = 'w',encoding={'time': \
                             {'units':'minutes since  ' + str(min(df_av.index).year) + '-' +  str(min(df_av.index).month) + '-01'}})
            else:
                ds_out.u.attrs['long_name']       = 'zonal wind velocity (m/s)'
                ds_out.v.attrs['long_name']       = 'meridional wind velocity (m/s)'
                ds_out.w.attrs['long_name']       = 'vertical wind velocity (m/s)'
                ds_out.to_netcdf(L2dir + file.replace('L1A','L2_t' + dT),mode = 'w',encoding={'time': \
                             {'units':'minutes since  ' + str(min(df_av.index).year) + '-' +  str(min(df_av.index).month) + '-01'}})

                
    if sensor == 'Young':
        
        
        # Process files sequentially
        for file in sorted(glob.glob('*' + sensor + "*" + inputdata + "*nc")):
            print(file)
            # Open net CDF
            ds = xr.open_dataset(file)
            
            # Convert to dataframe
            df = ds.to_dataframe()
            
            # Find sampling frequency of input data
            frequency = 1 / (df.index[1]-df.index[0]).total_seconds()
            
            # Max number of samples per time block
            N = int(pd.to_timedelta(dT).total_seconds() * frequency)
                  
            # Group original data in blocks
            grouped = df.groupby(pd.Grouper(freq=dT,label='right'))
            
            # Compute mean per block
            df_av   = grouped.mean()
 
            # Change the way mean wind direction is calculated
            df_av['WD']    =  grouped.apply(lambda x: meanWindDir((x['u']**2+x['v']**2)**(0.5),x['WD']))

            # Count number of missing data in runs
            df_av['pnanU']   =  grouped.apply(lambda x: np.sum(x['flagU'])/N)
            df_av['pnanw']   =  grouped.apply(lambda x: np.sum(x['flagw'])/N)
            df_av['pnanT1']  =  grouped.apply(lambda x: np.sum(x['flagT1'])/N)
            
            # Flag runs with distorted measurements 
            if WDvalid[0]>WDvalid[1]:
                df_av['flagUW'] = ~((df_av.WD>=WDvalid[0]) | (df_av.WD<=WDvalid[1]))
                df_av['flagH1'] = ~((df_av.WD>=WDvalid[0]) | (df_av.WD<=WDvalid[1]))
            else:
                df_av['flagUW'] = ~((df_av.WD>=WDvalid[0]) & (df_av.WD<=WDvalid[1]))  
                df_av['flagH1'] = ~((df_av.WD>=WDvalid[0]) & (df_av.WD<=WDvalid[1]))
                
            # Flag runs with not enough data 
            df_av['flagUW'] = (df_av['flagUW'] | ((df_av['pnanU']>maxnanp) | (df_av['pnanw']>maxnanp)))
            df_av['flagH1'] = (df_av['flagH1'] | ((df_av['pnanw']>maxnanp) | (df_av['pnanT1']>maxnanp)))
                
            # Delete column with fraction of missing data
            df_av.drop(['flagU','flagw','flagT1','pnanU','pnanw','pnanT1'] , axis=1, inplace=True)
                        
            # Detrending
            if ldetrend:
                print('Detrending...')
                df['u']        = grouped.apply(lambda x: nandetrend(x['u'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['v']        = grouped.apply(lambda x: nandetrend(x['v'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['w']        = grouped.apply(lambda x: nandetrend(x['w'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['T1']       = grouped.apply(lambda x: nandetrend(x['T1'])).reset_index(1).reset_index(drop=True).set_index('time')
            
            # Group detrended data in blocks
            print('Calculating covariances & spectra ...')
            grouped = df.groupby(pd.Grouper(freq=dT,label='right'))
    
            # Compute covariance per block
            df_av['wT1']   = grouped.apply(lambda x: x['T1'].cov(x['w']))
            df_av['uw']    = grouped.apply(lambda x: x['u'].cov(x['w']))
            df_av['vw']    = grouped.apply(lambda x: x['v'].cov(x['w']))
            df_av['ww']    = grouped.apply(lambda x: x['w'].cov(x['w']))
            df_av['uv']    = grouped.apply(lambda x: x['u'].cov(x['v']))
            df_av['uT1']   = grouped.apply(lambda x: x['u'].cov(x['T1']))
            df_av['uu']    = grouped.apply(lambda x: x['u'].cov(x['u']))
            df_av['vv']    = grouped.apply(lambda x: x['v'].cov(x['v']))
            df_av['T1T1']  = grouped.apply(lambda x: x['T1'].cov(x['T1']))


            # Compute friction velocity
            df_av['ustar'] = (df_av['uw']**2 + df_av['vw']**2)**(1/4)
            
            # Compute temperature scale
            df_av['tstar'] = np.divide(df_av['wT1'],df_av['ustar'])
            
            # Compute Obukhov length
            df_av['obh']   = -(df_av['ustar']**3)*df_av['T1']/(9.81*utils.kappa*df_av['wT1'])
            
            # Make dataset
            ds_out = xr.Dataset(df_av)
                            
            # Calculate spectra only if rotated input data is used
            if inputdata=='L1B':
                df['u']        = grouped.apply(lambda x: nandetrend(x['u'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['v']        = grouped.apply(lambda x: nandetrend(x['v'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['w']        = grouped.apply(lambda x: nandetrend(x['w'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['T1']       = grouped.apply(lambda x: nandetrend(x['T1'])).reset_index(1).reset_index(drop=True).set_index('time')
                
                grouped = df.groupby(pd.Grouper(freq=dT,label='right'))
                
                # Calculate cospectra
                SwT1      = grouped.apply(lambda x: cospectrum(x['T1'],x['w'],N=N,frequency=frequency))
                Suw       = grouped.apply(lambda x: cospectrum(x['w'],x['u'],N=N,frequency=frequency))
                Svw       = grouped.apply(lambda x: cospectrum(x['w'],x['v'],N=N,frequency=frequency))
                Sww       = grouped.apply(lambda x: cospectrum(x['w'],x['w'],N=N,frequency=frequency))
                Suu       = grouped.apply(lambda x: cospectrum(x['u'],x['u'],N=N,frequency=frequency))
                Svv       = grouped.apply(lambda x: cospectrum(x['v'],x['v'],N=N,frequency=frequency))
                ST1T1     = grouped.apply(lambda x: cospectrum(x['T1'],x['T1'],N=N,frequency=frequency))
                
                # Convert dataframe of lists to numpy arrays
                SwT1  = np.stack(np.asarray(SwT1.to_xarray()), axis=0)
                Suw   = np.stack(np.asarray(Suw.to_xarray()), axis=0)
                Svw   = np.stack(np.asarray(Svw.to_xarray()), axis=0)
                Sww   = np.stack(np.asarray(Sww.to_xarray()), axis=0)
                Suu   = np.stack(np.asarray(Suu.to_xarray()), axis=0)
                Svv   = np.stack(np.asarray(Svv.to_xarray()), axis=0)
                ST1T1 = np.stack(np.asarray(ST1T1.to_xarray()), axis=0)
    
                # Define dataset output coordinates
                index = df_av.index
                freq  = np.fft.rfftfreq(N, 1/frequency)
                
                # Remove zero frequency
                freq = np.delete(freq,0)
    
                # Convert to DataArrays
                SwT1  = xr.DataArray(SwT1, coords=[index,freq], dims=['time','freq'])
                Suw   = xr.DataArray(Suw, coords=[index,freq], dims=['time','freq'])
                Svw   = xr.DataArray(Svw, coords=[index,freq], dims=['time','freq'])
                Sww   = xr.DataArray(Sww, coords=[index,freq], dims=['time','freq'])
                Suu   = xr.DataArray(Suu, coords=[index,freq], dims=['time','freq'])
                Svv   = xr.DataArray(Svv, coords=[index,freq], dims=['time','freq'])
                ST1T1 = xr.DataArray(ST1T1, coords=[index,freq], dims=['time','freq'])
                
                # Add spectra to dataset          
                ds_out = ds_out.assign({'SwT1': SwT1, 'Suw': Suw, \
                                        'Svw': Svw, 'Sww': Sww, 'Suu': Suu, \
                                        'Svv': Svv, 'ST1T1': ST1T1})

            if dT != '30min':
                ds_out = ds_out.resample(time="30min").mean()
                
            # Go to folder where 30-min height data is stored
            if not lfixedheight:
                os.chdir(L1B_30min_dir) 
                file_30min = glob.glob("*WS*" + file[ -9:-3] + "*nc") 
                ds_30min = xr.open_dataset(file_30min[0])
                
                # Add height to dataset
                ds_out['zm']   = ds_30min['zm']
                ds_out['d']    = ds_30min['d']
                ds_out['H']    = ds_30min['H']
                ds_out['z']    = ds_30min['zm']+ds_30min['H']-ds_30min['d']+EC_hgt_offset
                
                # Go back to L1B dir
                os.chdir(L1Bdir)  
            else:
                z      = xr.DataArray(np.full(len(ds_out.time),float(zp + H - d)),dims=['time'])
                ds_out = ds_out.assign({'z': z})
                
            if store_spectra == True:
                # Calculate Kaimal spectra
                ds_out = Kaimal(ds_out,nml,level = 'L2')
        
                # Calculate transfer functions
                ds_out = transfer_functions(ds_out,nml, tauT = tauT, Aw = Aw, Lu = Lu,level = 'L2')
                        
            # Store old attribute
            att1                                    = ds.attrs['IceEddie_namelist_L0toL1A']
            att2                                    = ds.attrs['IceEddie_namelist_L1AtoL1B']

            # Add new attributes
            ds_out                                      = utils.Add_dataset_attributes(ds_out,'YOUNG_EC_L2.JSON')
            ds_out.attrs['location']                    = LOC + '_' + ID
            ds_out.attrs['file_creation_date_time']     = str(datetime.datetime.now())
            ds_out.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
            ds_out.attrs['IceEddie_namelist_L0toL1A']   = att1
            ds_out.attrs['IceEddie_namelist_L1AtoL1B']  = att2
            ds_out.attrs['IceEddie_namelist_L1BtoL2']   = str(nml['L1BtoL2'])
            ds_out.attrs['Averaging_time']              = dT
            
            # Export to net CDF
            if inputdata=='L1B':
                ds_out.to_netcdf(L2dir + file.replace('L1B','L2_t' + dT),mode = 'w',encoding={'time': \
                             {'units':'minutes since  ' + str(min(df_av.index).year) + '-' +  str(min(df_av.index).month) + '-01'}})
            else:
                ds_out.u.attrs['long_name']       = 'zonal wind velocity (m/s)'
                ds_out.v.attrs['long_name']       = 'meridional wind velocity (m/s)'
                ds_out.to_netcdf(L2dir + file.replace('L1A','L2_t' + dT),mode = 'w',encoding={'time': \
                             {'units':'minutes since  ' + str(min(df_av.index).year) + '-' +  str(min(df_av.index).month) + '-01'}})
    
        
        
    if sensor == 'Gill':
        
        
        # Process files sequentially
        for file in sorted(glob.glob("iWS*" + inputdata + "*nc")):
            print(file)
            # Open net CDF
            ds = xr.open_dataset(file)
            
            # Convert to dataframe
            df = ds.to_dataframe()
            
            # Find sampling frequency of input data
            frequency = 1 / (df.index[1]-df.index[0]).total_seconds()
            
            # Max number of samples per time block
            N = int(pd.to_timedelta(dT).total_seconds() * frequency)
                  
            # Group original data in blocks
            grouped = df.groupby(pd.Grouper(freq=dT,label='right'))
            
            # Compute mean per block
            df_av   = grouped.mean()

            # Count number of missing data in runs
            df_av['#nan']  =  grouped.apply(lambda x: np.sum(np.isnan(x['w']))/N)
            
            # Flag runs with substancial missing data 
            df_av['flag'] = (df_av['#nan']<0.10)
                
            # Delete column with amount of missing data
            df_av.drop(['#nan'] , axis=1, inplace=True)
                        
            # Detrending
            if ldetrend:
                print('Detrending...')
                df['w']        = grouped.apply(lambda x: nandetrend(x['w'])).reset_index(1).reset_index(drop=True).set_index('time')
                df['T2']       = grouped.apply(lambda x: nandetrend(x['T2'])).reset_index(1).reset_index(drop=True).set_index('time')
            
            # Group detrended data in blocks
            print('Calculating covariances & spectra ...')
            grouped = df.groupby(pd.Grouper(freq=dT,label='right'))
    
            # Compute covariance per block
            df_av['wT2']   = grouped.apply(lambda x: x['T2'].cov(x['w']))
            df_av['ww']    = grouped.apply(lambda x: x['w'].cov(x['w']))
            df_av['T2T2']  = grouped.apply(lambda x: x['T2'].cov(x['T2']))
            
            # Make dataset
            ds_out = xr.Dataset(df_av)
            
            # Store old attribute
            att1                                    = ds.attrs['IceEddie_namelist_L0toL1A']
            att2                                    = ds.attrs['IceEddie_namelist_L1AtoL1B']

            # Add new attributes
            ds_out                                      = utils.Add_dataset_attributes(ds_out,'GILL_EC_L2.JSON')
            ds_out.attrs['location']                    = LOC + '_' + ID
            ds_out.attrs['file_creation_date_time']     = str(datetime.datetime.now())
            ds_out.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
            ds_out.attrs['IceEddie_namelist_L0toL1A']   = att1
            ds_out.attrs['IceEddie_namelist_L1AtoL1B']  = att2
            ds_out.attrs['IceEddie_namelist_L1BtoL2']   = str(nml['L1BtoL2'])
            
            # Export to net CDF
            ds_out.to_netcdf(L2dir + file.replace('L1A','L2_t' + dT),mode = 'w',encoding={'time': \
                         {'units':'minutes since  ' + str(min(df_av.index).year) + '-' +  str(min(df_av.index).month) + '-01'}})    
     
#_______________________________________________________
def L2toL3(nml):
    """
    Reads all the uncorrected averaged data and raw spectra, applies spectral corrections 
    and estimates the roughness length using two different methods (eddy-correlation method and 
    flux-variance similarity method).
    If no paths are provided, this function reads all the files in the current working directory.
    
    Input
    ----------
    nml: python f90nml namelist 
        namelist containing all parameters
    
    Output
    ----------
    monthly nc files containing corrected fluxes and estimated roughness lengths
    
    Example
    ----------
    L2toL3('iWS')
    
    Required packages
    ----------
    glob, os, itertools, pandas
    
    Required functions
    ----------
    
    
    """     
    # np.warnings.filterwarnings('ignore')
    

    base_dir    = nml['global']['dir']
    L2dir       = base_dir + 'L2/'
    L3dir       = base_dir + 'L3/'
    sensor      = nml['global']['sensor'] 
    ID			= nml['global']['ID']    
    LOC			= nml['global']['LOC']
    L1B_30min_dir = nml['global']['l1b_30min_dir']
    L2_30min_dir = nml['global']['L2_30min_dir']
    input_type     = nml['global']['input_type']
    version = nml['global']['version']
    
    Aw_nml        = nml['L2toL3']['Aw']
    Lu_nml        = nml['L2toL3']['lu']
    tauT_nml      = nml['L2toL3']['tauT']
    ldtau         = nml['L2toL3']['ldtau']
    dAw           = nml['L2toL3']['dAw']
    dLu           = nml['L2toL3']['dLu']
    dtauT         = nml['L2toL3']['dtauT']
    WDoffset     = nml['L2toL3']['WDoffset']

    lshadowing     = nml['L2toL3']['lshadowing']
    lwpl           = nml['L2toL3']['lwpl']
    lwpl_method    = nml['L2toL3']['lwpl_method']
    lsnd           = nml['L2toL3']['lsnd']
    lsnd_method    = nml['L2toL3']['lsnd_method']
    lsidewind      = nml['L2toL3']['lsidewind']
    lsidewind_method = nml['L2toL3']['lsidewind_method']
    CESAR_dir      = nml['L2toL3']['CESAR_dir']
    nitera         = nml['L2toL3']['nitera']
    Tsensor        = nml['L2toL3']['Tsensor']
    toffset_min  = nml['L2toL3']['toffset_min']

    lfixedheight  = nml['L2toL3']['lfixedheight']
    EC_hgt_offset = nml['L2toL3']['EC_hgt_offset']    
    lcorrH        = nml['L2toL3']['lcorrH']    
    lconvhum      = nml['L2toL3']['lconvhum']
    lhumbiascorr      = nml['L2toL3']['lhumbiascorr']
    humbiascorr      = nml['L2toL3']['humbiascorr']
    zp            = nml['L1BtoL2']['z']    
    zpy           = nml['L1BtoL2']['zy']    
    gillcorr_method =  nml['L2toL3']['gillcorr_method']   
    tau_w_1 =  nml['L2toL3']['tau_w_1']    
    tau_w_2 =  nml['L2toL3']['tau_w_2']    
    tau_u =  nml['L2toL3']['tau_u']  
    lBo_LHF =  nml['L2toL3']['lBo_LHF']  
    dT =  nml['L2toL3']['dT']  

    if not os.path.exists(L3dir):
       os.makedirs(L3dir) 
       
    if not L2dir:
        os.chdir(os.getcwd())  
    else:
        os.chdir(L2dir)  
        
    if sensor == 'iWS':
        
        # Process files sequentially
        for file in sorted(glob.glob('*' + sensor + '*L2*' + "*nc")):
            print(file)
            # Open net CDF
            ds = xr.open_dataset(file)
            
            ds['time'] = ds.time.values + np.timedelta64(toffset_min,'m')
            
            # Store initial dataset
            ds_L2 = ds
            

            # Delete previously spectra 
            try:
                ds = ds.drop({'Gw','Gu','GT','GD','GwT','Guw',\
                        'Tww','Tuu','TTT','Tuw','TwT','TswT','Tsuw',\
                            'SnTT','Snuu','Snuw','SnwT','Snww',\
                                'Svw','Sww','Suu','Svv','Suw','ST1T1','SwT2','SwT1','ST2T2','freq','freq_corr'})
            except:
                try:
                    ds = ds.drop({'Svw','Sww','Suu','Svv','Suw','ST1T1','SwT2','SwT1','ST2T2','freq'})  
                except:
                        
                    print('error')
                pass
#           ds = ds.drop({'freq_corr','SnTT','Snuu','Snuw','SnwT','Snww',\
#                     'TTT','Tww','Tuu','Tuw','TwT'})              
            # Assign transfer function constants to new dimension
            if ldtau:
                ds['tauT'] = [tauT_nml,tauT_nml- dtauT,tauT_nml+dtauT]
                ds['Aw']   = [Aw_nml,Aw_nml - dAw,Aw_nml+dAw]
                ds['Lu']   = [Lu_nml,Lu_nml - dLu,Lu_nml+dLu]
                ds.set_coords(['tauT','Aw','Lu'])
                # Initialise array
                Aww    = np.full((len(ds.time),len(ds['tauT']),len(ds['Aw']),len(ds['Lu'])),np.nan)
                Auu    = np.full((len(ds.time),len(ds['tauT']),len(ds['Aw']),len(ds['Lu'])),np.nan)
                ATT    = np.full((len(ds.time),len(ds['tauT']),len(ds['Aw']),len(ds['Lu'])),np.nan)
                AwT    = np.full((len(ds.time),len(ds['tauT']),len(ds['Aw']),len(ds['Lu'])),np.nan)
                Auw    = np.full((len(ds.time),len(ds['tauT']),len(ds['Aw']),len(ds['Lu'])),np.nan)
                A_wT_H = np.full((len(ds.time),len(ds['tauT']),len(ds['Aw']),len(ds['Lu'])),np.nan)
                A_uw_H = np.full((len(ds.time),len(ds['tauT']),len(ds['Aw']),len(ds['Lu'])),np.nan)

            else:
                ds['tauT'] = tauT_nml
                ds['Aw']   = Aw_nml
                ds['Lu']   = Lu_nml
                ds.set_coords(['tauT','Aw','Lu'])
                # Initialise array
                Aww    = np.full((len(ds.time),1,1,1),np.nan)
                Auu    = np.full((len(ds.time),1,1,1),np.nan)
                ATT    = np.full((len(ds.time),1,1,1),np.nan)
                AwT    = np.full((len(ds.time),1,1,1),np.nan)
                Auw    = np.full((len(ds.time),1,1,1),np.nan)
                A_wT_H = np.full((len(ds.time),1,1,1),np.nan)
                A_uw_H = np.full((len(ds.time),1,1,1),np.nan)


            itera = 0
            
            # Iterate corrections
            while (itera<nitera):
                itera    = itera + 1
                
                print('Iteration '  +str(itera) + '/' + str(nitera))
                # Calculate Kaimal spectra
                ds = Kaimal(ds,nml,level = 'L3')
                
                if ldtau:
                    # Loop over response times
                    i = -1
                    for tauT in ds.tauT.values:
                        j = -1
                        i = i + 1
                        for Aw in ds.Aw.values:
                            k = -1
                            j = j + 1
                            for Lu in ds.Lu.values:
                                k = k + 1       
                                
                                # Calculate transfer functions
                                ds_tmp = transfer_functions(ds,nml, tauT = tauT, Aw = Aw, Lu = Lu,level = 'L3')
                            
                                # Calculate attenuation factors after Moore
                                Aww[:,i,j,k]  = np.trapz(ds.Snww * ds_tmp.Tww, ds_tmp.freq_corr) / np.trapz(ds.Snww, ds_tmp.freq_corr) 
                                Auu[:,i,j,k]  = np.trapz(ds.Snuu * ds_tmp.Tuu, ds_tmp.freq_corr) / np.trapz(ds.Snuu, ds_tmp.freq_corr) 
                                ATT[:,i,j,k]  = np.trapz(ds.SnTT * ds_tmp.TTT, ds_tmp.freq_corr) / np.trapz(ds.SnTT, ds_tmp.freq_corr) 
                                AwT[:,i,j,k]  = np.trapz(ds.SnwT * ds_tmp.TwT, ds_tmp.freq_corr) / np.trapz(ds.SnwT, ds_tmp.freq_corr) 
                                Auw[:,i,j,k]  = np.trapz(ds.Snuw * ds_tmp.Tuw, ds_tmp.freq_corr) / np.trapz(ds.Snuw, ds_tmp.freq_corr) 
                                
                                # Calculate attenuation factors after Horst
                                ds_tmp        = Horst_correction(ds, nml, tauT, Aw, Lu)
                                
                                A_uw_H[:,i,j,k] = ds_tmp.A_uw_H
                                A_wT_H[:,i,j,k] = ds_tmp.A_wT_H
                                
                                del ds_tmp
                else:
                    
                    # Calculate transfer functions
                    ds_tmp = transfer_functions(ds,nml, tauT = ds.tauT.values, Aw = ds.Aw.values, Lu = ds.Lu.values,level = 'L3')
                    
                    # Calculate attenuation factors after Moore
                    Aww  = np.trapz(ds.Snww * ds_tmp.Tww, ds_tmp.freq_corr) / np.trapz(ds.Snww, ds_tmp.freq_corr) 
                    Auu  = np.trapz(ds.Snuu * ds_tmp.Tuu, ds_tmp.freq_corr) / np.trapz(ds.Snuu, ds_tmp.freq_corr) 
                    ATT  = np.trapz(ds.SnTT * ds_tmp.TTT, ds_tmp.freq_corr) / np.trapz(ds.SnTT, ds_tmp.freq_corr) 
                    AwT  = np.trapz(ds.SnwT * ds_tmp.TwT, ds_tmp.freq_corr) / np.trapz(ds.SnwT, ds_tmp.freq_corr) 
                    Auw  = np.trapz(ds.Snuw * ds_tmp.Tuw, ds_tmp.freq_corr) / np.trapz(ds.Snuw, ds_tmp.freq_corr) 
                    
                    # Calculate attenuation factors after Horst
                    ds_tmp        = Horst_correction(ds, nml, tauT = ds.tauT.values, Aw = ds.Aw.values, Lu = ds.Lu.values)
                    
                    A_uw_H = ds_tmp.A_uw_H
                    A_wT_H = ds_tmp.A_wT_H
                    
    
                # Add corrections to dataset
                if ldtau:         
                    ds['Aww']  = xr.DataArray(Aww, coords=[ds.time,ds.tauT,ds.Aw,ds.Lu], dims=['time','tauT','Aw','Lu'])
                    ds['Auu']  = xr.DataArray(Auu, coords=[ds.time,ds.tauT,ds.Aw,ds.Lu], dims=['time','tauT','Aw','Lu'])
                    ds['ATT']  = xr.DataArray(ATT, coords=[ds.time,ds.tauT,ds.Aw,ds.Lu], dims=['time','tauT','Aw','Lu'])
                    ds['AwT']  = xr.DataArray(AwT, coords=[ds.time,ds.tauT,ds.Aw,ds.Lu], dims=['time','tauT','Aw','Lu'])
                    ds['Auw']  = xr.DataArray(Auw, coords=[ds.time,ds.tauT,ds.Aw,ds.Lu], dims=['time','tauT','Aw','Lu'])
                    ds['A_uw_H']  = xr.DataArray(A_uw_H, coords=[ds.time,ds.tauT,ds.Aw,ds.Lu], dims=['time','tauT','Aw','Lu'])
                    ds['A_wT_H']  = xr.DataArray(A_wT_H, coords=[ds.time,ds.tauT,ds.Aw,ds.Lu], dims=['time','tauT','Aw','Lu'])
                else:
                    ds['Aww']  = xr.DataArray(Aww, coords=[ds.time], dims=['time'])
                    ds['Auu']  = xr.DataArray(Auu, coords=[ds.time], dims=['time'])
                    ds['ATT']  = xr.DataArray(ATT, coords=[ds.time], dims=['time'])
                    ds['AwT']  = xr.DataArray(AwT, coords=[ds.time], dims=['time'])
                    ds['Auw']  = xr.DataArray(Auw, coords=[ds.time], dims=['time'])
                    ds['A_uw_H']  = xr.DataArray(A_uw_H, coords=[ds.time], dims=['time'])
                    ds['A_wT_H']  = xr.DataArray(A_wT_H, coords=[ds.time], dims=['time'])
    
                # Apply high-frequency correction
                ds['ww']    = ds['ww']  /ds['Aww']
                ds['wT2']   = ds['wT2'] /ds['AwT']
                ds['uw']    = ds['uw']  /ds['Auw']  
                ds['uu']    = ds['uu']  /ds['Auu']
                ds['T1T1']  = ds['T1T1']/ds['ATT']
                ds['T2T2']  = ds['T2T2']/ds['ATT']
                ds['wT1']   = ds['wT1'] /ds['AwT']
        
                # Compute corrected friction velocity   
                ds['ustar'] = (ds['uw']**2 + ds['vw']**2)**(1/4)
                # Compute corrected  Obukhov length and temperature scale
                if Tsensor == 'T1':
                    ds['obh']   = -(ds['ustar']**3)*ds['T1']/(9.81*utils.kappa*ds['wT1'])
                    ds['tstar'] = -np.divide(ds['wT1'],ds['ustar'])
                if Tsensor == 'T2':
                    ds['obh']   = -(ds['ustar']**3)*ds['T2']/(9.81*utils.kappa*ds['wT2'])
                    ds['tstar'] = -np.divide(ds['wT2'],ds['ustar'])
                    
                # calculate corrected stability parameter 
                ds['zeta']  = ds['z']/ds['obh']
            
            
            # apply final correction to L2 fluxes
            ds['ww']    = ds_L2['ww']  /ds['Aww']
            ds['uu']    = ds_L2['uu']  /ds['Auu']
            ds['T1T1']  = ds_L2['T1T1']/ds['ATT']
            ds['T2T2']  = ds_L2['T2T2']/ds['ATT']
            ds['wT1']   = ds_L2['wT1'] /ds['AwT']
            ds['wT2']   = ds_L2['wT2'] /ds['AwT']
            ds['uw']    = ds_L2['uw']  /ds['Auw']     

            # Compute corrected friction velocity
            ds['ustar'] = (ds['uw']**2 + ds['vw']**2)**(1/4)
        
            # Compute corrected temperature scale and Obukhov length
            if Tsensor == 'T1':
                ds['obh']   = -(ds['ustar']**3)*ds['T1']/(9.81*utils.kappa*ds['wT1'])
                ds['tstar'] = -np.divide(ds['wT1'],ds['ustar'])
                ds['wT'] = ds['wT1']
            if Tsensor == 'T2':
                ds['obh']   = -(ds['ustar']**3)*ds['T2']/(9.81*utils.kappa*ds['wT2'])
                ds['tstar'] = -np.divide(ds['wT2'],ds['ustar'])
                ds['wT'] = ds['wT2']
            
            # calculate corrected stability parameter 
            ds['zeta']  = ds['z']/ds['obh']
                
            # Initialise roughness length
            if ldtau:
                ds['z0_EC']  = xr.DataArray(np.full(np.shape(Aww),np.nan), coords=[ds.time,ds.tauT,ds.Aw,ds.Lu], dims=['time','tauT','Aw','Lu'])
                # Loop over response times
                i = -1
                for tauT in ds.tauT.values:
                    j = -1
                    i = i + 1
                    for Aw in ds.Aw.values:
                        k = -1
                        j = j + 1
                        for Lu in ds.Lu.values:
                            k = k + 1 
                                 
                            # Roughness length for momentum using the EC method
                            ds['z0_EC'][:,i,j,k] = ds['z'] / (np.exp((utils.kappa * ds['u'] / ds['ustar'][:,i,j,k])  +  Psim(ds['zeta'][:,i,j,k])  ))           
                            
            else:
                ds['z0_EC']  = xr.DataArray(np.full(np.shape(Aww),np.nan), coords=[ds.time], dims=['time'])
                ds['z0_EC']  = ds['z'] / (np.exp((utils.kappa * ds['u'] / ds['ustar'])  +  Psim(ds['zeta'])  ))    
            
            # Remove spectra from dataset
            ds = ds.drop({'SnTT','Snuu','SnwT', \
                          'Snww','Snuw','freq_corr'})
    
            # Store old attribute
            att1                                    = ds.attrs['IceEddie_namelist_L0toL1A']
            att2                                    = ds.attrs['IceEddie_namelist_L1AtoL1B']
            att3                                    = ds.attrs['IceEddie_namelist_L1BtoL2']
            # Add new attributes
            ds                                      = utils.Add_dataset_attributes(ds,'iWS_EC_L3.JSON')
            ds.attrs['location']                    = LOC + '_' + ID
            ds.attrs['file_creation_date_time']     = str(datetime.datetime.now())
            ds.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
            ds.attrs['IceEddie_namelist_L0toL1A']   = att1
            ds.attrs['IceEddie_namelist_L1AtoL1B']  = att2
            ds.attrs['IceEddie_namelist_L1BtoL2']   = att3
            ds.attrs['IceEddie_namelist_L2toL3']    = str(nml['L2toL3'])
            
            # Export to net CDF
            print(L3dir + file.replace('L2','L3'))
            ds.to_netcdf(L3dir + file.replace('L2','L3'),mode = 'w',encoding={'time': \
                             {'units':'minutes since  ' + file[-9:-5] + '-' +  file[-5:-3] + '-01'}})   
     

    elif sensor == 'Young':
        
        # Process files sequentially
        for file in sorted(glob.glob('*' + sensor + '*L2*' + "*nc")):
            print(file)
            # Open net CDF
            ds = xr.open_dataset(file)
            ds['time'] = ds.time.values + np.timedelta64(toffset_min,'m')
            
            try:
                ds = ds.drop({'Gw','Gu','GT','GD','GwT','Guw',\
                    'Tww','Tuu','TTT','Tuw','TwT'\
                    ,'SnTT','Snuu','Snuw','SnwT','Snww'\
                    ,'Suu','Svv','Sww','Suw','Svw','ST1T1','SwT1','freq','freq_corr','TswT','Tsuw'})
            except:
                try:
                    ds = ds.drop({'freq','Suu','Svv','Sww','Suw','Svw','ST1T1','SwT1'})
                except:
                    pass

            # Store initial dataset
            ds_L2 = ds
            
            # Delete previously calculated Kaimal spectra 
            # ds = ds.drop({'freq_corr','SnTT','Snuu','Snuw','SnwT','Snww',\
            #          'TTT','Tww','Tuu','Tuw','TwT'})
               

            # Assign transfer function constants to new dimension
            if ldtau:
                ds['tauT'] = [tauT_nml,tauT_nml- dtauT,tauT_nml+dtauT]
                ds['Aw']   = [Aw_nml,Aw_nml - dAw,Aw_nml+dAw]
                ds['Lu']   = [Lu_nml,Lu_nml - dLu,Lu_nml+dLu]
                ds.set_coords(['tauT','Aw','Lu'])
                # Initialise array
                Aww    = np.full((len(ds.time),len(ds['tauT']),len(ds['Aw']),len(ds['Lu'])),np.nan)
                Auu    = np.full((len(ds.time),len(ds['tauT']),len(ds['Aw']),len(ds['Lu'])),np.nan)
                ATT    = np.full((len(ds.time),len(ds['tauT']),len(ds['Aw']),len(ds['Lu'])),np.nan)
                AwT    = np.full((len(ds.time),len(ds['tauT']),len(ds['Aw']),len(ds['Lu'])),np.nan)
                Auw    = np.full((len(ds.time),len(ds['tauT']),len(ds['Aw']),len(ds['Lu'])),np.nan)

            else:
                ds['tauT'] = tauT_nml
                ds['Aw']   = Aw_nml
                ds['Lu']   = Lu_nml
                ds.set_coords(['tauT','Aw','Lu'])
                # Initialise array
                Aww    = np.full((len(ds.time),1,1,1),np.nan)
                Auu    = np.full((len(ds.time),1,1,1),np.nan)
                ATT    = np.full((len(ds.time),1,1,1),np.nan)
                AwT    = np.full((len(ds.time),1,1,1),np.nan)
                Auw    = np.full((len(ds.time),1,1,1),np.nan)

                
                
            itera = 0
            
            # Iterate corrections
            while (itera<nitera):
                itera    = itera + 1
                
                print('Iteration '  +str(itera) + '/' + str(nitera))
                # Calculate Kaimal spectra
                ds = Kaimal(ds,nml,level = 'L3')
                
                if ldtau:
        
                    # Loop over response times
                    i = -1
                    for tauT in ds.tauT.values:
                        j = -1
                        i = i + 1
                        for Aw in ds.Aw.values:
                            k = -1
                            j = j + 1
                            for Lu in ds.Lu.values:
                                k = k + 1       
                                
                                # Calculate transfer functions
                                ds_tmp = transfer_functions(ds,nml, tauT = tauT, Aw = Aw, Lu = Lu,level = 'L3')
                            
                                # Calculate attenuation factors
                                Aww[:,i,j,k]  = np.trapz(ds.Snww * ds_tmp.Tww, ds.freq_corr) / np.trapz(ds.Snww, ds.freq_corr) 
                                Auu[:,i,j,k]  = np.trapz(ds.Snuu * ds_tmp.Tuu, ds.freq_corr) / np.trapz(ds.Snuu, ds.freq_corr) 
                                ATT[:,i,j,k]  = np.trapz(ds.SnTT * ds_tmp.TTT, ds.freq_corr) / np.trapz(ds.SnTT, ds.freq_corr) 
                                AwT[:,i,j,k]  = np.trapz(ds.SnwT * ds_tmp.TwT, ds.freq_corr) / np.trapz(ds.SnwT, ds.freq_corr) 
                                Auw[:,i,j,k]  = np.trapz(ds.Snuw * ds_tmp.Tuw, ds.freq_corr) / np.trapz(ds.Snuw, ds.freq_corr) 
                                
                                del ds_tmp
                            
                else:
                    # Calculate transfer functions
                    ds_tmp = transfer_functions(ds,nml, tauT = ds.tauT.values, Aw = ds.Aw.values, Lu = ds.Lu.values,level = 'L3')
                    
                    # Calculate attenuation factors after Moore
                    Aww  = np.trapz(ds.Snww * ds_tmp.Tww, ds.freq_corr) / np.trapz(ds.Snww, ds.freq_corr) 
                    Auu  = np.trapz(ds.Snuu * ds_tmp.Tuu, ds.freq_corr) / np.trapz(ds.Snuu, ds.freq_corr) 
                    ATT  = np.trapz(ds.SnTT * ds_tmp.TTT, ds.freq_corr) / np.trapz(ds.SnTT, ds.freq_corr) 
                    AwT  = np.trapz(ds.SnwT * ds_tmp.TwT, ds.freq_corr) / np.trapz(ds.SnwT, ds.freq_corr) 
                    Auw  = np.trapz(ds.Snuw * ds_tmp.Tuw, ds.freq_corr) / np.trapz(ds.Snuw, ds.freq_corr) 
                        

                     
                # Add corrections to dataset
                if ldtau:         
                    ds['Aww']  = xr.DataArray(Aww, coords=[ds.time,ds.tauT,ds.Aw,ds.Lu], dims=['time','tauT','Aw','Lu'])
                    ds['Auu']  = xr.DataArray(Auu, coords=[ds.time,ds.tauT,ds.Aw,ds.Lu], dims=['time','tauT','Aw','Lu'])
                    ds['ATT']  = xr.DataArray(ATT, coords=[ds.time,ds.tauT,ds.Aw,ds.Lu], dims=['time','tauT','Aw','Lu'])
                    ds['AwT']  = xr.DataArray(AwT, coords=[ds.time,ds.tauT,ds.Aw,ds.Lu], dims=['time','tauT','Aw','Lu'])
                    ds['Auw']  = xr.DataArray(Auw, coords=[ds.time,ds.tauT,ds.Aw,ds.Lu], dims=['time','tauT','Aw','Lu'])
                else:
                    ds['Aww']  = xr.DataArray(Aww, coords=[ds.time], dims=['time'])
                    ds['Auu']  = xr.DataArray(Auu, coords=[ds.time], dims=['time'])
                    ds['ATT']  = xr.DataArray(ATT, coords=[ds.time], dims=['time'])
                    ds['AwT']  = xr.DataArray(AwT, coords=[ds.time], dims=['time'])
                    ds['Auw']  = xr.DataArray(Auw, coords=[ds.time], dims=['time'])

                # Apply high-frequency correction
                ds['ww']    = ds['ww']  /ds['Aww']
                ds['wT1']   = ds['wT1'] /ds['AwT']
                ds['uw']    = ds['uw']  /ds['Auw']  
                ds['uu']    = ds['uu']  /ds['Auu']
                ds['T1T1']  = ds['T1T1']/ds['ATT']
        
                # Compute corrected friction velocity
                ds['ustar'] = (ds['uw']**2 + ds['vw']**2)**(1/4)
                # Compute corrected  Obukhov length
                ds['obh']   = -(ds['ustar']**3)*ds['T1']/(9.81*utils.kappa*ds['wT1'])
                # calculate corrected stability parameter 
                ds['zeta']  = ds['z']/ds['obh']
            
            
#            # apply final correction to L2 fluxes
            ds['ww']    = ds_L2['ww']  /ds['Aww']
            ds['uu']    = ds_L2['uu']  /ds['Auu']
            ds['T1T1']  = ds_L2['T1T1']/ds['ATT']
            ds['wT1']   = ds_L2['wT1'] /ds['AwT']
            ds['uw']    = ds_L2['uw']  /ds['Auw']     

            # Compute corrected friction velocity
            ds['ustar'] = (ds['uw']**2 + ds['vw']**2)**(1/4)
            
            # Compute corrected temperature scale
            ds['tstar'] = -np.divide(ds['wT1'],ds['ustar'])
            
            # Compute corrected  Obukhov length
            ds['obh']   = -(ds['ustar']**3)*ds['T1']/(9.81*utils.kappa*ds['wT1'])
            
            # calculate corrected stability parameter 
            ds['zeta']  = ds['z']/ds['obh']
                
            # Initialise roughness length
            if ldtau:
                ds['z0_EC']  = xr.DataArray(np.full(np.shape(Aww),np.nan), coords=[ds.time,ds.tauT,ds.Aw,ds.Lu], dims=['time','tauT','Aw','Lu'])
                # Loop over response times
                i = -1
                for tauT in ds.tauT.values:
                    j = -1
                    i = i + 1
                    for Aw in ds.Aw.values:
                        k = -1
                        j = j + 1
                        for Lu in ds.Lu.values:
                            k = k + 1 
                                 
                            # Roughness length for momentum using the EC method
                            ds['z0_EC'][:,i,j,k] = ds['z'] / (np.exp((utils.kappa * ds['u'] / ds['ustar'][:,i,j,k])  +  Psim(ds['zeta'][:,i,j,k])  ))           
                            
            else:
                ds['z0_EC']  = xr.DataArray(np.full(np.shape(Aww),np.nan), coords=[ds.time], dims=['time'])
                ds['z0_EC']  = ds['z'] / (np.exp((utils.kappa * ds['u'] / ds['ustar'])  +  Psim(ds['zeta'])  ))   
                
            # Remove spectra from dataset
            ds = ds.drop({'SnTT','Snuu','SnwT', \
                          'Snww','Snuw','freq_corr'})
    
        
            # Store old attribute
            att1                                    = ds.attrs['IceEddie_namelist_L0toL1A']
            att2                                    = ds.attrs['IceEddie_namelist_L1AtoL1B']
            att3                                    = ds.attrs['IceEddie_namelist_L1BtoL2']
            # Add new attributes
            ds                                      = utils.Add_dataset_attributes(ds,'YOUNG_EC_L3.JSON')
            ds.attrs['location']                    = LOC + '_' + ID
            ds.attrs['file_creation_date_time']     = str(datetime.datetime.now())
            ds.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
            ds.attrs['IceEddie_namelist_L0toL1A']   = att1
            ds.attrs['IceEddie_namelist_L1AtoL1B']  = att2
            ds.attrs['IceEddie_namelist_L1BtoL2']   = att3
            ds.attrs['IceEddie_namelist_L2toL3']    = str(nml['L2toL3'])
            
            # Export to net CDF
            print(L3dir + file.replace('L2','L3'))
            ds.to_netcdf(L3dir + file.replace('L2','L3'),mode = 'w',encoding={'time': \
                             {'units':'minutes since  ' + file[-9:-5] + '-' +  file[-5:-3] + '-01'}})          
   
    elif sensor == 'CSAT': 
        
        ds_out = xr.Dataset()
        fid = 0
        
        # Process files sequentially
        for file in sorted(glob.glob('*' + sensor + '*L2*' + "*nc")):
            print(file)
            # Open net CDF
            ds = xr.open_dataset(file)
            ds['time'] = ds.time.values + np.timedelta64(toffset_min,'m')
            if 'q' in list(ds.variables):
                if lconvhum: # After this step, q units should be in kg/kg !!!!
                    # ds['q'] = (18/1000)*ds['q']
                    # ds['wq'] =  (18/1000)* ds['wq']
                    # ds['uq'] =  (18/1000)* ds['uq']
                    # ds['vq'] =  (18/1000)* ds['vq']
                    # ds['qq'] = (18/1000)*(18/1000)* ds['qq']
                    ds['q'] = (1/1000)*ds['q']
                    ds['wq'] =  (1/1000)* ds['wq']
                    # ds['uq'] =  (1/1000)* ds['uq']
                    # ds['vq'] =  (1/1000)* ds['vq']
                    ds['qq'] = (1/1000)*(1/1000)* ds['qq']
            if lhumbiascorr:
                ds['wq'] =  humbiascorr* ds['wq']   
                ds['q'] =humbiascorr*ds['q']
            try:
                ds = ds.drop({'Tww','Tuu','TTsTs','Tuw','TwTs','TTcTc','TwTc'\
                 ,'SnTT','Snuu','Snuw','SnwT','Snww'\
                 ,'Suu','Svv','Sww','Suw','Svw','STsTs','SwTs','STcTc','SwTc','Swq','SwCO2','Sqq','SCO2O2','freq','freq_corr'})
            except:
                try:
                                        ds = ds.drop({'Tww','Tuu','TTT','Tuw','TwT'\
                     ,'SnTT','Snuu','Snuw','SnwT','Snww'\
                     ,'Suu','Svv','Sww','Suw','Svw','STsTs','SwTs','STcTc','SwTc','Sqq','Swq','freq','freq_corr'})
                except: 
                    try:
                        ds = ds.drop({'Tww','Tuu','TTT','Tuw','TwT'\
                         ,'SnTT','Snuu','Snuw','SnwT','Snww'\
                         ,'Suu','Svv','Sww','Suw','Svw','STsTs','SwTs','STcTc','SwTc','freq','freq_corr'})
                    except:
                        try:
                            ds = ds.drop({'Tww','Tuu','TTsTs','Tuw','TwTs','TTcTc','TwTc'\
                                ,'SnTT','Snuu','Snuw','SnwT','Snww'\
                                ,'Suu','Svv','Sww','Suw','Svw','STsTs','SwTs','STcTc','SwTc','freq','freq_corr'})
                        except:
                            try:
                                ds = ds.drop({'Suu','Svv','Sww','Suw','Svw','STsTs','SwTs','STcTc','SwTc','Swq','Sqq','freq'})
                            except:
                                try:
                                    ds = ds.drop({'Suu','Svv','Sww','Suw','Svw','STsTs','SwTs','STcTc','SwTc','freq'})
                                except:
                                    pass
                                
            # Go to folder where 30-min height data is stored
            if not lfixedheight:
                os.chdir(L1B_30min_dir) 
                file_30min = glob.glob("*WS*nc")
                # file_30min = glob.glob("*WS*" + file[ -9:-3] + "*nc") 
                ds_30min = xr.open_dataset(file_30min[0])
                ds_30min = ds_30min.resample(time=dT).interpolate("linear")
                
                # remove previous
                try:
                    ds = ds.drop({'zm','d','H','z'})
                except:
                    ds = ds.drop({'z'})
                # print(ds)
                # Add height to dataset
                ds['zm']   = ds_30min['zm']
                ds['d']    = ds_30min['d']
                ds['H']    = ds_30min['H']
                if lcorrH:
                    if lfixedheight:
                        ds['z']    = -ds_30min['snowheight']+zp
                    else:
                        ds['z']    = ds_30min['zm']+ds_30min['H']-ds_30min['snowheight']-ds_30min['d']+EC_hgt_offset
                else:
                    ds['z']    = ds_30min['zm']+ds_30min['H']-ds_30min['d']+EC_hgt_offset
                
                # Go back to L1B dir
                os.chdir(L2dir) 
            else: 
                if lcorrH:
                    ds['z']    = -ds_30min['snowheight']+zp
                else:
                    ds['z']  = xr.DataArray(np.full(len(ds.time),float(zp)),dims=['time'])


                
            # Correction of sonic temperature for side wind
            if lsidewind:
                if lsidewind_method == 'Schotanus1983':
                    print('Sidewind correction after ' + lsidewind_method)
                    ds['wTs'] = ds.wTs + 2*ds.u*ds.T_couple*ds.uw / (340**2)
                if lsidewind_method == 'Liu2001':
                    print('Sidewind correction after ' + lsidewind_method)
                    print('Liu2001 not implemented')
            
            # Initialise array
            Aww   = np.full(len(ds.time),np.nan)
            Auu   = np.full(len(ds.time),np.nan)
            ATsTs = np.full(len(ds.time),np.nan)
            ATcTc = np.full(len(ds.time),np.nan)
            AwTs  = np.full(len(ds.time),np.nan)
            AwTc  = np.full(len(ds.time),np.nan)
            Auw   = np.full(len(ds.time),np.nan)

            # Calculate Kaimal spectra
            ds = Kaimal(ds,nml,level = 'L3')
                              
            # Scotanus correction for sonic temperature
            if lsnd == True:
                print('SND correction using ' + lsnd_method + ' data')
                
                if lsnd_method == 'CESAR':
                    # Find, load and downsample corresponding CESAR files
                   os.chdir(CESAR_dir + 'Surface_meteorology/LC1/') 
                   file_CESAR_meteo = glob.glob("*cesar_surface_meteo*" + file[ -9:-3] + "*nc") 
                   ds_CESAR_meteo   = xr.open_dataset(file_CESAR_meteo[0])
                   ds_CESAR_meteo   = ds_CESAR_meteo.resample(time='30min').mean()
                   
                   os.chdir(CESAR_dir + 'Surface_fluxes/LC1/') 
                   file_CESAR_flux  = glob.glob("*cesar_surface_flux*"  + file[ -9:-3] + "*nc") 
                   ds_CESAR_flux    = xr.open_dataset(file_CESAR_flux[0])
                   ds_CESAR_flux    = ds_CESAR_flux.resample(time='30min').mean()
                   
                   # Go back to L1B dir
                   os.chdir(L2dir) 
                   # Air density 
                   rho_a = utils.rho_air(ds.T_couple,ds_CESAR_meteo.P0*100,ds_CESAR_meteo.Q002/1e3)
                   
                   # Latent  heat flux
                   LE = ds_CESAR_flux.LE
                   q = ds_CESAR_flux.q
                   
                   # Humidity covariance
                   ds['wq'] = LE / (rho_a * utils.Lv())
                   
                   # Correction
                   ds['wTs']  = ds.wTs - 0.51 * ds.T_couple * ds.wq
                   ds['T_sonic']  = ds.T_sonic / (1 +0.51*q)
                   
                if lsnd_method == 'L1B': 
                    os.chdir(L1B_30min_dir) 
                    file_L1B_30min = glob.glob("*L1B*nc") 
                    # file_L1B_30min = glob.glob("*L1B*"  + file[ -9:-3] + "*nc") 
                    ds_L1B_30min    = xr.open_dataset(file_L1B_30min[0])
                    ds_L1B_30min['z'] = ds['z']
                    ds_L1B_30min['obl'] = ds['obh']
                    ds_L1B_30min['z0m'] = xr.DataArray(np.full(len(ds_L1B_30min.time),1e-3),dims=['time']) 
                    
                    ds_L1B_30min = ds_L1B_30min.resample(time=dT).interpolate("linear")
                    
                    Tv0 = (ds_L1B_30min['T0'])*(1+0.608*ds_L1B_30min['qv0'])
                    
                    # Surface pressure
                    ps = ds_L1B_30min['p0']/np.exp(-utils.grav*ds_L1B_30min['z']/(utils.rd*Tv0))
                    
                    # Surface potential temperature
                    ds_L1B_30min['ths'] = utils.T2thl(ds_L1B_30min['Ts'],ps,ql=0)
                    
                    # Surface humidity assuming saturation
                    ds_L1B_30min['qvs'] = utils.RH2qv(100,ps, ds_L1B_30min['Ts'])
                    
                    # Air density and heat capacity
                    ds_L1B_30min['rho_a'] = utils.rho_air(ds_L1B_30min.T0,ds_L1B_30min.p0,ds_L1B_30min.qv0)
        
                    os.chdir(L2dir) 
                                      
                    # Calculate bulk latent  heat flux
                    # ds_L1B_30min['z0'] = ['1mm']
                    # ds_L1B_30min.set_coords(['z0'])
                    # ds_L1B_30min['z0m'] = xr.DataArray(np.full((len(ds_L1B_30min.time),1),np.nan),dims=['time','z0']) 
                    # ds_L1B_30min.z0m[:,0]   = xr.DataArray(np.full(len(ds_L1B_30min.time),1.0e-3),dims=['time'])
                    bulk.bulkflux(ds_L1B_30min,SvdB=True,getOBL=False)

                    # Humidity covariance
                    ds['wq_bulk'] = ds_L1B_30min.wq
                    ds['qv0'] = ds_L1B_30min.qv0
                    
                    # Interpolate missing data
                    ds['wq_bulk']  = ds.wq_bulk.interpolate_na(dim="time", method="linear")
                    ds['qv0']  = ds.qv0.interpolate_na(dim="time", method="linear")
                    
                    # Correction
                    ds['wTs']  = ds.wTs - 0.51 * ds.T_sonic * ds.wq_bulk - 0.51 * ds.qv0 * ds.wTs
                    ds['T_sonic']  = ds.T_sonic / (1 +0.51*ds.qv0)

                    # Convert humidity concenttation to specific humidity 
                    rho_a = utils.rho_air(ds_L1B_30min['T0'],ds_L1B_30min['p0'],ds_L1B_30min['qv0'])
                    ds['q'] = (ds.q) / rho_a
                if lsnd_method == 'L2_AWS': 
                    os.chdir(L2_30min_dir) 
                    file_L2_30min = glob.glob("*L2*"  + file[ -9:-3] + "*nc") 
                    ds_L2_30min    = xr.open_dataset(file_L2_30min[0])
                    ds_L2_30min = ds_L2_30min.resample(time=dT).interpolate("linear")
                    
                    os.chdir(L2dir) 
                    
                    ds_L2_30min['z'] = ds['z']
                    ds_L2_30min['obl'] = ds['obh']
                    
                    bulk.bulkflux(ds_L2_30min,SvdB=True,getOBL=False)

                    # Humidity covariance
                    ds['wq_bulk'] = ds_L2_30min.wq[:,1]
                    ds['qv0'] = ds_L2_30min.qv0
                    
                    # Interpolate missing data
                    ds['wq_bulk']  = ds.wq_bulk.interpolate_na(dim="time", method="linear")
                    ds['qv0']  = ds.qv0.interpolate_na(dim="time", method="linear")
                    
                    # Correction
                    ds['wTs']  = ds.wTs - 0.51 * ds.T_sonic * ds.wq_bulk - 0.51 * ds.qv0 * ds.wTs
                    ds['T_sonic']  = ds.T_sonic / (1 +0.51*ds.qv0) 
                    
                if lsnd_method == 'L2': 
                    ## CAREFUL, q and wq should be in kg/kg and m/s !! 
                    rho_d = utils.rho_air_dry(ds.T_sonic,1e5)
                    # Correction
                    ds['wTs']      = ds.wTs - 0.51 * ds.T_sonic * (ds.wq) - 0.51 * (ds.q/rho_d) * ds.wTs
                    ds['T_sonic']  = ds.T_sonic / (1 +0.51*(ds.q))
                    
                if lsnd_method == 'L2_bulk': 
                    # Prepare variable for bulk flux calculation
                    ds['T0'] = ds['T_couple']
                    ds['p0'] = ds['p0']*100
                    ds['U'] = ds['u']
                    ds['wq_EC'] = ds['wq']
                    ds['qv0'] = ds['q']
                    Tv0 = (ds['T0'])*(1+0.608*ds['qv0'])
                    ps = ds['p0']/np.exp(-utils.grav*ds['z']/(utils.rd*Tv0))
                    ds['th0'] = utils.T2thl(ds['T0'],ds['p0'],ql=0)
                    ds['ths'] = utils.T2thl(ds['Ts'],ps,ql=0)
                    ds['qvs'] = utils.RH2qv(100,ps, ds['Ts'])
                    ds['rho_a'] = utils.rho_air(ds.T0,ds.p0,ds.qv0)
                    ds['Cp']    = utils.Cp(ds.qv0)
                    ds['obl'] = ds['obh']
                    ds['z0m'] = xr.DataArray(np.full(len(ds.time),1e-3),dims=['time']) 
                    bulk.bulkflux(ds,SvdB=True,getOBL=False)

                    # Humidity covariance
                    ds['wq_bulk'] = ds.wq
                    
                    # Interpolate missing data
                    ds['wq_bulk']  = ds.wq_bulk.interpolate_na(dim="time", method="linear")
                    
                    # Correction
                    ds['wTs']  = ds.wTs - 0.51 * ds.T_sonic * ds.wq_bulk - 0.51 * ds.qv0 * ds.wTs
                    ds['T_sonic']  = ds.T_sonic / (1 +0.51*ds.qv0) 
                    
            # WPL correction for density fluctuations
            if lwpl == True:
                if lwpl_method == 'L2':
                    ## CAREFUL, q and wq should be in kg/kg and m/s !! 
                    nu = 1.609 # m_a / m_v
                    rho_d = utils.rho_air_dry(ds.T_sonic,1e5)
                    sigma =   (ds.q)
                    print('WPL correction using ' + lwpl_method + ' data')
                    ds['wq']   = (1+nu*sigma) * ((ds.wq) + (ds.q) * ds.wTs / ds.T_sonic)
                    if 'wCO2' in list(ds.keys()):
                        ds['wCO2'] = (ds.wCO2) + nu*((ds.CO2) / rho_d) * (ds.wCO2) + (1+nu*sigma)*(ds.CO2)*ds.wTs / ds.T_sonic
                
            # Calculate transfer functions
            ds_tmp = transfer_functions(ds,nml, tauT = nml['L2toL3']['tauT'], Aw = 2, Lu = 2,level = 'L3')
        
            # Calculate attenuation factors
            Aww   = np.trapz(ds.Snww * ds_tmp.Tww, ds.freq_corr)   / np.trapz(ds.Snww, ds.freq_corr) 
            Auu   = np.trapz(ds.Snuu * ds_tmp.Tuu, ds.freq_corr)   / np.trapz(ds.Snuu, ds.freq_corr) 
            ATsTs = np.trapz(ds.SnTT * ds_tmp.TTsTs, ds.freq_corr) / np.trapz(ds.SnTT, ds.freq_corr) 
            ATcTc = np.trapz(ds.SnTT * ds_tmp.TTcTc, ds.freq_corr) / np.trapz(ds.SnTT, ds.freq_corr) 
            AwTs  = np.trapz(ds.SnwT * ds_tmp.TwTs, ds.freq_corr)  / np.trapz(ds.SnwT, ds.freq_corr) 
            AwTc  = np.trapz(ds.SnwT * ds_tmp.TwTc, ds.freq_corr)  / np.trapz(ds.SnwT, ds.freq_corr) 
            Auw   = np.trapz(ds.Snuw * ds_tmp.Tuw, ds.freq_corr)   / np.trapz(ds.Snuw, ds.freq_corr) 
            if 'wq' in list(ds.variables):
                Awq  = np.trapz(ds.SnwT * ds_tmp.Twq, ds.freq_corr)  / np.trapz(ds.SnwT, ds.freq_corr) 
                ds['Awq']  = xr.DataArray(Awq, coords=[ds.time], dims=['time'])


            # Add corrections to dataset
            ds['Aww']   = xr.DataArray(Aww, coords=[ds.time],  dims=['time'])
            ds['Auu']   = xr.DataArray(Auu, coords=[ds.time],  dims=['time'])
            ds['ATsTs'] = xr.DataArray(ATsTs, coords=[ds.time],dims=['time'])
            ds['ATcTc'] = xr.DataArray(ATcTc, coords=[ds.time],dims=['time'])
            ds['AwTs']  = xr.DataArray(AwTs, coords=[ds.time], dims=['time'])
            ds['AwTc']  = xr.DataArray(AwTc, coords=[ds.time], dims=['time'])
            ds['Auw']   = xr.DataArray(Auw, coords=[ds.time],  dims=['time'])

            # Apply high-frequency correction
            ds['ww']    = ds['ww']  /ds['Aww']
            ds['uu']    = ds['uu']  /ds['Auu']
            ds['TsTs']  = ds['TsTs']/ds['ATsTs']
            ds['TcTc']  = ds['TcTc']/ds['ATcTc']
            ds['wTs']   = ds['wTs'] /ds['AwTs']
            ds['wTc']   = ds['wTc'] /ds['AwTc']
            ds['uw']    = ds['uw']  /ds['Auw']
            
            if 'wq' in list(ds.variables):
                ds['wq']    = ds['wq']  /ds['Awq']

            if lshadowing:
                # correction according to Horst2015 when NO licor is present
                corr_uw = ((((-np.cos((ds['WD']-WDoffset)*(np.pi/180)*3*1))+1)/2*0.25+0.83))
                corr_wTs = 0.96
                corr_sigmaw = 0.965
                corr_sigmau = 0.985
                corr_u = 0.98
                ds['uw']   = ds['uw'] / corr_uw
                ds['wTs']   = ds['wTs'] / corr_wTs
                ds['ww']   = ds['ww'] / (corr_sigmaw)**0.5
                ds['uu']   = ds['uu'] / (corr_sigmau)**0.5
                ds['u']   = ds['u'] / corr_u
                
            # Compute corrected friction velocity
            ds['ustar'] = (ds['uw']**2 + ds['vw']**2)**(1/4)
            
            # Compute corrected temperature scale
            ds['tstar']   = -np.divide(ds['wTs'],ds['ustar'])
            ds['tstar_c'] = -np.divide(ds['wTc'],ds['ustar'])
            
            if 'wq' in list(ds.variables):
                ds['qstar'] = -np.divide(ds['wq'],ds['ustar'])
                    
            # Compute corrected  Obukhov length
            ds['obh']   = -(ds['ustar']**3)*ds['T_sonic']/(9.81*utils.kappa*ds['wTs'])
            
            # Calculate stability parameter
            ds['zeta']  = ds['z']/ds['obh']
            
            # Calculate roughness length for momentum using the EC method
            ds['z0_EC'] = ds['z']/(np.exp((utils.kappa*ds['u']/ds['ustar'])  +  Psim(ds['zeta'])  ))  
            
            # Remove spectra from dataset
            ds = ds.drop({'SnTT','Snuu','SnwT', \
                          'Snww','Snuw','freq_corr'})

            # Store old attribute
            try:
                att1                                    = ds.attrs['IceEddie_namelist_L0toL1A']
                att2                                    = ds.attrs['IceEddie_namelist_L1AtoL1B']
                att3                                    = ds.attrs['IceEddie_namelist_L1BtoL2']
                # Add new attributes
                ds                                      = utils.Add_dataset_attributes(ds,'CSAT_EC_L3.JSON')
                ds.attrs['location']                    = LOC + '_' + ID
                ds.attrs['file_creation_date_time']     = str(datetime.datetime.now())
                ds.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
                ds.attrs['IceEddie_namelist_L0toL1A']   = att1
                ds.attrs['IceEddie_namelist_L1AtoL1B']  = att2
                ds.attrs['IceEddie_namelist_L1BtoL2']   = att3
                ds.attrs['IceEddie_namelist_L2toL3']    = str(nml['L2toL3'])
            except:
                print('No old attributes found')
                ds                                      = utils.Add_dataset_attributes(ds,'CSAT_EC_L3.JSON')
                ds.attrs['location']                    = LOC + '_' + ID
                ds.attrs['file_creation_date_time']     = str(datetime.datetime.now())
                ds.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
                ds.attrs['IceEddie_namelist_L2toL3']    = str(nml['L2toL3'])
                
            if fid == 0:
                ds_out = ds
            else:
                ds_out = xr.concat([ds_out,ds], "time")
            fid = fid + 1
            
        # Export to net CDF
        file_out = L3dir + LOC + '_' + ID  + '_' + sensor + '_' + input_type + '_' + "L3"  + '_' + version +  '_' + 'all' + ".nc"
        print(file_out)
        ds_out.to_netcdf(file_out,mode = 'w')
            
            
    elif sensor == 'CSAT_Young': 
        
        ds_out = xr.Dataset()
        fid = 0
        
        # Process files sequentially
        for file in sorted(glob.glob('*' + sensor + '*L2*' + "*nc")):
            print(file)
            # Open net CDF
            ds = xr.open_dataset(file)
            ds['time'] = ds.time.values + np.timedelta64(toffset_min,'m')
            if 'q' in list(ds.variables):
                if lconvhum: # After this step, q units should be in kg/kg !!!!
                    # ds['q'] = (18/1000)*ds['q']
                    # ds['wq'] =  (18/1000)* ds['wq']
                    # ds['uq'] =  (18/1000)* ds['uq']
                    # ds['vq'] =  (18/1000)* ds['vq']
                    # ds['qq'] = (18/1000)*(18/1000)* ds['qq']
                    ds['q'] = (1/1000)*ds['q']
                    ds['wq'] =  (1/1000)* ds['wq']
                    # ds['uq'] =  (1/1000)* ds['uq']
                    # ds['vq'] =  (1/1000)* ds['vq']
                    ds['qq'] = (1/1000)*(1/1000)* ds['qq']
            ds = ds.drop_dims(['freq','freq_corr'],errors='ignore') 

                                
            # Go to folder where 30-min height data is stored
            if not lfixedheight:
                os.chdir(L1B_30min_dir) 
                file_30min = glob.glob("*WS*nc")
                # file_30min = glob.glob("*WS*" + file[ -9:-3] + "*nc") 
                ds_30min = xr.open_dataset(file_30min[0])
                ds_30min = ds_30min.resample(time=dT).interpolate("linear")
                
                # remove previous
                try:
                    ds = ds.drop({'zm','d','H','z'})
                except:
                    ds = ds.drop({'z'})
                # print(ds)
                # Add height to dataset
                ds['zm']   = ds_30min['zm']
                ds['d']    = ds_30min['d']
                ds['H']    = ds_30min['H']
                if lcorrH:
                    ds['z']    = -ds_30min['snowheight']+zp
                    ds['zy']   = -ds_30min['snowheight']+zpy
                    # if lfixedheight:
                    #     ds['z']    = -ds_30min['snowheight']+zp
                    #     ds['zy']    = -ds_30min['snowheight']+zpy
                    # else:
                    #     ds['z']    = ds_30min['zm']+ds_30min['H']-ds_30min['snowheight']-ds_30min['d']+EC_hgt_offset
                else:
                    ds['z']    = ds_30min['zm']+ds_30min['H']-ds_30min['d']+EC_hgt_offset
                
                # Go back to L1B dir
                os.chdir(L2dir)  

                
            # Correction of sonic temperature for side wind
            if lsidewind:
                if lsidewind_method == 'Schotanus1983':
                    print('Sidewind correction after ' + lsidewind_method)
                    ds['wTs'] = ds.wTs + 2*ds.u*ds.T_couple*ds.uw / (340**2)
                if lsidewind_method == 'Liu2001':
                    print('Sidewind correction after ' + lsidewind_method)
                    print('Liu2001 not implemented')
            
            # Initialise array
            Aww   = np.full(len(ds.time),np.nan)
            Auu   = np.full(len(ds.time),np.nan)
            ATsTs = np.full(len(ds.time),np.nan)
            ATcTc = np.full(len(ds.time),np.nan)
            AwTs  = np.full(len(ds.time),np.nan)
            AwTc  = np.full(len(ds.time),np.nan)
            Auw   = np.full(len(ds.time),np.nan)

            # Assign transfer function constants to new dimension
            if ldtau:
                ds['tauT'] = [tauT_nml,tauT_nml- dtauT,tauT_nml+dtauT]
                ds['Aw']   = [Aw_nml,Aw_nml - dAw,Aw_nml+dAw]
                ds['Lu']   = [Lu_nml,Lu_nml - dLu,Lu_nml+dLu]
                ds.set_coords(['tauT','Aw','Lu'])
                # Initialise array
                Awgwg  = np.full((len(ds.time),len(ds['tauT']),len(ds['Aw']),len(ds['Lu'])),np.nan)
                Auyuy  = np.full((len(ds.time),len(ds['tauT']),len(ds['Aw']),len(ds['Lu'])),np.nan)
                AwgTc  = np.full((len(ds.time),len(ds['tauT']),len(ds['Aw']),len(ds['Lu'])),np.nan)
                Auywg  = np.full((len(ds.time),len(ds['tauT']),len(ds['Aw']),len(ds['Lu'])),np.nan)

            else:
                ds['tauT'] = tauT_nml
                ds['Aw']   = Aw_nml
                ds['Lu']   = Lu_nml
                ds.set_coords(['tauT','Aw','Lu'])
                # Initialise array
                Awgwg    = np.full((len(ds.time),1,1,1),np.nan)
                Auyuy    = np.full((len(ds.time),1,1,1),np.nan)
                Auywg    = np.full((len(ds.time),1,1,1),np.nan)
                Auywg    = np.full((len(ds.time),1,1,1),np.nan)

            
            # Calculate Kaimal spectra
            ds = Kaimal(ds,nml,level = 'L3')
                              
            # Scotanus correction for sonic temperature
            if lsnd == True:
                print('SND correction using ' + lsnd_method + ' data')
                
                if lsnd_method == 'CESAR':
                    # Find, load and downsample corresponding CESAR files
                   os.chdir(CESAR_dir + 'Surface_meteorology/LC1/') 
                   file_CESAR_meteo = glob.glob("*cesar_surface_meteo*" + file[ -9:-3] + "*nc") 
                   ds_CESAR_meteo   = xr.open_dataset(file_CESAR_meteo[0])
                   ds_CESAR_meteo   = ds_CESAR_meteo.resample(time="'30min'").mean()
                   
                   os.chdir(CESAR_dir + 'Surface_fluxes/LC1/') 
                   file_CESAR_flux  = glob.glob("*cesar_surface_flux*"  + file[ -9:-3] + "*nc") 
                   ds_CESAR_flux    = xr.open_dataset(file_CESAR_flux[0])
                   ds_CESAR_flux    = ds_CESAR_flux.resample(time='30min').mean()
                   
                   # Go back to L1B dir
                   os.chdir(L2dir) 
                   # Air density 
                   rho_a = utils.rho_air(ds.T_couple,ds_CESAR_meteo.P0*100,ds_CESAR_meteo.Q002/1e3)
                   
                   # Latent  heat flux
                   LE = ds_CESAR_flux.LE
                   q = ds_CESAR_flux.q
                   
                   # Humidity covariance
                   ds['wq'] = LE / (rho_a * utils.Lv())
                   
                   # Correction
                   ds['wTs']  = ds.wTs - 0.51 * ds.T_couple * ds.wq
                   ds['T_sonic']  = ds.T_sonic / (1 +0.51*q)
                   
                if lsnd_method == 'L1B': 
                    os.chdir(L1B_30min_dir) 
                    file_L1B_30min = glob.glob("*L1B*nc") 
                    # file_L1B_30min = glob.glob("*L1B*"  + file[ -9:-3] + "*nc") 
                    ds_L1B_30min    = xr.open_dataset(file_L1B_30min[0])
                    ds_L1B_30min['z'] = ds['z']
                    ds_L1B_30min['obl'] = ds['obh']
                    ds_L1B_30min['z0m'] = xr.DataArray(np.full(len(ds_L1B_30min.time),1e-3),dims=['time']) 
                    
                    ds_L1B_30min = ds_L1B_30min.resample(time=dT).interpolate("linear")
                    
                    Tv0 = (ds_L1B_30min['T0'])*(1+0.608*ds_L1B_30min['qv0'])
                    
                    # Surface pressure
                    ps = ds_L1B_30min['p0']/np.exp(-utils.grav*ds_L1B_30min['z']/(utils.rd*Tv0))
                    
                    # Surface potential temperature
                    ds_L1B_30min['ths'] = utils.T2thl(ds_L1B_30min['Ts'],ps,ql=0)
                    
                    # Surface humidity assuming saturation
                    ds_L1B_30min['qvs'] = utils.RH2qv(100,ps, ds_L1B_30min['Ts'])
                    
                    # Air density and heat capacity
                    ds_L1B_30min['rho_a'] = utils.rho_air(ds_L1B_30min.T0,ds_L1B_30min.p0,ds_L1B_30min.qv0)
        
                    os.chdir(L2dir) 
                                      
                    # Calculate bulk latent  heat flux
                    # ds_L1B_30min['z0'] = ['1mm']
                    # ds_L1B_30min.set_coords(['z0'])
                    # ds_L1B_30min['z0m'] = xr.DataArray(np.full((len(ds_L1B_30min.time),1),np.nan),dims=['time','z0']) 
                    # ds_L1B_30min.z0m[:,0]   = xr.DataArray(np.full(len(ds_L1B_30min.time),1.0e-3),dims=['time'])
                    bulk.bulkflux(ds_L1B_30min,SvdB=True,getOBL=False)

                    # Humidity covariance
                    ds['wq_bulk'] = ds_L1B_30min.wq
                    ds['qv0'] = ds_L1B_30min.qv0
                    
                    # Interpolate missing data
                    ds['wq_bulk']  = ds.wq_bulk.interpolate_na(dim="time", method="linear")
                    ds['qv0']  = ds.qv0.interpolate_na(dim="time", method="linear")
                    
                    # Correction
                    ds['wTs']  = ds.wTs - 0.51 * ds.T_sonic * ds.wq_bulk - 0.51 * ds.qv0 * ds.wTs
                    ds['T_sonic']  = ds.T_sonic / (1 +0.51*ds.qv0)

                    # Convert humidity concenttation to specific humidity 
                    rho_a = utils.rho_air(ds_L1B_30min['T0'],ds_L1B_30min['p0'],ds_L1B_30min['qv0'])
                    ds['q'] = (ds.q) / rho_a
                            
                if lsnd_method == 'L2_AWS': 
                    os.chdir(L2_30min_dir) 
                    file_L2_30min = glob.glob("*L2*"  + file[ -9:-3] + "*nc") 
                    ds_L2_30min    = xr.open_dataset(file_L2_30min[0])
                    ds_L2_30min = ds_L2_30min.resample(time=dT).interpolate("linear")
                    
                    os.chdir(L2dir) 
                    
                    ds_L2_30min['z'] = ds['z']
                    ds_L2_30min['obl'] = ds['obh']
                    
                    bulk.bulkflux(ds_L2_30min,SvdB=True,getOBL=False)

                    # Humidity covariance
                    ds['wq_bulk'] = ds_L2_30min.wq[:,1]
                    ds['qv0'] = ds_L2_30min.qv0
                    
                    # Interpolate missing data
                    ds['wq_bulk']  = ds.wq_bulk.interpolate_na(dim="time", method="linear")
                    ds['qv0']  = ds.qv0.interpolate_na(dim="time", method="linear")
                    
                    # Correction
                    ds['wTs']  = ds.wTs - 0.51 * ds.T_sonic * ds.wq_bulk - 0.51 * ds.qv0 * ds.wTs
                    ds['T_sonic']  = ds.T_sonic / (1 +0.51*ds.qv0) 
                    
                if lsnd_method == 'L2': 
                    ## CAREFUL, q and wq should be in kg/kg and m/s !!
                    
                    ds['q'] = ds.q
                    # Correction
                    ds['wTs']      = ds.wTs - 0.51 * ds.T_sonic * (ds.wq) - 0.51 * (ds.q) * ds.wTs
                    ds['T_sonic']  = ds.T_sonic / (1 +0.51*(ds.q))
                    
                if lsnd_method == 'L2_bulk': 
                    # Prepare variable for bulk flux calculation
                    ds['T0'] = ds['T_couple']
                    ds['p0'] = ds['p0']*100
                    ds['U'] = ds['u']
                    ds['wq_EC'] = ds['wq']
                    ds['qv0'] = ds['q']
                    Tv0 = (ds['T0'])*(1+0.608*ds['qv0'])
                    ps = ds['p0']/np.exp(-utils.grav*ds['z']/(utils.rd*Tv0))
                    ds['th0'] = utils.T2thl(ds['T0'],ds['p0'],ql=0)
                    ds['ths'] = utils.T2thl(ds['Ts'],ps,ql=0)
                    ds['qvs'] = utils.RH2qv(100,ps, ds['Ts'])
                    ds['rho_a'] = utils.rho_air(ds.T0,ds.p0,ds.qv0)
                    ds['Cp']    = utils.Cp(ds.qv0)
                    ds['obl'] = ds['obh']
                    ds['z0m'] = xr.DataArray(np.full(len(ds.time),1e-3),dims=['time']) 
                    bulk.bulkflux(ds,SvdB=True,getOBL=False)

                    # Humidity covariance
                    ds['wq_bulk'] = ds.wq
                    
                    # Interpolate missing data
                    ds['wq_bulk']  = ds.wq_bulk.interpolate_na(dim="time", method="linear")
                    
                    # Correction
                    ds['wTs']  = ds.wTs - 0.51 * ds.T_sonic * ds.wq_bulk - 0.51 * ds.qv0 * ds.wTs
                    ds['T_sonic']  = ds.T_sonic / (1 +0.51*ds.qv0) 
                    
            # WPL correction for density fluctuations
            if lwpl == True:
                if lwpl_method == 'L2':
                    nu = 1.609 # m_a / m_v
                    rho_d = utils.rho_air_dry(ds.T_sonic,1e5)
                    sigma = (ds.q)
                    print('WPL correction using ' + lwpl_method + ' data')
                    ds['wq']   = (1+nu*sigma) * ((ds.wq) + (ds.q) * ds.wTs / ds.T_sonic)
                    if 'wCO2' in list(ds.keys()):
                        ds['wCO2'] = (ds.wCO2) + nu*((ds.CO2) / rho_d) * (ds.wCO2) + (1+nu*sigma)*(ds.CO2)*ds.wTs / ds.T_sonic
            


            # Calculate transfer functions
            ds_tmp = transfer_functions(ds,nml, tauT = nml['L2toL3']['tauT'], Aw = ds.Aw.values, Lu = ds.Lu.values,level = 'L3')
        
            # Calculate attenuation factors
            Aww   = np.trapz(ds.Snww * ds_tmp.Tww, ds.freq_corr)   / np.trapz(ds.Snww, ds.freq_corr) 
            Auu   = np.trapz(ds.Snuu * ds_tmp.Tuu, ds.freq_corr)   / np.trapz(ds.Snuu, ds.freq_corr) 
            ATsTs = np.trapz(ds.SnTT * ds_tmp.TTsTs, ds.freq_corr) / np.trapz(ds.SnTT, ds.freq_corr) 
            ATcTc = np.trapz(ds.SnTT * ds_tmp.TTcTc, ds.freq_corr) / np.trapz(ds.SnTT, ds.freq_corr) 
            AwTs  = np.trapz(ds.SnwT * ds_tmp.TwTs, ds.freq_corr)  / np.trapz(ds.SnwT, ds.freq_corr) 
            AwTc  = np.trapz(ds.SnwT * ds_tmp.TwTc, ds.freq_corr)  / np.trapz(ds.SnwT, ds.freq_corr) 
            Auw   = np.trapz(ds.Snuw * ds_tmp.Tuw, ds.freq_corr)   / np.trapz(ds.Snuw, ds.freq_corr) 

            if 'wq' in list(ds.variables):
                Awq  = np.trapz(ds.SnwT * ds_tmp.Twq, ds.freq_corr)  / np.trapz(ds.SnwT, ds.freq_corr) 
                ds['Awq']  = xr.DataArray(Awq, coords=[ds.time], dims=['time'])

            # Add corrections to dataset
            ds['Aww']   = xr.DataArray(Aww, coords=[ds.time],  dims=['time'])
            ds['Auu']   = xr.DataArray(Auu, coords=[ds.time],  dims=['time'])
            ds['ATsTs'] = xr.DataArray(ATsTs, coords=[ds.time],dims=['time'])
            ds['ATcTc'] = xr.DataArray(ATcTc, coords=[ds.time],dims=['time'])
            ds['AwTs']  = xr.DataArray(AwTs, coords=[ds.time], dims=['time'])
            ds['AwTc']  = xr.DataArray(AwTc, coords=[ds.time], dims=['time'])
            ds['Auw']   = xr.DataArray(Auw, coords=[ds.time],  dims=['time'])

            # do the same for VPEC data
            del ds_tmp
            ds = ds.drop_dims(['freq','freq_corr'],errors='ignore') 
            ds = Kaimal(ds,nml,level = 'L3',height='zy')
            ds_tmp = transfer_functions(ds,nml, tauT = nml['L2toL3']['tauT'], Aw = ds.Aw.values, Lu = ds.Lu.values,level = 'L3')

            Awgwg   = np.trapz(ds.Snww * ds_tmp.Twgwg, ds.freq_corr)   / np.trapz(ds.Snww, ds.freq_corr) 
            Auyuy   = np.trapz(ds.Snuu * ds_tmp.Tuyuy, ds.freq_corr)   / np.trapz(ds.Snuu, ds.freq_corr) 
            AwgTc   = np.trapz(ds.SnwT * ds_tmp.TwgTc, ds.freq_corr)   / np.trapz(ds.SnwT, ds.freq_corr) 
            Auywg   = np.trapz(ds.Snuw * ds_tmp.Tuywg, ds.freq_corr)   / np.trapz(ds.Snuw, ds.freq_corr) 

            ds['Awgwg'] = xr.DataArray(Awgwg, coords=[ds.time],dims=['time'])
            ds['Auyuy'] = xr.DataArray(Auyuy, coords=[ds.time], dims=['time'])
            ds['AwgTc'] = xr.DataArray(AwgTc, coords=[ds.time], dims=['time'])
            ds['Auywg'] = xr.DataArray(Auywg, coords=[ds.time],  dims=['time'])

            # Apply high-frequency correction
            ds['ww']    = ds['ww']  /ds['Aww']
            ds['uu']    = ds['uu']  /ds['Auu']
            ds['TsTs']  = ds['TsTs']/ds['ATsTs']
            ds['TcTc']  = ds['TcTc']/ds['ATcTc']
            ds['wTs']   = ds['wTs'] /ds['AwTs']
            ds['wTc']   = ds['wTc'] /ds['AwTc']
            ds['uw']    = ds['uw']  /ds['Auw']

            ds['wgwg']  = ds['wgwg']  /ds['Awgwg']
            ds['uyuy']  = ds['uyuy']  /ds['Auyuy']
            ds['T2T2']  = ds['T2T2']/ds['ATcTc']
            ds['wgT2']  = ds['wgT2'] /ds['AwgTc']
            ds['uywg']  = ds['uywg']  /ds['Auywg']

            if 'wq' in list(ds.variables):
                ds['wq']    = ds['wq']  /ds['Awq']

            if lshadowing:
                # correction according to Horst2015 when NO licor is present
                corr_uw = ((((-np.cos((ds['WD']-WDoffset)*(np.pi/180)*3*1))+1)/2*0.25+0.83))
                corr_wTs = 0.96
                corr_sigmaw = 0.965
                corr_sigmau = 0.985
                corr_u = 0.98
                ds['uw']   = ds['uw'] / corr_uw
                ds['wTs']   = ds['wTs'] / corr_wTs
                ds['ww']   = ds['ww'] / (corr_sigmaw)**0.5
                ds['uu']   = ds['uu'] / (corr_sigmau)**0.5
                ds['u']   = ds['u'] / corr_u
                
            # Compute corrected friction velocity
            ds['ustar'] = (ds['uw']**2 + ds['vw']**2)**(1/4)
            ds['ustary'] = (ds['uywg']**2 + ds['vywg']**2)**(1/4)
            
            # Compute corrected temperature scale
            ds['tstar']   = -np.divide(ds['wTs'],ds['ustar'])
            ds['tstar_c'] = -np.divide(ds['wTc'],ds['ustar'])
            ds['tstar_g'] = -np.divide(ds['wgT2'],ds['ustary'])
            
            if 'wq' in list(ds.variables):
                ds['qstar'] = -np.divide(ds['wq'],ds['ustar'])
                    
            # Compute corrected  Obukhov length
            ds['obh']   = -(ds['ustar']**3)*ds['T_sonic']/(9.81*utils.kappa*ds['wTs'])
            ds['obhy']   = -(ds['ustary']**3)*ds['T_couple2']/(9.81*utils.kappa*ds['wgT2'])
            
            # Calculate stability parameter
            ds['zeta']  = ds['z']/ds['obh']
            ds['zetay']  = ds['zy']/ds['obhy']
            
            # Calculate roughness length for momentum using the EC method
            ds['z0_EC'] = ds['z']/(np.exp((utils.kappa*ds['u']/ds['ustar'])  +  Psim(ds['zeta'])  ))  
            ds['z0_ECy'] = ds['zy']/(np.exp((utils.kappa*ds['uy']/ds['ustary'])  +  Psim(ds['zetay'])  ))  

            # Calculate LHF from SHF and apparent bowen ratio (see https://doi.org/10.1007/s10546-024-00864-y)
            if lBo_LHF:
                os.chdir(L1B_30min_dir) 
                file_L1B_30min = glob.glob("*L1B*nc") 
                # file_L1B_30min = glob.glob("*L1B*"  + file[ -9:-3] + "*nc") 
                ds_L1B_30min    = xr.open_dataset(file_L1B_30min[0])
                ds_L1B_30min['z'] = ds['z']
                ds_L1B_30min['obl'] = ds['obh']
                ds_L1B_30min['z0m'] = xr.DataArray(np.full(len(ds_L1B_30min.time),1e-3),dims=['time']) 
                
                ds_L1B_30min = ds_L1B_30min.resample(time="30min").interpolate("linear")
                
                Tv0 = (ds_L1B_30min['T0'])*(1+0.608*ds_L1B_30min['qv0'])
                
                # Surface pressure
                ps = ds_L1B_30min['p0']/np.exp(-utils.grav*ds_L1B_30min['z']/(utils.rd*Tv0))
                
                # potential temperatures
                ds_L1B_30min['th0'] = utils.T2thl(ds_L1B_30min['T0'],ds_L1B_30min['p0'],ql=0)
                ds_L1B_30min['ths'] = utils.T2thl(ds_L1B_30min['Ts'],ps,ql=0)
                
                # Surface humidity assuming saturation
                ds_L1B_30min['qvs'] = utils.RH2qv(100,ps, ds_L1B_30min['Ts'])
                
                # Air density and heat capacity
                ds_L1B_30min['rho_a'] = utils.rho_air(ds_L1B_30min.T0,ds_L1B_30min.p0,ds_L1B_30min.qv0)
    
                Bo = (ds_L1B_30min['th0'] - ds_L1B_30min['ths']) / ((ds_L1B_30min['qv0'] - ds_L1B_30min['qvs']))

                ds['rho_a'] = ds_L1B_30min['rho_a']
                ds['wq_Bo'] = ds['wTs'] / Bo
                ds['wgq_Bo'] = ds['wgT2'] / Bo

                os.chdir(L2dir) 


            # Remove spectra from dataset
            ds = ds.drop_dims(['freq','freq_corr'],errors='ignore') 


            # Store old attribute
            try:
                att1                                    = ds.attrs['IceEddie_namelist_L0toL1A']
                att2                                    = ds.attrs['IceEddie_namelist_L1AtoL1B']
                att3                                    = ds.attrs['IceEddie_namelist_L1BtoL2']
                # Add new attributes
                ds                                      = utils.Add_dataset_attributes(ds,'CSAT_EC_L3.JSON')
                ds.attrs['location']                    = LOC + '_' + ID
                ds.attrs['file_creation_date_time']     = str(datetime.datetime.now())
                ds.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
                ds.attrs['IceEddie_namelist_L0toL1A']   = att1
                ds.attrs['IceEddie_namelist_L1AtoL1B']  = att2
                ds.attrs['IceEddie_namelist_L1BtoL2']   = att3
                ds.attrs['IceEddie_namelist_L2toL3']    = str(nml['L2toL3'])
            except:
                print('No old attributes found')
                ds                                      = utils.Add_dataset_attributes(ds,'CSAT_EC_L3.JSON')
                ds.attrs['location']                    = LOC + '_' + ID
                ds.attrs['file_creation_date_time']     = str(datetime.datetime.now())
                ds.attrs['IceEddie_namelist_GLOBAL']    = str(nml['global'])
                ds.attrs['IceEddie_namelist_L2toL3']    = str(nml['L2toL3'])
                
            # ds = ds.drop_dims(['freq'],errors='ignore') 
            if fid == 0:
                ds_out = ds
            else:
                ds_out = xr.concat([ds_out,ds], "time")
            fid = fid + 1

        ds_out = ds_out.sortby('time')
        _, index = np.unique(ds_out['time'], return_index=True)
        ds_out = ds_out.isel(time=index)
        
        # Export to net CDF
        file_out = L3dir + LOC + '_' + ID  + '_' + sensor + '_' + input_type + '_' + "L3"  + '_' + version +  '_' + 'all' + ".nc"
        print(file_out)
        ds_out.to_netcdf(file_out,mode = 'w')       
#_______________________________________________________        
def Psim(zeta,unstable='Högström',stable='BeljaarsHoltslag'):
    """
    Returns the integrated flux-profile function for momentum evaluated at zeta
    
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
        if out[zeta<0].size: 
            out[zeta<0]  = np.log( ((1 + x[zeta<0]**2)/2) * ((1 + x[zeta<0])/2)**2 ) \
            - 2*np.arctan(x[zeta<0]) + np.pi/2
           
    if stable == 'BeljaarsHoltslag':
        a = 1
        b = 2/3
        c = 5
        d = 0.35
        if out[zeta>=0].size:
            out[zeta>=0] = -(b*c/d + a*zeta[zeta>=0] + b*(zeta[zeta>=0] - (c/d))*np.exp(-d*zeta[zeta>=0]))
   
    return out
        
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

            
#_______________________________________________________
def Kaimal(ds, nml,level = 'L2',height = 'z'):
    """
    Appends the Kaimal normalized cospectra to the input dataset
    
    Input
    ----------
    ds: xarray dataset
        dataset that contains wind velocity and obukhov length
    nml: python f90nml namelist 
        namelist containing all parameters
    level: str
        Level of input data
    height: str
        Name of height variable 

    
    Output
    ----------
    ds: xarray dataset
        Input dataset with indexed normalised Kaimal spectra
    
    Example
    ----------
    ds = Kaimal(ds,z):
    
    Required packages
    ----------
    glob, os, itertools, pandas, xarray
    
    Required functions
    ----------
    
    
    """ 
    zi         = 1000
    
    if level == 'L2':
        freq_range = nml['L1BtoL2']['freq_range']
        spectra    = nml['L1BtoL2']['spectra']
    elif level == 'L3':
        freq_range = nml['L2toL3']['freq_range']
        spectra    = nml['L2toL3']['spectra']
        
    
    print('Calculating Kaimal (co)spectra')
    sizet = np.size(ds.time.values)
    
    # Mapping frequencies
    if freq_range   == 'CSAT':
        freq_corr   = np.linspace(5.5556e-04, 5,9000)
        freq        = np.tile(freq_corr,(sizet,1))
        sizef       = 9000
    elif freq_range == 'Input':
        freq_corr   = ds.freq.values
        freq        = np.tile(freq_corr,(sizet,1))
        sizef       = np.size(ds.freq.values)
    elif freq_range == 'Wide': 
        freq_corr   =  10**(np.linspace(np.log10(1/3600),np.log10(100),1000))
        freq        = np.tile(freq_corr,(sizet,1))
        sizef       = 1000
    

    
    # Build matrices for calculations in frequency domain
    if height == 'z':
        z    = np.tile((ds.z.values),(sizef,1)).T 
        if np.ndim(ds.obh) > 1:
            zeta = np.tile((ds.z.values/ds.obh[:,0,0,0].values),(sizef,1)).T # Stability parameter
        else:
            zeta = np.tile((ds.z.values/ds.obh.values),(sizef,1)).T
        U    = np.tile(((ds.u.values**2+ds.v.values**2)**(1/2)),(sizef,1)).T # Wind velocity
    elif height == 'zy':
        z    = np.tile((ds.zy.values),(sizef,1)).T 
        if np.ndim(ds.obhy) > 1:
            zeta = np.tile((ds.zy.values/ds.obhy[:,0,0,0].values),(sizef,1)).T # Stability parameter
        else:
            zeta = np.tile((ds.zy.values/ds.obhy.values),(sizef,1)).T
        U    = np.tile(((ds.uy.values**2+ds.vy.values**2)**(1/2)),(sizef,1)).T # Wind velocity
    # Define normalised frequency
    n    = (z/U) * freq
    
    # Prevent stability parameter from reaching abnormal values 
    zeta[np.abs(zeta) > 10] = np.sign(zeta[np.abs(zeta) > 10])*10
    
    if spectra=='Kaimal':
        # Write reference Kaimal spectra derived from the Kansas experiment
        ## _____ Normalised fSww(f) and fSuu(f) spectrum _____ ##
        
        # Initialize variables
        SnTT = np.full((sizet, sizef), np.nan)
        Snww = np.full((sizet, sizef), np.nan)
        Snuu = np.full((sizet, sizef), np.nan)
        SnwT = np.full((sizet, sizef), np.nan)
        Snuw = np.full((sizet, sizef), np.nan)
    
        # Stable case 
        Aw  = 0.838 + 1.172*zeta[zeta>0]
        Bw  = 3.124 * Aw **(-2/3)
        
        Au  = 0.2*Aw
        Bu  = 3.124 * Au **(-2/3)
        
        Snww[zeta>0] = n[zeta>0]/(Aw + Bw * n[zeta>0]**(5/3))
        Snuu[zeta>0] = n[zeta>0]/(Au + Bu * n[zeta>0]**(5/3))
                
        # Unstable case
        chi   = (-zeta[zeta<=0])**(2/3)
        xi    = (z[zeta<=0]/zi)**(5/3)
        
        Cw    = 0.7285 + 1.4115 * chi
        Cu    = 9.546  + 1.235  * chi * xi**(-2/5)
        
        Snww[zeta<=0]  = ((n[zeta<=0]/(1+5.3*n[zeta<=0]**(5/3))) + \
                    (16*n[zeta<=0]*chi/(1+17*n[zeta<=0])**(5/3)))/Cw
            
        Snuu[zeta<=0]  = ((210 * n[zeta<=0]/(1+33*n[zeta<=0]**(5/3))) + \
                    (n[zeta<=0]*chi/(xi+2.2*n[zeta<=0]**(5/3))))/Cu
            
        
        ## _____ Normalised fSTT(f), fSwT(f) and fSuw(f) spectrum _____ ##
        # Stable case 
        At   = 0.0961 + 0.644* zeta[zeta>0]**0.6
        Bt   = 3.124 * At **(-2/3)
        
        Awt  = 0.284 * (1 + 6.4 * zeta[zeta>0])**(0.75)
        Bwt  = 2.34 * Awt **(-1.1) 
        
        Auw  = 0.124 * (1 + 7.9 * zeta[zeta>0])**(0.75)
        Buw  = 2.34 * Auw **(-1.1) 
        
        SnTT[zeta>0] = n[zeta>0]/(At  + Bt  * n[zeta>0]**(5/3))
        SnwT[zeta>0] = n[zeta>0]/(Awt + Bwt * n[zeta>0]**(2.1))
        Snuw[zeta>0] = n[zeta>0]/(Auw + Buw * n[zeta>0]**(2.1))
                
        # Unstable case (line per line calculation is required due to case dependance of Sxy on n)
        for idxt in range(0,sizet):
            if zeta[idxt,0] < 0:  
                tmp = n[idxt,:]
                SnTT[idxt,tmp<0.15]  = 14.94 * tmp[tmp<0.15]  / (1 + 24   * tmp[tmp<0.15]) **(5/3) 
                SnTT[idxt,tmp>=0.15] = 6.827 * tmp[tmp>=0.15] / (1 + 12.5 * tmp[tmp>=0.15])**(5/3)
                
                SnwT[idxt,tmp<0.54]  = 12.92 * tmp[tmp<0.54]  / (1 + 26.7 * tmp[tmp<0.54]) **(1.375) 
                SnwT[idxt,tmp>=0.54] = 4.378 * tmp[tmp>=0.54] / (1 + 3.8  * tmp[tmp>=0.54])**(2.4)
                
                Snuw[idxt,tmp<0.24]  = 20.78 * tmp[tmp<0.24]  / (1 + 31   * tmp[tmp<0.24]) **(1.575) 
                Snuw[idxt,tmp>=0.24] = 12.66 * tmp[tmp>=0.24] / (1 + 9.6  * tmp[tmp>=0.24])**(2.4)
                
    elif spectra=='Cabauw':
        # Write reference spectra adapted to Cabauw
        
        # Parameters  (A0, fx, m, mu) for (Sww, Suu, STT, Suw, Swt)
        x = np.array([[0.2024,    0.4394,    1.5000,    0.4713], \
                      [1.3384,    0.0258,    1.5000,    0.3194], \
                      [0.0716,    0.1236,    1.5000,    0.2917], \
                      [0.0958,    0.0673,    0.7500,    0.4696], \
                      [0.0330,    0.1915,    0.7500,    0.3349]])
    
        # Slope and interception parameters for peak normalized frequency of model spectra
        a = np.array([[1.4897,    0.4255], \
                      [0.4333,    0.0091], \
                      [1.4527,    0.0033], \
                      [0.5211,    0.0491], \
                      [1.5514,    0.0906]])
    
        # Frequency of spectral peak (Adaptation to Horst 1997) in the 0-0.2 stability range
        # Vertical velocity variance
        fmw              = (U / z)                          * (a[0,1] + a[0,0] * zeta)
        fmw[zeta < 0.0]  = (U[zeta < 0.0] / z[zeta < 0.0])  * a[0,1]
        fmw[zeta > 0.2]  = (U[zeta > 0.2] / z[zeta > 0.2])  * (a[0,1] + 0.2 * a[0,0])
        # Horizontal velocity variance
        fmu              = (U / z)                          * (a[1,1] + a[1,0] * zeta)
        fmu[zeta < 0.0]  = (U[zeta < 0.0] / z[zeta < 0.0])  * a[1,1]
        fmu[zeta > 0.2]  = (U[zeta > 0.2] / z[zeta > 0.2])  * (a[1,1] + 0.2 * a[1,0])
        # Temperature variance
        fmT              = (U / z)                          * (a[2,1] + a[2,0] * zeta)
        fmT[zeta < 0.0]  = (U[zeta < 0.0] / z[zeta < 0.0])  * a[2,1]
        fmT[zeta > 0.2]  = (U[zeta > 0.2] / z[zeta > 0.2])  * (a[2,1] + 0.2 * a[2,0])
        # Momentum flux
        fmuw             = (U / z)                          * (a[3,1] + a[3,0] * zeta)
        fmuw[zeta < 0.04]= (U[zeta < 0.04] / z[zeta < 0.0]) * (a[3,1] + 0.04 * a[3,0])
        fmuw[zeta > 0.2] = (U[zeta > 0.2] / z[zeta > 0.2])  * (a[3,1] + 0.2 * a[3,0])
        # Sensible heat flux
        fmwT             = (U / z)                          * (a[4,1] + a[4,0] * zeta)
        fmwT[zeta < 0.04]= (U[zeta < 0.04] / z[zeta < 0.0]) * (a[4,1] + 0.04 * a[4,0])
        fmwT[zeta > 0.2] = (U[zeta > 0.2] / z[zeta > 0.2])  * (a[4,1] + 0.2 * a[4,0])
         
        # Reference spectra          
        Snww = x[0,0] * (freq/fmw)  / (1 + x[0,2]*(freq/fmw)**(2*x[0,3]))  ** (1/(2*x[0,3]) * (x[0,2]+1)/x[0,2])
        Snuu = x[1,0] * (freq/fmu)  / (1 + x[1,2]*(freq/fmu)**(2*x[1,3]))  ** (1/(2*x[1,3]) * (x[1,2]+1)/x[1,2])
        SnTT = x[2,0] * (freq/fmT)  / (1 + x[2,2]*(freq/fmT)**(2*x[2,3]))  ** (1/(2*x[2,3]) * (x[2,2]+1)/x[2,2])
        Snuw = x[3,0] * (freq/fmuw) / (1 + x[3,2]*(freq/fmuw)**(2*x[3,3])) ** (1/(2*x[3,3]) * (x[3,2]+1)/x[3,2])
        SnwT = x[4,0] * (freq/fmwT) / (1 + x[4,2]*(freq/fmwT)**(2*x[4,3])) ** (1/(2*x[4,3]) * (x[4,2]+1)/x[4,2])  
          
    elif spectra=='S10':
        # Write reference spectra adapted to Cabauw
        
        # Parameters  (A0, fx, m, mu) for (Sww, Suu, STT, Suw, Swt)
        x = np.array([[0.2701,    1.3786,    1.5000,    0.4772], \
                      [2.2130,    0.1034,    1.5000,    0.2772], \
                      [2.2130,    0.1034,    1.5000,    0.2530], \
                      [0.1680,    0.2559,    0.7500,    0.4029], \
                      [0.0736,    0.6130,    0.7500,    0.3055]])
    
        # Slope and interception parameters for peak normalized frequency of model spectra
        a = np.array([[2.4925,    0.5264], \
                      [0.6145,    0.0264], \
                      [0.6145,    0.0264], \
                      [1.1698,    0.0586], \
                      [1.9161,    0.1977]])
    
        # Frequency of spectral peak (Adaptation to Horst 1997) in the 0-0.2 stability range
        # Vertical velocity variance
        fmw              = (U / z)                          * (a[0,1] + a[0,0] * zeta)
        fmw[zeta < 0.0]  = (U[zeta < 0.0] / z[zeta < 0.0])  *  a[0,1]
        fmw[zeta > 0.2]  = (U[zeta > 0.2] / z[zeta > 0.2])  * (a[0,1] + 0.2 * a[0,0])
        # Horizontal velocity variance
        fmu              = (U / z)                          * (a[1,1] + a[1,0] * zeta)
        fmu[zeta < 0.0]  = (U[zeta < 0.0] / z[zeta < 0.0])  *  a[1,1]
        fmu[zeta > 0.2]  = (U[zeta > 0.2] / z[zeta > 0.2])  * (a[1,1] + 0.2 * a[1,0])
        # Temperature variance
        fmT              = (U / z)                          * (a[2,1] + a[2,0] * zeta)
        fmT[zeta < 0.0]  = (U[zeta < 0.0] / z[zeta < 0.0])  *  a[2,1]
        fmT[zeta > 0.2]  = (U[zeta > 0.2] / z[zeta > 0.2])  * (a[2,1] + 0.2 * a[2,0])
        # Momentum flux
        fmuw             = (U / z)                          * (a[3,1] + a[3,0] * zeta)
        fmuw[zeta < 0.0] = (U[zeta < 0.0] / z[zeta < 0.0])  *  a[3,1] 
        fmuw[zeta > 0.2] = (U[zeta > 0.2] / z[zeta > 0.2])  * (a[3,1] + 0.2 * a[3,0])
        # Sensible heat flux
        fmwT             = (U / z)                          * (a[4,1] + a[4,0] * zeta)
        fmwT[zeta < 0.0] = (U[zeta < 0.0] / z[zeta < 0.0])  *  a[4,1] 
        fmwT[zeta > 0.2] = (U[zeta > 0.2] / z[zeta > 0.2])  * (a[4,1] + 0.2 * a[4,0])
         
        # Reference spectra          
        Snww = x[0,0] * (freq/fmw)  / (1 + x[0,2]*(freq/fmw)**(2*x[0,3]))  ** (1/(2*x[0,3]) * (x[0,2]+1)/x[0,2])
        Snuu = x[1,0] * (freq/fmu)  / (1 + x[1,2]*(freq/fmu)**(2*x[1,3]))  ** (1/(2*x[1,3]) * (x[1,2]+1)/x[1,2])
        SnTT = x[2,0] * (freq/fmT)  / (1 + x[2,2]*(freq/fmT)**(2*x[2,3]))  ** (1/(2*x[2,3]) * (x[2,2]+1)/x[2,2])
        Snuw = x[3,0] * (freq/fmuw) / (1 + x[3,2]*(freq/fmuw)**(2*x[3,3])) ** (1/(2*x[3,3]) * (x[3,2]+1)/x[3,2])
        SnwT = x[4,0] * (freq/fmwT) / (1 + x[4,2]*(freq/fmwT)**(2*x[4,3])) ** (1/(2*x[4,3]) * (x[4,2]+1)/x[4,2])          
    
    
    # Divide by frequency to obtain Sxy(f)
    Snww = Snww/freq
    SnTT = SnTT/freq
    Snuu = Snuu/freq
    SnwT = SnwT/freq
    Snuw = Snuw/freq     
            
    # Convert to data array
    Snww = xr.DataArray(Snww,coords=[ds.time,freq_corr], dims=['time','freq_corr'])
    SnTT = xr.DataArray(SnTT,coords=[ds.time,freq_corr], dims=['time','freq_corr'])
    Snuu = xr.DataArray(Snuu,coords=[ds.time,freq_corr], dims=['time','freq_corr'])
    SnwT = xr.DataArray(SnwT,coords=[ds.time,freq_corr], dims=['time','freq_corr'])
    Snuw = xr.DataArray(Snuw,coords=[ds.time,freq_corr], dims=['time','freq_corr'])

    return ds.assign({'Snww': Snww, 'SnTT': SnTT, 'Snuu': Snuu, 'SnwT': SnwT, 'Snuw': Snuw})
   
#_______________________________________________________
def transfer_functions(ds, nml, tauT = 0.14, Aw = 0.4, Lu = 2,level = 'L2'):
    """
    Appends the estimated frequency transfer functions to the input dataset
    
    Input
    ----------
    ds: xarray dataset
        dataset that contains wind velocity
    nml: python f90nml namelist 
        namelist containing all parameters
    tauT = float
        thermocouple response time (s) 
    Aw = float
        Gill propeller response time length (m)  {~0.45m for EPS and way more for CFT}
    Lu = float
        Young propeller response length (m) {1.2m for EPS and 2m for CFT and 3.2m for PP}
    suw = float
        separation distance between uYoung and wGill (m)

    
    Output
    ----------
    ds: xarray dataset
        Input dataset with estimated tranfers functions 
    
    Example
    ----------
    ds = Kaimal(ds,z):
    
    Required packages
    ----------
    glob, os, itertools, pandas, xarray
    
    Required functions
    ----------
    
    
    """ 
    print('Calculating transfer functions')
     
    sensor     = nml['global']['sensor']
    
    if level == 'L2':
        freq_range = nml['L1BtoL2']['freq_range']
        ldetrend   = nml['L1BtoL2']['ldetrend']
        lhighpass  = nml['L1BtoL2']['lhighpass']
    elif level == 'L3':
        freq_range = nml['L2toL3']['freq_range']
        ldetrend   = nml['L2toL3']['ldetrend']
        lhighpass  = nml['L2toL3']['lhighpass']
    
    suw        = nml['L2toL3']['suw']
    swT        = nml['L2toL3']['swT']
    swq        = nml['L2toL3']['swq']
    pw         = nml['L2toL3']['pw']
    pu         = nml['L2toL3']['pu']
    pT         = nml['L2toL3']['pT']
    pq         = nml['L2toL3']['pq']
    dT         = nml['L2toL3']['dT']
    tau_wT      = nml['L2toL3']['tau_wT']
    tau_uw      = nml['L2toL3']['tau_uw']
    gillcorr_method = nml['L2toL3']['gillcorr_method']
    youngcorr_method = nml['L2toL3']['youngcorr_method']
    tau_w_nml_1 = nml['L2toL3']['tau_w_1']
    tau_w_nml_2 = nml['L2toL3']['tau_w_2']
    tau_u_nml = nml['L2toL3']['tau_u']
    Lu_nml = nml['L2toL3']['Lu']

    
    T = pd.to_timedelta(dT).total_seconds()
    
    sizet = np.size(ds.time.values)
    
    # Mapping frequencies
    if freq_range   == 'CSAT':
        freq_corr   = np.linspace(5.5556e-04, 5,9000)
        freq        = np.tile(freq_corr,(sizet,1))
        sizef       = 9000
    elif freq_range == 'Input':
        freq        = np.tile(ds.freq.values,(sizet,1))
        sizef       = np.size(ds.freq.values)
    elif freq_range == 'Wide': 
        freq_corr   =  10**(np.linspace(np.log10(1/3600),np.log10(100),1000))
        freq        = np.tile(freq_corr,(sizet,1))
        sizef       = 1000
        
    if ((sensor == 'iWS') | (sensor == 'Young')):
        
        # Initialize variables   
        GT  = np.full([len(ds.time),sizef], np.nan)
        Gw  = np.full([len(ds.time),sizef], np.nan)
        Gu  = np.full([len(ds.time),sizef], np.nan)
        GwT = np.full([len(ds.time),sizef], np.nan)
        Guw = np.full([len(ds.time),sizef], np.nan)
        
        # Wind velocity
        U    = np.tile(((ds.u.values**2+ds.v.values**2)**(1/2)),(sizef,1)).T 
        
        # Response length of vertical propeller (after Fichtl & Kumar 1973)
        #inclusindg factor 2 to take into account attenuation of ww by vertical EPS propeller
        if np.ndim(ds.ww) > 1:
            sigma_w = np.tile(((2*ds.ww[:,0,0,0].values)**(1/2)),(sizef,1)).T 
        else:
            sigma_w = np.tile(((2*ds.ww.values)**(1/2)),(sizef,1)).T 
        
        Lw   = Aw * (sigma_w / U)**(-2/3)
        
        # Sensor gain functions
        GT = np.sqrt(1/(1+(2*np.pi*freq*tauT)**2))
        Gw = np.sqrt(1/(1+(2*np.pi*freq*Lw/U)**2))
        Gu = np.sqrt(1/(1+(2*np.pi*freq*Lu/U)**2))
        
        # Transfer function from correlated measurements (Horst 1997) ingoring phase differences
        GwT = (1+(2*np.pi*freq)**2 * Lw/U * tauT ) / ((1+(2*np.pi*freq)**2 * (Lw/U)**2) * (1+(2*np.pi*freq)**2 * (tauT)**2))
        Guw = (1+(2*np.pi*freq)**2 * Lw/U * Lu/U ) / ((1+(2*np.pi*freq)**2 * (Lw/U)**2) * (1+(2*np.pi*freq)**2 * (Lu/U)**2))
            
        # Sensor separation normalised frequency
        nsuw    = (suw/U) * freq
        nswT    = (swT/U) * freq
        # Sensor separation transfer function
        Tsuw = np.exp(-9.9 * nsuw **1.5)
        TswT = np.exp(-9.9 * nswT **1.5)
        
        # Time delay correction
        ns_tau_wT = freq * tau_wT
        Ts_tau_wT = np.cos(2*np.pi*ns_tau_wT)
        ns_tau_uw = freq * tau_uw
        Ts_tau_uw = np.cos(2*np.pi*ns_tau_uw)
        

        
        # High-pass filter (averaging)
        if lhighpass:
            if ldetrend:
                GD = 1 - ((np.sin(np.pi*freq*T)**2/(np.pi*freq*T)**2) - \
                          3 * ((np.sin(np.pi*freq*T)/(np.pi*freq*T))-np.cos(np.pi*freq*T))**2 / (np.pi*freq*T)**2)
            else:
                GD = 1 - np.sin(np.pi*freq*T)**2/(np.pi*freq*T)**2 
        else:
            GD = 1
        
        # Write total transfer functions
        Tww = GD * Gw**2
        Tuu = GD * Gu**2
        TTT = GD * GT**2
        TwT = GD * GwT * TswT * Ts_tau_wT
        Tuw = GD * Guw * Tsuw * Ts_tau_uw
                    
        # Convert to data array
        if freq_range == 'Input':
            Tww = xr.DataArray(Tww,coords=[ds.time,ds.freq], dims=['time','freq'])
            Tuu = xr.DataArray(Tuu,coords=[ds.time,ds.freq], dims=['time','freq'])
            TTT = xr.DataArray(TTT,coords=[ds.time,ds.freq], dims=['time','freq'])
            TwT = xr.DataArray(TwT,coords=[ds.time,ds.freq], dims=['time','freq'])
            Tuw = xr.DataArray(Tuw,coords=[ds.time,ds.freq], dims=['time','freq'])
            GD = xr.DataArray(GD,coords=[ds.time,ds.freq], dims=['time','freq'])
            GwT = xr.DataArray(GwT,coords=[ds.time,ds.freq], dims=['time','freq'])
            Guw = xr.DataArray(Guw,coords=[ds.time,ds.freq], dims=['time','freq'])
            TswT = xr.DataArray(TswT,coords=[ds.time,ds.freq], dims=['time','freq'])
            Tsuw = xr.DataArray(Tsuw,coords=[ds.time,ds.freq], dims=['time','freq'])
            Gw = xr.DataArray(Gw,coords=[ds.time,ds.freq], dims=['time','freq'])
            Gu = xr.DataArray(Gu,coords=[ds.time,ds.freq], dims=['time','freq'])
            GT = xr.DataArray(GT,coords=[ds.time,ds.freq], dims=['time','freq'])
        else:
            Tww = xr.DataArray(Tww,coords=[ds.time,freq_corr], dims=['time','freq_corr'])
            Tuu = xr.DataArray(Tuu,coords=[ds.time,freq_corr], dims=['time','freq_corr'])
            TTT = xr.DataArray(TTT,coords=[ds.time,freq_corr], dims=['time','freq_corr'])
            TwT = xr.DataArray(TwT,coords=[ds.time,freq_corr], dims=['time','freq_corr'])
            Tuw = xr.DataArray(Tuw,coords=[ds.time,freq_corr], dims=['time','freq_corr'])
            GD = xr.DataArray(GD,coords=[ds.time,freq_corr], dims=['time','freq_corr'])
            GwT = xr.DataArray(GwT,coords=[ds.time,freq_corr], dims=['time','freq_corr'])
            Guw = xr.DataArray(Guw,coords=[ds.time,freq_corr], dims=['time','freq_corr'])
            TswT = xr.DataArray(TswT,coords=[ds.time,freq_corr], dims=['time','freq_corr'])
            Tsuw = xr.DataArray(Tsuw,coords=[ds.time,freq_corr], dims=['time','freq_corr'])
            Gw = xr.DataArray(Gw,coords=[ds.time,freq_corr], dims=['time','freq_corr'])
            Gu = xr.DataArray(Gu,coords=[ds.time,freq_corr], dims=['time','freq_corr'])
            GT = xr.DataArray(GT,coords=[ds.time,freq_corr], dims=['time','freq_corr'])
    
        if level == 'L3':
            return ds.assign({'Tww': Tww, 'Tuu': Tuu, 'TTT': TTT, 'TwT': TwT, 'Tuw': Tuw})
        else:
            return ds.assign({'Tww': Tww, 'Tuu': Tuu, 'TTT': TTT, 'TwT': TwT, 'Tuw': Tuw, \
                              'GD': GD, 'GwT': GwT, 'Guw': Guw, 'TswT': TswT, 'Tsuw': Tsuw, \
                              'Gw': Gw, 'Gu': Gu, 'GT': GT})
    
    elif sensor == 'CSAT':
        # Initialize variables   
        GT  = np.full([len(ds.time),sizef], np.nan)
  
        # Wind velocity
        U    = np.tile(((ds.u.values**2+ds.v.values**2)**(1/2)),(sizef,1)).T 
        
        # Sensor gain functions
        GT = np.sqrt(1/(1+(2*np.pi*freq*tauT)**2))
             
        
        # Sensor separation normalised frequency
        nswT    = (swT/U) * freq
        
        # Sensor separation transfer function
        TswT = np.exp(-9.9 * nswT **1.5)
        
        if 'wq' in list(ds.variables):
             nswq = (swq/U) * freq
             Tswq = np.exp(-9.9 * nswq **1.5)
        
        # Sensor path  normalised frequency
        npw    = (pw/U) * freq
        npu    = (pu/U) * freq
        npT    = (pT/U) * freq
        
        if 'wq' in list(ds.variables):
            npq    = (pq/U) * freq
            Tpq    = (1/(2 * np.pi * npT) ) * (3 + (np.exp(-2*np.pi*npq)) \
                                               -  (4*(1-np.exp(-2*np.pi*npq))/(2*np.pi*npq)))
        
        # Path averaging transfer functions
        Tpw    = (2/(np.pi * npw) ) * (1 + (np.exp(-2*np.pi*npw)/2) \
               -  (3*(1-np.exp(-2*np.pi*npw))/(4*np.pi*npw)))
        Tpu    = (2/(np.pi * npu) ) * (1 + (np.exp(-2*np.pi*npu)/2) \
               -  (3*(1-np.exp(-2*np.pi*npu))/(4*np.pi*npu)))
        TpT    = (1/(2 * np.pi * npT) ) * (3 + (np.exp(-2*np.pi*npT)) \
               -  (4*(1-np.exp(-2*np.pi*npT))/(2*np.pi*npT)))
        
        # High-pass filter
        if lhighpass:
            if ldetrend:
                GD = 1 - ((np.sin(np.pi*freq*T)**2/(np.pi*freq*T)**2) - \
                          3 * ((np.sin(np.pi*freq*T)/(np.pi*freq*T))-np.cos(np.pi*freq*T))**2 / (np.pi*freq*T)**2)
            else:
                GD = 1-np.sin(np.pi*freq*T)**2./(np.pi*freq*T)**2
        else:
            GD = 1
            
        # Write total transfer functions
        Tww   = GD * Tpw
        Tuu   = GD * Tpu
        TTsTs = GD * TpT
        TTcTc = GD * GT**2
        TwTs  = GD * (Tpw * TpT)**0.5
        TwTc  = GD * GT * TswT * Tpw **0.5
        Tuw   = GD * (Tpu * Tpw)**0.5
        
        if 'wq' in list(ds.variables):
            Twq  = GD * Tswq * (Tpw * Tpq)**0.5
                    
        # Convert to data array
        if freq_range == 'Input':
            Tww   = xr.DataArray(Tww,coords=[ds.time,ds.freq],  dims=['time','freq'])
            Tuu   = xr.DataArray(Tuu,coords=[ds.time,ds.freq],  dims=['time','freq'])
            TTsTs = xr.DataArray(TTsTs,coords=[ds.time,ds.freq],dims=['time','freq'])
            TwTs  = xr.DataArray(TwTs,coords=[ds.time,ds.freq], dims=['time','freq'])
            TTcTc = xr.DataArray(TTcTc,coords=[ds.time,ds.freq],dims=['time','freq'])
            TwTc  = xr.DataArray(TwTc,coords=[ds.time,ds.freq], dims=['time','freq'])
            Tuw   = xr.DataArray(Tuw,coords=[ds.time,ds.freq],  dims=['time','freq'])
            if 'wq' in list(ds.variables):
                 Twq  = xr.DataArray(Twq,coords=[ds.time,ds.freq],  dims=['time','freq'])
        else:
            Tww   = xr.DataArray(Tww,coords=[ds.time,freq_corr],  dims=['time','freq_corr'])
            Tuu   = xr.DataArray(Tuu,coords=[ds.time,freq_corr],  dims=['time','freq_corr'])
            TTsTs = xr.DataArray(TTsTs,coords=[ds.time,freq_corr],dims=['time','freq_corr'])
            TwTs  = xr.DataArray(TwTs,coords=[ds.time,freq_corr], dims=['time','freq_corr'])
            TTcTc = xr.DataArray(TTcTc,coords=[ds.time,freq_corr],dims=['time','freq_corr'])
            TwTc  = xr.DataArray(TwTc,coords=[ds.time,freq_corr], dims=['time','freq_corr'])
            Tuw   = xr.DataArray(Tuw,coords=[ds.time,freq_corr],  dims=['time','freq_corr'])   
            if 'wq' in list(ds.variables):
                 Twq  = xr.DataArray(Twq,coords=[ds.time,ds.freq_corr],  dims=['time','freq_corr'])
                     
        
        if 'wq' in list(ds.variables):
            return ds.assign({'Tww': Tww, 'Tuu': Tuu, 'TTsTs': TTsTs, 'TwTs': TwTs, 'TTcTc': TTcTc, 'TwTc': TwTc, 'Tuw': Tuw, 'Twq': Twq})
        else:
            return ds.assign({'Tww': Tww, 'Tuu': Tuu, 'TTsTs': TTsTs, 'TwTs': TwTs, 'TTcTc': TTcTc, 'TwTc': TwTc, 'Tuw': Tuw})
        
    elif sensor == 'CSAT_Young':
        # Initialize variables   
        GT  = np.full([len(ds.time),sizef], np.nan)
        Gw  = np.full([len(ds.time),sizef], np.nan)
        Gu  = np.full([len(ds.time),sizef], np.nan)
        GwT = np.full([len(ds.time),sizef], np.nan)
        Guw = np.full([len(ds.time),sizef], np.nan)

        # Wind velocity
        U    = np.tile(((ds.u.values**2+ds.v.values**2)**(1/2)),(sizef,1)).T 
        Uy   = np.tile(((ds.uy.values**2+ds.vy.values**2)**(1/2)),(sizef,1)).T 
        Lw = np.full([len(ds.time),sizef], np.nan)

        if np.ndim(ds.ww) > 1:
            sigma_w = np.tile(((2*ds.wgwg[:,0,0,0].values)**(1/2)),(sizef,1)).T 
        else:
            sigma_w = np.tile(((2*ds.wgwg.values)**(1/2)),(sizef,1)).T 

        if gillcorr_method == 'linear':
            Lw   = Uy * (tau_w_nml_1[0] + tau_w_nml_1[1] * (1 / Uy))
        elif gillcorr_method == 'two_linear':
            Lw[Uy > 10]   = Uy[Uy > 10] * (tau_w_nml_1[0] + tau_w_nml_1[1] * (1 / Uy[Uy > 10]))
            Lw[Uy <= 10]  = Uy[Uy <= 10] * (tau_w_nml_2[0] + tau_w_nml_2[1] * (1 / Uy[Uy <= 10]))
        elif gillcorr_method == 'none':
            Lw = 0
        else: 
            Lw   = Aw * (sigma_w / Uy)**(-2/3)
        if youngcorr_method == 'linear':
            Lu   = Uy * (tau_u_nml[0] + tau_u_nml[1] * (1 / Uy))
        elif youngcorr_method == 'none':
            Lu = 0
        else: 
            Lu = Lu_nml

        # Sensor gain functions 
        GT = np.sqrt(1/(1+(2*np.pi*freq*tauT)**2))
        Gw = np.sqrt(1/(1+(2*np.pi*freq*Lw/Uy)**2))
        Gu = np.sqrt(1/(1+(2*np.pi*freq*Lu/Uy)**2))

        # Transfer function from correlated measurements (Horst 1997) ingoring phase differences
        GwT = (1+(2*np.pi*freq)**2 * Lw/Uy * tauT ) / ((1+(2*np.pi*freq)**2 * (Lw/Uy)**2) * (1+(2*np.pi*freq)**2 * (tauT)**2))
        Guw = (1+(2*np.pi*freq)**2 * Lw/Uy * Lu/Uy ) / ((1+(2*np.pi*freq)**2 * (Lw/Uy)**2) * (1+(2*np.pi*freq)**2 * (Lu/Uy)**2))

        # Sensor separation normalised frequency
        nsuw    = (suw/Uy) * freq
        nswT    = (swT/Uy)  * freq
        
        # Sensor separation transfer function
        Tsuw = np.exp(-9.9 * nsuw **1.5)
        TswT = np.exp(-9.9 * nswT **1.5)
        
        if 'wq' in list(ds.variables):
             nswq = (swq/U) * freq
             Tswq = np.exp(-9.9 * nswq **1.5)
        
        # Sensor path  normalised frequency
        npw    = (pw/U) * freq
        npu    = (pu/U) * freq
        npT    = (pT/U) * freq
        
        if 'wq' in list(ds.variables):
            npq    = (pq/U) * freq
            Tpq    = (1/(2 * np.pi * npT) ) * (3 + (np.exp(-2*np.pi*npq)) \
                                               -  (4*(1-np.exp(-2*np.pi*npq))/(2*np.pi*npq)))
        
        # Path averaging transfer functions
        Tpw    = (2/(np.pi * npw) ) * (1 + (np.exp(-2*np.pi*npw)/2) \
               -  (3*(1-np.exp(-2*np.pi*npw))/(4*np.pi*npw)))
        Tpu    = (2/(np.pi * npu) ) * (1 + (np.exp(-2*np.pi*npu)/2) \
               -  (3*(1-np.exp(-2*np.pi*npu))/(4*np.pi*npu)))
        TpT    = (1/(2 * np.pi * npT) ) * (3 + (np.exp(-2*np.pi*npT)) \
               -  (4*(1-np.exp(-2*np.pi*npT))/(2*np.pi*npT)))
        
        # High-pass filter
        if lhighpass:
            if ldetrend:
                GD = 1 - ((np.sin(np.pi*freq*T)**2/(np.pi*freq*T)**2) - \
                          3 * ((np.sin(np.pi*freq*T)/(np.pi*freq*T))-np.cos(np.pi*freq*T))**2 / (np.pi*freq*T)**2)
            else:
                GD = 1-np.sin(np.pi*freq*T)**2./(np.pi*freq*T)**2
        else:
            GD = 1
            
        # Write total transfer functions
        Tww   = GD * Tpw
        Tuu   = GD * Tpu
        TTsTs = GD * TpT
        TTcTc = GD * GT**2
        TwTs  = GD * (Tpw * TpT)**0.5
        TwTc  = GD * GT * TswT * Tpw **0.5
        Tuw   = GD * (Tpu * Tpw)**0.5

        Twgwg = GD * Gw**2
        Tuyuy = GD * Gu**2
        TTT = GD * GT**2
        TwgTc = GD * GwT * TswT
        Tuywg = GD * Guw * Tsuw 
        if 'wq' in list(ds.variables):
            Twq  = GD * Tswq * (Tpw * Tpq)**0.5
                    
        # Convert to data array
        if freq_range == 'Input':
            Tww   = xr.DataArray(Tww,coords=[ds.time,ds.freq],  dims=['time','freq'])
            Tuu   = xr.DataArray(Tuu,coords=[ds.time,ds.freq],  dims=['time','freq'])
            TTsTs = xr.DataArray(TTsTs,coords=[ds.time,ds.freq],dims=['time','freq'])
            TwTs  = xr.DataArray(TwTs,coords=[ds.time,ds.freq], dims=['time','freq'])
            TTcTc = xr.DataArray(TTcTc,coords=[ds.time,ds.freq],dims=['time','freq'])
            TwTc  = xr.DataArray(TwTc,coords=[ds.time,ds.freq], dims=['time','freq'])
            Tuw   = xr.DataArray(Tuw,coords=[ds.time,ds.freq],  dims=['time','freq'])
            Twgwg  = xr.DataArray(Twgwg,coords=[ds.time,ds.freq],  dims=['time','freq'])
            Tuyuy  = xr.DataArray(Tuyuy,coords=[ds.time,ds.freq],  dims=['time','freq'])
            TTT  = xr.DataArray(TTT,coords=[ds.time,ds.freq],  dims=['time','freq'])
            TwgTc  = xr.DataArray(TwgTc,coords=[ds.time,ds.freq],  dims=['time','freq'])
            Tuywg  = xr.DataArray(Tuywg,coords=[ds.time,ds.freq],  dims=['time','freq'])
            if 'wq' in list(ds.variables):
                Twq  = xr.DataArray(Twq,coords=[ds.time,ds.freq],  dims=['time','freq'])
        else:
            Tww   = xr.DataArray(Tww,coords=[ds.time,freq_corr],  dims=['time','freq_corr'])
            Tuu   = xr.DataArray(Tuu,coords=[ds.time,freq_corr],  dims=['time','freq_corr'])
            TTsTs = xr.DataArray(TTsTs,coords=[ds.time,freq_corr],dims=['time','freq_corr'])
            TwTs  = xr.DataArray(TwTs,coords=[ds.time,freq_corr], dims=['time','freq_corr'])
            TTcTc = xr.DataArray(TTcTc,coords=[ds.time,freq_corr],dims=['time','freq_corr'])
            TwTc  = xr.DataArray(TwTc,coords=[ds.time,freq_corr], dims=['time','freq_corr'])
            Tuw   = xr.DataArray(Tuw,coords=[ds.time,freq_corr],  dims=['time','freq_corr'])   
            Twgwg  = xr.DataArray(Twgwg,coords=[ds.time,ds.freq_corr],  dims=['time','freq_corr'])
            Tuyuy  = xr.DataArray(Tuyuy,coords=[ds.time,ds.freq_corr],  dims=['time','freq_corr'])
            TTT  = xr.DataArray(TTT,coords=[ds.time,ds.freq_corr],  dims=['time','freq_corr'])
            TwgTc  = xr.DataArray(TwgTc,coords=[ds.time,ds.freq_corr],  dims=['time','freq_corr'])
            Tuywg  = xr.DataArray(Tuywg,coords=[ds.time,ds.freq_corr],  dims=['time','freq_corr'])
            if 'wq' in list(ds.variables):
                Twq  = xr.DataArray(Twq,coords=[ds.time,ds.freq_corr],  dims=['time','freq_corr'])
                     
        if 'wq' in list(ds.variables):
            return ds.assign({'Tww': Tww, 'Tuu': Tuu, 'TTsTs': TTsTs, 'TwTs': TwTs, 'TTcTc': TTcTc, 'TwTc': TwTc, 'Tuw': Tuw, \
                              'Twgwg': Twgwg, 'Tuyuy': Tuyuy, 'TTT': TTT, 'TwgTc': TwgTc, 'Tuywg': Tuywg, 'Twq': Twq})
        else:
            return ds.assign({'Tww': Tww, 'Tuu': Tuu, 'TTsTs': TTsTs, 'TwTs': TwTs, 'TTcTc': TTcTc, 'TwTc': TwTc, 'Tuw': Tuw, \
                              'Twgwg': Twgwg, 'Tuyuy': Tuyuy, 'TTT': TTT, 'TwgTc': TwgTc, 'Tuywg': Tuywg})
        
#_______________________________________________________
def Horst_correction(ds, nml, tauT = 0.14, Aw = 0.4, Lu = 2):
    """
    Appends the estimated high-frequency attenuation correction using Horst (1996)
    
    Input
    ----------
    ds: xarray dataset
        dataset that contains wind velocity
    nml: python f90nml namelist 
        namelist containing all parameters
    tauT = float
        thermocouple response time (s) 
    Aw = float
        Gill propeller response time length (m)  {~0.45m for EPS and way more for CFT}
    Lu = float
        Young propeller response length (m) {1.2m for EPS and 2m for CFT and 3.2m for PP}

    
    Output
    ----------
    ds: xarray dataset
        Input dataset with estimated tranfers functions 
    
    Example
    ----------
    ds = Kaimal(ds,z):
    
    Required packages
    ----------
    glob, os, itertools, pandas, xarray
    
    Required functions
    ----------
    
    
    """ 
    print('Calculating attenuation coefficient after Horst (1997)')
       

    # Prepare correction
    U          = (ds.u.values**2+ds.v.values**2)**(1/2)
    sigma_w    = (2*ds.ww.values)**(1/2)
    z          = ds.z.values
    
    if np.ndim(ds.obh) > 1:
        zeta = (ds.z.values/ds.obh[:,0,0,0].values) # Stability parameter
    else:
        zeta = (ds.z.values/ds.obh.values)
   
    # Write propeller response times
    tau_w      = Aw * (sigma_w / U)**(-2/3) / U
    tau_u      = Lu / U

    # Slope and interception parameters for peak normalized frequency of model cospectra Snuw and SnwT
    a = np.array([[0.41,      0.08], \
                  [0.8,      0.19]])
    
    # Peak frequency parameterization
    # Momentum flux
    fmuw             = (U / z)                          * (a[0,1] + a[0,0] * zeta)
    fmuw[zeta < 0.0] = (U[zeta < 0.0] / z[zeta < 0.0])  *  a[0,1] 
    fmuw[zeta > 0.2] = (U[zeta > 0.2] / z[zeta > 0.2])  * (a[0,1] + 0.2 * a[0,0])
    # Sensible heat flux
    fmwT             = (U / z)                          * (a[1,1] + a[1,0] * zeta)
    fmwT[zeta < 0.0] = (U[zeta < 0.0] / z[zeta < 0.0])  *  a[1,1] 
    fmwT[zeta > 0.2] = (U[zeta > 0.2] / z[zeta > 0.2])  * (a[1,1] + 0.2 * a[1,0])
        
    # Attenuation factor    
    wmuw = 2*np.pi*fmuw
    wmwT = 2*np.pi*fmwT
    
    rTuw = tau_w / tau_u
    rTwT = tau_w / tauT
    
    
    Guw = 1 + ( (rTuw*wmuw*tau_u)/(1+rTuw*wmuw*tau_u) ) * (1-rTuw)/(1+rTuw)
    GwT = 1 + ( (rTwT*wmwT*tauT)/(1+rTwT*wmwT*tauT) ) * (1-rTwT)/(1+rTwT)
    
    A_uw_H = (1/(1+wmuw*tau_u)) * Guw
    A_wT_H = (1/(1+wmwT*tauT)) * GwT

    A_uw_H   = xr.DataArray(A_uw_H,coords=[ds.time],  dims='time')   
    A_wT_H   = xr.DataArray(A_wT_H,coords=[ds.time],  dims='time')   

    return ds.assign({'A_uw_H': A_uw_H, 'A_wT_H': A_wT_H})
    
    
#_______________________________________________________
def dis_hgt(ds,method='FV'):
    """
    Estimated the displacement height using the method from Martano(2000)
    
    Input
    ----------
    ds: xarray dataset
    method: string
        Switch for method to estimate displacement height
        - FP-IT-1 : Iterative minimization of the variance of flux-profile similarity
        - FV-RE-1 : Regression on the flux-variance similarity


    """ 
    print('Calculating displacement height')
    
    if method == 'FP-IT-1' :
        # Initialise arrays
        z0m     = 1e-4
        d       = np.linspace(-5, 5, num=101)
        sigmaS2 = np.full((len(d),len(ds.time)), np.nan)
        
        # Calculate variance of S for different displacement heights
        for i in range(0,len(d)):
            S             = (0.4*ds.u/ds.ustar) + Psim((ds.z-d[i])/ds.obh) - Psim(z0m/ds.obh)
            sigmaS2[i,:]  = ((S-np.nanmean(S))**2).values
            
        ds['d'] = d[np.argmin(sigmaS2, axis = 0)]
    
    elif method == 'FV-RE-1':
        C1  = 1.3
        C2  = 2.0
        rms = np.full((len(d),len(ds.time)), np.nan)
        
        # Calculate variance of S for different displacement heights
        for i in range(0,len(d)):
            lhs       = ds.ww**0.5 / ds.ustar
            rhs       = C1*(1-C2*(ds.z-d[i]/ds.obh))**(1/3)
            rms[i,:]  = (lhs-rhs)**2
            
        ds['d'] = d[np.argmin(rms, axis = 0)]        
        
        

    
    return ds