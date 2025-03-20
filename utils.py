#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Filename: utils.py
#Description: Some utility functions and constants
"""
Created on Mon Mar 11 17:13:09 2019

Maurice van Tiggelen
IMAU, Utrecht University, the Netherlands
m.vantiggelen@uu.nl
"""

####### LIBRARIES ##########

import numpy as np
import xarray as xr
import json
import collections

### FOLDERS ###
JSON_path = '/Users/Tigge006/surfdrive/04_Scripts/Python/IMAU-IceEddie/JSON formats/' 


### CONSTANTS ###
boltz  = 5.67e-8     # Stefan-Boltzmann constant (W/m^2/K^4)
grav   = 9.81        # Acceleration due to Earth's gravity at sea level (m/s^2 )
R      = 8.3143      # Universal gas constant (J/kg/mol)
rd     = 287.05      # Gas constant for dry air (J/kg/K)
rv     = 461.5       # Gas constant for water vapour (J/kg/K)
cpd    = 1004.7      # Specific heat capacity of dry air at constant pressure (J/kg/K)
cpv    = 1849        # Specific heat capacity of water vapor at constant pressure (J/kg/K)
cv     = 719         # Specific heat capacity of dry air at constant volume (J/kg/K)
gammad = cpd/cv      # CInstant used for the speed of sound in air
ci     = 2106        # Specific heat capacity of ice at 0 degrees C (J/kg/K)
p0     = 1000e2      # Reference pressure in exner function (Pa)
lv0    = 2.5008e6    # Latent heat of vaporization at 0 degrees C (J/kg)
lv100  = 2.2500e6    # Latent heat of vaporization at 100 degrees C (J/kg)
ls0    = 2.8345e6    # Latent heat of sublimation at 0 degrees C (J/kg) 
lf     = 3.3400e5    # Latent heat of fusion (J/kg) 
fkar   = 0.40        # Von Karman constant (-)
es0    = 6.11657e2   # Saturation water vapor partial pressure  at 0 degrees C (Pa)
Tt     = 273.16      # Triple point temperature of water (K)
at     = 17.2694     # Teten's formula constant (-)
bt     = 35.86       # Teten's formula constant (K)
kappa  = 0.4         # Von Karman constant
rho_s  = 400         # fresh snow density [kg m-3]
rho_i  = 910         # ice density [kg m-3]
rho_w  = 1000        # water denisty [kg m-3]
####### CUSTOM FUNCTIONS ##########
## FILE FORMATS ##
#_______________________________________________________
def read_ordered_json(file):
    """Return json file as an ordered dict"""
#    path = relative_path(path)
    decoder = json.JSONDecoder(object_pairs_hook=collections.OrderedDict)
    with open(file) as stream:
        return decoder.decode(stream.read())
    

#_______________________________________________________
def Add_dataset_attributes(ds,JSON_type):
    '''
    Open attributes in a JSON file and add attributes to the dataset, and to all it's known variables 
    '''
    file      = JSON_path + JSON_type
    attr_dict = read_ordered_json(file)
    ds.attrs  = attr_dict.pop('attributes')
    
    for key, value in attr_dict.items():
        for key1, value1 in value.items():
            if (key1 in ds.variables):
                for key2, value2 in value1.items():
                    if key2 == 'type':
                        pass
                    else:
                        ds[key1].attrs = value2.items()
            if (key1 in ds.coords):
                for key2, value2 in value1.items():
                    if key2 == 'type':
                        pass
                    else:
                        ds[key1].attrs = value2.items()

    return ds


## THERMODYNAMIC VARIABLES AND PARAMETERIZATIONS ##
#_______________________________________________________
def Cp(q=0):
    '''
    Returns the specific heat of moist air at constant pressure
    
    Input
    ----------
    q: float
        specific humidity (kg/kg)
    
    Output
    ----------
    Cp: float
        pecific heat at constant pressure (J/kg/K)
    '''
    return cpd * (1 - q) + cpv * q
    
#_______________________________________________________
def Lv(T=273.15):
    '''
    Returns the latent heat of vaporization at temperature T
    
    Input
    ----------
    T: float
        atmospheric temperature (K)
    
    Output
    ----------
    Lv: float
        latent heat of vaporization (J/kg)
    '''
    return lv0


#_______________________________________________________
def Ls(T=273.15):
    '''
    Returns the latent heat of sublimation at temperature T
    
    Input
    ----------
    T: float
        atmospheric temperature (K)
    
    Output
    ----------
    Ls: float
        latent heat of sublimation (J/kg)
    '''
    return ls0

    
#_______________________________________________________   
def nu_air(T,p,qv):
    """
    Calculates the kinematic viscosity of air using Sutherland's law
    nu = mu / rho
    where:
    nu: kinematic viscosity (m^2/s)
    mu: dynamic viscosity (kg/m/s)
    rho: density (kg/m^3)
    
    Input
    ----------
    T: float 
        atmospheric temperature (K)
    p: float
        atmospheric pressure (Pa)
    qv: float
        water vapor specific humidity (kg/kg)

    
    Output
    ----------
    nu: float
        kinematic viscosity  (m^2/s)
        
    """
    # Global constants
    rd    = 287.05

    # Sutherland's law parameters for air
    mu0  = 1.716e-5 #1.7894e-5 # (kg/m/s)
    T0   = 273.11    # (K)
    S    = 110.56    # (K)
    
    # Sutherland's law
    mu = mu0 * (T/T0)**(3/2) * (T0 + S)/(T + S) 
    
    # Definition of virtual temperature
    Tv    = T * (1. + (rv/rd - 1.) * qv)
    
    # Air density
    rho   = p/(rd*Tv)
    
    # Kinematic viscosity
    return mu / rho

## THERMODYNAMIC VARIABLE CONVERSIONS ##
#_______________________________________________________
def RH2qv(RH,p,T):
    """
    Returns the water vapor specific humidity
    Different relations for the saturation water vapor pressure can be implemented
    If temperatures are below freezing, saturation is calculated with respect to ice
    
    Input
    ----------
    RH: float
        relative humidity (%)
    p: float
        atmospheric pressure (Pa)
    T: float 
        atmospheric temperature (K)
    
    Output
    ----------
    qv: float
        specific humidity  (kg/kg)
        
    """
    
    es = np.full(len(T), np.nan)
    
    
    # Convert data arrays to numpy arrays
    RH = np.array(RH)
    p  = np.array(p)
    T  = np.array(T)    
    
    

    # Water vapor pressure at saturation (Pa)
    
    # with respect to ice (Magnus formula, Springer Handbook of Atmsopheric measurements)
    es[T<273.15] = es0 * np.exp(22.46*(T[T<273.15]-Tt)/(T[T<273.15]-Tt+272.62))
    # with respect to liquid water (Magnus formula, Springer Handbook of Atmsopheric measurements)
    es[T>=273.15] = es0 * np.exp(17.62*(T[T>=273.15]-Tt)/(T[T>=273.15]-Tt+243.12))
  
    # Saturaton specific humidity
    qs = (rd/rv) * es/(p+es*(rd/rv-1))
      
    # Water vapour specific humidity
    return xr.DataArray((RH/100)*qs, dims=['time'])

#_______________________________________________________
def RH2qv_water(RH,p,T):
    """
    Returns the water vapor specific humidity with respect to liquid water
    
    Input
    ----------
    RH: float
        relative humidity with respect to liquid water (%)
    p: float
        atmospheric pressure (Pa)
    T: float 
        atmospheric temperature (K)
    
    Output
    ----------
    qv: float
        specific humidity  (kg/kg)
        
    """
    
    es = np.full(len(T), np.nan)
    
    
    # Convert data arrays to numpy arrays
    RH = np.array(RH)
    p  = np.array(p)
    T  = np.array(T)    
    
    
    # Water vapor pressure at saturation (Pa)
    # with respect to liquid water
    es = es0 * np.exp(17.62*(T-Tt)/(T-Tt+243.12))

    # Saturaton specific humidity
    qs = (rd/rv) * es/(p+es*(rd/rv-1))
      
    # Water vapour specific humidity
    return xr.DataArray((RH/100)*qs, dims=['time'])    


#_______________________________________________________
def qv2RH(qv0,p,T):
    """
    Returns the water vapor specific humidity
    Different relations for the saturation water vapor pressure can be implemented
    If temperatures are below freezing, saturation is calculated with respect to ice
    
    Input
    ----------
    qv0: float
        specific humidity  (kg/kg)
    p: float
        atmospheric pressure (Pa)
    T: float 
        atmospheric temperature (K)
    
    Output
    ----------
    RH: float
        relative humidity (%)
        
    """
    
    es = np.full(len(T), np.nan)
    
    
    # Convert data arrays to numpy arrays
    qv0  = np.array(qv0)
    p  = np.array(p)
    T  = np.array(T)    
    
    
    # Water vapor pressure at saturation (Pa)
    # with respect to ice (Magnus formula, Springer Handbook of Atmsopheric measurements)
    es[T<273.15] = es0 * np.exp(22.46*(T[T<273.15]-Tt)/(T[T<273.15]-Tt+272.62))
    # with respect to liquid water (Magnus formula, Springer Handbook of Atmsopheric measurements)
    es[T>=273.15] = es0 * np.exp(17.62*(T[T>=273.15]-Tt)/(T[T>=273.15]-Tt+243.12))

    # Saturaton specific humidity
    qs = (rd/rv) * es/(p+es*(rd/rv-1))
      
    # Water vapour specific humidity
    return xr.DataArray((qv0/qs)*100, dims=['time'])    

#_______________________________________________________
def T2thl(T,p,ql=0):
    '''
    Converts temperature to liquid water potential temperature
    
    Input
    ----------
    T: float
        atmospheric temperature (K)
    p: float
        atmospheric pressure (Pa)
    ql: float
        liquid water specific humidity (kg/kg)
    
    Output
    ----------
    thl: float
        liquid water potential temperature (K)
    '''
    # exner function
    exner = (p/p0)**(rd/Cp(q=0))  
    # potential temperature
    th    = T/exner  
    # liquid water potential temperature
    thl   = th*np.exp(-Lv(T)*ql/(cpd*T));
        
    return thl

#_______________________________________________________
def rho_air_dry(T,p):
    '''
    Calculates dry air density using the gas law
    
    Input
    ----------
    T: float
        atmospheric temperature (K)
    p: float
        atmospheric pressure (Pa)
    
    Output
    ----------
    rho_air: float
        density of dry air (kg/m^3)
    '''
    # Air density (gas law)
    return p / (rd * T)

#_______________________________________________________
def rho_air(T,p,qv):
    '''
    Calculates air density using the gas law
    
    Input
    ----------
    T: float
        atmospheric temperature (K)
    p: float
        atmospheric pressure (Pa)
    qv: float
        water vapor specific humidity (kg/kg)
    
    Output
    ----------
    rho_air: float
        density of moist air (kg/m^3)
    '''
    # Virtual temperature
    Tv   = T * (1. + (rv/rd - 1.) * qv)
    # Air density (gas law)
    return p / (rd * Tv)


#_______________________________________________________
def rv2qv(rv):
    '''
    Converts the mixing ratio to specific humidity
    '''
    return rv/(1+rv)
#_______________________________________________________
def qv2rv(qv):
    '''
    Converts the specific humidity to the mixing ratio
    '''
    return qv/(1-qv)    
#_______________________________________________________
def ev2rv(ev,p):
    '''
    Converts partial pressure of water vapour
    to mixing ratio using Dalton's law
    '''
    return (rd/rv) * ev/(p+ev*(rd/rv-1))

#_______________________________________________________
def TD2RH(Td,T):
    '''
    Converts dewpoint temperature to relative humidity 
    '''

    RH = np.full(len(T), np.nan)

    # with respect to ice (Magnus formula, Springer Handbook of Atmsopheric measurements)
    RH[T<273.15] = 100 * np.exp(22.46 * (Td[T<273.15]-Tt)/(Td[T<273.15]+272.62-Tt)) / np.exp(22.46 * (T[T<273.15]-Tt)/(T[T<273.15]+272.62-Tt))

    # with respect to liquid water (Magnus formula, Springer Handbook of Atmsopheric measurements)
    RH[T>=273.15] = 100 * np.exp(17.62 * (Td[T>=273.15]-Tt)/(Td[T>=273.15]+243.12-Tt)) / np.exp(17.62 * (T[T>=273.15]-Tt)/(T[T>=273.15]+243.12-Tt))

    return RH

#_______________________________________________________
def p2z_std(p):
    '''
    Converts dewpoint temperature to relative humidity using standard US atmosphere
    '''

    t0 = 288.0
    p0 = 1013.25
    gamma = 6.5

    return (t0 / gamma) * (1 - (p / p0)**(rd * gamma / grav))

#_______________________________________________________
def z2p( z,T,qv,ps,method):
    # z2p Calculates pressure at different altitudes assuming hydrostatic equilibirum and
    # the gas law:
    # p(i+1) = p[i]exp(-gdz/(RdTv))
    #  OR using potential temperatures:
    # p(i+1)**(Rd/Cp)=p[i]**(Rd/Cp)-g(p0)**(Rd/Cp)*dz/(Cp*thv)

    # Allocate memory
    p = np.zeros((len(z),1))
    # Constants
    g=9.81
    Rd=287.06
    p0=1e5
    Cp=1004.703
    if method == 'T': # Use standard temperature
        # Virtual temperature [K]
        Tv=T*(1+0.608*qv)
        for i in range(len(z)):
            if i==0:
                Tv_mid = Tv[0]
                dz = z[0]
                p[0]=ps*np.exp(-g*dz/(Rd*Tv_mid))
            else:
                Tv_mid = (Tv[i]+Tv[i-1])/2
                dz = z[i]-z[i-1]
                p[i]=p[i-1]*np.exp(-g*dz/(Rd*Tv_mid))

            
    if method == 'theta': # Use potential temperature
        thv=T*(1+0.608*qv)
        for i in range(len(z)):
            if i==0:
                thv_mid = thv[0]
                dz = z[0]
                p[i]=ps**(Rd/Cp)-g*p0**(Rd/Cp)*dz/(Cp*thv_mid)
                p[i]=p[i]**(1/(Rd/Cp))
            else:
                thv_mid = (thv[i]+thv[i-1])/2
                dz = z[i]-z[i-1]
                p[i]=p[i-1]**(Rd/Cp)-g*p0**(Rd/Cp)*dz/(Cp*thv_mid)
                p[i]=p[i]**(1/(Rd/Cp))
    return p

#_______________________________________________________
def p2z( p,T,qv):
    # p2z Calculates the thickness of a layer using the hypsometric equation


    # Allocate memory
    z = np.zeros((len(p),1))

    # Virtual temperature [K]
    Tv=T*(1+0.608*qv)
    for i in range(len(p)):
        if i==0:
            z[0]=p0
        else:
            Tv_mid = (Tv[i]+Tv[i-1])/2
            dz = z[i]-z[i-1]
            z[i] = z[i-1] + (rd*Tv_mid/grav)*np.log(p[i-1]/p[i])
    return z

## FLUX-PROFILE STABILITY FUNCTIONS ##
#_______________________________________________________        
def Psim(zeta,unstable='Högström',stable='BeljaarsHoltslag'):
    """
    Returns the integrated flux-profile function for momentum evaluated at zeta
    
    Input
    ----------
    zeta: float
        stability parameter
    unstable: str
        function to use in case of unstable stratification
    stable: str
        function to use in case of stable stratification
    
    Output
    ----------
    float 
    
    Example
    ----------
    Psim(z/L)
    
    Required packages
    ----------
    glob, os, itertools, pandas
    
    Required functions
    ----------
    
    
    """           
    
    if isinstance(zeta,float):
        if zeta < 0:
                
            if unstable == 'Högström':     
                x             = (1 - 16*zeta)**(1/4)
                return np.log( ((1 + x**2)/2) * ((1 + x)/2)**2 ) \
                - 2*np.arctan(x) + np.pi/2
               
        elif zeta >= 0:
            if stable == 'BeljaarsHoltslag':
                a = 1
                b = 2/3
                c = 5
                d = 0.35
                
                return -(b*c/d + a*zeta + b*(zeta - (c/d))*np.exp(-d*zeta))
            
    else:
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
            
            out[zeta>=0] = -(b*c/d + a*zeta[zeta>=0] + b*(zeta[zeta>=0] - (c/d))*np.exp(-d*zeta[zeta>=0]))
       
        return out



        
#_______________________________________________________        
def Psih(zeta,unstable='Högström',stable='BeljaarsHoltslag'):
    """
    Returns the integrated flux-profile function for heat evaluated at zeta
    
    Input
    ----------
    zeta: float 
        stability parameter
    unstable: str
        function to use in case of unstable stratification
    stable: str
        function to use in case of stable stratification
    
    Output
    ----------
    float 
    
    Example
    ----------
    Psim(z/L)
    
    Required packages
    ----------
    glob, os, itertools, pandas
    
    Required functions
    ----------
    """
    
    if isinstance(zeta, float):  
        
        if zeta < 0:
            
            if unstable == 'Högström':     
                x             = (1 - 16*zeta)**(1/4)
                return 2*np.log( (1 + x**2)/2 ) 
               
        elif zeta >= 0:
            
            if stable == 'BeljaarsHoltslag':
                a = 1
                b = 2/3
                c = 5
                d = 0.35
                return -(b * (zeta - c/d)*np.exp(-d * zeta) +  \
                (1 + b * a * zeta) ** (1.5) + (b*c/d) - 1)
                
    else:

        out = np.full(np.shape(zeta), np.nan)
        
        if unstable == 'Högström':     
            x             = (1 - 16*zeta)**(1/4)
            out[zeta<0]  = 2*np.log( (1 + x[zeta<0]**2)/2 ) 
               
        if stable == 'BeljaarsHoltslag':
            a = 1
            b = 2/3
            c = 5
            d = 0.35
            out[zeta>=0] = -(b * (zeta[zeta>=0] - c/d)*np.exp(-d * zeta[zeta>=0]) +  \
            (1 + b * a * zeta[zeta>=0]) ** (1.5) + (b*c/d) - 1)
       
        return out
    
