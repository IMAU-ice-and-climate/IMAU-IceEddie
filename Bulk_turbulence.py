#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Filename: Bulk_method.py
#Description: Some useful functions used to calculate surface fluxes using the bulk method
"""
Created on Fri Nov  9 15:58:19 2018

Maurice van Tiggelen
IMAU, Utrecht University, the Netherlands
m.vantiggelen@uu.nl
"""

####### LIBRARIES ##########
import numpy as np
import xarray as xr
import utils 

####### CUSTOM FUNCTIONS ##########

#_______________________________________________________   
def bulkflux(ds,SvdB=False,getOBL=True):
    """
    Calculates the turbulent surface fluxes using the bulk method

    
    Input
    ----------
    ds: xarray dataset
        dataset containing L1B corrected AWS data
    
    Output
    ----------
    ds: xarray dataset
        dataset containing L1B corrected AWS data and bulk fluxes
    
    """   

    # Find obukhov length
    if getOBL:
        ds['obl']      = getobl(ds,SvdB)
        
    
    # Initialise arrays
    Cm = np.full(np.shape(ds.z0m), np.nan)
    Ch = np.full(np.shape(ds.z0m), np.nan)
    Cq = np.full(np.shape(ds.z0m), np.nan)
    
    ram = np.full(np.shape(ds.z0m), np.nan)
    rah = np.full(np.shape(ds.z0m), np.nan)
    raq = np.full(np.shape(ds.z0m), np.nan)
    
    z0h = np.full(np.shape(ds.z0m), np.nan)
    z0q = np.full(np.shape(ds.z0m), np.nan)
    
    ustar   = np.full(np.shape(ds.z0m), np.nan)
    nu      = np.full(np.shape(ds.z0m), np.nan)
    Re_star = np.full(np.shape(ds.z0m), np.nan)
    wT      = np.full(np.shape(ds.z0m), np.nan)
    wq      = np.full(np.shape(ds.z0m), np.nan)
    H_bulk  = np.full(np.shape(ds.z0m), np.nan)
    LE_bulk = np.full(np.shape(ds.z0m), np.nan)
    
    # print(SvdB)
    
    
    k = -1
    try:
        # Loop over roughness lengths
        for (z0m, obl) in zip(ds.z0m.T.values, ds.obl.T.values):
            k = k+1
            print('     Iterating over z0: ' + str(k+1) + '/' + str(np.shape(ds.z0m.values)[1]))
            
            # Drag coefficient for momentum
            Cm[:,k]             = utils.fkar ** 2. / (np.log(ds.z / z0m) - utils.Psim(ds.z / obl) + utils.Psim(z0m / obl)) ** 2.
            
            # Aerodynamic resistance
            ram[:,k]            = 1. / ( Cm[:,k] * ds.U )
            
            # Friction velocity
            ustar[:,k]          = np.sqrt(ds.U / ram[:,k])
            
            # Air kinematic viscosity
            nu[:,k] = utils.nu_air(ds.T0,ds.p0,ds.qv0) 
        
            # Roughness Reynolds number
            Re_star[:,k] = ustar[:,k]*z0m/nu[:,k]
            
            # Scalar roughness lengths
            z0h[:,k], z0q[:,k]  = z0h_Andreas(z0m,ustar[:,k],ds.p0,ds.T0,ds.qv0,SvdB = SvdB)  
            
             # Drag coefficient for scalars
            Ch[:,k]             = utils.fkar ** 2. / ((np.log(ds.z / z0m) - utils.Psim(ds.z / obl) + utils.Psim(z0m / obl)) * \
                              (np.log(ds.z / z0h[:,k]) - utils.Psih(ds.z / obl) + utils.Psih(z0h[:,k] / obl)))
              
            Cq[:,k]             = utils.fkar ** 2. / ((np.log(ds.z / z0m) - utils.Psim(ds.z / obl) + utils.Psim(z0m / obl)) * \
                              (np.log(ds.z / z0q[:,k]) - utils.Psih(ds.z / obl) + utils.Psih(z0q[:,k] / obl)))
            
            # aerodynamic resistance for scalars
            rah[:,k]           = 1. / ( Ch[:,k] * ds.U )
            raq[:,k]           = 1. / ( Cq[:,k] * ds.U )
            
            # Trubulent Fluxes 
            wT[:,k]            = (ds.ths - ds.th0) / rah[:,k]
            wq[:,k]            = (ds.qvs - ds.qv0) / raq[:,k]
            
            # Convert turbulent fluxes to energy fluxes
            H_bulk[:,k]  = ds.rho_a * ds.Cp *  wT[:,k]
            LE_bulk[:,k] = ds.rho_a * utils.Ls(ds.T0)   *  wq[:,k]
        
        # Save arrays to dataset
        ds['Cm']      = xr.DataArray(Cm, dims=['time','z0'])
        ds['Ch']      = xr.DataArray(Ch, dims=['time','z0'])
        ds['Cq']      = xr.DataArray(Cq, dims=['time','z0'])
        
        ds['ustar']   = xr.DataArray(ustar, dims=['time','z0'])
        ds['wT']      = xr.DataArray(wT, dims=['time','z0'])
        ds['wq']      = xr.DataArray(wq, dims=['time','z0'])
        ds['H_bulk']  = xr.DataArray(H_bulk, dims=['time','z0'])
        ds['LE_bulk'] = xr.DataArray(LE_bulk, dims=['time','z0'])
        
        ds['nu']      = xr.DataArray(nu, dims=['time','z0'])
        ds['Re_star'] = xr.DataArray(Re_star, dims=['time','z0'])
        ds['z0h']     = xr.DataArray(z0h, dims=['time','z0'])
        ds['z0q']     = xr.DataArray(z0q, dims=['time','z0'])
        
    except:


        z0m = ds.z0m.values
        obl = ds.obl.values
        # Drag coefficient for momentum
        Cm            = utils.fkar ** 2. / (np.log(ds.z / z0m) - utils.Psim(ds.z / obl) + utils.Psim(z0m / obl)) ** 2.
        
        # Aerodynamic resistance
        ram           = 1. / ( Cm * ds.U )
        
        # Friction velocity
        ustar        = np.sqrt(ds.U / ram)
        
        # Air kinematic viscosity
        nu = utils.nu_air(ds.T0,ds.p0,ds.qv0) 
    
        # Roughness Reynolds number
        Re_star = ustar*z0m/nu
        
        # Scalar roughness lengths
        z0h, z0q  = z0h_Andreas(z0m,ustar,ds.p0,ds.T0,ds.qv0,SvdB = SvdB)  
        
         # Drag coefficient for scalars
        Ch            = utils.fkar ** 2. / ((np.log(ds.z / z0m) - utils.Psim(ds.z / obl) + utils.Psim(z0m / obl)) * \
                          (np.log(ds.z / z0h) - utils.Psih(ds.z / obl) + utils.Psih(z0h / obl)))
          
        Cq             = utils.fkar ** 2. / ((np.log(ds.z / z0m) - utils.Psim(ds.z / obl) + utils.Psim(z0m / obl)) * \
                          (np.log(ds.z / z0q) - utils.Psih(ds.z / obl) + utils.Psih(z0q / obl)))
        
        # aerodynamic resistance for scalars
        rah           = 1. / ( Ch * ds.U )
        raq           = 1. / ( Cq * ds.U )
        
        # Trubulent Fluxes 
        wT            = (ds.ths - ds.th0) / rah
        wq            = (ds.qvs - ds.qv0) / raq
        
        # Convert turbulent fluxes to energy fluxes
        H_bulk  = ds.rho_a * ds.Cp *  wT
        LE_bulk = ds.rho_a * utils.Ls(ds.T0)   *  wq
        
        # Save arrays to dataset
        ds['Cm']      = xr.DataArray(Cm, dims='time')
        ds['Ch']      = xr.DataArray(Ch, dims='time')
        ds['Cq']      = xr.DataArray(Cq, dims='time')
        
        ds['ustar']   = xr.DataArray(ustar, dims='time')
        ds['wT']      = xr.DataArray(wT, dims='time')
        ds['wq']      = xr.DataArray(wq, dims='time')
        ds['H_bulk']  = xr.DataArray(H_bulk, dims='time')
        ds['LE_bulk'] = xr.DataArray(LE_bulk, dims='time')
        
        ds['nu']      = xr.DataArray(nu, dims='time')
        ds['Re_star'] = xr.DataArray(Re_star, dims='time')
        ds['z0h']     = xr.DataArray(z0h, dims='time')
        ds['z0q']     = xr.DataArray(z0q, dims='time')


#_______________________________________________________   
def z0h_Andreas(z0m,ustar,p0,T0,qv0=0,SvdB = False):
    """
    Calculates the roughness length for heat and moisture using the Andreas(1987) parametrization

    
    Input
    ----------
    z0m: float 
        roughness length for momentum (m)_
    ustar: float
        friction velocity (m/s)
    p0: float
        atmospheric pressure (Pa)
    T0: float
        atmospheric temperature (K)
    SvdB: boolean
        switch to use the parameterisation from Smeets & van den Broeke(2008)
        
    
    Output
    ----------
    z0h: float
        roughness length for heat
    z0q: float
        roughness length for moisture
    
    Example
    ----------
    getobl()
    
    Required packages
    ----------
    glob, os, itertools, pandas
    
    Required functions
    ----------
    
    
    """   

    # Constants
    ch1 = 0.317    
    ch2 = -0.5651
    ch3 = -0.183
    ch4 = 0.149
    ch5 = -0.550
    ch6 = 1.250
    cq1 = 0.396
    cq2 = -0.512
    cq3 = -0.180
    cq4 = 0.351
    cq5 = -0.628
    cq6 = 1.610

    if SvdB:
        # Smeets and VdBroeke 2008
        # ch1 = 1.5    
        # ch2 = -0.2
        # ch3 = -0.11
        # cq1 = 1.5
        # cq2 = -0.2
        # cq3 = -0.11
        # New fit 2021 
        ch1 = 1.2    
        ch2 = -0.4
        ch3 = -0.15
        cq1 = 1.2
        cq2 = -0.4
        cq3 = -0.15
    z0h_min=1.0E-10
    z0q_min=1.0E-10
    
    # Air kinematic viscosity
    nu = utils.nu_air(T0,p0,qv0) 
    
    # Roughness Reynolds number
    Re = ustar*z0m/nu
    
    if isinstance(z0m, float):  
        if Re > 2.5:
            z0h = z0m*np.exp(ch1+ch2*np.log(Re)+ch3*(np.log(Re))**2)
            z0q = z0m*np.exp(cq1+cq2*np.log(Re)+cq3*(np.log(Re))**2)
        elif Re > 0.135:
            z0h = z0m*np.exp(ch4+ch5*np.log(Re))
            z0q = z0m*np.exp(cq4+cq5*np.log(Re))
        else:
            z0h = z0m*np.exp(ch6)
            z0q = z0m*np.exp(cq6)
    
        if (z0h<z0h_min): z0h = z0h_min
        if (z0q<z0q_min): z0q = z0q_min
        
        return z0h, z0q
        
    else:
        z0h = np.full(len(z0m),np.nan)
        z0q = np.full(len(z0m),np.nan)
        
        z0h[Re>2.5] = z0m[Re>2.5]*np.exp(ch1+ch2*np.log(Re[Re>2.5])+ch3*(np.log(Re[Re>2.5]))**2)
        z0q[Re>2.5] = z0m[Re>2.5]*np.exp(cq1+cq2*np.log(Re[Re>2.5])+cq3*(np.log(Re[Re>2.5]))**2)
        
        z0h[(Re<=2.5) & (Re>=0.135)] = z0m[(Re<=2.5) & (Re>=0.135)]*np.exp(ch4+ch5*np.log(Re[(Re<=2.5) & (Re>=0.135)]))
        z0q[(Re<=2.5) & (Re>=0.135)] = z0m[(Re<=2.5) & (Re>=0.135)]*np.exp(cq4+cq5*np.log(Re[(Re<=2.5) & (Re>=0.135)]))

        z0h[Re<0.135] = z0m[Re<0.135]*np.exp(ch6)
        z0q[Re<0.135] = z0m[Re<0.135]*np.exp(cq6)
        
        z0h[z0h < z0h_min] == z0h_min
        z0q[z0q < z0q_min] == z0q_min
        
        return  z0h, z0q
    

#_______________________________________________________   
def getobl(ds,SvdB=False):
    """
    Returns the estimated obukhov length using single-level and surface observations
    
    The calculated obukhov length minimises the difference between two different
    expressions for the bulk Richardson number (aka the Louis method)
    
    Input
    ----------
    ds: xarray dataset containing the following variables:
        p0: atmospheric pressure (Pa)
        T0: atmospheric temperature (K)
        Th0: atmospheric potential temperature (K)
        qv0: specific humidity (kg/kg)
        ths: surface potential temperature (K)
        qvs: surface specific humidity (kg/kg)
        U: atmospheric wind velocity (m/s)
        z0m: roughness length for momentum (m)

    
    Output
    ----------
    L: float
        Obukhov length (m)
    
    Example
    ----------
    getobl()
    
    Required packages
    ----------
    glob, os, itertools, pandas
    
    Required functions
    ----------
    
    
    """   
    
    print('Calculating Obukhov length')

      
    # Initiailise variables
    obl = np.full(np.shape(ds.z0m), np.nan)
    obl[0,:] = 0.1
    
    p0  = np.array(ds.p0)
    T0  = np.array(ds.T0)
    th0 = np.array(ds.th0)
    qv0 = np.array(ds.qv0)
    ths = np.array(ds.ths)
    qvs = np.array(ds.qvs)
    U   = np.array(ds.U)
    z0 = np.array(ds.z0m)
    z   = np.array(ds.z)

    k = -1
    # Loop over all roughness lengths
    for z0m in z0.T:
        k = k+1
        print('Iterating over z0: ' + str(k+1) + '/' + str(np.shape(z0)[1]))
        for i in range(len(ds.time)):  
            if any(np.isnan([p0[i],T0[i],th0[i],qv0[i],ths[i],qvs[i],U[i],z0m[i],z[i]])): continue     
            try:    
                # Defintion of virtual potential temperature
                thv0    = th0[i] * (1. + (utils.rv/utils.rd - 1.) * qv0[i])
                thvs    = ths[i] * (1. + (utils.rv/utils.rd - 1.) * qvs[i])
                
                U2 = U[i]**2
            
                # Definition of bulk Richardson number
                Rib   = utils.grav / thvs * z[i] * (thv0 - thvs) / U2
            
                # Initialisation 
                itera = 0
                L    = obl[i,k]
                if np.isnan(obl[i,k]): L = 0.1
                
                 
               
               
                if((Rib * L < 0) | (abs(L) == 1e5)):
                  if(Rib > 0): L = 0.01
                  if(Rib < 0): L = -0.01
            
            
                while (True):
                  itera    = itera + 1
                  
                  Lold = L
                  # Drag coefficient for momentum
                  Cm = utils.fkar ** 2. / (np.log(z[i] / z0m[i]) - utils.Psim(z[i] / L) + utils.Psim(z0m[i] / L)) ** 2
                  
                  # Aerodynamic resistance for momentum
                  ram = 1. / ( Cm * U[i] )      
                  
                  # Friction velocity
                  ustar = np.sqrt(U[i] / ram)      
                  
                  # Scalar roughness lengths
                  z0h, z0q = z0h_Andreas(z0m[i],ustar,p0[i],T0[i],qv0[i],SvdB = SvdB)  
                  
                  # Calculate new obukhov length
                  fx       = Rib - z[i] / L * (np.log(z[i] / z0h) - utils.Psih(z[i]/ L) + utils.Psih(z0h / L)) / \
                  (np.log(z[i] / z0m[i]) - utils.Psim(z[i] / L) + utils.Psim(z0m[i] / L)) ** 2
                  Lstart   = L - 0.001*L
                  Lend     = L + 0.001*L
                  fxdif    = ( (- z[i] / Lstart * (np.log(z[i] / z0h) - utils.Psih(z[i] / Lstart) + utils.Psih(z0h / Lstart)) / \
                  (np.log(z[i] / z0m[i]) - utils.Psim(z[i] / Lstart) + utils.Psim(z0m[i] / Lstart)) ** 2.) - (-z[i] / Lend * (np.log(z[i] / z0h) \
                  - utils.Psih(z[i] / Lend) + utils.Psih(z0h / Lend)) / (np.log(z[i] / z0m[i]) - utils.Psim(z[i] / Lend) \
                  + utils.Psim(z0m[i] / Lend)) ** 2.) ) / (Lstart - Lend)
                  L        = L - fx / fxdif
                  if(((Rib * L) < 0.) | (abs(L) == 1e5)):
                    if(Rib > 0): L = 0.01
                    if(Rib < 0): L = -0.01
                  if(abs((L - Lold)/L) < 1e-3): break
                  if(itera > 1000):
                      L = np.nan
                      print('Obukhov length calculation does not converge!')
                      break 
                if (abs(L)>1e6): L = 1.0e6 * np.sign(L)
            
            
                obl[i,k] = L

        
            except TypeError:
                continue
        
    return xr.DataArray(obl, dims=['time','z0'])

#_______________________________________________________
def is_snow(ds):
    '''
    Find when surface is snow covered based on accumulated albedo
    '''
    
    issnow = np.ones(len(ds.time), dtype=bool)
    
    for i in range(len(ds.time)):  
        
        if np.isnan(ds.albedo_acc[i]):                  issnow[i] = issnow[i-1]
        if (ds.albedo_acc[i]>1) | (ds.albedo_acc[i]<0): issnow[i] = issnow[i-1]
        if (ds.albedo_acc[i] >= 0.7):                   issnow[i] = True
        if (ds.albedo_acc[i] < 0.7):                    issnow[i] = False
        
    ds['issnow']     = xr.DataArray(issnow, dims=['time'])