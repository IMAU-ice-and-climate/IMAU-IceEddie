#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IceEddie main script

Main function that executes subparts of IceEddie. 
The switches in the namoptions file determine which part of the code is activated

Created on Sun Sep 29 18:58:32 2019

@author: Maurice van Tiggelen (Utrecht University)
"""

import read_EC as rec
import read_AWS as raws
import f90nml

nml = f90nml.read('namoptions')

if (nml['global']['input_type'] == 'EC'):

    if nml['global']['ll0tol1a']:
        print('L0---->L1A')
        rec.L0toL1A(nml)
        
    if nml['global']['lmergeL1A']:
        print('Merging L1A files')
        rec.mergeL1A(nml)
        
    if nml['global']['lmergeL3']:
        print('Merging L3 files')
        rec.mergeL3(nml)
        
    if nml['global']['ll0tol2']:
        print('L0---->L2')
        rec.L0toL2(nml)
        
    if nml['global']['ll1atol1b']:
        print('L1A---->L1B')
        rec.L1AtoL1B(nml)
        
    if nml['global']['ll1btol2']:
        print('L1B---->L2')
        rec.L1BtoL2(nml)
        
    if nml['global']['l2tol3']:
        print('L2---->L3')
        rec.L2toL3(nml)

    if nml['global']['l3tol3B']:
        print('L3---->L3B')
        raws.L3toL3B(nml)


if (nml['global']['input_type'] == 'WS'):

    if nml['global']['ll0tol1a']:
        print('L0---->L1A')
        raws.L0toL1A(nml)
        
    if nml['global']['lmergeL1A']:
        print('Merging L1A files')
        raws.mergeL1A(nml)
    
    if nml['global']['ll1atol1b']:
        print('L1A---->L1B')
        raws.L1AtoL1B(nml)
            
    if nml['global']['lL1B_to_EBM']:
        raws.L1B_to_EBM(nml)

    if nml['global']['lL1B_to_snowpack']:
        raws.L1B_to_snowpack(nml)

    
    if nml['global']['lEBM_AWS_to_csv']:
        if nml['global']['LOC'] == 'ANT':
            raws.EBM_AWS_to_csv(nml)
        elif nml['global']['LOC'] == 'GRL':
            raws.EBM_AWS_to_csv_GRL(nml)
        elif nml['global']['LOC'] == 'SVB':
            raws.EBM_AWS_to_csv_GRL(nml)
        else:
            print('no code implemented')

    if nml['global']['lEBM_to_csv']:
        if nml['global']['LOC'] == 'ANT':
            raws.EBM_to_csv(nml)
        else:
            print('no code implemented')

    if nml['global']['L1B_to_grl_awsID']:
        raws.L1B_to_grl_awsID(nml)
        
    if nml['global']['ll1btol2']:
        print('L1B---->L2')
        raws.L1BtoL2(nml)
    
    if nml['global']['l3tol3B']:
        print('L3---->L3B')
        raws.L3toL3B(nml)

    if nml['global']['lav_csv']:
        print('Averaging csv files')
        raws.lav_csv(nml)
        
        
if (nml['global']['input_type'] == 'snowfox'):

    if nml['global']['ll0tol1a']:
        print('L0---->L1A')
        raws.L0toL1A(nml)
    
    if nml['global']['ll1atol1b']:
        print('L1A---->L1B')
        raws.L1AtoL1B_snowfox(nml)

    if nml['global']['lmergeL1A']:
        print('Merging L1A files')
        raws.mergeL1A(nml)