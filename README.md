
![image info](./media/Eddie.png)
*Eddie, from the Ice Age movie*

# IMAU-IceEddie

Python toolkit to process data from automated weather stations and eddy covariance stations.

## Disclaimer
This code is not developped for universal usage and needs to be adapted for specific projects. 

## Introduction
Sometimes, the data that comes out of a data logger is not a nice continous, corrected, processed file. Instead it can contain values in Volts or in pulses, wind speeds in dubious coordinate systems, noise, etc...
This modest toolkit groups some useful functions to cope with these issues under the same package. This toolkit can be used as a modular framework for working with weather station or eddy covariance measurements.
 There is no GUI. Instead, switches and options are passed through a ```namoptions``` file that may be stored for later usage or reproducibility.

## Overview
This toolkit contains useful functions to process data from:
- Automatic Weather Stations 
- Eddy covariance data
- Cosmic ray neutron sensors 

The processing steps are grouped in several levels:

- ```L0``` : raw untouched data in original format (e.g. daily csv files or TOA5 files)
- ```L1A``` : raw uncorrected data, stored in  NetCDF files (monthly for eddy covariance data)
- ```L1B``` : corrected data

For the weather station data, one additionnal level is available:
- ```L2``` : corrected data including bulk turbulet fluxes

For the eddy-covariance data, two additionnal levels are available:
- ```L2``` : uncorrected time averages, (co)variances and raw (co)spectra
- ```L3``` : corrected time averages and (co)variances

Eddy covariance and AWS data can be merged to the same dataset:
- ```L3B``` : merged AWS and eddy covariance data, possibly including additional calculation such as fetch footprints

The code contains functions to process data from one level to a higher level.
![image info](./media/Code_Structure.png)


## Supported data
+ Weather station data (```input_type = 'WS'```)
    - IMAU Automatic weather station data
+ Eddy covariance data (```input_type = 'EC'```)
    - CSAT3/CSAT3B (Campbell) sonic anemoeters
    - Li-7500 (LiCOR) open path gas analyser
    - EC150 (Campbell) open path gas analyser
    - vertical propeller eddy covariance (VPEC)
+ Cosmic ray neutron sensor data (```input_type = 'snowfox'```)
    - SnowFox (Hydroinnova) data



## List of corrections / processing steps

### Eddy covariance data (```input_type = 'EC'```)

+  Level ```L0``` &rarr; Level ```L1A``` (```ll0tol1a = .true.```)
    - Correct wind direction for sensor orientation
    - Compute wind direction from 2D wind measurements
    - Correct units and variable names
    - Remove duplicated timestamps or unused data
    - Fill missing samples by NAN
    - Scale raw data based on calibration factors
    - Apply time offset

+  Level ```L1A``` &rarr; Level ```L1B``` (```ll1atol1b = .true.```)
    - Consistency checks based on physical thresholds and sensor quality flags
    - Outlier removal (de-spiking) using a median absolute deviation filter
    - Flagging of wind-distorted or missing data
    - Possibility for resampling raw data
    - Coordinate rotation based on planar-fit or double rotation per time interval

+  Level ```L1B``` &rarr; Level ```L2``` (```ll1btol2 = .true.```)
    - Detrending raw data per time interval
    - Computing (co)spectra, times averages and (co)variances
    - Import sensor height from level L1B AWS data
    - Computing normalised (co)spectra 
    - Computing transfer functions

+  Level ```L2``` &rarr; Level ```L3``` (```ll2tol3 = .true.```)
    - SND correction for sensible heat flux and sonic temperature
    - WPL correction for latent heat/CO2 flux
    - Spectral attenuation corrections
    - Computation of Obukhov length, roughness length, stability parameter

### AWS data  (```input_type = 'WS'```)
+ Level ```L0``` &rarr; Level ```L1A``` (```ll0atol1a = .true.```)
    - Correct units and variable names
    - Remove duplicated timestamps or unused data
    - Fill missing samples by NAN
    - Correct wind direction for sensor orientation
    - Apply time offset
+ Level ```L1A``` &rarr; Level ```L1B``` (```ll1atol1b = .true.```)
    - Shortwave radiation corrections (incination & bias)
    - Longwave radiation corrections (window heating & bias), only if not already applied to raw data
    - Temperature correction of sonic height ranger, only if not already applied to raw data

+  Level ```L1B``` &rarr; Level ```L2``` (```ll1btol2 = .true.```)
    - Computation of turbulent fluxes using Monin-Obukhov similarity theory

## How to use
* Make the following directory structure and place the raw data in the ```L0``` folder
```
LOC
├── ID
│   ├── L0
```
with ```LOC``` the geographical location (e.g. ```ANT``` for Antarctica) and with ```ID``` the station identifier  (e.g. ```AWS14```)
* Modify the ```namoptions``` file with your chosen settings and place it in the same folder as ```IceEddie.py```
* Run ```IceEddie.py```


## Example application 1: process custom AWS data 
+ Convert raw data to netCDF
    - Set ```input_type = 'WS``` and ```ll0tol1a = .true.``` in ```namoptions```
    - Change ```LOC```  and ```ID``` in ```namoptions```
    - Write a function in ```read_AWS.py``` that is called by the ```L0toL1A``` function for a certain value of ```Input_fname``` (set in ```namoptions```), e.g. ```Input_fname = "my_AWS_station_format"```
    - Run ```python IceEddie.py```

+ Apply corrections to your own AWS data
    - Set ```ll0tol1a = .false.``` and ```lL1AtoL1B = .true.``` in ```namoptions```
    - Change the ```sensor``` in ```namoptions```to your own custom sensor. E.g. ```sensor = 'PROMICE_v04'```
    - Adapt the ```L1AtoL1B``` function in ```read_AWS.py``` such that whe ```sensor = 'PROMICE_v04'```, correct variable names are set and all output variables are written to the output L1B netCDF file
    - Change any parameters in the ```AWScorr``` part of the ```namoptions``` depending on your applixation. For instance, include a meta.csv file that is used when ```lmetafile = .true.```
    - Run ```python IceEddie.py```

## Example application 2: process custom raw eddy covariance data 
+  Convert raw data to netCDF
    - Set ```input_type = 'EC``` and ```ll0tol1a = .true.``` in ```namoptions```
    - Change ```LOC```  and ```ID``` in ```namoptions```
    - Write a function in ```read_EC.py``` that is called by the ```L0toL1A``` function for a certain value of ```sensor``` (set in ```namoptions```), e.g. ```sensor = "CSAT3B"```.
    - Run ```python IceEddie.py```
+  Apply corrections on raw data
    - Set ```lll0tol1a = .false.```l and ```lL1AtoL1B = .true.``` in ```namoptions```
    - Adapt the ```L1AtoL1B``` function in ```read_EC.py``` such that the code functions for your entry of ```sensor```
    - Adapt the parameters in the ```L1AtoL1B``` part of ```namoptions``` for your application
    - Run ```python IceEddie.py```
+  Compute spectra and fluxes
    - Set ```lL1AtoL1B = .false.``` and ```lL1BtoL2 = .true.``` in ```namoptions```
    - Adapt the ```L1BtoL2``` function in ```read_EC.py``` such that the code functions for your entry of ```sensor```
    - Adapt the parameters in the ```L1BtoL2``` part of ```namoptions``` for your application
    - Run ```python IceEddie.py```


## Namoptions file
#### GLOBAL
| Item              | Default/example value  | Description | Possible entries |
| :---------------- | :------: | ----: |  ----: | 
| DIR       |   ''   | Main directory location containing eddy covariance data subfolders |
| L0dir           |   ''   | folder with level 0 raw eddy covariance data |
| L1B_30min_dir    |  ''   | folder with L1B AWS data used for crrecting eddy covariance data|
| L2_30min_dir |  ''   | folder with L2 AWS data used for SND correction of eddy covariance data |
| LOC |  'ANT'   | Geographical label ('ANT' for Antarctica, 'GRL' for Greenland, ...) |
| ID | 'AWS14'   |  Location identifier |   |
| sensor | 'ANT_AWS_Paul'   |  Switch for sensor type |
| version | '1.0'   |  Version number |
| input_type | 'WS' | Switch for data type |'EC'/'WS' / 'snowfox'|
|lL0toL2      |   .false.| Switch to run L0toL2 script | .true./.false. |
|lL0toL1A	 |    .false.| Switch to run L0toL1A script | .true./.false. |
|lmergeL1A	|	.false.| Switch to  merge L1A files  | .true./.false. |
|lmergeL3	|	 .false.| Switch to  merge L3 files  | .true./.false. |
|lL1AtoL1B	|	 .false.| Switch to run L1AtoL1B script | .true./.false. |
|lL1BtoL2	 |    .false.| Switch to run L1BtoL2 script | .true./.false. |
|l2toL3	   |  	 .false.| Switch to run L2toL3 script | .true./.false. |
|l3toL3B	 |    	 .false.| Switch to run L3toL3B script | .true./.false. |
|lL1B_to_EBM  |    .false.| Switch to convert L1B data to SEB model input | .true./.false. |
|lL1B_to_snowpack | .false. | Switch to convert L1B data to SNOWPACK input | .true./.false. |
|lEBM_to_csv   |   .false.| Switch to convert SEB model output to single csv file | .true./.false. |
|lEBM_AWS_to_csv  |   .false.| Switch to merge SEB model output and AWS data to single csv file | .true./.false. |
|L1B_to_grl_awsID | .false.|  Switch to convert L1B data to old grl_awsID format | .true./.false. |
|lav_csv|  .false.| Switch to compute 10 aily averages from merged csv files | .true./.false. |

#### AWScorr
| Item              | Default/example value  | Description | Possible entries | 
| :---------------- | :------: | ----: |  ----: | 
DIR		|		'' | Main directory location containing AWS data | |
L1B_precip_dir  | '' | folder with different L1B AWS data containing precipitation data  | |
L1B_WS_dir     | '' | folder with different L1B AWS data containing wind speed data  | |
L1B_RH_dir    | '' | folder with different L1B AWS data containing relative humidity  data  | |
L0dir		  | '' | folder with level 0 AWS data   | |
L0_raddir	| '' | folder with radiation data to complmemnt GC-Net data  | |
Input_fname	| '' | Switch for level 0 file name  | |
toffset_min    | 0 | time offset correction in minutes for L0toL1A script | |
d   			 | 0.0 | displacement height in meters for effective height calculation | |
H			| 0.0 | obstacle height in meters for effective height calculation | |
lSvdB			| .false | Switch for Smeets & van den Broeke (2008) parameterisation for scalar roughness lengths | .true./.false. |
z0_WD_table    | '' | Look-up table file containing z0m values per wind direction | |
lmetafile       | .true. | Switch for metadatafile used for L1AtoL1B script | .true./.false. |
lcalCNR4       | .false. | Switch to correct for calibration constants radiometer | .true./.false. |
lSMB            | .true. | Switch to compute SMB components| .true./.false. |
SMB             | .daily. |  Time interval for SMB componetns | 'daily'/'3hourly' |
lsolar          | .true. | Switch to compute solar angles | .true./.false. |
lcorr_lwheating  | .false. | Switch to correct longwave radiation for heating and bias  | .true./.false. |
pcorr_lwheating | 0.0 0.0 | coefficients for correction of longwave radiation for heating and bias  (only if lmeta = .false.) | a b |
lcorr_SWcoldbias | .false. | Switch to correct shortwave radiation bias  | .true./.false. |
pcorr_SWu_coldbias | 0.0 0.0 | coefficients for correction of upwards shortwave radiation bias  (only if lmeta = .false.)  | a b |
pcorr_SWd_coldbias| 0.0 0.0 | coefficients for correction of downwards shortwave radiation bias  (only if lmeta = .false.) | a b |
ldownsample     | .false. | Switch to compute 3hourly and daily data | .true./.false. |
pu              | 0.0 0.0 0.0 | coefficients for upper polynomial to compute cloud cover from LWd and T  | a b c |
pl              | 0.0 0.0 0.0 | coefficients for lower polynomial to compute  cloud cover from LWd and T  | a b c |
zrange          = 1.0 3.0  | bounds for sonic height ranger data (only if lmeta = .false.) | a b |
yyyymmdd_start  | "YYYY-MM-DD" | starting date  |  |
yyyymmdd_end   | "YYYY-MM-DD" | ending date  |  |
zfill         | 3.0 | value to fill missing height data  |  |
lcorrect_SWd   | .false. | Switch to correct SWd data using geometrical angles  | .true./.false. |
lfilter_height_histogram  | .false. | Switch to automatically remove secondary reflections from sonic height ranger  | .true./.false. |
l_correctRH     | .false. | Switch to correct RH data   | .true./.false. |
zsnow_start      | 0.0 | Initial value of surface height timeseries  | |
ldrop_ABL        | .false. | Switch to remove draw wire data   | .true./.false. |
zm_factor        | 0.0 | correction factor for sonic height ranger data (only if lmeta = .false.)  | |
alb_ice_certain  | 0.0 | albedo below which ice is certain at the surface, used for ablation calculation  | |
p0_constant    | 980 | fill value for air pressure  | |
EBMdir          | '' | location of SEB model output run 1  | |
EBMdir2         | '' | location of SEB model output run 2  | |
L1B_REF_dir    | '' | location of rerefece neutron data for snofox SWE calculation  | |
tref_start      | "YYYY-MM-DD" | starting date of reference period for snowfox SWE calculation  |  |
tref_end        | "YYYY-MM-DD" | end date of reference period for SWE calculation  |  |
lshift_time_EBM | 1 | Switch to move SEB model output by 1 timestep (fortran SEB model only)   | 1/0 |
lref_method    | 'L1B_ref_NMDB' | method used as referenc neutron data for snowfox SWE calculation  | 'L1B_ref'/'L1B_ref_NMDB'/'L1A' |
lref_method_var | 'N1Cts' | variable containing reference neutron data in L1A data (only if lref_method = 'L1A')  | 'N1Cts'/'L1B_ref_NMDB'/'L1A' |
sensor_height | 'surface_level_zm' | Switch to estoimate sensor height in time   | 'surface_level_zm'/'' |
csvdir  | '' | folder containing csv data for av_csv script | |
file_aws_locations  | '' | file containing LAT/LON data EBM_AWS_to_csv scripts | |
heights_dir | '' | folder containing heights data for EBM_AWS_to_csv_GRL script | |
L3B_dir | '' | folder containing L3B_dir data for EBM_AWS_to_csv_GRL script | |

#### AWS_to_SEB
| Item              | Default/example value  | Description | Possible entries | 
| :---------------- | :------: | ----: |  ----: | 
zfill     |     2.65 | fill value for sensor height data in SEB model input | 
z0fill     |     1e-4  | fill value for z0m data in SEB model input | 
yyyymmdd_start  | "YYYY-MM-DD" | starting date SEB model input |  |
yyyymmdd_end   | "YYYY-MM-DD" | ending date SEB model input  |  |
lSWdcorr | .false. | Switch to use SWd_corr data in SEB model input  | .true./.false. |
lSWucorr | .false. | Switch to use SWu_corr data in SEB model input  | .true./.false. |,
lalbcorr | .true. | Switch to use corrected albedo in SEB model input  | .true./.false. |
lzfixed | .false. | Switch to use fixed sensor height in SEB model input  | .true./.false. |
zfixed | 2.65 | Sensor height (if lzfixed = .true.) |  |
lperturb_alb  | .false. | Switch to perturb albedo in SEB model input  | .true./.false. |
perturb_alb | -0.02 | Albedo perturbation (if lperturb_alb = .true.) |  |
z_snow  | 0.0 | Initial value of Serie in SEB model input|  |
dT | '1H'  |SEB model input frequency |  |
serie | 'surface_level_zm' | Variable used for Serie variable | 'surface_level_zm'/'surface_level'/'surface_level_zs'
luse_LWraw | .false. | Switch to use raw LW data in SEB model input  | .true./.false. |
lcorr_dL | .false. | Switch to correct for dL in LW data in SEB model input  | .true./.false. |
luse_Traw | .false. | Switch to use raw T data in SEB model input  | .true./.false. |
luse_RHraw | .false. | Switch to use raw RH data in SEB model input  | .true./.false. |
luse_SWdraw | .false. | Switch to use raw SW data in SEB model input  | .true./.false. |
TSG | 255.0700 | Ground temeprature for SNOWPACK input |  |


#### L0toL1A
| Item              | Default/example value  | Description | Possible entries | 
| :---------------- | :------: | ----: |  ----: | 
toffset_method  | 'min' | method to shift L0 data in time | 'samples'/'min' |
toffset         | 0 | value for time offset | |
WDoffset   		| 0 | wind direction offset | | 
WDoffset_iWS   | 0 | wind direction offset for VPEC data (only if sensor == 'iWS+CSAT') | | 
WDoffset_CSAT	| 0  | wind direction offset for VPEC data (only if sensor == 'iWS+CSAT') | | 
frequency    	|'0.1S' | sampling period of L0 data  | | 
Gillfactor	    | 1.25 | correction factor Gill VPEC data | |
Youngfactor     | 1  | correction factor Young VPEC data | |
fnameprefix     | TOA5 | L0 files prefix (only if sensor == 'Custom') | |
sensorname      | 'CSAT'  | Type of sensor| 'CSAT'/'Young' |
ifiletype      | 'TOA' | Input file type (only for L0toL2 script)  | |
headers        | "" | header names of L0 files | |
yearstart      | 2021 | Start year tio convert time from doy (only for L0toL2 script)  | |
lreplace_24h | .false. | Switch to replace 24 by 00 | .true./.false. |


#### L1AtoL1B
| Item              | Default/example value  | Description | Possible entries | 
| :---------------- | :------: | ----: |  ----: | 
dT		    	| '30min' | Averaring period | |
WDvalid    	  | 0 360  |  Bounds of unobstructed wind directions | |
frequency   	| 10 |  Input frequency  | |
ldownsample     | .false. |  Switch to dowsample raw data  | .true./.false. |
lrolling        | .false. |  Switch to apply rolling average to raw data  | .true./.false. |
laverage        | .false.  |  Switch to average raw data  | .true./.false. |
shift_w_sample | 0 |  Number of sample to shifht vertical velocity samples (Gill data only)  |  |
Fd				| '4S' | Downsamling time period | | 
maxnanp         | 5 | Maximum percentage of allowed missing datab per time interval | |
Rotation 		| 'DR' | Method of coordinate rotation | 'PF'/'DR'/'' |
dT_PF 	| '1M'  | PLanar fit time period |  |
qMAD			| 7 	|  Tolerance factor for despiking filter  |  |
LoopsMAD	    | 1 |  Number of iteration of  despiking filter  |  |
maxdiag        | 100  |  Maximum diagnostic of valid raw data |  |

#### L1BtoL2
| Item              | Default/example value  | Description | Possible entries | 
| :---------------- | :------: | ----: |  ----: | 
dT		    	| '30min' | Averaging time | | 
WDvalid    	 	| 0 360 | Unobstructed wind directions | | 
inputdata		| L1B | Level of input data for block averaging | L1A/L1B |
lwritespectra   | .false. | Switch to write spectra to output | .true./.false. | 
spectra         | Kaimal | Type of reference normalised spectra | Kaimal/Cabauw/S10 | 
freq_range		| Wide | Frquency range for spectra calcultions  | CSAT/Input/Wide
lheightinfile   | .false. | Switch if sensor height is already present in input files |  .true./.false. | 
lfixedheight    | .false. | Switch if sensor height is constant |  .true./.false. | 
EC_hgt_offset   | 0.0 | height offset in meters applied to AWS height data (only used if lheightinfile = .false. and lfixedheight =  .false.)  |  | 
z			 	| 4.00 | height of sensors (only if lfixedheight = .true.)|  | 
H				| 0.0 | Obstacle height |  | 
d				| 0.0 | Displacement height |  | 
lcorrH          | .true. | Switch to correct sensor height for snow depth in AWS L1B data |  .true./.false. | 
ignore_samples_rhs | 0 | Number of sample to ignore on right hand side of L1B data (iWS data only)  | | 
ldetrend     	| .false. | Switch to apply  linear dentrending to data | | 
lhighpass		| .true. | Switch to include  high pass filter in spectra corrections| | 
Tsensor         | 'T1' | Variable name of temperature used to compute fluxes (VPEC only) |

#### L2toL3
| Item              | Default/example value  | Description | Possible entries | 
| :---------------- | :------: | ----: |  ----: | 
dT		    	| '30min' | Averaging time | | 
WDoffset   		| 0 | Wind direction offset correction | | 
toffset_min     | 0 | Time offset correction | | 
lfixedheight    | .false. | Switch if sensor height is constant |  .true./.false. | 
EC_hgt_offset   | 0.0 | height offset in meters applied to AWS height data (only used if lheightinfile = .false. and lfixedheight =  .false.)  |  | 
lcorrH          | .true. | Switch to correct sensor height for snow depth in AWS L1B data |  .true./.false. | 
ldetrend     	| .false. | Switch to apply  linear dentrending to data | | 
lhighpass		| .true. | Switch to include  high pass filter in spectra corrections| | 
spectra         | Kaimal | Type of reference normalised spectra | Kaimal/Cabauw/S10 | 
freq_range		| Wide | Frequency range for spectra calcultions  | CSAT/Input/Wide |
Aw				| 0.45 | Gill propeller response length  (VPEC only) | | 
Lu				| 1.78 | Young propeller response length  (VPEC only) | | 
tauT			| 0.13 | Thermocouple response time   (VPEC only) | | 
tau_wT          | 0.2 | Time delay between Gill and Thermocouple (VPEC only) | | 
tau_uw          | 0.5 | Time delay between Yonug and Gill (VPEC only) | | 
ldtau			| .false. | Switch to test different repsonse times |  .true./.false. | 
dAw 			| 0.10 | Perturbation for Aw (only if .ldtau. = .true.) | | 
dLu				| 0.2 | Perturbation for Lu (only if .ldtau. = .true.) | | 
dtauT			| 0.04 |  Perturbation for LutauT (only if .ldtau. = .true.) | | 
suw				| 0.5  |  Separation distance between u and w sensor   | | 
swT				| 0.0 |  Separation distance between w and T sensor   | | 
swq  			| 0.25 |  Separation distance between w and q sensor   | | 
pw				| 0.12  |  Path averaging of w sensor   | | 
pu				| 0.12 |  Path averaging of u sensor   | | 
pT				| 0.12 |  Path averaging of T sensor   | | 
pq				| 0.125 |  Path averaging of q sensor   | | 
lshadowing		| .false. | Switch to correct for sensor shadowing |  .true./.false. | 
lwpl			| .true.  | Switch to apply WPL correction |  .true./.false. | 
lwpl_method     | 'L2' | WPL correction method |  L2 | 
lsnd			| .true. | Switch to apply SND correction |  .true./.false. | 
lsnd_method     | 'L1B' | SND correction method |  CESAR/L2/L2_bulk/L2_AWS/L1B | 
lsidewind       | .false. | Switch to apply sidewind correction |  .true./.false. | 
lsidewind_method | 'Schotanus1983' |  sidewind correction method | Schotanus1983/Liu2001 | 
CESAR_dir       | ''  |  Directory containing  CESAR data (only if lsnd_method= CESAR)  | | 
nitera          | 1 |   Numr o iteration for VPEC attenuation corrections  | | 
Tsensor         | 'T1' | Variable name of temperature used to compute fluxes (VPEC only) |

#### L3toL3B
| Item              | Default/example value  | Description | Possible entries | 
| :---------------- | :------: | ----: |  ----: | 
L3sensor    | 'CSAT' | Sensor used to merge EC data iwht AWS data | CSAT/iWS/Young | 
lfp         | .false. | Switch to compute foorpint  |  .true./.false. | 
fp_domain   | -500 500 -500 500  | Domain of footprint calculation (omly if lfp = .true.)  |   | 
lbin_WD     | .false.   | Switch to compute wind drection averages   |  .true./.false. | 
fp_dx       | 5 | Footprint resolution (only if lfp = .true.)  |   | 
fp_WDbins   | 0 360 | Bounds of footprint wind direction averages   |  | 
fp_dWDbins  | 22.5 | Resolution of footprint wind direction averages   |  | 
fp_r        | 10 50 80  | Footprint flux radius   |  | 
lsave_all_fp | .false. | Switch to save all foorpints   |  .true./.false. | 
toffset_min     | 0 | Time offset correction | | 
toffset_min     | 0 | Height offset correction | | 
ldownsample | .false. | Switch to compute 3hourly and daily averages |  .true./.false. | 

### Additionnal information

For similar projects, please check:

* EddyPro: https://github.com/LI-COR-Environmental/eddypro-engine
* pypromice: https://github.com/GEUS-Glaciology-and-Climate/pypromice
* Pymicra: https://github.com/tomchor/pymicra
* hesseflux: https://mcuntz.github.io/hesseflux/html/index.html


If you have any questions, feel free to contact Maurice van Tiggelen, m.vantiggelen[@]uu.nl 