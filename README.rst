##########
obsio
##########

**obsio** is an Python package that provides a consistent generic interface for
accessing weather and climate observations from multiple different data 
providers. All station and observation data are returned using pandas data
structures. **obsio** is currently in pre-alpha stage and undergoing active development.

Installation
=============
**obsio** has the following dependencies:

* lxml_
* netCDF4_
* numpy_
* pandas_
* pycurl_
* pytz_
* scipy_
* suds_
* tzwhere_
* xray_

The easiest method to install the required dependencies is with a combination
of conda_ and pip_:

::

	conda create -n obsio_env python=2 lxml ipython netCDF4 numpy pandas pycurl pytz scipy xray
	pip install suds
	pip install tzwhere

And then install obsio from source:

::

	python setup.py install

.. _lxml: http://lxml.de/
.. _netCDF4: https://github.com/Unidata/netcdf4-python
.. _numpy: http://www.numpy.org/
.. _pandas: http://pandas.pydata.org/
.. _pycurl: http://pycurl.sourceforge.net/
.. _pytz: http://pythonhosted.org/pytz/
.. _scipy: http://www.scipy.org/
.. _suds: https://pypi.python.org/pypi/suds
.. _tzwhere: https://pypi.python.org/pypi/tzwhere/
.. _xray: http://xray.readthedocs.org/en/stable/
.. _conda: http://conda.pydata.org/docs/
.. _pip: https://pypi.python.org/pypi/pip

Available Data Providers
=============
**obsio** currently has full or partial support for a number of climate and
weather data providers. Only daily and monthly elements are supported at this
time, but hourly and sub-hourly can easily be added.

+---------------+-----------------------------------------+--------------------+
| Provider Name | Currently Supported Elements            | Req. Local Storage |
+===============+=========================================+====================+
| ACIS_	        | tmin,tmax,prcp,tobs_tmin,tobs_tmax,	  |	No             |
|               | tobs_prcp                               |                    |
+---------------+-----------------------------------------+--------------------+
| GHCN-D_       | tmin,tmax,prcp,tobs_tmin,tobs_tmax,     | Yes                |
|               | tobs_prcp                               |                    |
+---------------+-----------------------------------------+--------------------+
| ISDLite_      | tmin,tmax,tdew,tdewmin,tdewmax,vpd,     | No                 |
|               | vpdmin,vpdmax,rh,rhmin,rhmax,prcp       |                    |
+---------------+-----------------------------------------+--------------------+
| MADIS_        | tmin,tmax,prcp,tdew,tdewmin,tdewmax,    | Yes                |
|               | vpd,vpdmin,vpdmax,rh,rhmin,rhmax,srad,  |                    |
|               | wspd                                    |                    |
+---------------+-----------------------------------------+--------------------+
| NRCS_         | tmin,tmax,prcp,snwd,swe                 | No                 |
+---------------+-----------------------------------------+--------------------+
| USHCN_	| \*\_mth_raw,\*\_mth_tob,\*\_mth_fls     | Yes                |
+---------------+-----------------------------------------+--------------------+
| WRCC_		| tmin,tmax,tdew,tdewmin,tdewmax,vpd,     | No                 |
|               | vpdmin,vpdmax,rh,rhmin,rhmax,prcp,srad, |                    |
|               | wspd                                    |                    |
+---------------+-----------------------------------------+--------------------+

Element definitions:

* tmin : daily minimum temperature (C)
* tmax : daily maximum temperature (C)
* tdew : daily average dewpoint (C)
* tdewmin : daily minimum dewpoint (C)
* tdewmax : daily maximum dewpoint (C)
* vpd : daily average vapor pressure deficit (Pa)
* vpdmin : daily minimum vapor pressure deficit (Pa)
* vpdmax : daily maximum vapor pressure deficit (Pa)
* rh : daily average relative humidity (%)
* rhmin : daily minimum relative humidity (%)
* rhmax : daily maximum relative humidity (%)
* prcp : daily total precipitation (mm)
* srad : daily 24-hr average incoming solar radiation (w m-2)
* wspd : daily average windspeed (m s-1)
* snwd : snow depth (mm)
* swe : snow water equivalent (mm)
* tobs_tmin : time-of-observation for daily tmin (local hr)
* tobs_tmax : time-of-observation for daily tmax (local hr)
* tobs_prcp : time-of-observation for daily prcp (local hr)
* \*_mth_raw : USHCN-specific elements. Original, raw monthly elements: 

  * tmin_mth_raw (C)
  * tmax_mth_raw (C)
  * tavg_mth_raw(C)
  * prcp_mth_raw (mm)

* \*_mth_tob : USHCN-specific elements. Time-of-observation adjusted elements:

  * tmin_mth_tob (C)
  * tmax_mth_tob (C)
  * tavg_mth_tob (C)

* \*_mth_fls : USHCN-specific elements. Homogenized and infilled elements:
  
  * tmin_mth_fls (C)
  * tmax_mth_fls (C)
  * tavg_mth_fls (C)
  * prcp_mth_fls (mm)

.. _ACIS: http://www.rcc-acis.org/
.. _GHCN-D: https://www.ncdc.noaa.gov/oa/climate/ghcn-daily/
.. _ISDLite: https://www.ncdc.noaa.gov/isd
.. _MADIS: https://madis.noaa.gov/
.. _NRCS: http://www.wcc.nrcs.usda.gov/web_service/AWDB_Web_Service_Reference.htm
.. _USHCN: http://www.ncdc.noaa.gov/oa/climate/research/ushcn/
.. _WRCC: http://www.raws.dri.edu/

Usage
=============
The main entry point for using **obsio** is through **ObsIoFactory**. **ObIoFactory** is
used to build **ObsIO** objects for accessing station metadata and observations
from specific providers.

::

	# Example code for accessing NRCS SNOTEL/SCAN observations in the Pacific
	# Northwest for January 2015
	
	import obsio
	import pandas as pd
	
	# List of elements to obtain
	elems = ['tmin', 'tmax', 'swe']
	
	# Lat/Lon bounding box for the Pacific Northwest
	bbox = obsio.BBox(west_lon=-126, south_lat=42, east_lon=-111, north_lat=50)
	
	# Start, end dates as pandas Timestamp objects
	start_date = pd.Timestamp('2015-01-01')
	end_date = pd.Timestamp('2015-01-31')
	
	# Initialize factory with specified parameters
	obsiof = obsio.ObsIoFactory(elems, bbox, start_date, end_date)
	
	# Create ObsIO object for accessing daily NRCS observations
	nrcs_io = obsiof.create_obsio_dly_nrcs()
	
	# All ObsIO objects contain a stns attribute. This is a pandas DataFrame
	# containing metadata for all stations that met the specified parameters.
	print nrcs_io.stns
	
	# Access observations using read_obs() method. By default, read_obs() will
	# return observations for all stations in the stns attribute
	obs = nrcs_io.read_obs()
	
	# Observations are provided in a pandas DataFrame. Observation values are 
	# indexed by a 3 level multi-index: station_id, elem, time
	print obs
	
	# To access observations for only a few specific stations, send in a list
	# of station ids to read_obs()
	obs = nrcs_io.read_obs(['11E07S', '11E31S'])

In contrast to the NRCS SNOTEL/SCAN example, some **ObsIO** provider objects
require all observation data to first be downloaded and stored locally, and
then parsed (see provider table above). The data directory for local storage
can be pre-specified in a 'OBSIO_DATA' environmental variable or specified
as a parameter when creating the **ObsIO** object. If no directory is specified,
obsio will default to a standard temporary directory. Example:

::

	# Example code for accessing GHCN-D observations in the Pacific
	# Northwest for January 2015. GHCN-D is a data provider that requires
	# local storage.
	
	import obsio
	import pandas as pd
	
	# List of elements to obtain
	elems = ['tmin', 'tmax']
	
	# Lat/Lon bounding box for the Pacific Northwest
	bbox = obsio.BBox(west_lon=-126, south_lat=42, east_lon=-111, north_lat=50)
	
	# Start, end dates as pandas Timestamp objects
	start_date = pd.Timestamp('2015-01-01')
	end_date = pd.Timestamp('2015-01-31')
	
	# Initialize factory with specified parameters
	obsiof = obsio.ObsIoFactory(elems, bbox, start_date, end_date)
	
	# Create ObsIO object for accessing GHCN-D observations. A local data path
	# can be specified in the create_obsio_dly_ghcnd() call. If not specified,
	# the 'OBSIO_DATA' environmental variable will be checked. If 'OBSIO_DATA'
	# doesn't exist, a default temporary directory will be used.
	ghcnd_io = obsiof.create_obsio_dly_ghcnd()
			
	# Access observations for first 10 stations using the read_obs() method.
	# First call to read_obs() will take several minutes due to initial data
	# download.
	obs = ghcnd_io.read_obs(ghcnd_io.stns.station_id.iloc[0:10])

	
