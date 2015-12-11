from .providers.acis import AcisObsIO
from .providers.ghcnd import GhcndObsIO
from .providers.isd import IsdLiteObsIO
from .providers.madis import MadisObsIO
from .providers.nrcs import NrcsObsIO
from .providers.ushcn import UshcnObsIO
from .providers.wrcc import WrccRawsObsIO


class ObsIoFactory(object):
    """Builds ObsIO objects for different observation data sources.

    ObsIoFactory is the main entry point for using the obsio package.
    """

    def __init__(self, elems, bbox=None, start_date=None, end_date=None):
        """
        Parameters
        ----------
        elems : list
            Observation elements to access. Currently available elements:
            - 'tmin' : daily minimum temperature (C)
            - 'tmax' : daily maximum temperature (C)
            - 'tdew' : daily average dewpoint (C)
            - 'prcp' : daily total precipitation (mm)
            - 'srad' : daily 24-hr average incoming solar radiation (w m-2)
            - 'wspd' : daily average windspeed (m s-1)
            - 'snwd' : snow depth (mm)
            - 'swe' : snow water equivalent (mm)
            - 'tobs_tmin' : time-of-observation for daily tmin (local hr)
            - 'tobs_tmax' : time-of-observation for daily tmax (local hr)
            - 'tobs_prcp' : time-of-observation for daily prcp (local hr)
            - 'tmin_mth_raw' : original, raw monthly average minimum
                               temperature (C; USHCN specific)
            - 'tmin_mth_tob' : time-of-observation adjusted monthly average minimum
                               temperature (C; USHCN specific)
            - 'tmin_mth_fls' : homogenized and infilled monthly average minimum
                               temperature (C; USHCN specific)
            - 'tmax_mth_raw' : original, raw monthly average maximum
                               temperature (C; USHCN specific)
            - 'tmax_mth_tob' : time-of-observation adjusted monthly average maximum
                               temperature (C; USHCN specific)
            - 'tmax_mth_fls' : homogenized and infilled monthly average maximum
                               temperature (C; USHCN specific)
            - 'tavg_mth_raw' : original, raw monthly average
                               temperature (C; USHCN specific)
            - 'tavg_mth_tob' : time-of-observation adjusted monthly average
                               temperature (C; USHCN specific)
            - 'tavg_mth_fls' : homogenized and infilled monthly average
                               temperature (C; USHCN specific)
            - 'prcp_mth_raw' : original, raw monthly total precipitation
                               (mm; USHCN specific)
            - 'prcp_mth_fls' : homogenized and infilled monthly total
                               precipitation (mm; USHCN specific)
            Not all elements are available for every ObsIO data source.
        bbox : obsio.BBox, optional
            Lat/lon bounding box of desired spatial domain. Only stations and 
            observations within the bounding box will be returned by the
            ObsIOs.
        start_date : pandas.Timestamp, optional
            Start date of desired date range. If used, end_date must also be
            specified. Only stations and observations within the date range
            will be returned by the ObsIOs.
        end_date : pandas.Timestamp, optional
            End date of desired date range. If used, start_date must also be
            specified. Only stations and observations within the date range
            will be returned by the ObsIOs.        
        """

        self.elems = elems
        self.bbox = bbox
        self.start_date = start_date
        self.end_date = end_date

    def create_obsio_dly_nrcs(self):
        """Create ObsIO to access daily observations from NRCS AWDB.

        The National Resources Conservation Service (NRCS) Air-Water Database
        (AWDB) contains observations from the NRCS Snow Telemetry and Soil
        Climate Analysis networks (SNOTEL, SCAN). This ObsIO accesses daily
        SNOTEL and SCAN observations directly via the AWDB SOAP web service
        (http://www.wcc.nrcs.usda.gov/web_service/awdb_web_service_landing.htm)
        and does not require the data to be stored locally. Currently 
        available elements:
        - 'tmin' : daily minimum temperature (C)
        - 'tmax' : daily maximum temperature (C)
        - 'prcp' : daily total precipitation (mm)
        - 'snwd' : snow depth (mm)
        - 'swe' : snow water equivalent (mm)

        Returns
        ----------
        obsio.ObsIO
        """

        return NrcsObsIO(elems=self.elems, bbox=self.bbox,
                         start_date=self.start_date,
                         end_date=self.end_date)

    def create_obsio_dly_acis(self):
        """Create ObsIO to access daily observations from RCC ACIS.

        The NOAA Regional Climate Centers (RCCs) Applied Climate Information
        System (ACIS) contains observations from a variety of federal,
        regional, state, and local networks. One of the main sources of
        observations (http://www.rcc-acis.org/docs_datasets.html) is the
        Global Historical Climatology Network - Daily (http://www.ncdc.noaa.
        gov/oa/climate/ghcn-daily). This ObsIO accesses daily observations
        directly via the ACIS web services (http://www.rcc-acis.org/
        docs_webservices.html) and does not require the data to be stored
        locally. Currently available elements:
        - 'tmin' : daily minimum temperature (C)
        - 'tmax' : daily maximum temperature (C)
        - 'prcp' : daily total precipitation (mm)
        - 'tobs_tmin' : time-of-observation for daily tmin (local hr)
        - 'tobs_tmax' : time-of-observation for daily tmax (local hr)
        - 'tobs_prcp' : time-of-observation for daily prcp (local hr)

        Returns
        ----------
        obsio.ObsIO
        """

        return AcisObsIO(elems=self.elems, bbox=self.bbox,
                         start_date=self.start_date,
                         end_date=self.end_date)

    def create_obsio_dly_ghcnd(self, local_data_path=None):
        """Create ObsIO to access daily observations from NCEI's GHCN-D.

        NOAA's National Centers for Environmental Information (NCEI) Global
        Historical Climatology Network Daily (GHCN-D) is an integrated
        database of daily climate summaries from land surface stations across
        the globe. This ObsIO accesses GHCN-D observations via the GHCN-D FTP
        site: ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/.
        Currently available elements:
        - 'tmin' : daily minimum temperature (C)
        - 'tmax' : daily maximum temperature (C)
        - 'prcp' : daily total precipitation (mm)
        - 'tobs_tmin' : time-of-observation for daily tmin (local hr)
        - 'tobs_tmax' : time-of-observation for daily tmax (local hr)
        - 'tobs_prcp' : time-of-observation for daily prcp (local hr)

        Parameters
        ----------
        local_data_path : str, optional
            The local path for downloading and storing GHCN-D data from the
            FTP site. If not specified, will use and create a GHCND directory
            in the path specified by the OBSIO_DATA environmental variable. If
            OBSIO_DATA is not set, a default temporary path will be used. On
            a call to obsio.ObsIO.download_local(), the ObsIO locally mirrors
            the full ghcnd_all.tar.gz and yearly files with time-of-observation
            data.

        Returns
        ----------
        obsio.ObsIO
        """

        return GhcndObsIO(local_data_path=local_data_path,
                          elems=self.elems, bbox=self.bbox,
                          start_date=self.start_date,
                          end_date=self.end_date)

    def create_obsio_mthly_ushcn(self, local_data_path=None):
        """Create ObsIO to access monthly observations from NCEI's USHCN.

        NOAA's National Centers for Environmental Information (NCEI) U.S.
        Historical Climatology Network (USHCN) is a subset of the Cooperative
        Observer Network (COOP) and is used to quantify national and regional
        scale temperature changes in the contiguous U.S. USHCN sites were 
        selected according to their spatial coverage, record length, data
        completeness, and historical stability. USHCN is a monthly dataset. 
        USHCN details: http://www.ncdc.noaa.gov/oa/climate/research/ushcn/
        This ObsIO accesses USHCN observations via the USHCN FTP site: 
        ftp://ftp.ncdc.noaa.gov/pub/data/ushcn/
        Currently available elements:
        - 'tmin_mth_raw' : original, raw monthly average minimum temperature (C)
        - 'tmin_mth_tob' : time-of-observation adjusted monthly average minimum
                           temperature (C)
        - 'tmin_mth_fls' : homogenized and infilled monthly average minimum
                           temperature (C)
        - 'tmax_mth_raw' : original, raw monthly average maximum temperature (C)
        - 'tmax_mth_tob' : time-of-observation adjusted monthly average maximum
                           temperature (C)
        - 'tmax_mth_fls' : homogenized and infilled monthly average maximum
                           temperature (C)
        - 'tavg_mth_raw' : original, raw monthly average temperature (C)
        - 'tavg_mth_tob' : time-of-observation adjusted monthly average temperature (C)
        - 'tavg_mth_fls' : homogenized and infilled monthly average temperature (C)
        - 'prcp_mth_raw' : original, raw monthly total precipitation (mm)
        - 'prcp_mth_fls' : homogenized and infilled monthly total precipitation (mm)
                           
        Parameters
        ----------
        local_data_path : str, optional
            The local path for downloading and storing USHCN data from the
            FTP site. If not specified, will use and create a USHCN directory
            in the path specified by the OBSIO_DATA environmental variable. If
            OBSIO_DATA is not set, a default temporary path will be used. On
            a call to obsio.ObsIO.download_local(), the ObsIO locally mirrors
            the USHCN FTP site.

        Returns
        ----------
        obsio.ObsIO
        """

        return UshcnObsIO(local_data_path=local_data_path,
                          elems=self.elems, bbox=self.bbox,
                          start_date=self.start_date,
                          end_date=self.end_date)

    def create_obsio_dly_madis(self, local_data_path=None, data_version=None,
                               username=None, password=None,
                               madis_datasets=None, local_time_zones=None,
                               min_hrly_for_dly=None, nprocs=1,
                               temp_path=None, handle_dups=True):
        """Create ObsIO to access daily observations from MADIS.

        MADIS (Meteorological Assimilation Data Ingest System) is a 
        meteorological observation database (https://madis.noaa.gov) provided
        by NWS's National Centers for Environmental Prediction Central
        Operations. MADIS ingests observations from a wide range of public
        and private data sources. Observations go back to 2001 and are provided
        in hourly netCDF files. This ObsIO aggregates the hourly observations
        to daily based on the local calendar day of a respective station.
        Currently available elements:
        - 'tmin' : daily minimum temperature (C)
        - 'tmax' : daily maximum temperature (C)
        - 'tdew' : daily average dewpoint (C)
        - 'tdewmin' : daily minimum dewpoint (C)
        - 'tdewmax' : daily maximum dewpoint (C)
        - 'vpd' : daily average vapor pressure deficit (Pa)
        - 'vpdmin' : daily minimum vapor pressure deficit (Pa)
        - 'vpdmax' : daily maximum vapor pressure deficit (Pa)
        - 'rh' : daily average relative humidity (Pa)
        - 'rhmin' : daily minimum relative humidity (Pa)
        - 'rhmax' : daily minimum relative humidity (Pa)
        - 'prcp' : daily total precipitation (mm)
        - 'srad' : daily 24-hr average incoming solar radiation (w m-2)
        - 'wspd' : daily average windspeed (m s-1)
        
        Parameters
        ----------
        local_data_path : str, optional
            The local path for downloading and storing MADIS hourly netCDF
            files. If not specified, will use and create a MADIS directory in
            the path specified by the OBSIO_DATA environmental variable. If
            OBSIO_DATA is not set, a default temporary path will be used.
        data_version : str, optional
            MADIS dataset version (public, research, noaa-only, etc.) Some
            dataset versions are restricted and require a username and
            password (see https://madis.noaa.gov/madis_restrictions.shtml).
            Currently available data versions:
            - 'madisPublic1'
            - 'madisPublic2'
            - 'madisPublic3'
            - 'madisResearch'
            - 'madisResearch2'
            - 'madisNoaa'
            - 'madisGov'
            If not specified, the MADIS dataset version will be set to
            'madisPublic1'.            
        username : str, optional
            MADIS username for accessing a restricted data version
        password : str, optional
            MADIS password for accessing restricted data version
        madis_datasets: list, optional
            MADIS datasets to access. Currently supported datasets:
            - 'LDAD/mesonet/netCDF'
            - 'point/metar/netcdf'
            - 'LDAD/coop/netCDF'
            - 'LDAD/crn/netCDF'
            - 'LDAD/hcn/netCDF'
            - 'LDAD/hydro/netCDF'
            - 'LDAD/nepp/netCDF'
            - 'point/sao/netcdf'
            If not specified, all available datasets will be accessed.
        local_time_zones : list, optional
            List of time zones used to determine what MADIS UTC hourly files
            should be loaded to cover the local calendar date(s) specified by
            the factory's date range (e.g. ['US/Mountain','US/Eastern']). If
            not provided, will default to loading MADIS UTC hourly files that
            cover the local calendar dates of all global time zones.
        min_hrly_for_dly : dict, optional
            The number of hourly observations required to calculate a daily
            value for each element, e.g: {'tmin':20,'tmax':20,'tdew':4}. If
            not specified, the default values are:
            - 'tmin' : 20
            - 'tmax' : 20
            - 'tdew' : 4
            - 'srad' : 24
            - 'prcp' : 24
            - 'wspd' : 24
            - 'tdewmin' : 18
            - 'tdewmax' : 18
            - 'vpd' : 18
            - 'vpdmin' : 18
            - 'vpdmax' : 18
            - 'rh' : 18
            - 'rhmin' : 18
            - 'rhmax' : 18
            - 'prcp' : 24
        nprocs : int, optional
            The number of processes to use. Increasing the processor count 
            can decrease the time required to decompress and load the MADIS
            hourly netCDF files.
        temp_path : str, optional
            Temporary directory to decompress MADIS netCDF files for reading
            if they cannot be decompressed on the fly. If not specified,
            will use a 'tmp' directory in the local_data_path. MADIS data are 
            provided in externally gzipped netCDF3 files. These files can 
            typically be decompressed on the fly, but in some instance on the
            fly decompression fails. In these cases, the netCDF3 file will be
            temporarly decompressed to temp_path and read into memory.
        handle_dups : boolean, optional
            Handle duplicate station ids. Default: True. In MADIS, a specific
            station's location is allowed to change, but the station still 
            retains the same station id. If a station location changes during
            the dates being processed and handle_dups is True, the new location
            of the station will be considered a new station and '_dup##' will
            be appended to the station id. If handle_dups is False, only the
            latest location information for the station will be returned
            and all observations for the station will be associated with this
            latest location.
              
        Returns
        ----------
        obsio.ObsIO
        """

        return MadisObsIO(local_data_path=local_data_path,
                          data_version=data_version,
                          username=username, password=password,
                          madis_datasets=madis_datasets,
                          local_time_zones=local_time_zones,
                          min_hrly_for_dly=min_hrly_for_dly,
                          elems=self.elems, bbox=self.bbox,
                          start_date=self.start_date, end_date=self.end_date,
                          nprocs=nprocs, temp_path=temp_path,
                          handle_dups=handle_dups)

    def create_obsio_dly_isdlite(self, min_hrly_for_dly=None):
        """Create ObsIO to access daily observations from NCEI's ISD-Lite.

        NOAA's National Centers for Environmental Information (NCEI)
        Integrated Surface Database (ISD) contains global hourly and synoptic
        observations compiled from numerous sources 
        (https://www.ncdc.noaa.gov/isd). This ObsIO uses the ISD Lite version
        of the database and accesses the data via the ISD Lite FTP:
        ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-lite/ . ISD Lite hourly
        observations are aggregated to daily based on the local calendar day 
        of a respective station. Currently available elements:
        - 'tmin' : daily minimum temperature (C)
        - 'tmax' : daily maximum temperature (C)
        - 'tdew' : daily average dewpoint (C)
        - 'tdewmin' : daily minimum dewpoint (C)
        - 'tdewmax' : daily maximum dewpoint (C)
        - 'vpd' : daily average vapor pressure deficit (Pa)
        - 'vpdmin' : daily minimum vapor pressure deficit (Pa)
        - 'vpdmax' : daily maximum vapor pressure deficit (Pa)
        - 'rh' : daily average relative humidity (Pa)
        - 'rhmin' : daily minimum relative humidity (Pa)
        - 'rhmax' : daily minimum relative humidity (Pa)
        - 'prcp' : daily total precipitation (mm)

        Parameters
        ----------
        min_hrly_for_dly : dict, optional
            The number of hourly observations required to calculate a daily
            value for each element, e.g: {'tmin':20,'tmax':20,'tdew':4}. If
            not specified, the default values are:
            - 'tmin' : 20
            - 'tmax' : 20
            - 'tdew' : 4
            - 'tdewmin' : 18
            - 'tdewmax' : 18
            - 'vpd' : 18
            - 'vpdmin' : 18
            - 'vpdmax' : 18
            - 'rh' : 18
            - 'rhmin' : 18
            - 'rhmax' : 18
            - 'prcp' : 24

        Returns
        ----------
        obsio.ObsIO
        """

        return IsdLiteObsIO(min_hrly_for_dly=min_hrly_for_dly,
                            elems=self.elems, bbox=self.bbox,
                            start_date=self.start_date,
                            end_date=self.end_date)

    def create_obsio_dly_wrcc_raws(self, nprocs=1, hrly_pwd=None,
                                   min_hrly_for_dly=None):
        """Create ObsIO to access daily observations from WRCC RAWS.

        The Western Regional Climate Center (WRCC; http://www.wrcc.dri.edu)
        provides observations from the Remote Automated Weather Stations (RAWS)
        network (http://www.raws.dri.edu). This ObsIO accesses daily
        observations directly from the WRCC RAWS daily webform. It does not
        require data to be stored locally. Currently available elements:
        - 'tmin' : daily minimum temperature (C)
        - 'tmax' : daily maximum temperature (C)
        - 'tdew' : daily average dewpoint (C)
        - 'tdewmin' : daily minimum dewpoint (C)
        - 'tdewmax' : daily maximum dewpoint (C)
        - 'vpd' : daily average vapor pressure deficit (Pa)
        - 'vpdmin' : daily minimum vapor pressure deficit (Pa)
        - 'vpdmax' : daily maximum vapor pressure deficit (Pa)
        - 'rh' : daily average relative humidity (Pa)
        - 'rhmin' : daily minimum relative humidity (Pa)
        - 'rhmax' : daily minimum relative humidity (Pa)
        - 'prcp' : daily total precipitation (mm)
        - 'srad' : daily 24-hr average incoming solar radiation (w m-2)
        - 'wspd' : daily average windspeed (m s-1)

        Parameters
        ----------
        nprocs : int, optional
            The number of concurrent processes to use for downloading
            observations from the WRCC RAWS webform. Default: 1
        hrly_pwd : str, optional
            Password for accessing hourly RAWS observations to get a more 
            accurate estimate of following humidity variables: tdew, tdewmin,
            tdewmax, vpd, vpdmin, vpdmax. Daily RAWS observations from WRCC are
            free, but historical RAWS hourly observations are "restricted".
            WRCC only provides relative humidity at a daily aggregation.
            To convert to daily values of tdew and vpd, several assumptions
            must be made and an accurate reading might not always be possible.
            If hrly_pwd is passed, the ObsIO will access the hourly WRCC webform
            to get a more accurate estimate of daily tdew and vpd elements.
        min_hrly_for_dly : dict, optional
            If hrly_pwd specified, the number of hourly observations required to
            calculate a daily value, e.g: {'tdew':4}. If not specified,
            the default values are:
            - 'tdew' : 4
            - 'tdewmin' : 18
            - 'tdewmax' : 18
            - 'vpd' : 18
            - 'vpdmin' : 18
            - 'vpdmax' : 18
        
        Returns
        ----------
        obsio.ObsIO
        """

        return WrccRawsObsIO(nprocs=nprocs, hrly_pwd=hrly_pwd,
                             min_hrly_for_dly=min_hrly_for_dly, elems=self.elems,
                             bbox=self.bbox, start_date=self.start_date,
                             end_date=self.end_date)
