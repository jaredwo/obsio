from .providers.acis import AcisObsIO
from .providers.ghcnd import GhcndObsIO
from .providers.isd import IsdLiteObsIO
from .providers.madis import MadisObsIO
from .providers.nrcs import NrcsObsIO


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

    def create_obsio_dly_madis(self, local_data_path=None, username=None,
                               password=None, madis_datasets=None,
                               local_time_zones=None, fname_tz_geonames=None,
                               min_hrly_for_dly=None,
                               nprocs=1):
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
        username : str, optional
            MADIS username for accessing restricted datasets (e.g.--mesonets)
        password : str, optional
            MADIS password for accessing restricted datasets (e.g.--mesonets)
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
            cover the local calendar dates of all conterminous U.S. time zones.
        fname_tz_geonames : str, optional
            Geonames username. If provided, station time zone information that
            cannot be determined locally via the python tzwhere package
            will be looked up in the geonames time zone web service. If the
            time zone of a station point still can't be determined, the time zone
            of the nearest neighboring station with a time zone will be used.
            Time zone location information only needs to be looked up once
            per station and will then be cached in the MADIS local data directory.
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
        nprocs : int, optional
            The number of processes to use. Increasing the processor count 
            can decrease the time required to decompress and load the MADIS
            hourly netCDF files.

        Returns
        ----------
        obsio.ObsIO
        """

        return MadisObsIO(local_data_path=local_data_path,
                          username=username, password=password,
                          madis_datasets=madis_datasets,
                          local_time_zones=local_time_zones,
                          fname_tz_geonames=fname_tz_geonames,
                          min_hrly_for_dly=min_hrly_for_dly,
                          elems=self.elems, bbox=self.bbox,
                          start_date=self.start_date, end_date=self.end_date,
                          nprocs=nprocs)

    def create_obsio_dly_isdlite(self, local_data_path=None,
                                 min_hrly_for_dly=None, fname_tz_geonames=None):
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

        Parameters
        ----------
        local_data_path : str, optional
            The local path for downloading and storing ISD-Lite data files
            from the ISD-Lite FTP. If not specified, will use and create an
            ISD-Lite directory in the path specified by the OBSIO_DATA 
            environmental variable. If OBSIO_DATA is not set, a default 
            temporary path will be used.
        min_hrly_for_dly : dict, optional
            The number of hourly observations required to calculate a daily
            value for each element, e.g: {'tmin':20,'tmax':20,'tdew':4}. If
            not specified, the default values are:
            - 'tmin' : 20
            - 'tmax' : 20
            - 'tdew' : 4
        fname_tz_geonames : str, optional
            Geonames username. If provided, station time zone information that
            cannot be determined locally via the python tzwhere package
            will be looked up in the geonames time zone web service. If the
            time zone of a station point still can't be determined, the time zone
            of the nearest neighboring station with a time zone will be used.
            Time zone location information only needs to be looked up once
            per station and will then be cached in the ISD-Lite local data directory.

        Returns
        ----------
        obsio.ObsIO
        """

        return IsdLiteObsIO(local_data_path=local_data_path,
                            min_hrly_for_dly=min_hrly_for_dly,
                            fname_tz_geonames=fname_tz_geonames,
                            elems=self.elems, bbox=self.bbox,
                            start_date=self.start_date,
                            end_date=self.end_date)
