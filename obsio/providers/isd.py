from ..util.humidity import calc_pressure, convert_tdew_to_vpd, \
    convert_tdew_to_rh
from ..util.misc import TimeZones, open_remote_file
from .generic import ObsIO
from ftplib import FTP
from gzip import GzipFile
from io import BytesIO
from multiprocessing.pool import Pool
from pytz.exceptions import NonExistentTimeError, AmbiguousTimeError
import numpy as np
import pandas as pd
import sys
import urllib

_RPATH_ISD = 'ftp://ftp.ncdc.noaa.gov/pub/data/noaa/'
_RPATH_ISD_LITE = 'ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-lite/'
_ISD_FWF_COLSPECS = [(0, 4), (5, 7), (8, 10), (11, 13), (13, 19), (19, 25),
                     (49, 55)]
_ISD_FWF_COLNAMES = ['year', 'month', 'day', 'hour', 'tair', 'tdew', 'prcp']

def _get_begin_end_utc(a_date, tz_name):

    # Get begin/end local time bounds for the local calendar day
    # On daylight savings time transition days, the time period is
    # still limited to 24 hours and may include an hour in the next calendar
    # day or one less hour in the current calendar day.
    try:
        begin_time = a_date.tz_localize(tz_name)
    except NonExistentTimeError:
        # Time does not exist because of clocks are set forward
        # for daylight savings time at midnight for this time zone.
        # This only happens in the America/Havana time zone. Add one hour
        # to the begin_time
        begin_time = (
            a_date + pd.Timedelta(hours=1)).tz_localize(tz_name)
    except AmbiguousTimeError:
        # Time is ambiguous because clocks are set backward for daylight
        # savings time at midnight for this time zone. This only happens in
        # the America/Havana time zone. Set ambiguous=True so that dst=True
        # and the time is considered to be the first occurrence.
        begin_time = a_date.tz_localize(tz_name, ambiguous=True)

    end_time = begin_time + pd.Timedelta(days=1)

    begin_time = begin_time.tz_convert('UTC')
    end_time = end_time.tz_convert('UTC')
    
    return begin_time, end_time

def _init_worker(func):
    '''
    http://stackoverflow.com/questions/10117073/
    how-to-use-initializer-to-set-up-my-multiprocess-pool
    '''
    
    func.ftp = FTP('ftp.ncdc.noaa.gov')
    func.ftp.login()

def _download_obs(args):

    a_stn, start_date, end_date, elems, min_hrly_for_dly = args
    stn_id = a_stn.station_id
    tz_name = a_stn.time_zone
    elev = a_stn.elevation
    has_start_end = start_date is not None and end_date is not None
    yrs = np.arange(a_stn.start_year,a_stn.end_year+1)
    
    try:
    
        if has_start_end:
            
            #subset to years that only intersect with start/end date years requested
            start_hr_utc = _get_begin_end_utc(start_date, tz_name)[0]
            end_hr_utc = _get_begin_end_utc(end_date, tz_name)[1]
            yrs_req = np.unique(pd.date_range(start_hr_utc.date(), end_hr_utc.date(),
                                              freq='D').year)
            yrs = np.intersect1d(yrs, yrs_req, True)
    
        obs_hrly = []
        
        for yr in yrs:
            
            try:
                        
                fileobj = BytesIO()
                cmd_ftp = 'RETR pub/data/noaa/isd-lite/%d/%s-%d.gz'%(yr, stn_id, yr)
                _download_obs.ftp.retrbinary(cmd_ftp, fileobj.write)
                fileobj.seek(0)
                fileobj = GzipFile(fileobj=fileobj,mode='rb')
            
            except Exception as e:
                
                if e.args[0][-25:] == 'No such file or directory':
                    # Year file does not exist for station. Continue to next year
                    continue
                else:
                    raise
                    
            df_obs = pd.read_fwf(fileobj, _ISD_FWF_COLSPECS, header=None,
                                 names=_ISD_FWF_COLNAMES, na_values=['-9999'])
            
            # https://github.com/pydata/pandas/issues/8158
            # http://stackoverflow.com/questions/19350806/
            # how-to-convert-columns-into-one-datetime-column-in-pandas
            y = np.array(df_obs.year - 1970, dtype='<M8[Y]')
            m = np.array(df_obs.month - 1, dtype='<m8[M]')
            d = np.array(df_obs.day - 1, dtype='<m8[D]')
            
            a_time = (pd.to_datetime(y + m + d).values + 
                      pd.to_timedelta(df_obs.hour, 'h').values)
            
            df_obs.set_index(pd.to_datetime(a_time, utc=True), inplace=True)
            
            df_obs['tair'] = df_obs['tair'] / 10.0
            df_obs['tdew'] = df_obs['tdew'] / 10.0
            
            # Trace prcp is represented with -1. Set to 0 for now
            df_obs.loc[df_obs.prcp == -1, 'prcp'] = 0
            df_obs['prcp'] = df_obs['prcp'] / 10.0
    
            if has_start_end:
                
                mask_time = ((df_obs.index >= start_hr_utc) & 
                             (df_obs.index < end_hr_utc))
                df_obs.drop(df_obs.index[~mask_time], axis=0, inplace=True)
            
            if not df_obs.empty:
            
                obs_hrly.append(df_obs)
        
        if len(obs_hrly) > 0:
    
            obs_hrly = pd.concat(obs_hrly)
                        
            obs_hrly.index = obs_hrly.index.tz_convert(tz_name).tz_localize(None)
            
            stn_pres = calc_pressure(elev)
            obs_hrly['rh'] = convert_tdew_to_rh(obs_hrly.tdew, obs_hrly.tair,
                                                stn_pres)
            obs_hrly['vpd'] = convert_tdew_to_vpd(obs_hrly.tdew, obs_hrly.tair,
                                                  stn_pres)
            
            obs_dly = obs_hrly[['tair', 'tdew', 'rh',
                                'vpd', 'prcp']].resample('D', how=['mean', 'min',
                                                                   'max', 'count',
                                                                   'sum'])
            # http://stackoverflow.com/questions/14507794/
            # python-pandas-how-to-flatten-a-hierarchical-index-in-columns
            obs_dly.columns = ['_'.join(col) for col in obs_dly.columns.values]
            
            obs_dly.rename(columns={'tair_min':'tmin', 'tair_max':'tmax',
                                    'tdew_mean':'tdew', 'tdew_min':'tdewmin',
                                    'tdew_max':'tdewmax', 'vpd_mean':'vpd',
                                    'vpd_min':'vpdmin', 'vpd_max':'vpdmax',
                                    'rh_mean':'rh', 'rh_min':'rhmin',
                                    'rh_max':'rhmax', 'prcp_sum':'prcp'},
                           inplace=True)
                    
            # Set days that don't have minimum number of hourly obs to missing
            # Need to account for daylight savings days that only have 23 hours
            # Get the number of hours in each day
            hr_cnt = pd.Series(0,pd.date_range(obs_dly.index[0].date(),
                                               obs_dly.index[-1].date()+pd.Timedelta(days=1),
                                               freq='h',closed='left',tz=tz_name),
                               name='hr_cnt')
            hr_cnt = hr_cnt.tz_localize(None)
            hr_cnt = hr_cnt.resample('D',how='count')
            #Subtract hour count from 24 to get offset for minimum number of observation
            #thresholds
            hr_cnt = 24 - hr_cnt
            #Set any negative offsets (i.e.-day with 25 hours) to 0
            hr_cnt[hr_cnt < 0] = 0
            obs_dly = obs_dly.join(hr_cnt)
            
            for a_elem in elems:
                
                cnt_vname = IsdLiteObsIO._elem_to_cnt_vname[a_elem]
            
                mask_low_cnt = (obs_dly[cnt_vname] <
                                (min_hrly_for_dly[a_elem]- obs_dly.hr_cnt)).values
            
                obs_dly.loc[mask_low_cnt, a_elem] = np.nan
            
            obs_dly.drop(obs_dly.columns[~obs_dly.columns.isin(elems)], axis=1,
                         inplace=True)
            
            obs_dly = obs_dly.stack().reset_index()
            obs_dly.rename(columns={'level_0':'time', 'level_1':'elem',
                                    0:'obs_value'}, inplace=True)
            obs_dly['station_id'] = stn_id
            
            return obs_dly
        
        else:
            
            return None

    except Exception as e:
        
        print("Error for station "+stn_id+": "+str(e))
        
        

class IsdLiteObsIO(ObsIO):

    _avail_elems = ['tmin', 'tmax', 'tdew', 'tdewmin', 'tdewmax', 'vpd',
                    'vpdmin', 'vpdmax', 'rh', 'rhmin', 'rhmax','prcp']
    
    name = 'ISDLITE'
    
    _elem_to_cnt_vname =  {'tmin':'tair_count','tmax':'tair_count','tdew':'tdew_count',
                           'tdewmin':'tdew_count', 'tdewmax':'tdew_count',
                           'vpd':'vpd_count','vpdmin':'vpd_count','vpdmax':'vpd_count',
                           'rh':'rh_count','rhmin':'rh_count','rhmax':'rh_count',
                           'prcp':'prcp_count'}
    
    _MIN_HRLY_FOR_DLY_DFLT = {'tmin': 20, 'tmax': 20, 'tdew': 4, 'tdewmin': 18,
                              'tdewmax': 18, 'vpd':18, 'vpdmin':18, 'vpdmax':18,
                              'rh':18, 'rhmin':18, 'rhmax':18, 'prcp': 24}

    def __init__(self, nprocs=1, min_hrly_for_dly=None, **kwargs):

        super(IsdLiteObsIO, self).__init__(**kwargs)
        
        if nprocs < 0 or nprocs > 3:
            raise ValueError("Number of processes must be between 1 and 3.")
        
        self.nprocs = nprocs
        self.min_hrly_for_dly = (min_hrly_for_dly if min_hrly_for_dly
                                 else self._MIN_HRLY_FOR_DLY_DFLT)
        # check to make sure there is an entry in min_hrly_for_dly for each
        # elem
        for a_elem in self.elems:

            try:

                self.min_hrly_for_dly[a_elem]

            except KeyError:

                self.min_hrly_for_dly[
                    a_elem] = self._MIN_HRLY_FOR_DLY_DFLT[a_elem]

        self._a_tz = None

    @property
    def _tz(self):

        if self._a_tz is None:

            self._a_tz = TimeZones()

        return self._a_tz

    def _read_stns(self):
        
        fstns = open_remote_file(urllib.parse.urljoin(_RPATH_ISD, 'isd-history.csv'))

        stns = pd.read_csv(fstns, dtype={'USAF': np.str, 'WBAN': np.str})
        stns['BEGIN'] = pd.to_datetime(stns.BEGIN, format="%Y%m%d")
        stns['END'] = pd.to_datetime(stns.END, format="%Y%m%d")
        stns['station_id'] = stns.USAF + "-" + stns.WBAN
        stns['station_name'] = stns['STATION NAME'].astype(np.str)
        stns['provider'] = 'ISD-Lite'
        stns['sub_provider'] = 'WBAN'
        stns.loc[stns.WBAN == '99999', 'sub_provider'] = ''

        stns = stns.rename(columns={'LAT': 'latitude', 'LON': 'longitude',
                                    'ELEV(M)': 'elevation'})
        # Get start and end year of station records
        # Add 1 year buffer on both ends to account for period-of-record
        # metadata not always being up-to-date
        stns['start_year'] = stns.BEGIN.dt.year - 1
        stns['end_year'] = stns.END.dt.year + 1

        stns = stns.drop(['USAF', 'WBAN', 'STATION NAME', 'CTRY', 'STATE',
                          'ICAO','BEGIN','END'], axis=1)
        
        stns = stns.set_index('station_id', drop=False)
        
        # Limit years to what is available on the FTP
        # Get year list
        afile = open_remote_file(_RPATH_ISD_LITE)
        yrs_avail = pd.read_fwf(afile, header=None, usecols=[8], names=['year'])
        yrs_avail = pd.to_numeric(yrs_avail.year,
                                  errors='coerce').dropna().astype(np.int).values
        min_yr = np.min(yrs_avail)
        max_yr = np.max(yrs_avail)
        
        stns.loc[stns.start_year < min_yr, 'start_year'] = min_yr
        stns.loc[stns.end_year > max_yr, 'end_year'] = max_yr
        
        if self.bbox is not None:

            mask_bnds = ((stns.latitude >= self.bbox.south) & 
                         (stns.latitude <= self.bbox.north) & 
                         (stns.longitude >= self.bbox.west) & 
                         (stns.longitude <= self.bbox.east))

            stns = stns[mask_bnds].copy()

        if self.has_start_end_dates:
            
            # Get stations that overlap with the year(s) of the start/end dates
            yr_start = self.start_date.year
            yr_end = self.end_date.year
            
            mask_por = (((yr_start <= stns.start_year) & 
                         (stns.start_year <= yr_end)) | 
                        ((stns.start_year <= yr_start) & 
                         (yr_start <= stns.end_year)))

            stns = stns[mask_por].copy()

        
        self._tz.set_tz(stns)

        return stns

    def _read_obs(self, stns_ids=None):

        # Saw extreme decreased performance due to garbage collection when
        # pandas ran checks for a chained assignment. Turn off this check
        # temporarily.
        opt_val = pd.get_option('mode.chained_assignment')
        pd.set_option('mode.chained_assignment', None)

        try:
            
            if stns_ids is None:
                stns_obs = self.stns
            else:
                stns_obs = self.stns.loc[stns_ids]
            
            nstns = len(stns_obs.station_id)
            nprocs = self.nprocs if nstns >= self.nprocs else nstns
            
            if self.has_start_end_dates:
                start_date = self.start_date
                end_date = self.end_date
            else:
                start_date = None
                end_date = None
                                            
            iter_stns = [(row[1], start_date, end_date, self.elems,
                          self.min_hrly_for_dly) for row in stns_obs.iterrows()]
                        
            if nprocs > 1:
                
                # http://stackoverflow.com/questions/24171725/
                # scikit-learn-multicore-attributeerror-stdin-instance-
                # has-no-attribute-close
                if not hasattr(sys.stdin, 'close'):
                    def dummy_close():
                        pass
                    sys.stdin.close = dummy_close
    
                pool = Pool(processes=nprocs,initializer=_init_worker,
                            initargs=[_download_obs])
                obs_all = pool.map(_download_obs, iter_stns, chunksize=1)
                pool.close()
                pool.join()
                
            else:
                
                obs_all = []
                
                _init_worker(_download_obs)
                
                for a_stn in iter_stns:
                    
                    obs_stn = _download_obs(a_stn)
                    obs_all.append(obs_stn)
                
                _download_obs.ftp.close()
            
            try:
                obs_all = pd.concat(obs_all, ignore_index=True)
            except ValueError:
                # No valid observations
                obs_all = pd.DataFrame({'station_id':[], 'elem':[], 'time':[],
                                        'obs_value':[]})
            
        finally:

            pd.set_option('mode.chained_assignment', opt_val)
            
        obs_all = obs_all.set_index(['station_id', 'elem', 'time'])
        obs_all = obs_all.sort_index(0, sort_remaining=True)

        return obs_all
