from ..util.humidity import calc_pressure, convert_tdew_to_vpd, \
    convert_tdew_to_rh
from ..util.misc import TimeZones, open_remote_file, open_remote_gz
from .generic import ObsIO
from pytz.exceptions import NonExistentTimeError, AmbiguousTimeError
from urlparse import urljoin
import numpy as np
import pandas as pd

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

def _parse_obs(stn_id, elev, tz_name, start_date, end_date, elems,
               min_hrly_for_dly):
    
    start_hr_utc = _get_begin_end_utc(start_date, tz_name)[0]
    end_hr_utc = _get_begin_end_utc(end_date, tz_name)[1]
    
    uyrs = np.unique(pd.date_range(start_hr_utc.date(), end_hr_utc.date(),
                                   freq='D').year)
    
    obs_hrly = []

    for yr in uyrs:
        
        url = urljoin(_RPATH_ISD_LITE, '%d/%s-%d.gz' % (yr, stn_id, yr))
        
        try:
            f_stnyr = open_remote_gz(url)
        except Exception:
            print "Warning: Could not find remote file, will skip: %s" % url
            continue
        
        df_obs = pd.read_fwf(f_stnyr, _ISD_FWF_COLSPECS, header=None,
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
        
        # Trace prcp is represented with -1. Set to na for now
        df_obs.loc[df_obs.prcp == -1, 'prcp'] = np.nan
        df_obs['prcp'] = df_obs['prcp'] / 10.0

        obs_hrly.append(df_obs)

    if len(obs_hrly) > 0:

        obs_hrly = pd.concat(obs_hrly)
        
        mask_time = ((obs_hrly.index >= start_hr_utc) & 
                     (obs_hrly.index <= end_hr_utc))
        
        obs_hrly.drop(obs_hrly.index[~mask_time], axis=0, inplace=True)
        obs_hrly.index = obs_hrly.index.tz_convert(tz_name)
        
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
        obs_dly.loc[(obs_dly.tair_count < min_hrly_for_dly['tmin']).values,
                    'tmin'] = np.nan
        obs_dly.loc[(obs_dly.tair_count < min_hrly_for_dly['tmax']).values,
                    'tmax'] = np.nan
        obs_dly.loc[(obs_dly.prcp_count < min_hrly_for_dly['prcp']).values,
                    'prcp'] = np.nan
        obs_dly.loc[(obs_dly.tdew_count < min_hrly_for_dly['tdew']).values,
                    'tdew'] = np.nan
        obs_dly.loc[(obs_dly.tdew_count < min_hrly_for_dly['tdewmin']).values,
                    'tdewmin'] = np.nan
        obs_dly.loc[(obs_dly.tdew_count < min_hrly_for_dly['tdewmax']).values,
                    'tdewmax'] = np.nan  
        obs_dly.loc[(obs_dly.vpd_count < min_hrly_for_dly['vpd']).values,
                    'vpd'] = np.nan
        obs_dly.loc[(obs_dly.vpd_count < min_hrly_for_dly['vpdmin']).values,
                    'vpdmin'] = np.nan
        obs_dly.loc[(obs_dly.vpd_count < min_hrly_for_dly['vpdmax']).values,
                    'vpdmax'] = np.nan
        obs_dly.loc[(obs_dly.rh_count < min_hrly_for_dly['rh']).values,
                    'rh'] = np.nan
        obs_dly.loc[(obs_dly.rh_count < min_hrly_for_dly['rhmin']).values,
                    'rhmin'] = np.nan
        obs_dly.loc[(obs_dly.rh_count < min_hrly_for_dly['rhmax']).values,
                    'rhmax'] = np.nan
                
        obs_dly.drop(obs_dly.columns[~obs_dly.columns.isin(elems)], axis=1,
                     inplace=True)
        obs_dly.index = obs_dly.index.tz_localize(None)
        obs_dly = obs_dly.stack().reset_index()
        obs_dly.rename(columns={'level_0':'time', 'level_1':'elem', 0:'obs_value'},
                       inplace=True)
        obs_dly['station_id'] = stn_id
        
        return obs_dly
    
    else:
        
        return None

class IsdLiteObsIO(ObsIO):

    _avail_elems = ['tmin', 'tmax', 'tdew', 'tdewmin', 'tdewmax', 'vpd',
                    'vpdmin', 'vpdmax', 'rh', 'rhmin', 'rhmax','prcp']

    _ISD_FWF_COLSPECS = [(0, 4), (5, 7), (8, 10), (11, 13), (13, 19), (19, 25),
                         (49, 55)]
    _ISD_FWF_COLNAMES = ['year', 'month', 'day', 'hour', 'temperature',
                         'dewpoint', 'precipitation']

    _MIN_HRLY_FOR_DLY_DFLT = {'tmin': 20, 'tmax': 20, 'tdew': 4, 'tdewmin': 18,
                              'tdewmax': 18, 'vpd':18, 'vpdmin':18, 'vpdmax':18,
                              'rh':18, 'rhmin':18, 'rhmax':18, 'prcp': 24}

    def __init__(self, min_hrly_for_dly=None, **kwargs):

        super(IsdLiteObsIO, self).__init__(**kwargs)
                
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
        
        fstns = open_remote_file(urljoin(_RPATH_ISD, 'isd-history.csv'))

        stns = pd.read_csv(fstns, dtype={'USAF': np.str, 'WBAN': np.str})
        stns['BEGIN'] = pd.to_datetime(stns.BEGIN, format="%Y%m%d")
        stns['END'] = pd.to_datetime(stns.END, format="%Y%m%d")
        stns['station_id'] = stns.USAF + "-" + stns.WBAN
        stns['station_name'] = (stns['STATION NAME'].astype(np.str).
                                apply(unicode, errors='ignore'))
        stns['provider'] = 'ISD-Lite'
        stns['sub_provider'] = 'WBAN'
        stns.loc[stns.WBAN == '99999', 'sub_provider'] = ''

        stns = stns.rename(columns={'LAT': 'latitude', 'LON': 'longitude',
                                    'ELEV(M)': 'elevation', 'BEGIN': 'start_date',
                                    'END': 'end_date'})

        stns = stns.drop(['USAF', 'WBAN', 'STATION NAME', 'CTRY', 'STATE',
                          'ICAO'], axis=1)
        
        stns = stns.set_index('station_id', drop=False)

        if self.bbox is not None:

            mask_bnds = ((stns.latitude >= self.bbox.south) & 
                         (stns.latitude <= self.bbox.north) & 
                         (stns.longitude >= self.bbox.west) & 
                         (stns.longitude <= self.bbox.east))

            stns = stns[mask_bnds].copy()

        if self.has_start_end_dates:

            max_enddate = stns.end_date.max()
            start_date = self.start_date
            end_date = self.end_date

            if end_date > max_enddate:
                end_date = max_enddate

                if start_date > end_date:
                    start_date = end_date


            mask_por = (((start_date <= stns.start_date) & 
                         (stns.start_date <= end_date)) | 
                        ((stns.start_date <= start_date) & 
                         (start_date <= stns.end_date)))

            stns = stns[mask_por].copy()

        
        self._tz.set_tz(stns)

        return stns

    def read_obs(self, stns_ids=None):

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
            
            obs_all = []

            for stn_id, a_stn in stns_obs.iterrows():
                
                if self.has_start_end_dates:
                    start_date = self.start_date
                    end_date = self.end_date
                else:
                    start_date = a_stn.start_date
                    end_date = a_stn.end_date
                    
                obs_stn = _parse_obs(stn_id, a_stn.elevation, a_stn.time_zone,
                                     start_date, end_date, self.elems,
                                     self.min_hrly_for_dly)
                        
                obs_all.append(obs_stn)

            obs_all = pd.concat(obs_all, ignore_index=True)

        finally:

            pd.set_option('mode.chained_assignment', opt_val)

        obs_all = obs_all.set_index(['station_id', 'elem', 'time'])
        obs_all = obs_all.sortlevel(0, sort_remaining=True)

        return obs_all
