from .. import LOCAL_DATA_PATH
from ..util.misc import TimeZones
from .generic import ObsIO
from pytz.exceptions import NonExistentTimeError, AmbiguousTimeError
from urlparse import urljoin
import errno
import ftplib
import numpy as np
import os
import pandas as pd
import re
import subprocess


class IsdLiteObsIO(ObsIO):

    _avail_elems = ['tmin', 'tmax', 'tdew']

    _RPATH_ISD = 'ftp://ftp.ncdc.noaa.gov/pub/data/noaa/'
    _RPATH_ISD_LITE = 'ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-lite/'
    _ISD_FWF_COLSPECS = [
        (0, 4), (5, 7), (8, 10), (11, 13), (13, 19), (19, 25), (49, 55)]
    _ISD_FWF_COLNAMES = ['year', 'month', 'day', 'hour', 'temperature',
                         'dewpoint', 'precipitation']

    min_hrly_for_dly_DFLT = {'tmin': 20, 'tmax': 20, 'tdew': 4}

    def __init__(self, local_data_path=None, min_hrly_for_dly=None,
                 fname_tz_geonames=None, **kwargs):

        super(IsdLiteObsIO, self).__init__(**kwargs)

        self.local_data_path = (local_data_path if local_data_path
                                else LOCAL_DATA_PATH)
        self.path_isd_data = os.path.join(self.local_data_path, 'ISD-Lite')
        
        if not os.path.isdir(self.path_isd_data):
            os.mkdir(self.path_isd_data)
        
        self._fname_tz_geonames = fname_tz_geonames
        self.min_hrly_for_dly = (min_hrly_for_dly if min_hrly_for_dly
                                 else self.min_hrly_for_dly_DFLT)
        # check to make sure there is an entry in min_hrly_for_dly for each
        # elem
        for a_elem in self.elems:

            try:

                self.min_hrly_for_dly[a_elem]

            except KeyError:

                self.min_hrly_for_dly[
                    a_elem] = self.min_hrly_for_dly_DFLT[a_elem]

        if self.has_start_end_dates:

            self._years = np.arange(self.start_date.year,
                                    self.end_date.year + 1)

        else:

            fnames = np.array(os.listdir(self.path_isd_data))

            if fnames.size > 0:

                mask = np.array([re.match('^-?[0-9]+$', a_name) is None
                                 for a_name in fnames])
                fnames = fnames[~mask]

                if fnames.size > 0:

                    self._years = np.sort(fnames.astype(np.int))

                else:

                    self._years = np.array([])

            else:

                self._years = np.array([])

        self._fpath_stns_cache = os.path.join(self.path_isd_data,
                                              'stns_cache.pkl')

        try:

            self._stns_cache = pd.read_pickle(self._fpath_stns_cache)

        except IOError:

            self._stns_cache = pd.DataFrame(columns=['latitude', 'longitude',
                                                     'elevation', 'start_date',
                                                     'end_date', 'station_id',
                                                     'station_name', 'provider',
                                                     'sub_provider',
                                                     'time_zone'])

        self._a_tz = None

    def _merge_with_stn_cache(self, stns_new):

        stns_cache = self._stns_cache

        if stns_cache.size != 0:

            stns_merge = stns_new.merge(stns_cache[['station_id', 'time_zone']],
                                        on='station_id', how='left', sort=False)
        else:

            stns_merge = stns_new

        if not 'time_zone' in stns_merge.columns:
            stns_merge['time_zone'] = np.nan

        if stns_merge.shape[0] != stns_new.shape[0]:
            raise ValueError("Non-unique station id.")

        mask_no_tz = stns_merge['time_zone'].isnull()

        if mask_no_tz.any():

            self._tz.set_tz(stns_merge)

        stns_merge = stns_merge.set_index('station_id', drop=False)

        return stns_merge

    @property
    def _tz(self):

        if self._a_tz is None:

            self._a_tz = TimeZones(self._fname_tz_geonames)

        return self._a_tz

    def _read_stns(self):

        stns = pd.read_csv(os.path.join(self.path_isd_data, 'isd-history.csv'),
                           dtype={'USAF': np.str, 'WBAN': np.str})
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

                print ("IsdLiteObsIO: Warning: Max end date %s of stations is"
                       " < than the specified end date of %s. Respecifying "
                       "end date to %s since isd-history.csv is not always "
                       "up-to-date." % (max_enddate.strftime('%Y-%m-%d'),
                                        self.start_date.strftime('%Y-%m-%d'),
                                        end_date.strftime('%Y-%m-%d')))

            mask_por = (((start_date <= stns.start_date) &
                         (stns.start_date <= end_date)) |
                        ((stns.start_date <= start_date) &
                         (start_date <= stns.end_date)))

            stns = stns[mask_por].copy()

        stns = stns.reset_index(drop=True)

        stns = self._merge_with_stn_cache(stns)
        self._update_stn_cache(stns)

        return stns

    def _update_stn_cache(self, stns):

        mask_exist = stns.index.isin(self._stns_cache.index)

        self._stns_cache = pd.concat([self._stns_cache,
                                      stns.drop(stns.index[mask_exist])])

        pd.to_pickle(self._stns_cache, self._fpath_stns_cache)

    def download_local(self):

        def mkdir_p(path):
            try:
                os.makedirs(path)
            except OSError as exc:  # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(path):
                    pass
                else:
                    raise

        def get_years():

            aftp = ftplib.FTP('ftp.ncdc.noaa.gov')
            aftp.login()

            aftp.cwd('/pub/data/noaa/isd-lite')
            fnames = np.array(aftp.nlst())
            mask = np.array([re.match('^-?[0-9]+$', a_name) is None
                             for a_name in fnames])
            years_ftp = np.sort(fnames[~mask].astype(np.int))

            if self.has_start_end_dates:

                years = np.arange(self.start_date.year, self.end_date.year + 1)
                years = years[np.in1d(years, years_ftp, False)]

            else:

                years = years_ftp

            aftp.close()

            return years

        local_path = self.path_isd_data

        subprocess.call(['wget', '-N', '--directory-prefix=' + local_path,
                         urljoin(self._RPATH_ISD, 'isd-history.csv')])

        years = get_years().astype(np.str)

        for yr in years:

            path_yr = os.path.join(local_path, yr)
            mkdir_p(path_yr)

            subprocess.call(['wget', '-m', '-nd',
                             '--directory-prefix=' + path_yr,
                             urljoin(self._RPATH_ISD_LITE, yr)])

        if not self.has_start_end_dates:
            # update year list
            fnames = np.array(os.listdir(self.path_isd_data))
            mask = np.array([re.match('^-?[0-9]+$', a_name) is None
                             for a_name in fnames])
            self._years = np.sort(fnames[~mask].astype(np.int))

    def _parse_hrly_stn_obs(self, stn_id, local_tz):

        obs_all = []

        for yr in self._years.astype(np.str):

            try:

                fpath = os.path.join(self.path_isd_data, yr,
                                     '%s-%s.gz' % (stn_id, yr))

                df_obs = pd.read_fwf(fpath, self._ISD_FWF_COLSPECS, header=None,
                                     names=self._ISD_FWF_COLNAMES,
                                     na_values=['-9999'])

                a_time = list((df_obs.year.apply(lambda x: '%d' % x) +
                               df_obs.month.apply(lambda x: '%.2d' % x) +
                               df_obs.day.apply(lambda x: '%.2d' % x) +
                               df_obs.hour.apply(lambda x: '%.2d' % x)))

                df_obs.index = pd.to_datetime(a_time, format='%Y%m%d%H',
                                              utc=True)
                df_obs['temperature'] = df_obs['temperature'] / 10.0
                df_obs['dewpoint'] = df_obs['dewpoint'] / 10.0
                df_obs['precipitation'] = df_obs['precipitation'] / 10.0

                obs_all.append(df_obs)

            except IOError:
                # No observations in year for station
                continue

        if len(obs_all) > 0:

            obs_all = pd.concat(obs_all)

            dt_i = (obs_all.index.tz_convert(local_tz))
            obs_all['time_local'] = dt_i
            obs_all['hour_local'] = dt_i.hour
            obs_all['day_local'] = dt_i.day
            obs_all['month_local'] = dt_i.month
            obs_all['year_local'] = dt_i.year
            obs_all.index = obs_all.time_local

            if self.has_start_end_dates:

                begin = self._localize(self.start_date, local_tz)
                end = self._localize(self.end_date, local_tz)

                mask_time = ((obs_all.time_local >= begin) &
                             (obs_all.time_local <= end))
                obs_all = obs_all[mask_time].copy()

        return obs_all

    def _localize(self, a_date, tz_local):

        # Get begin/end local time bounds for the local calendar day
        # On daylight savings time transition days, the time period is
        # still limited to 24 hours and may include an hour in the next calendar
        # day or one less hour in the current calendar day.
        try:
            a_time = a_date.tz_localize(tz_local)
        except NonExistentTimeError:
            # Time does not exist because of clocks are set forward
            # for daylight savings time at midnight for this time zone.
            # This only happens in the America/Havana time zone. Add one hour
            # to the begin_time
            a_time = (a_date + pd.Timedelta(hours=1)).tz_localize(tz_local)
        except AmbiguousTimeError:
            # Time is ambiguous because clocks are set backward for daylight
            # savings time at midnight for this time zone. This only happens in
            # the America/Havana time zone. Set ambiguous=True so that dst=True
            # and the time is considered to be the first occurrence.
            a_time = a_date.tz_localize(tz_local, ambiguous=True)

        return a_time

    def _to_daily(self, obs_hrly):

        cnts = obs_hrly[['temperature', 'dewpoint']].resample('D', how='count')

        obs_dly = []

        if 'tdew' in self.elems:

            tdew_avg = obs_hrly['dewpoint'].resample('D', how='mean')
            tdew_avg = tdew_avg[cnts['dewpoint'] >=
                                self.min_hrly_for_dly['tdew']]

            df_tdew_avg = pd.DataFrame({'obs_value': tdew_avg,
                                        'elem': 'tdew',
                                        'time': tdew_avg.index.
                                        tz_localize(None)}).reset_index(drop=True)
            obs_dly.append(df_tdew_avg)

        if 'tmin' in self.elems:

            tmin = obs_hrly['temperature'].resample('D', how='min')
            tmin = tmin[cnts['temperature'] >= self.min_hrly_for_dly['tmin']]

            df_tmin = pd.DataFrame({'obs_value': tmin,
                                    'elem': 'tmin',
                                    'time': tmin.index.
                                    tz_localize(None)}).reset_index(drop=True)

            obs_dly.append(df_tmin)

        if 'tmax' in self.elems:

            tmax = obs_hrly['temperature'].resample('D', how='max')
            tmax = tmax[cnts['temperature'] >= self.min_hrly_for_dly['tmax']]

            df_tmax = pd.DataFrame({'obs_value': tmax,
                                    'elem': 'tmax',
                                    'time': tmax.index.
                                    tz_localize(None)}).reset_index(drop=True)

            obs_dly.append(df_tmax)

        obs_dly = pd.concat(obs_dly, ignore_index=True)

        return obs_dly

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

            for a_id, a_tz in zip(stns_obs.station_id, stns_obs.time_zone):

                obs_hrly_stn = self._parse_hrly_stn_obs(a_id, a_tz)

                if len(obs_hrly_stn) > 0:

                    obs_dly = self._to_daily(obs_hrly_stn)
                    obs_dly['station_id'] = a_id

                    obs_all.append(obs_dly)

            obs_all = pd.concat(obs_all, ignore_index=True)

        finally:

            pd.set_option('mode.chained_assignment', opt_val)

        obs_all = obs_all.set_index(['station_id', 'elem', 'time'])
        obs_all = obs_all.sortlevel(0, sort_remaining=True)

        return obs_all
