from .. import LOCAL_DATA_PATH
from ..util.humidity import calc_dew
from ..util.misc import TimeZones
from .generic import ObsIO
from StringIO import StringIO
from datetime import datetime, timedelta
from lxml.html import parse as html_parse
from multiprocessing import Pool
from pytz.exceptions import NonExistentTimeError, AmbiguousTimeError
from time import sleep
from urlparse import urlunsplit, urlsplit
import errno
import gzip
import numpy as np
import os
import pandas as pd
import pycurl
import pytz
import subprocess
import sys
import tempfile
import xray

_URL_BASE_MADIS = 'https://madis-data.ncep.noaa.gov/madisResearch/data/'

_MADIS_SFC_DATASETS = ['LDAD/mesonet/netCDF', 'point/metar/netcdf',
                       'LDAD/coop/netCDF', 'LDAD/crn/netCDF',
                       'LDAD/hcn/netCDF',  # 'LDAD/hydro/netCDF',
                       'LDAD/nepp/netCDF', 'point/sao/netcdf']

_DATE_FMT_MADIS_FILE = '%Y%m%d_%H%M.gz'

_UNIQ_STN_COLUMNS = ['station_id_orig', 'provider', 'sub_provider',
                     'station_name', 'elevation', 'longitude', 'latitude']

# QA Flags for MADIS: https://madis.ncep.noaa.gov/madis_sfc_qc.shtml
# Z = "No QC applied" ;
# C = "Passed QC stage 1" ;
# S = "Passed QC stages 1 and 2" ;
# V = "Passed QC stages 1, 2 and 3" ;
# X = "Failed QC stage 1" ;
# Q = "Passed QC stage 1, but failed stages 2 or 3 " ;
# K = "Passed QC stages 1, 2, 3, and 4" ;
# k = "Passed QC stage 1,2, and 3, failed stage 4 " ;
# G = "Included in accept list" ;
# B = "Included in reject list" ;

_QA_GOOD_TAIR_MADIS = np.array(['C', 'S', 'V', 'G'])

_DSNAME_TO_TRANSFORM = {'mesonet': '_to_dataframe_MADIS_MESONET',
                        'metar': '_to_dataframe_MADIS_METAR',
                        'coop': '_to_dataframe_MADIS_COOP',
                        'sao': '_to_dataframe_MADIS_SAO'}

_k_to_c = lambda k: k - 273.15


def _mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def _madis_file_to_df(fpath, dly_elem, bbox):

    def get_transform_func(ds):

        try:

            transform_func = globals()['_to_dataframe_%s' % ds.id]

        except AttributeError:

            ds_name = fpath.split(os.path.sep)[-3]
            transform_func = globals()[_DSNAME_TO_TRANSFORM[ds_name]]

        return transform_func

    print "Reading " + fpath

    try:
        ds = xray.open_dataset(fpath, decode_cf=True,
                               mask_and_scale=False,
                               concat_characters=True,
                               decode_times=False)

        transform_func = get_transform_func(ds)

        df = transform_func(ds, dly_elem, bbox)
        ds.close()

    except ValueError as e:

        if e.args[0] == 'string size must be a multiple of element size':

            # read tempfile
            gzfile = gzip.open(fpath)
            path_tmp = os.path.join(LOCAL_DATA_PATH, 'tmp')
            _mkdir_p(path_tmp)
            tmpfile, fpath_tmp = tempfile.mkstemp(dir=path_tmp)
            tmpfile = os.fdopen(tmpfile, 'rb+')
            tmpfile.write(gzfile.read())
            gzfile.close()
            tmpfile.close()

            ds = xray.open_dataset(fpath_tmp, decode_cf=True,
                                   mask_and_scale=False,
                                   concat_characters=True,
                                   decode_times=False)

            transform_func = get_transform_func(ds)
            df = transform_func(ds, dly_elem, bbox)
            ds.close()

            os.remove(fpath_tmp)

        else:
            raise e

    return df


def _process_one_path_mp(args):

    path, dly_elem, bbox = args

    return _madis_file_to_df(path, dly_elem, bbox)


class _DailyElem(object):

    def __init__(self, min_hrly_for_dly):

        self.min_hrly_for_dly = min_hrly_for_dly

    @property
    def vnames(self):
        return self._vnames

    def mask_qa(self, df, ds_id, rm_inplace=False):
        raise NotImplementedError

    def convert_units(self, df):
        raise NotImplementedError

    def transform_to_daily(self, obs_tz, a_date):
        raise NotImplementedError

    def _get_mask_rm(self, df, vname, vname_qa,
                     qa_good_flgs=_QA_GOOD_TAIR_MADIS):

        try:

            mask_rm = ((~df[vname_qa].isin(qa_good_flgs)) |
                       (df[vname].isnull()))

        except KeyError:

            mask_rm = np.ones(df.shape[0], dtype=np.bool)

        return mask_rm

    def _get_mask_rm_noqa(self, df, vname):

        try:

            mask_rm = df[vname].isnull()

        except KeyError:

            mask_rm = np.ones(df.shape[0], dtype=np.bool)

        return mask_rm

    def _set_nan(self, df, mask, vname):

        try:
            df.loc[mask, vname] = np.nan
        except KeyError:
            pass


class _Tmin(_DailyElem):

    _vnames = ['temperature', 'temperatureDD', 'minTemp24Hour',
               'minTemp24HourDD', 'timeNominal']

    def mask_qa(self, df, rm_inplace=False):

        mask_rm_t = self._get_mask_rm(df, 'temperature', 'temperatureDD')
        mask_rm_t24 = self._get_mask_rm(df, 'minTemp24Hour', 'minTemp24HourDD')

        self._set_nan(df, mask_rm_t, 'temperature')
        self._set_nan(df, mask_rm_t24, 'minTemp24Hour')

        mask_rm = np.logical_and(mask_rm_t, mask_rm_t24)

        if rm_inplace:

            idx_rm = df.index[mask_rm]

            if idx_rm.size > 0:

                df.drop(idx_rm, axis=0, inplace=True)

        else:

            return mask_rm

    def convert_units(self, df):

        if ('temperature' in df.columns and
                'temperature_C' not in df.columns):

            df['temperature_C'] = _k_to_c(df['temperature'])

        if ('minTemp24Hour' in df.columns and
                'minTemp24Hour_C' not in df.columns):

            df['minTemp24Hour_C'] = _k_to_c(df['minTemp24Hour'])

    def transform_to_daily(self, obs_tz, a_date):

        tz_name = obs_tz.name

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
            begin_time = (a_date + pd.Timedelta(hours=1)).tz_localize(tz_name)
        except AmbiguousTimeError:
            # Time is ambiguous because clocks are set backward for daylight
            # savings time at midnight for this time zone. This only happens in
            # the America/Havana time zone. Set ambiguous=True so that dst=True
            # and the time is considered to be the first occurrence.
            begin_time = a_date.tz_localize(tz_name, ambiguous=True)

        end_time = begin_time + pd.Timedelta(days=1)

        begin_time = begin_time.tz_convert('UTC').tz_convert(None)
        end_time = end_time.tz_convert('UTC').tz_convert(None)

        # Create mask of observations that fall within specified local calendar day
        # and subset observations by mask
        mask_day = ((obs_tz.index >= begin_time) &
                    (obs_tz.index < end_time))
        obs_tz_day = obs_tz[mask_day]

        # Group observations by site_number, day, and hour
        obs_grped_hrs_stnid = obs_tz_day.groupby([obs_tz_day.uid,
                                                  obs_tz_day.index.hour,
                                                  obs_tz_day.index.day])
        # For each stationId and hour, get min and max temperature
        obs_hrly = obs_grped_hrs_stnid['temperature_C'].agg({'tmin': np.min})

        # Group hourly min and max by stationId
        obs_hrly_grped_stns = obs_hrly.groupby(level=0)

        # Create mask of stationIds that have minimum number of
        # of min/max hourly observations to calculate reliable daily
        # Tmin/Tmax
        mask_nhrs = np.array(
            obs_hrly_grped_stns['tmin'].count() >= self.min_hrly_for_dly)

        # Get the min of all of hourly Tmin observations
        # and the max of all hourly Tmax observations
        # This is final daily Tmin, Tmax
        tmin = obs_hrly_grped_stns['tmin'].min()

        # Remove stations that did not meet minimum number of hourly
        # observation requirements
        tmin = tmin[mask_nhrs]

        df_out = pd.DataFrame({'obs_value': tmin,
                               'elem': 'tmin',
                               'time': a_date}).reset_index()

        mask_metar = obs_tz.provider == 'MADIS_METAR'

        if mask_metar.any():
            df_metar = _metar_24hr(obs_tz[mask_metar], tz_name, a_date,
                                   vname_24hr='minTemp24Hour_C')

            df_out = df_out[~df_out.uid.isin(df_metar.uid)]
            df_out = pd.concat([df_out, df_metar], ignore_index=True)

        return df_out


class _Tmax(_DailyElem):

    _vnames = ['temperature', 'temperatureDD', 'maxTemp24Hour',
               'maxTemp24HourDD', 'timeNominal']

    def mask_qa(self, df, rm_inplace=False):

        mask_rm_t = self._get_mask_rm(df, 'temperature', 'temperatureDD')
        mask_rm_t24 = self._get_mask_rm(df, 'maxTemp24Hour', 'maxTemp24HourDD')

        self._set_nan(df, mask_rm_t, 'temperature')
        self._set_nan(df, mask_rm_t24, 'maxTemp24Hour')

        mask_rm = np.logical_and(mask_rm_t, mask_rm_t24)

        if rm_inplace:

            idx_rm = df.index[mask_rm]

            if idx_rm.size > 0:

                df.drop(idx_rm, axis=0, inplace=True)

        else:

            return mask_rm

    def convert_units(self, df):

        if ('temperature' in df.columns and
                'temperature_C' not in df.columns):

            df['temperature_C'] = _k_to_c(df['temperature'])

        if ('maxTemp24Hour' in df.columns and
                'maxTemp24Hour_C' not in df.columns):

            df['maxTemp24Hour_C'] = _k_to_c(df['maxTemp24Hour'])

    def transform_to_daily(self, obs_tz, a_date):

        tz_name = obs_tz.name

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
            begin_time = (a_date + pd.Timedelta(hours=1)).tz_localize(tz_name)
        except AmbiguousTimeError:
            # Time is ambiguous because clocks are set backward for daylight
            # savings time at midnight for this time zone. This only happens in
            # the America/Havana time zone. Set ambiguous=True so that dst=True
            # and the time is considered to be the first occurrence.
            begin_time = a_date.tz_localize(tz_name, ambiguous=True)

        end_time = begin_time + pd.Timedelta(days=1)

        begin_time = begin_time.tz_convert('UTC').tz_convert(None)
        end_time = end_time.tz_convert('UTC').tz_convert(None)

        # Create mask of observations that fall within specified local calendar day
        # and subset observations by mask
        mask_day = ((obs_tz.index >= begin_time) &
                    (obs_tz.index < end_time))
        obs_tz_day = obs_tz[mask_day]

        # Group observations by site_number, day, and hour
        obs_grped_hrs_stnid = obs_tz_day.groupby([obs_tz_day.uid,
                                                  obs_tz_day.index.hour,
                                                  obs_tz_day.index.day])
        # For each stationId and hour, get min and max temperature
        obs_hrly = obs_grped_hrs_stnid['temperature_C'].agg({'tmax': np.max})

        # Group hourly min and max by stationId
        obs_hrly_grped_stns = obs_hrly.groupby(level=0)

        # Create mask of stationIds that have minimum number of
        # of min/max hourly observations to calculate reliable daily
        # Tmin/Tmax
        mask_nhrs = np.array(
            obs_hrly_grped_stns['tmax'].count() >= self.min_hrly_for_dly)

        # Get the min of all of hourly Tmin observations
        # and the max of all hourly Tmax observations
        # This is final daily Tmin, Tmax
        tmax = obs_hrly_grped_stns['tmax'].max()

        # Remove stations that did not meet minimum number of hourly
        # observation requirements
        tmax = tmax[mask_nhrs]

        df_out = pd.DataFrame({'obs_value': tmax,
                               'elem': 'tmax',
                               'time': a_date}).reset_index()

        mask_metar = obs_tz.provider == 'MADIS_METAR'

        if mask_metar.any():
            df_metar = _metar_24hr(obs_tz[mask_metar], tz_name, a_date,
                                   vname_24hr='maxTemp24Hour_C')

            df_out = df_out[~df_out.uid.isin(df_metar.uid)]
            df_out = pd.concat([df_out, df_metar], ignore_index=True)

        return df_out


class _Tdew(_DailyElem):

    _vnames = ['dewpoint', 'dewpointDD', 'relHumidity', 'relHumidityDD',
               'temperature', 'temperatureDD']

    def mask_qa(self, df, rm_inplace=False):

        mask_rm_tdew = self._get_mask_rm(df, 'dewpoint', 'dewpointDD')
        mask_rm_rh = self._get_mask_rm(df, 'relHumidity', 'relHumidityDD')
        mask_rm_t = self._get_mask_rm(df, 'temperature', 'temperatureDD')

        self._set_nan(df, mask_rm_tdew, 'dewpoint')
        self._set_nan(df, mask_rm_rh, 'relHumidity')
        self._set_nan(df, mask_rm_t, 'temperature')

        mask_rm = np.logical_and(np.logical_and(mask_rm_tdew, mask_rm_rh),
                                 mask_rm_t)

        if rm_inplace:

            idx_rm = df.index[mask_rm]

            if idx_rm.size > 0:

                df.drop(idx_rm, axis=0, inplace=True)

        else:

            return mask_rm

    def convert_units(self, df):

        if ('dewpoint' in df.columns and
                'dewpoint_C' not in df.columns):

            df['dewpoint_C'] = _k_to_c(df['dewpoint'])

        if ('temperature' in df.columns and
                'temperature_C' not in df.columns):

            df['temperature_C'] = _k_to_c(df['temperature'])

        if ('relHumidity' in df.columns and
                'dewpoint_rh_C' not in df.columns):

            df['dewpoint_rh_C'] = calc_dew(df['relHumidity'],
                                           df['temperature_C'])

    def transform_to_daily(self, obs_tz, a_date):

        tz_name = obs_tz.name

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
            begin_time = (a_date + pd.Timedelta(hours=1)).tz_localize(tz_name)
        except AmbiguousTimeError:
            # Time is ambiguous because clocks are set backward for daylight
            # savings time at midnight for this time zone. This only happens in
            # the America/Havana time zone. Set ambiguous=True so that dst=True
            # and the time is considered to be the first occurrence.
            begin_time = a_date.tz_localize(tz_name, ambiguous=True)

        end_time = begin_time + pd.Timedelta(days=1)

        begin_time = begin_time.tz_convert('UTC').tz_convert(None)
        end_time = end_time.tz_convert('UTC').tz_convert(None)

        # Create mask of observations that fall within specified local calendar day
        # and subset observations by mask
        mask_day = ((obs_tz.index >= begin_time) &
                    (obs_tz.index < end_time))
        obs_tz_day = obs_tz[mask_day]

        # Group observations by site_number, day, and hour
        obs_grped_hrs_stnid = obs_tz_day.groupby([obs_tz_day.uid,
                                                  obs_tz_day.index.hour,
                                                  obs_tz_day.index.day])
        # For each stationId and hour, get mean dewpoint_C, mean dewpoint_rh_C
        obs_hrly = (obs_grped_hrs_stnid[['dewpoint_C', 'dewpoint_rh_C']].
                    agg(np.mean))

        # Group  by stationId
        obs_hrly_grped_stns = obs_hrly.groupby(level=0)

        # Count the number of hourly obs for dewpoint_C, dewpoint_rh_C
        nobs = obs_hrly_grped_stns[['dewpoint_C', 'dewpoint_rh_C']].count()

        # Create masks of stationIds that have minimum number of
        # of hourly obs to calculate reliable average dewpoint_C, dewpoint_rh_C
        mask_nhrs_tdew = nobs['dewpoint_C'] >= self.min_hrly_for_dly
        mask_nhrs_tdewrh = nobs['dewpoint_rh_C'] >= self.min_hrly_for_dly

        # Get the mean of all of hourly dewpoint_C, dewpoint_rh_C observations
        # These are the daily average values
        obs_avg_tdew = obs_hrly_grped_stns[['dewpoint_C',
                                            'dewpoint_rh_C']].mean()

        # Remove stations that did not meet minimum number of hourly
        # observation requirements
        tdew = obs_avg_tdew['dewpoint_C'][mask_nhrs_tdew]
        tdewrh = obs_avg_tdew['dewpoint_rh_C'][mask_nhrs_tdewrh]

        # Remove tdewrh obs that are already in tdew
        tdewrh = tdewrh[~tdewrh.index.isin(tdew.index)]

        tdew_fnl = pd.concat([tdew, tdewrh])

        df_out = pd.DataFrame({'obs_value': tdew_fnl,
                               'elem': 'tdew',
                               'time': a_date}).reset_index()

        return df_out


class _Wspd(_DailyElem):

    _vnames = ['windSpeed', 'windSpeedDD']

    def mask_qa(self, df, rm_inplace=False):

        mask_rm = self._get_mask_rm(df, 'windSpeed', 'windSpeedDD')

        self._set_nan(df, mask_rm, 'windSpeed')

        if rm_inplace:

            idx_rm = df.index[mask_rm]

            if idx_rm.size > 0:

                df.drop(idx_rm, axis=0, inplace=True)

        else:

            return mask_rm

    def convert_units(self, df):

        pass

    def transform_to_daily(self, obs_tz, a_date):

        tz_name = obs_tz.name

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
            begin_time = (a_date + pd.Timedelta(hours=1)).tz_localize(tz_name)
        except AmbiguousTimeError:
            # Time is ambiguous because clocks are set backward for daylight
            # savings time at midnight for this time zone. This only happens in
            # the America/Havana time zone. Set ambiguous=True so that dst=True
            # and the time is considered to be the first occurrence.
            begin_time = a_date.tz_localize(tz_name, ambiguous=True)

        end_time = begin_time + pd.Timedelta(days=1)

        begin_time = begin_time.tz_convert('UTC').tz_convert(None)
        end_time = end_time.tz_convert('UTC').tz_convert(None)

        # Create mask of observations that fall within specified local calendar day
        # and subset observations by mask
        mask_day = ((obs_tz.index >= begin_time) &
                    (obs_tz.index < end_time))
        obs_tz_day = obs_tz[mask_day]

        # Group observations by site_number, day, and hour
        obs_grped_hrs_stnid = obs_tz_day.groupby([obs_tz_day.uid,
                                                  obs_tz_day.index.hour,
                                                  obs_tz_day.index.day])

        # For each stationId and hour, get mean wind speed
        obs_hrly = (obs_grped_hrs_stnid[['windSpeed']].agg(np.mean))

        # Group by stationId
        obs_hrly_grped_stns = obs_hrly.groupby(level=0)

        # Count the number of hourly obs
        nobs = obs_hrly_grped_stns[['windSpeed']].count()

        # Create masks of stationIds that have minimum number of
        # of hourly obs to calculate reliable avg wind speed
        mask_nhrs = nobs['windSpeed'] >= self.min_hrly_for_dly

        # Get the mean of all of hourly wind observations
        # These are the daily average values
        obs_avg = obs_hrly_grped_stns[['windSpeed']].mean()

        # Remove stations that did not meet minimum number of hourly
        # observation requirements
        wspd = obs_avg['windSpeed'][mask_nhrs]

        if wspd.empty:

            df_out = pd.DataFrame(columns=['obs_value', 'elem', 'time'])

        else:

            df_out = pd.DataFrame({'obs_value': wspd,
                                   'elem': 'wspd',
                                   'time': a_date}).reset_index()

        return df_out


class _Srad(_DailyElem):

    _vnames = ['solarRadiation']

    def mask_qa(self, df, rm_inplace=False):

        mask_rm = self._get_mask_rm_noqa(df, 'solarRadiation')

        try:

            # Remove invalid values greater than 1500 watts and less than zero
            # http://mesowest.utah.edu/cgi-bin/droman/variable_select.cgi
            mask_invalid = ((df['solarRadiation'] > 1500) |
                            (df['solarRadiation'] < 0))
            self._set_nan(df, mask_invalid, 'solarRadiation')

            mask_rm = np.logical_or(mask_rm, mask_invalid)

        except KeyError:
            pass

        if rm_inplace:

            idx_rm = df.index[mask_rm]

            if idx_rm.size > 0:

                df.drop(idx_rm, axis=0, inplace=True)

        else:

            return mask_rm

    def convert_units(self, df):

        pass

    def transform_to_daily(self, obs_tz, a_date):

        tz_name = obs_tz.name

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
            begin_time = (a_date + pd.Timedelta(hours=1)).tz_localize(tz_name)
        except AmbiguousTimeError:
            # Time is ambiguous because clocks are set backward for daylight
            # savings time at midnight for this time zone. This only happens in
            # the America/Havana time zone. Set ambiguous=True so that dst=True
            # and the time is considered to be the first occurrence.
            begin_time = a_date.tz_localize(tz_name, ambiguous=True)

        end_time = begin_time + pd.Timedelta(days=1)

        begin_time = begin_time.tz_convert('UTC').tz_convert(None)
        end_time = end_time.tz_convert('UTC').tz_convert(None)

        # Create mask of observations that fall within specified local calendar day
        # and subset observations by mask
        mask_day = ((obs_tz.index >= begin_time) &
                    (obs_tz.index < end_time))
        obs_tz_day = obs_tz[mask_day]

        # Group observations by site_number, day, and hour
        obs_grped_hrs_stnid = obs_tz_day.groupby([obs_tz_day.uid,
                                                  obs_tz_day.index.hour,
                                                  obs_tz_day.index.day])

        # For each stationId and hour, get mean srad
        obs_hrly = (obs_grped_hrs_stnid[['solarRadiation']].agg(np.mean))

        # Group by stationId
        obs_hrly_grped_stns = obs_hrly.groupby(level=0)

        # Count the number of hourly obs
        nobs = obs_hrly_grped_stns[['solarRadiation']].count()

        # Create masks of stationIds that have minimum number of
        # of hourly obs to calculate reliable avg srad
        mask_nhrs = nobs['solarRadiation'] >= self.min_hrly_for_dly

        # Get the mean of all of hourly srad observations
        # These are the daily average values
        obs_avg = obs_hrly_grped_stns[['solarRadiation']].mean()

        # Remove stations that did not meet minimum number of hourly
        # observation requirements
        srad = obs_avg['solarRadiation'][mask_nhrs]

        try:
            # Remove invalid srad values == 0
            srad = srad[srad > 0].copy()
        except ValueError:
            # srad is empty even before masking
            pass

        if srad.empty:

            df_out = pd.DataFrame(columns=['obs_value', 'elem', 'time'])

        else:

            df_out = pd.DataFrame({'obs_value': srad,
                                   'elem': 'srad',
                                   'time': a_date}).reset_index()

        return df_out


class _Prcp(_DailyElem):

    # Does not include observations from METAR

    _vnames = ['precipAccum', 'precipAccumDD', 'precipRate', 'precipRateDD']

    def mask_qa(self, df, rm_inplace=False):

        mask_rm_accum = self._get_mask_rm(df, 'precipAccum', 'precipAccumDD')
        mask_rm_rate = self._get_mask_rm(df, 'precipRate', 'precipRateDD')

        self._set_nan(df, mask_rm_accum, 'precipAccum')
        self._set_nan(df, mask_rm_rate, 'precipRate')

        mask_rm = np.logical_and(mask_rm_accum, mask_rm_rate)

        if rm_inplace:

            idx_rm = df.index[mask_rm]

            if idx_rm.size > 0:

                df.drop(idx_rm, axis=0, inplace=True)

        else:

            return mask_rm

    def convert_units(self, df):

        if ('precipRate' in df.columns and
                'precipRate_mm' not in df.columns):

            # convert meters per second to millimeters per second
            df['precipRate_mm'] = df['precipRate'] * 1000.0

    def transform_to_daily(self, obs_tz, a_date):

        tz_name = obs_tz.name

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
            begin_time = (a_date + pd.Timedelta(hours=1)).tz_localize(tz_name)
        except AmbiguousTimeError:
            # Time is ambiguous because clocks are set backward for daylight
            # savings time at midnight for this time zone. This only happens in
            # the America/Havana time zone. Set ambiguous=True so that dst=True
            # and the time is considered to be the first occurrence.
            begin_time = a_date.tz_localize(tz_name, ambiguous=True)

        end_time = begin_time + pd.Timedelta(days=1)

        begin_time = begin_time.tz_convert('UTC').tz_convert(None)
        end_time = end_time.tz_convert('UTC').tz_convert(None)

        # subtract another hour from beginTime to get precip accum delta
        # at the true begin time
        begin_time_accum = begin_time - pd.Timedelta(hours=1)

        # Create mask of observations that fall within specified local calendar day
        # and subset observations by mask
        mask_day = ((obs_tz.index >= begin_time_accum) &
                    (obs_tz.index < end_time))
        obs_tz_day = obs_tz[mask_day]

        try:
            # Calculate first difference of precipAccum at each station to get
            # precipitation deltas
            obs_tz_day = obs_tz_day.set_index([obs_tz_day.uid,
                                               obs_tz_day.index])
            obs_tz_day = obs_tz_day.sortlevel(0, sort_remaining=True)

            obs_accum = (obs_tz_day.groupby(level=0)[['precipAccum']].
                         transform(lambda x: x.diff()))

            obs_tz_day['precipAccum'] = obs_accum

            uid = obs_tz_day.index.get_level_values(0)
            obs_tz_day.index = obs_tz_day.index.droplevel(0)
            obs_tz_day['uid'] = uid

            # Remove observations that where >= begin_time_accum but not
            # end_time
            obs_tz_day = obs_tz_day[obs_tz_day.index >= begin_time].copy()
            # Set any negative precipAccum deltas to 0. A negative value means
            # that the precipAccum was reset back to 0
            obs_tz_day.loc[obs_tz_day.precipAccum < 0, 'precipAccum'] = 0

            # Group observations by site_number, day, and hour
            obs_grped_hrs_stnid = obs_tz_day.groupby([obs_tz_day.uid,
                                                      obs_tz_day.index.hour,
                                                      obs_tz_day.index.day])

            # For each stationId and hour, get the sum of precipAccum and the
            # mean of precipRate
            obs_hrly = (obs_grped_hrs_stnid.agg({'precipAccum': np.sum,
                                                 'precipRate_mm': np.mean}))

            # Group  by stationId
            obs_hrly_grped_stns = obs_hrly.groupby(level=0)

            # Count the number of hourly obs for precipAccum, precipRate_mm
            nobs = obs_hrly_grped_stns[
                ['precipAccum', 'precipRate_mm']].count()

            # Create masks of stationIds that have minimum number of
            # of hourly obs to calculate reliable total daily prcp
            mask_nhrs_accum = nobs['precipAccum'] >= self.min_hrly_for_dly
            mask_nhrs_rate = nobs['precipRate_mm'] >= self.min_hrly_for_dly

            # Get the sum of all precipAccum deltas to get total prcp for day
            # Get the daily mean of precipRate hourly values and multiply by
            # number of seconds in the day to derive total precip from
            # precipRate
            rate_total_func = lambda x: x.mean() * 86400
            obs_total_prcp = obs_hrly_grped_stns.agg({'precipAccum': np.sum,
                                                      'precipRate_mm':
                                                      rate_total_func})

            # Remove stations that did not meet minimum number of hourly
            # observation requirements
            accum_total = obs_total_prcp['precipAccum'][mask_nhrs_accum]
            rate_total = obs_total_prcp['precipRate_mm'][mask_nhrs_rate]

            # Prefer to use precipAccum-derived total than precipRate-derived total
            # Remove all precipRate-derived totals that are already calculated via
            # precipAccum
            rate_total = rate_total[~rate_total.index.isin(accum_total.index)]

            prcp_fnl = pd.concat([accum_total, rate_total])

            # Remove invalid observations. 1143mm = 45in
            # http://mesowest.utah.edu/cgi-bin/droman/variable_select.cgi
            mask_valid = ((prcp_fnl >= 0) & (prcp_fnl <= 1143))

            prcp_fnl = prcp_fnl[mask_valid].copy()

            df_out = pd.DataFrame({'obs_value': prcp_fnl,
                                   'elem': 'prcp',
                                   'time': a_date}).reset_index()
        except ValueError as e:

            if e.args[0] == 'All objects passed were None':

                # No valid prcp obs. Return empty dataframe
                df_out = pd.DataFrame(columns=['obs_value', 'elem', 'time'])

            else:

                raise e

        return df_out


class _MultiElem(_DailyElem):

    _ELEM_TO_DAILYELEM = {'tmin': _Tmin,
                          'tmax': _Tmax,
                          'tdew': _Tdew,
                          'prcp': _Prcp,
                          'srad': _Srad,
                          'wspd': _Wspd}

    def __init__(self, elems, min_hrly_for_dly):

        self._dly_elems = [self._ELEM_TO_DAILYELEM[a_elem](min_hrly_for_dly[a_elem])
                           for a_elem in elems]

    @property
    def vnames(self):

        return list(np.unique(np.concatenate([a_var.vnames for a_var in
                                              self._dly_elems])))

    def mask_qa(self, df, rm_inplace=False):

        mask_null = np.ones(df.shape[0], dtype=np.bool)

        for a_dly_elem in self._dly_elems:
            mask_null = np.logical_and(a_dly_elem.mask_qa(df,
                                                          rm_inplace=False),
                                       mask_null)

        if rm_inplace:

            idx_rm = df.index[mask_null]

            if idx_rm.size > 0:

                df.drop(idx_rm, axis=0, inplace=True)

        else:

            return mask_null

    def convert_units(self, df):

        for a_dly_elem in self._dly_elems:
            a_dly_elem.convert_units(df)

    def transform_to_daily(self, obs_tz, a_date):

        all_obs = [a_elem.transform_to_daily(obs_tz, a_date)
                   for a_elem in self._dly_elems]

        all_obs = pd.concat(all_obs, ignore_index=True)

        return all_obs


def _get_utc_hours(start_date, end_date, tzs):

    end_date = end_date + timedelta(days=1)

    start_hr = datetime(start_date.year, start_date.month, start_date.day, 0)
    end_hr = datetime(end_date.year, end_date.month, end_date.day, 3)

    hrs_utc = [pd.Timestamp(start_hr, tz=a_tz).tz_convert('UTC')
               for a_tz in tzs]
    hrs_utc.extend([pd.Timestamp(end_hr, tz=a_tz).tz_convert('UTC')
                    for a_tz in tzs])

    hrs_utc = pd.DatetimeIndex(hrs_utc)

    min_hr = hrs_utc.min()
    max_hr = hrs_utc.max()

    if min_hr.minute != 0:

        min_hr = min_hr - pd.offsets.Minute(min_hr.minute)

    if max_hr.minute != 0:
        max_hr = max_hr - pd.offsets.Minute(max_hr.minute) + pd.offsets.Hour()

    hrs_all_utc = pd.date_range(min_hr, max_hr, freq='H')

    return hrs_all_utc


def _to_dataframe_MADIS_METAR(ds, dly_elem, bbox=None):

    if ds.idVariables != 'stationName':
        raise ValueError("Expected stationName for idVariables. "
                         "Received: " + str(ds.idVariables))

    if ds.timeVariables != 'timeObs':
        raise ValueError("Expected timeObs for timeVariables. "
                         "Received: " + str(ds.timeVariables))

#     if ds.stationLocationVariables != 'latitude,longitude,elevation':
#         raise ValueError("Expected latitude,longitude,elevation for "
#                          "stationLocationVariables. Received: " +
#                          str(ds.stationLocationVariables))

#     if ds.stationDescriptionVariable != 'locationName':
#         raise ValueError("Expected locationName for "
#                          "stationDescriptionVariable. Received: " +
#                          str(ds.stationDescriptionVariable))

    vnames_ds = _combine_vnames(['stationName', 'locationName', 'latitude',
                                 'longitude', 'elevation', 'timeObs'],
                                dly_elem.vnames, ds.variables.keys())

    ds = xray.Dataset(ds[vnames_ds])
    ds = xray.decode_cf(ds)

    df = ds.to_dataframe()
    df.rename(columns={'stationName': 'station_id_orig', 'timeObs':
                       'time', 'locationName': 'station_name'}, inplace=True)
    df['provider'] = 'MADIS_METAR'
    df['sub_provider'] = ''

    vnames_df = _combine_vnames(['station_id_orig', 'station_name', 'provider',
                                 'sub_provider', 'elevation', 'longitude',
                                 'latitude', 'time'], dly_elem.vnames,
                                df.columns)

    df = df[vnames_df]

    if bbox is not None:

        df = bbox.remove_outbnds_df(df)

    dly_elem.mask_qa(df, rm_inplace=True)
    dly_elem.convert_units(df)

    return df


def _to_dataframe_MADIS_SAO(ds, dly_elem, bbox=None):

    if ds.idVariables != 'stationName':
        raise ValueError("Expected stationName for idVariables. "
                         "Received: " + str(ds.idVariables))

    if ds.timeVariables != 'timeObs':
        raise ValueError("Expected timeObs for timeVariables. "
                         "Received: " + str(ds.timeVariables))

#     if ds.stationLocationVariables != 'latitude,longitude,elevation':
#         raise ValueError("Expected latitude,longitude,elevation for "
#                          "stationLocationVariables. Received: " +
#                          str(ds.stationLocationVariables))

#     if ds.stationDescriptionVariable != 'locationName':
#         raise ValueError("Expected locationName for "
#                          "stationDescriptionVariable. Received: " +
#                          str(ds.stationDescriptionVariable))

    vnames_ds = _combine_vnames(['stationName', 'latitude',
                                 'longitude', 'elevation', 'timeObs'],
                                dly_elem.vnames, ds.variables.keys())

    ds = xray.Dataset(ds[vnames_ds])
    ds = xray.decode_cf(ds)

    df = ds.to_dataframe()
    df.rename(columns={'stationName': 'station_id_orig', 'timeObs': 'time'},
              inplace=True)
    df['provider'] = 'MADIS_SAO'
    df['sub_provider'] = ''
    df['station_name'] = ''
    df['station_id_orig'] = df['station_id_orig'].str.strip()
    vnames_df = _combine_vnames(['station_id_orig', 'station_name', 'provider',
                                 'sub_provider', 'elevation', 'longitude',
                                 'latitude', 'time'], dly_elem.vnames,
                                df.columns)

    df = df[vnames_df]

    if bbox is not None:

        df = bbox.remove_outbnds_df(df)

    dly_elem.mask_qa(df, rm_inplace=True)
    dly_elem.convert_units(df)

    return df


def _to_dataframe_MADIS_MESONET(ds, dly_elem, bbox=None):

    if ds.idVariables != 'providerId,dataProvider':
        raise ValueError("Expected providerId,dataProvider for idVariables. "
                         "Received: " + str(ds.idVariables))

    if ds.timeVariables != 'observationTime,reportTime,receivedTime':
        raise ValueError("Expected observationTime,reportTime,receivedTime "
                         "for timeVariables.Received: " +
                         str(ds.timeVariables))

#     if ds.stationLocationVariables != 'latitude,longitude,elevation':
#         raise ValueError("Expected latitude,longitude,elevation for "
#                          "stationLocationVariables. Received: " +
#                          str(ds.stationLocationVariables))

#     if ds.stationDescriptionVariable != 'stationName':
#         raise ValueError("Expected stationName for stationDescriptionVariable."
#                          " Received: " + str(ds.stationDescriptionVariable))

    vnames_ds = _combine_vnames(['providerId', 'dataProvider', 'stationName',
                                 'latitude', 'longitude', 'elevation',
                                 'observationTime'], dly_elem.vnames,
                                ds.variables.keys())

    ds = xray.Dataset(ds[vnames_ds])

    for a_var in ds.data_vars.values():

        try:
            del a_var.attrs['missing_value']
        except KeyError:
            pass

    ds = xray.decode_cf(ds)

    df = ds.to_dataframe()
    df['station_id_orig'] = df['dataProvider'] + "_" + df['providerId']

    df.rename(columns={'stationName': 'station_name', 'observationTime':
                       'time', 'dataProvider': 'sub_provider'}, inplace=True)
    df['provider'] = 'MADIS_MESONET'

    vnames_df = _combine_vnames(['station_id_orig', 'station_name', 'provider',
                                 'sub_provider', 'elevation', 'longitude',
                                 'latitude', 'time'], dly_elem.vnames,
                                df.columns)

    df = df[vnames_df]

    if bbox is not None:

        df = bbox.remove_outbnds_df(df)

    dly_elem.mask_qa(df, rm_inplace=True)
    dly_elem.convert_units(df)

    return df


def _combine_vnames(vnames_base, vnames_add, ds_vnames):

    vnames = list(vnames_base)
    vnames.extend(vnames_add)
    vnames = np.array(vnames)

    mask_vnames = np.in1d(vnames, np.array(ds_vnames),
                          assume_unique=False)

    vnames = list(vnames[mask_vnames])

    return vnames


def _to_dataframe_MADIS_COOP(ds, dly_elem, bbox=None):

    if ds.idVariables != 'providerId,dataProvider':
        raise ValueError("Expected providerId,dataProvider for idVariables. "
                         "Received: " + str(ds.idVariables))

    if ds.timeVariables != 'observationTime,reportTime,receivedTime':
        raise ValueError("Expected observationTime,reportTime,receivedTime "
                         "for timeVariables. Received: " +
                         str(ds.timeVariables))

#     if ds.stationLocationVariables != 'latitude,longitude,elevation':
#         raise ValueError("Expected latitude,longitude,elevation for "
#                          "stationLocationVariables. Received: " +
#                          str(ds.stationLocationVariables))

#     if ds.stationDescriptionVariable != 'stationName':
#         raise ValueError("Expected stationName for stationDescriptionVariable."
#                          " Received: " + str(ds.stationDescriptionVariable))

    vnames_ds = _combine_vnames(['providerId', 'dataProvider', 'stationName',
                                 'latitude', 'longitude', 'elevation',
                                 'observationTime'], dly_elem.vnames,
                                ds.variables.keys())

    ds = xray.Dataset(ds[vnames_ds])

    for a_var in ds.data_vars.values():

        try:
            del a_var.attrs['missing_value']
        except KeyError:
            pass

    ds = xray.decode_cf(ds)

    df = ds.to_dataframe()
    df['station_id_orig'] = df['dataProvider'] + "_" + df['providerId']

    df.rename(columns={'stationName': 'station_name', 'observationTime':
                       'time', 'dataProvider': 'sub_provider'}, inplace=True)
    df['provider'] = 'MADIS_COOP'

    vnames_df = _combine_vnames(['station_id_orig', 'station_name', 'provider',
                                 'sub_provider', 'elevation', 'longitude',
                                 'latitude', 'time'], dly_elem.vnames,
                                df.columns)

    df = df[vnames_df]

    if bbox is not None:

        df = bbox.remove_outbnds_df(df)

    dly_elem.mask_qa(df, rm_inplace=True)
    dly_elem.convert_units(df)

    return df


def _metar_24hr(a_obs_metar, tz_name, a_date, vname_24hr):

    vname24hr_to_elem = {'minTemp24Hour_C': 'tmin', 'maxTemp24Hour_C': 'tmax'}

    next_date = a_date + timedelta(days=1)
    hr_minmax = pd.Timestamp(datetime(next_date.year,
                                      next_date.month,
                                      next_date.day,
                                      1), tz=tz_name)
    hr_minmax = hr_minmax.tz_convert('UTC').tz_convert(None)

    mask_hr = a_obs_metar['timeNominal'] == hr_minmax
    obs_minmax = a_obs_metar[mask_hr]
    obs_minmax = obs_minmax.groupby('uid').mean()

    df_out = obs_minmax.loc[~obs_minmax[vname_24hr].isnull(),
                            [vname_24hr]].reset_index()
    df_out['elem'] = vname24hr_to_elem[vname_24hr]
    df_out['time'] = a_date
    df_out = df_out.rename(columns={vname_24hr: 'obs_value'})

    return df_out


def _round_coords(df_stns, decimals=10):

    df_stns = df_stns.copy()

    df_stns['longitude'] = df_stns.longitude.round(decimals)
    df_stns['latitude'] = df_stns.latitude.round(decimals)
    df_stns['elevation'] = df_stns.elevation.round(decimals)

    return df_stns


class MadisObsIO(ObsIO):

    _avail_elems = ['tmin', 'tmax', 'tdew', 'srad', 'prcp', 'wspd']
    _requires_local = True

    _MIN_HRLY_FOR_DLY_DFLT = {'tmin': 20, 'tmax': 20, 'tdew': 4, 'srad': 24,
                              'prcp': 24, 'wspd': 24}

    def __init__(self, local_data_path=None, username=None, password=None,
                 madis_datasets=None, local_time_zones=None,
                 fname_tz_geonames=None, min_hrly_for_dly=None, nprocs=1,
                 **kwargs):

        super(MadisObsIO, self).__init__(**kwargs)

        self.local_data_path = (local_data_path if local_data_path
                                else LOCAL_DATA_PATH)
        self.madis_datasets = (madis_datasets if madis_datasets
                               else _MADIS_SFC_DATASETS)
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

        self._fname_tz_geonames = fname_tz_geonames
        self._username = username
        self._password = password
        self.nprocs = nprocs
        self.path_madis_data = os.path.join(self.local_data_path, 'MADIS')
        
        if not os.path.isdir(self.path_madis_data):
            os.mkdir(self.path_madis_data)
        
        self._fpath_stns_cache = os.path.join(self.path_madis_data,
                                              'stns_cache.pkl')

        try:

            self._stns_cache = _round_coords(pd.read_pickle(self.
                                                            _fpath_stns_cache))

        except IOError:

            self._stns_cache = pd.DataFrame(columns=_UNIQ_STN_COLUMNS + ['uid',
                                                                         'time_zone'])

        if self.has_start_end_dates:

            start_date = self.start_date
            end_date = self.end_date

        else:
            # hardcode begin date for now
            start_date = pd.Timestamp('2001-07-01')
            end_date = pd.Timestamp.now()

        local_time_zones = (local_time_zones if local_time_zones
                            else pytz.all_timezones)
        self._utc_hrs = _get_utc_hours(start_date, end_date, local_time_zones)
        self._dates = pd.date_range(start_date, end_date, freq='D')
        self._dly_elem = _MultiElem(self.elems, self.min_hrly_for_dly)
        self._a_df_obs = None
        self._a_tz = None
        self._a_fpaths_madis = None

    @property
    def _fpaths_madis(self):

        if self._a_fpaths_madis is None:

            hrs_fnames = (self._utc_hrs.map(lambda x:
                                            x.strftime(_DATE_FMT_MADIS_FILE)).
                          astype(np.str))

            hrs_s = pd.Series(hrs_fnames, index=self._utc_hrs)

            hrs_grpd = hrs_s.groupby([lambda x: x.year,
                                      lambda x: x.month,
                                      lambda x: x.day])

            fpaths = []
            fpaths_exist = []

            for a_dspath in self.madis_datasets:

                for a_day, day_fnames in hrs_grpd:

                    yr, mth, day = a_day
                    syr, smth, sday = "%d" % yr, "%.2d" % mth, "%.2d" % day

                    for a_fname in day_fnames:

                        fpath = os.path.join(self.path_madis_data, syr, smth,
                                             sday, a_dspath, a_fname)
                        fpaths.append(fpath)

                        fpaths_exist.append(os.path.exists(fpath))

            fpaths_madis = pd.DataFrame({'fpath': fpaths,
                                         'exists': fpaths_exist})

            self._a_fpaths_madis = fpaths_madis

        return self._a_fpaths_madis

    @property
    def _tz(self):

        if self._a_tz is None:

            self._a_tz = TimeZones(self._fname_tz_geonames)

        return self._a_tz

    @property
    def _df_obs(self):

        if self._a_df_obs is None:

            print "MadisObsIO: Reading in hourly MADIS netCDF files......"

            fpaths_fnd = self._fpaths_madis.fpath[self._fpaths_madis.exists]
            fpaths_miss = self._fpaths_madis.fpath[~self._fpaths_madis.exists]

            print ("MadisObsIO: Found the following data files for specified datasets:\n" +
                   "\n".join(fpaths_fnd.values.astype(np.str)))

            print ("MadisObsIO: Missing data files for specified datasets:\n" +
                   "\n".join(fpaths_miss.values.astype(np.str)))

            if self.nprocs > 1:
                # http://stackoverflow.com/questions/24171725/
                # scikit-learn-multicore-attributeerror-stdin-instance-
                # has-no-attribute-close
                if not hasattr(sys.stdin, 'close'):
                    def dummy_close():
                        pass
                    sys.stdin.close = dummy_close

                pool = Pool(processes=self.nprocs)

                iter_fpaths = [(a_fpath, self._dly_elem, self.bbox) for
                               a_fpath in fpaths_fnd]

                datasets = pool.map(_process_one_path_mp,
                                    iter_fpaths, chunksize=1)
                pool.close()
                pool.join()

            else:

                datasets = [_madis_file_to_df(p, self._dly_elem, self.bbox)
                            for p in fpaths_fnd]

            combined = pd.concat(datasets)
            combined.set_index(np.arange(combined.shape[0]), inplace=True)

            combined = _round_coords(combined)

            self._a_df_obs = combined
            self._a_df_obs.set_index('time', inplace=True)

        return self._a_df_obs

    @property
    def _df_obs_with_uid(self):

        if 'uid' not in self._df_obs.columns:
            
            print ("MadisObsIO: Initializing loaded observations "
                   "for daily aggregation...")

            self._df_obs.reset_index(inplace=True)

            self._a_df_obs = self._df_obs.merge(self.
                                                stns[_UNIQ_STN_COLUMNS +
                                                     ['uid', 'time_zone']],
                                                on=_UNIQ_STN_COLUMNS,
                                                how='left', sort=False)

            self._a_df_obs.set_index('time', inplace=True)

        return self._a_df_obs

    def _read_stns(self):

        mask_dup = self._df_obs.duplicated(_UNIQ_STN_COLUMNS)
        stns = (self._df_obs.loc[~mask_dup, _UNIQ_STN_COLUMNS].
                reset_index(drop=True))
        stns = self._merge_with_stn_cache(stns)
        self._update_stn_cache(stns)

        return stns

    def read_obs(self, stns_ids=None):

        if stns_ids is None:
            stns_obs = self.stns
            df_obs = self._df_obs_with_uid
        else:

            stns_obs = self.stns.loc[stns_ids]

            df_obs = self._df_obs_with_uid[self._df_obs_with_uid.
                                           uid.isin(stns_obs.uid)]

        grp_tz = df_obs.groupby('time_zone')
        all_obs = []

        for a_date in self._dates:

            df_dly = grp_tz.apply(self._dly_elem.transform_to_daily,
                                  a_date=a_date)
            df_dly.index = df_dly.index.droplevel(0)
            df_dly = df_dly.reset_index(drop=True)
            all_obs.append(df_dly)

        all_obs = pd.concat(all_obs, ignore_index=True)

        obs_merge = pd.merge(all_obs, self.stns[['uid', 'station_id']],
                             how='left', on='uid', sort=False)

        obs_merge = obs_merge.set_index(['station_id', 'elem', 'time'])
        obs_merge = obs_merge.sortlevel(0, sort_remaining=True)

        return obs_merge

    def download_local(self):

        def _url_path_join(*parts):
            """
            Normalize url parts and join them with a slash.
            http://codereview.stackexchange.com/questions/13027/
            joining-url-path-components-intelligently
            """
            def _first_of_each(*sequences):
                return (next((x for x in sequence if x), '') for
                        sequence in sequences)

            schemes, netlocs, paths, queries, fragments = zip(
                *(urlsplit(part) for part in parts))
            scheme, netloc, query, fragment = _first_of_each(
                schemes, netlocs, queries, fragments)
            path = '/'.join(x.strip('/') for x in paths if x)
            return urlunsplit((scheme, netloc, path, query, fragment))

        hrs_fnames = (self._utc_hrs.map(lambda x: x.
                                        strftime('%Y%m%d_%H%M.gz')).
                      astype(np.str))

        hrs_s = pd.Series(hrs_fnames, index=self._utc_hrs)
        hrs_grpd = hrs_s.groupby(
            [lambda x: x.year, lambda x: x.month, lambda x: x.day])

        print ("Starting MADIS downloads for %s to %s to local directory "
               "%s..." % (self._utc_hrs.min().strftime("%Y-%m-%d"),
                          self._utc_hrs.max().strftime("%Y-%m-%d"),
                          self.path_madis_data))

        def _get_madis_file_list(url, usrname, passwd):

            ntries = 0

            while 1:

                try:

                    buf = StringIO()

                    c = pycurl.Curl()
                    c.setopt(pycurl.WRITEDATA, buf)
                    c.setopt(pycurl.URL, url)
                    c.setopt(pycurl.USERPWD, "%s:%s" % (usrname, passwd))
                    c.setopt(pycurl.SSL_VERIFYPEER, 0)
                    c.setopt(pycurl.FAILONERROR, True)
                    c.setopt(pycurl.FOLLOWLOCATION, 1)
                    c.perform()
                    c.close()

                    break

                except pycurl.error as e:

                    error_code = e.args[0]

                    ntries += 1

                    # Only try again if below 5 tries
                    # and error code = 7 (connection time out)
                    if ntries == 5 or error_code != 7:

                        raise

                    sleep(5)

            doc = html_parse(StringIO(buf.getvalue()))

            fnames = np.array([elem.text_content() for elem in doc.xpath(
                '//tr//td[position() = 1]')][1:], dtype=np.str)

            return fnames

        def _wget_madis_file(url, path_local, usrname, passwd):

            print "WGET: Downloading %s to %s..." % (url, path_local)

            subprocess.call(['wget', '-N', '--no-check-certificate', '--quiet',
                             '--user=%s' % usrname, '--password=%s' % passwd,
                             url, '-P%s' % path_local])

        for a_dspath in self.madis_datasets:

            # Get list of realtime files for this dataset
            rt_fnames = _get_madis_file_list(_url_path_join(_URL_BASE_MADIS,
                                                            a_dspath),
                                             self._username, self._password)

            print "Processing downloads for %s" % a_dspath

            for a_day, day_fnames in hrs_grpd:

                yr, mth, day = a_day
                syr, smth, sday = "%d" % yr, "%.2d" % mth, "%.2d" % day
                url_archive_fnames = _url_path_join(
                    _URL_BASE_MADIS, 'archive', syr, smth, sday, a_dspath)

                try:

                    archive_fnames = _get_madis_file_list(url_archive_fnames,
                                                          self._username,
                                                          self._password)
                except pycurl.error as e:

                    # Raise if error code 7 (connection time out)
                    if e.args[0] == 7:
                        raise

                    print ("Warning: Received error trying to list files at "
                           "%s | %s" % (url_archive_fnames, str(e)))

                    archive_fnames = np.array([], dtype=np.str)

                avail_archive_fnames = (day_fnames
                                        [day_fnames.isin(archive_fnames)])

                avail_rt_fnames = day_fnames[
                    day_fnames.isin(rt_fnames) &
                    ~day_fnames.isin(archive_fnames)]

                n_avail = avail_archive_fnames.size + avail_rt_fnames.size

                print ("Found %d files out of %d possible for %s on date: "
                       "%s%s%s" % (n_avail, day_fnames.size, a_dspath, syr,
                                   smth, sday))
                print ("%d of these files are from the archive. %d are from "
                       "realtime." % (avail_archive_fnames.size,
                                      avail_rt_fnames.size))
                if n_avail > 0:

                    path_local = os.path.join(self.path_madis_data, syr, smth,
                                              sday, a_dspath)
                    _mkdir_p(path_local)

                    for a_fname in avail_archive_fnames:

                        url = _url_path_join(_URL_BASE_MADIS, 'archive', syr,
                                             smth, sday, a_dspath, a_fname)
                        _wget_madis_file(url, path_local, self._username,
                                         self._password)

                    for a_fname in avail_rt_fnames:

                        url = _url_path_join(
                            _URL_BASE_MADIS, a_dspath, a_fname)
                        _wget_madis_file(url, path_local, self._username,
                                         self._password)

    def _merge_with_stn_cache(self, stns_new):

        stns_new = _round_coords(stns_new)
        stns_cache = self._stns_cache

        if stns_cache.size != 0:

            stns_merge = stns_new.merge(stns_cache[_UNIQ_STN_COLUMNS +
                                                   ['uid', 'time_zone']],
                                        on=_UNIQ_STN_COLUMNS, how='left',
                                        sort=False)

        else:

            stns_merge = stns_new

        if not 'uid' in stns_merge.columns:
            stns_merge['uid'] = np.nan

        if not 'time_zone' in stns_merge.columns:
            stns_merge['time_zone'] = np.nan

        if stns_merge.shape[0] != stns_new.shape[0]:
            raise ValueError("Non-unique station location.")

        mask_no_uid = stns_merge.uid.isnull()

        if mask_no_uid.any():

            n_no_uid = mask_no_uid.sum()
            max_uid = -1 if stns_cache.size == 0 else stns_cache.uid.max()
            stns_merge.loc[mask_no_uid, 'uid'] = np.arange(max_uid + 1,
                                                           max_uid + 1 +
                                                           n_no_uid)

        mask_no_tz = stns_merge['time_zone'].isnull()

        if mask_no_tz.any():

            self._tz.set_tz(stns_merge)

        stns_merge['station_id'] = (stns_merge.station_id_orig + "_" +
                                    stns_merge.uid.apply(lambda x: '%d' % x))
        stns_merge = stns_merge.set_index('station_id', drop=False)
        return stns_merge

    def _update_stn_cache(self, stns):

        mask_exist = stns.index.isin(self._stns_cache.index)

        self._stns_cache = pd.concat([self._stns_cache,
                                      stns.drop(stns.index[mask_exist])])

        pd.to_pickle(self._stns_cache, self._fpath_stns_cache)
