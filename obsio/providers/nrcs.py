from ..util.humidity import convert_rh_to_tdew, convert_tdew_to_rh, \
    calc_pressure, convert_rh_to_vpd, convert_tdew_to_vpd
from .generic import ObsIO
from suds.client import Client
from time import sleep
import numpy as np
import pandas as pd

_URL_AWDB_WSDL = 'http://www.wcc.nrcs.usda.gov/awdbWebService/services?WSDL'

# Daily NRCS elements to download for each obsio element
_ELEMS_TO_NRCS_DAILY = {'tmin': ['TMIN'],
                        'tmax': ['TMAX'],
                        'prcp': ['PRCP'],
                        'snwd': ['SNWD'],
                        'swe': ['WTEQ'],
                        'tdew':[],
                        'rh':['RHUMV', 'RHUMN', 'RHUMX'],
                        'vpd':[],
                        'tdewmin':[],
                        'rhmin':['RHUMV', 'RHUMN', 'RHUMX'],
                        'vpdmin':[],
                        'tdewmax':[],
                        'rhmax':['RHUMV', 'RHUMN', 'RHUMX'],
                        'vpdmax':[]}

# Hourly NRCS elements to download for each obsio element
_ELEMS_TO_NRCS_HOURLY = {'tmin': [],
                         'tmax': [],
                         'prcp': [],
                         'snwd': [],
                         'swe': [],
                         'tdew':['DPTP', 'RHUMV', 'RHUM', 'TAVG', 'TOBS'],
                         'rh':['RHUMN', 'RHUMX', 'RHUMV', 'RHUM', 'DPTP', 'TAVG', 'TOBS'],
                         'vpd':['PVPV', 'SVPV', 'RHUMV', 'RHUM', 'DPTP', 'TAVG', 'TOBS'],
                         'tdewmin':['DPTP', 'RHUMV', 'RHUM', 'TAVG', 'TOBS'],
                         'rhmin':['RHUMN', 'RHUMX', 'RHUMV', 'RHUM', 'DPTP', 'TAVG', 'TOBS'],
                         'vpdmin':['DPTP', 'RHUMV', 'RHUM', 'TAVG', 'TOBS'],
                         'tdewmax':['DPTP', 'RHUMV', 'RHUM', 'TAVG', 'TOBS'],
                         'rhmax':['RHUMX', 'RHUMN', 'RHUMV', 'RHUM', 'DPTP', 'TAVG', 'TOBS'],
                         'vpdmax':['DPTP', 'RHUMV', 'RHUM', 'TAVG', 'TOBS']}

def _get_daily_nrcs_elems(elems):
    
    return list(np.unique(np.concatenate([_ELEMS_TO_NRCS_DAILY[a_elem] for
                                          a_elem in elems])))

def _get_hourly_nrcs_elems(elems):
    
    return list(np.unique(np.concatenate([_ELEMS_TO_NRCS_HOURLY[a_elem] for
                                          a_elem in elems])))

# Unit conversion functions
_f_to_c = lambda f: (f - 32.0) / 1.8
_in_to_mm = lambda i: i * 25.4
_ft_to_m = lambda ft: ft * 0.3048
_no_conv = lambda x: x
_kpa_to_pa = lambda kpa : kpa * 1000

_convert_funcs = {'TMIN': _f_to_c, 'TMAX': _f_to_c, 'PRCP': _in_to_mm,
                  'SNWD': _in_to_mm, 'WTEQ': _in_to_mm, 'DPTP':_f_to_c,
                  'RHUMV':_no_conv, 'RHUM':_no_conv, 'PVPV':_kpa_to_pa,
                  'SVPV':_kpa_to_pa, 'RHUMN':_no_conv, 'RHUMX':_no_conv,
                  'TAVG':_f_to_c, 'TOBS':_f_to_c}

# Element extraction functions
def _extract_tmin(obs_dly, obs_hrly, min_hrly_for_dly, stns):
    
    tmin = obs_dly[['time', 'stationTriplet', 'TMIN']].copy()
    tmin.dropna(subset=['TMIN'], inplace=True)
    tmin.rename(columns={'TMIN':'obs_value'}, inplace=True)
    tmin['elem'] = 'tmin'
    return tmin

def _extract_tmax(obs_dly, obs_hrly, min_hrly_for_dly, stns):
    
    tmax = obs_dly[['time', 'stationTriplet', 'TMAX']].copy()
    tmax.dropna(subset=['TMAX'], inplace=True)
    tmax.rename(columns={'TMAX':'obs_value'}, inplace=True)
    tmax['elem'] = 'tmax'
    return tmax

def _extract_prcp(obs_dly, obs_hrly, min_hrly_for_dly, stns):
    
    prcp = obs_dly[['time', 'stationTriplet', 'PRCP']].copy()
    prcp.dropna(subset=['PRCP'], inplace=True)
    prcp.rename(columns={'PRCP':'obs_value'}, inplace=True)
    prcp['elem'] = 'prcp'
    return prcp

def _extract_snwd(obs_dly, obs_hrly, min_hrly_for_dly, stns):
    
    snwd = obs_dly[['time', 'stationTriplet', 'SNWD']].copy()
    snwd.dropna(subset=['SNWD'], inplace=True)
    snwd.rename(columns={'SNWD':'obs_value'}, inplace=True)
    snwd['elem'] = 'snwd'
    return snwd

def _extract_swe(obs_dly, obs_hrly, min_hrly_for_dly, stns):
    
    swe = obs_dly[['time', 'stationTriplet', 'WTEQ']].copy()
    swe.dropna(subset=['WTEQ'], inplace=True)
    swe.rename(columns={'WTEQ':'obs_value'}, inplace=True)
    swe['elem'] = 'swe'
    return swe
    
def _extract_tdew(obs_dly, obs_hrly, min_hrly_for_dly, stns):
    
    obs_hrly['DPTP-RHUMV_TAVG'] = convert_rh_to_tdew(obs_hrly.RHUMV,
                                                     obs_hrly.TAVG)
    obs_hrly['DPTP-RHUM_TOBS'] = convert_rh_to_tdew(obs_hrly.RHUM,
                                                    obs_hrly.TOBS)
    obs_hrly['DPTP-RHUMV_TOBS'] = convert_rh_to_tdew(obs_hrly.RHUMV,
                                                     obs_hrly.TOBS)
    obs_hrly['DPTP-RHUM_TAVG'] = convert_rh_to_tdew(obs_hrly.RHUM,
                                                    obs_hrly.TAVG)
    
    obs_hrly_grp = obs_hrly.groupby([obs_hrly.time.dt.year,
                                     obs_hrly.time.dt.month,
                                     obs_hrly.time.dt.day,
                                     obs_hrly.stationTriplet])
    
    obs_dlyr = obs_hrly_grp[['DPTP',
                             'DPTP-RHUMV_TAVG',
                             'DPTP-RHUM_TOBS',
                             'DPTP-RHUMV_TOBS',
                             'DPTP-RHUM_TAVG']].agg(['mean', 'min',
                                                     'max', 'count'])
    
    for a_tdew in ['DPTP', 'DPTP-RHUMV_TAVG', 'DPTP-RHUM_TOBS', 'DPTP-RHUMV_TOBS',
                   'DPTP-RHUM_TAVG']:
        
        mask_cnt_avg = obs_dlyr[a_tdew]['count'] < min_hrly_for_dly['tdew']
        mask_cnt_min = obs_dlyr[a_tdew]['count'] < min_hrly_for_dly['tdewmin']
        mask_cnt_max = obs_dlyr[a_tdew]['count'] < min_hrly_for_dly['tdewmax']
        
        obs_dlyr.loc[mask_cnt_avg, (a_tdew, 'mean')] = np.nan
        obs_dlyr.loc[mask_cnt_min, (a_tdew, 'min')] = np.nan
        obs_dlyr.loc[mask_cnt_max, (a_tdew, 'max')] = np.nan
    
    tdew = obs_dlyr[[('DPTP', 'mean'), ('DPTP', 'min'), ('DPTP', 'max')]].copy()
    
    for a_agg in ['mean', 'min', 'max']:
        
        tdew[('DPTP', a_agg)].fillna(obs_dlyr[('DPTP-RHUMV_TAVG', a_agg)],
                                    inplace=True)
        tdew[('DPTP', a_agg)].fillna(obs_dlyr[('DPTP-RHUM_TOBS', a_agg)],
                                    inplace=True)
        tdew[('DPTP', a_agg)].fillna(obs_dlyr[('DPTP-RHUMV_TOBS', a_agg)],
                                    inplace=True)
        tdew[('DPTP', a_agg)].fillna(obs_dlyr[('DPTP-RHUM_TAVG', a_agg)],
                                    inplace=True)
        
    tdew.columns = ["_".join(a_col) for a_col in tdew.columns.values]
    tdew.rename(columns={'DPTP_mean':'tdew',
                         'DPTP_min':'tdewmin',
                         'DPTP_max':'tdewmax'}, inplace=True)
    tdew.index.names = ['year', 'month', 'day', 'stationTriplet']
    tdew.reset_index(inplace=True)
    
    y = np.array(tdew.year - 1970, dtype='<M8[Y]')
    m = np.array(tdew.month - 1, dtype='<m8[M]')
    d = np.array(tdew.day - 1, dtype='<m8[D]')
    
    tdew['time'] = pd.to_datetime(y + m + d)
    tdew.drop(['year', 'month', 'day'], axis=1, inplace=True)
    tdew.set_index(['time', 'stationTriplet'], inplace=True)
    tdew = tdew.stack().reset_index()
    tdew.rename(columns={'level_2':'elem', 0:'obs_value'}, inplace=True)
    
    return tdew

def _extract_rh(obs_dly, obs_hrly, min_hrly_for_dly, stns):
    
    obs_hrly = pd.merge(obs_hrly, stns[['pressure', 'stationTriplet']],
                        how='left', on='stationTriplet')
    
    obs_hrly['RH-DPTP_TAVG'] = convert_tdew_to_rh(obs_hrly.DPTP,
                                                  obs_hrly.TAVG,
                                                  obs_hrly.pressure)
    obs_hrly['RH-DPTP_TOBS'] = convert_tdew_to_rh(obs_hrly.DPTP,
                                                  obs_hrly.TOBS,
                                                  obs_hrly.pressure)
    
    obs_hrly_grp = obs_hrly.groupby([obs_hrly.time.dt.year,
                                     obs_hrly.time.dt.month,
                                     obs_hrly.time.dt.day,
                                     obs_hrly.stationTriplet])
        
    obs_dlyr = obs_hrly_grp[['RHUM',
                             'RHUMV',
                             'RHUMN',
                             'RHUMX',
                             'RH-DPTP_TAVG',
                             'RH-DPTP_TOBS']].agg(['mean', 'min',
                                                   'max', 'count'])
    
    for a_rh in ['RHUM', 'RHUMV', 'RHUMN', 'RHUMX', 'RH-DPTP_TAVG',
                 'RH-DPTP_TOBS']:
        
        mask_cnt_avg = obs_dlyr[a_rh]['count'] < min_hrly_for_dly['rh']
        mask_cnt_min = obs_dlyr[a_rh]['count'] < min_hrly_for_dly['rhmin']
        mask_cnt_max = obs_dlyr[a_rh]['count'] < min_hrly_for_dly['rhmax']
        
        obs_dlyr.loc[mask_cnt_avg, (a_rh, 'mean')] = np.nan
        obs_dlyr.loc[mask_cnt_min, (a_rh, 'min')] = np.nan
        obs_dlyr.loc[mask_cnt_max, (a_rh, 'max')] = np.nan
    
    rh = obs_dly[['time', 'stationTriplet', 'RHUMV', 'RHUMN', 'RHUMX']].copy()
    rh.rename(columns={'RHUMV':'rh', 'RHUMN':'rhmin', 'RHUMX':'rhmax'},
              inplace=True)
    rh.set_index(['time', 'stationTriplet'], inplace=True)
    
    obs_dlyr.columns = ["_".join(a_col) for a_col in obs_dlyr.columns.values]
    obs_dlyr.index.names = ['year', 'month', 'day', 'stationTriplet']
    obs_dlyr.reset_index(inplace=True)
    
    y = np.array(obs_dlyr.year - 1970, dtype='<M8[Y]')
    m = np.array(obs_dlyr.month - 1, dtype='<m8[M]')
    d = np.array(obs_dlyr.day - 1, dtype='<m8[D]')
    obs_dlyr['time'] = pd.to_datetime(y + m + d)
    obs_dlyr.drop(['year', 'month', 'day'], axis=1, inplace=True)
    obs_dlyr.set_index(['time', 'stationTriplet'], inplace=True)
    
    rh = rh.join(obs_dlyr, how='outer')
    
    rh['rh'].fillna(rh.RHUMV_mean, inplace=True)
    rh['rh'].fillna(rh.RHUM_mean, inplace=True)
    rh['rh'].fillna(rh['RH-DPTP_TOBS_mean'], inplace=True)
    rh['rh'].fillna(rh['RH-DPTP_TAVG_mean'], inplace=True)
    
    rh['rhmin'].fillna(rh.RHUMN_min, inplace=True)
    rh['rhmin'].fillna(rh.RHUMV_min, inplace=True)
    rh['rhmin'].fillna(rh.RHUM_min, inplace=True)
    rh['rhmin'].fillna(rh['RH-DPTP_TOBS_min'], inplace=True)
    rh['rhmin'].fillna(rh['RH-DPTP_TAVG_min'], inplace=True)
    
    rh['rhmax'].fillna(rh.RHUMN_max, inplace=True)
    rh['rhmax'].fillna(rh.RHUMV_max, inplace=True)
    rh['rhmax'].fillna(rh.RHUM_max, inplace=True)
    rh['rhmax'].fillna(rh['RH-DPTP_TOBS_max'], inplace=True)
    rh['rhmax'].fillna(rh['RH-DPTP_TAVG_max'], inplace=True)
    
    rh.drop(rh.columns[~rh.columns.isin(['rh', 'rhmin', 'rhmax'])],
            axis=1, inplace=True)
    rh = rh.stack().reset_index()
    rh.rename(columns={'level_2':'elem', 0:'obs_value'}, inplace=True)
         
    return rh

def _extract_vpd(obs_dly, obs_hrly, min_hrly_for_dly, stns):
    
    obs_hrly = pd.merge(obs_hrly, stns[['pressure', 'stationTriplet']],
                        how='left', on='stationTriplet')
        
        
    obs_hrly['VPD'] = obs_hrly.SVPV - obs_hrly.PVPV
    obs_hrly['VPD-RHUMV_TAVG'] = convert_rh_to_vpd(obs_hrly.RHUMV,
                                                   obs_hrly.TAVG,
                                                   obs_hrly.pressure)
    obs_hrly['VPD-RHUM_TOBS'] = convert_rh_to_vpd(obs_hrly.RHUM,
                                                   obs_hrly.TOBS,
                                                   obs_hrly.pressure)    
    obs_hrly['VPD-RHUMV_TOBS'] = convert_rh_to_vpd(obs_hrly.RHUMV,
                                                   obs_hrly.TOBS,
                                                   obs_hrly.pressure)
    obs_hrly['VPD-RHUM_TAVG'] = convert_rh_to_vpd(obs_hrly.RHUM,
                                                  obs_hrly.TAVG,
                                                  obs_hrly.pressure)
    obs_hrly['VPD-DPTP_TOBS'] = convert_tdew_to_vpd(obs_hrly.DPTP,
                                                    obs_hrly.TOBS,
                                                    obs_hrly.pressure)
    obs_hrly['VPD-DPTP_TAVG'] = convert_tdew_to_vpd(obs_hrly.DPTP,
                                                    obs_hrly.TAVG,
                                                    obs_hrly.pressure)
    
    obs_hrly_grp = obs_hrly.groupby([obs_hrly.time.dt.year,
                                     obs_hrly.time.dt.month,
                                     obs_hrly.time.dt.day,
                                     obs_hrly.stationTriplet])
    
    obs_dlyr = obs_hrly_grp[['VPD',
                             'VPD-RHUMV_TAVG',
                             'VPD-RHUM_TOBS',
                             'VPD-RHUMV_TOBS',
                             'VPD-RHUM_TAVG',
                             'VPD-DPTP_TOBS',
                             'VPD-DPTP_TAVG']].agg(['mean', 'min',
                                                    'max', 'count'])
    
    for a_vpd in ['VPD', 'VPD-RHUMV_TAVG', 'VPD-RHUM_TOBS',
                  'VPD-RHUMV_TOBS', 'VPD-RHUM_TAVG', 'VPD-DPTP_TOBS',
                  'VPD-DPTP_TAVG']:
        
        mask_cnt_avg = obs_dlyr[a_vpd]['count'] < min_hrly_for_dly['vpd']
        mask_cnt_min = obs_dlyr[a_vpd]['count'] < min_hrly_for_dly['vpdmin']
        mask_cnt_max = obs_dlyr[a_vpd]['count'] < min_hrly_for_dly['vpdmax']
        
        obs_dlyr.loc[mask_cnt_avg, (a_vpd, 'mean')] = np.nan
        obs_dlyr.loc[mask_cnt_min, (a_vpd, 'min')] = np.nan
        obs_dlyr.loc[mask_cnt_max, (a_vpd, 'max')] = np.nan
    
    vpd = obs_dlyr[[('VPD', 'mean'), ('VPD', 'min'), ('VPD', 'max')]].copy()
    
    for a_agg in ['mean', 'min', 'max']:
        
        vpd[('VPD', a_agg)].fillna(obs_dlyr[('VPD-RHUMV_TAVG', a_agg)], inplace=True)
        vpd[('VPD', a_agg)].fillna(obs_dlyr[('VPD-RHUM_TOBS', a_agg)], inplace=True)
        vpd[('VPD', a_agg)].fillna(obs_dlyr[('VPD-RHUMV_TOBS', a_agg)], inplace=True)
        vpd[('VPD', a_agg)].fillna(obs_dlyr[('VPD-RHUM_TAVG', a_agg)], inplace=True)
        vpd[('VPD', a_agg)].fillna(obs_dlyr[('VPD-DPTP_TOBS', a_agg)], inplace=True)
        vpd[('VPD', a_agg)].fillna(obs_dlyr[('VPD-DPTP_TAVG', a_agg)], inplace=True)
        
    vpd.columns = ["_".join(a_col) for a_col in vpd.columns.values]
    vpd.rename(columns={'VPD_mean':'vpd',
                        'VPD_min':'vpdmin',
                        'VPD_max':'vpdmax'}, inplace=True)
    vpd.index.names = ['year', 'month', 'day', 'stationTriplet']
    vpd.reset_index(inplace=True)
    
    y = np.array(vpd.year - 1970, dtype='<M8[Y]')
    m = np.array(vpd.month - 1, dtype='<m8[M]')
    d = np.array(vpd.day - 1, dtype='<m8[D]')
    
    vpd['time'] = pd.to_datetime(y + m + d)
    vpd.drop(['year', 'month', 'day'], axis=1, inplace=True)
    vpd.set_index(['time', 'stationTriplet'], inplace=True)
    vpd = vpd.stack().reset_index()
    vpd.rename(columns={'level_2':'elem', 0:'obs_value'}, inplace=True)
    
    return vpd

_ELEM_EXTRACT_FUNCS = {'tmin': _extract_tmin,
                       'tmax': _extract_tmax,
                       'prcp': _extract_prcp,
                       'snwd': _extract_snwd,
                       'swe': _extract_swe,
                       'tdew':_extract_tdew,
                       'rh':_extract_rh,
                       'vpd':_extract_vpd,
                       'tdewmin':_extract_tdew,
                       'rhmin':_extract_rh,
                       'vpdmin':_extract_vpd,
                       'tdewmax':_extract_tdew,
                       'rhmax':_extract_rh,
                       'vpdmax':_extract_vpd} 
    
def _execute_awdb_call(a_func, ntries_max=3, sleep_sec=5, **kwargs):

    ntries = 0

    while 1:

        try:

            a_result = a_func(**kwargs)
            break

        except Exception as e:

            ntries += 1

            if ntries == ntries_max:

                raise

            else:

                print ("WARNING: Received error executing AWDB function %s:"
                       " %s. Sleeping %d seconds and trying again." % 
                       (str(a_func.method.name), str(e), sleep_sec))

                sleep(sleep_sec)

    return a_result


class _NrcsHourly():

    def __init__(self, awdb_client, nrcs_elems):

        self.nrcs_elems = nrcs_elems
        self._client = awdb_client

    def read_obs(self, start_date, end_date, stns):

        begin = start_date.strftime("%Y-%m-%d")
        end = (end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        obs_all = []

        for a_elem in self.nrcs_elems:
                        
            datas = _execute_awdb_call(self._client.service.getHourlyData,
                                       stationTriplets=list(stns.stationTriplet),
                                       elementCd=[a_elem],
                                       ordinal=1,
                                       beginDate=begin,
                                       endDate=end)

            obs_elem = []

            for a_data in datas:

                try:
                    
                    obs_stn = pd.DataFrame([(a_val.dateTime, a_val.value)
                                            for a_val in a_data.values],
                                       columns=['time', 'obs_value'])
                    obs_stn['time'] = pd.to_datetime(obs_stn.time)
                    obs_stn['stationTriplet'] = a_data.stationTriplet
                    obs_stn['obs_value'] = _convert_funcs[a_elem](obs_stn['obs_value'])
                    obs_stn['elem'] = a_elem
                                       
                    obs_elem.append(obs_stn)
                    
                except AttributeError:
                    # No observations for station
                    continue
            
            try:
                
                obs_elem = pd.concat(obs_elem, axis=0, ignore_index=True)
            
            except ValueError:
            
                continue  # no observations for elem
            
            obs_all.append(obs_elem)
        
        try:
            
            obs_all = pd.concat(obs_all, axis=0, ignore_index=True)
            
            # Remove possible duplicate entries
            obs_all.drop_duplicates(['time', 'stationTriplet', 'elem'],
                                    keep='first', inplace=True)
            
            obs_all.set_index(['time', 'stationTriplet', 'elem'], inplace=True)
            obs_all = obs_all.unstack(level=2)
            obs_all.columns = [a_col[1] for a_col in obs_all.columns.values]
            obs_all.reset_index(inplace=True)
            
            nrcs_elems = np.array(self.nrcs_elems)
            elems_miss = nrcs_elems[~np.in1d(nrcs_elems,
                                             obs_all.columns.values)]
            
            for a_elem in elems_miss:
                obs_all[a_elem] = np.nan
        
        except ValueError as e:
            
            if e.args[0] == 'No objects to concatenate':
                
                obs_all = pd.DataFrame(None, columns=['time', 'stationTriplet'] + 
                                       self.nrcs_elems)
            
            else:
                
                raise
            
        
        return obs_all

class _NrcsDaily():

    def __init__(self, awdb_client, nrcs_elems):

        self.nrcs_elems = nrcs_elems
        self._client = awdb_client

    def read_obs(self, start_date, end_date, stns):

        begin = start_date.strftime("%Y-%m-%d")
        end = (end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        obs_all = []

        for a_elem in self.nrcs_elems:

            datas = _execute_awdb_call(self._client.service.getData,
                                       stationTriplets=list(stns.
                                                            stationTriplet),
                                       elementCd=[a_elem],
                                       ordinal=1,
                                       duration='DAILY',
                                       getFlags=False,
                                       beginDate=begin,
                                       endDate=end,
                                       alwaysReturnDailyFeb29=False)

            obs_elem = []

            for a_data in datas:

                try:
                    obs_stn = pd.DataFrame(a_data.values, columns=['obs_value'])
                    obs_stn['time'] = pd.date_range(a_data.beginDate, a_data.endDate)
                    obs_stn['stationTriplet'] = a_data.stationTriplet
                    obs_stn['obs_value'] = _convert_funcs[a_elem](obs_stn['obs_value'])
                    obs_stn['elem'] = a_elem
                    obs_elem.append(obs_stn)
                except AttributeError:
                    continue

            try:
            
                obs_elem = pd.concat(obs_elem, axis=0, ignore_index=True)
            
            except ValueError:
                continue  # no observations for elem

            obs_all.append(obs_elem)
        
        try:
        
            obs_all = pd.concat(obs_all, axis=0, ignore_index=True)
            obs_all.set_index(['time', 'stationTriplet', 'elem'], inplace=True)
            obs_all = obs_all.unstack(level=2)
            obs_all.columns = [a_col[1] for a_col in obs_all.columns.values]
            obs_all.reset_index(inplace=True)
                        
            nrcs_elems = np.array(self.nrcs_elems)
            elems_miss = nrcs_elems[~np.in1d(nrcs_elems, obs_all.columns.values)]
            
            for a_elem in elems_miss:
                obs_all[a_elem] = np.nan
        
        except ValueError as e:
            
            if e.args[0] == 'No objects to concatenate':
                
                obs_all = pd.DataFrame(None, columns=['time', 'stationTriplet'] + 
                                       self.nrcs_elems)
            
            else:
                
                raise
            

        return obs_all

class NrcsObsIO(ObsIO):

    _avail_elems = ['tmin', 'tmax', 'prcp', 'snwd', 'swe', 'tdew', 'tdewmin',
                    'tdewmax', 'vpd', 'vpdmin', 'vpdmax', 'rh', 'rhmin', 'rhmax']
    _requires_local = False
    
    _MIN_HRLY_FOR_DLY_DFLT = {'tdew': 4, 'tdewmin': 18, 'tdewmax': 18,
                              'vpd':18, 'vpdmin':18, 'vpdmax':18,
                              'rh':18, 'rhmin':18, 'rhmax':18}

    def __init__(self, min_hrly_for_dly=None, **kwargs):

        super(NrcsObsIO, self).__init__(**kwargs)
        
        self.min_hrly_for_dly = (min_hrly_for_dly if min_hrly_for_dly
                                 else self._MIN_HRLY_FOR_DLY_DFLT)
        
        # check to make sure there is an entry in min_hrly_for_dly for each
        # elem
        for a_elem in self.elems:

            try:

                self.min_hrly_for_dly[a_elem]

            except KeyError:

                try:
                    
                    self.min_hrly_for_dly[a_elem] = self._MIN_HRLY_FOR_DLY_DFLT[a_elem]
                
                except KeyError:
                    
                    continue
                
        self._elems_nrcs_dly = _get_daily_nrcs_elems(self.elems)
        self._elems_nrcs_hrly = _get_hourly_nrcs_elems(self.elems)
        self._elems_nrcs_all = list(np.unique(self._elems_nrcs_hrly + 
                                              self._elems_nrcs_dly))
        
        self._client = Client(_URL_AWDB_WSDL)
        self._stnmeta_attrs = (self._client.factory.
                               create('stationMetaData').__keylist__)
        
        self._nrcs_dly = _NrcsDaily(self._client, self._elems_nrcs_dly)
        self._nrcs_hrly = _NrcsHourly(self._client, self._elems_nrcs_hrly)
        
        self._elem_funcs = np.unique(np.array([_ELEM_EXTRACT_FUNCS[a_elem] for
                                               a_elem in self.elems]))

    def _read_stns(self):

        if self.bbox is None:

            stn_triplets = _execute_awdb_call(self._client.service.getStations,
                                              logicalAnd=True,
                                              networkCds=['SNTL', 'SCAN'],
                                              elementCds=self._elems_nrcs_all)

        else:

            stn_triplets = _execute_awdb_call(self._client.service.getStations,
                                              logicalAnd=True,
                                              minLatitude=self.bbox.south,
                                              maxLatitude=self.bbox.north,
                                              minLongitude=self.bbox.west,
                                              maxLongitude=self.bbox.east,
                                              networkCds=['SNTL', 'SCAN'],
                                              elementCds=self._elems_nrcs_all)

        print "NrcsObsIO: Getting station metadata..."
        stn_metas = _execute_awdb_call(self._client.service.
                                       getStationMetadataMultiple,
                                       stationTriplets=stn_triplets)

        stn_tups = [self._stationMetadata_to_tuple(a) for a in stn_metas]
        df_stns = pd.DataFrame(stn_tups, columns=self._stnmeta_attrs)

        stns = df_stns.rename(columns={'actonId': 'station_id',
                                       'name': 'station_name'})
        stns['station_id'] = stns.station_id.fillna(stns.shefId)
        stns = stns[~stns.station_id.isnull()]
        stns['beginDate'] = pd.to_datetime(stns.beginDate)
        stns['endDate'] = pd.to_datetime(stns.endDate)
        stns['elevation'] = _ft_to_m(stns.elevation)
        stns['provider'] = 'NRCS'
        stns['sub_provider'] = ''
        stns = stns.sort_values('station_id')

        if self.has_start_end_dates:

            mask_dates = (((self.start_date <= stns.beginDate) & 
                           (stns.beginDate <= self.end_date)) | 
                          ((stns.beginDate <= self.start_date) & 
                           (self.start_date <= stns.endDate)))

            stns = stns[mask_dates].copy()

        stns = stns.reset_index(drop=True)
        stns = stns.set_index('station_id', drop=False)
        stns['pressure'] = calc_pressure(stns.elevation)
        
        return stns

    def _read_obs(self, stns_ids=None):

        if stns_ids is None:
            stns_obs = self.stns
        else:
            stns_obs = self.stns.loc[stns_ids]

        if self.has_start_end_dates:

            start_date_obs = self.start_date
            end_date_obs = self.end_date

        else:

            start_date_obs = stns_obs.beginDate.min()
            end_date_obs = stns_obs.endDate.max()
             
        obs_dly = self._nrcs_dly.read_obs(start_date_obs, end_date_obs, stns_obs)
        obs_hrly = self._nrcs_hrly.read_obs(start_date_obs, end_date_obs, stns_obs)
        
        def check_empty(df_obs):
            
            if df_obs.empty:
                
                df_obs = pd.DataFrame(np.nan, index=np.arange(len(stns_obs)),
                                      columns=df_obs.columns)
                df_obs.time = pd.Timestamp.now()
                df_obs.stationTriplet = stns_obs.stationTriplet.values
            
            return df_obs
        
        # So aggregation functions don't fail, 
        # make sure observation dataframes aren't empty and at least have rows
        # of nan values
        obs_dly = check_empty(obs_dly)
        obs_hrly = check_empty(obs_hrly)
        
        obs = [a_func(obs_dly, obs_hrly, self.min_hrly_for_dly, stns_obs) for
               a_func in self._elem_funcs]
        obs = pd.concat(obs, axis=0, ignore_index=True)
        
        # Drop elems that were not requested
        obs.drop(obs.index[~(obs.elem.isin(self.elems))], axis=0, inplace=True)
        
        # Replace stationTriplet with station_id
        obs_merge = pd.merge(obs, self.stns[['station_id', 'stationTriplet']],
                             how='left', on='stationTriplet', sort=False)

        if obs_merge.shape[0] != obs.shape[0]:
            raise ValueError("Non-unique station ids.")
        if obs_merge.station_id.isnull().any():
            raise ValueError("stationTriplet without a station_id")

        obs_merge = obs_merge.drop('stationTriplet', axis=1)
        obs_merge = obs_merge.set_index(['station_id', 'elem', 'time'])
        obs_merge = obs_merge.sortlevel(0, sort_remaining=True)
        return obs_merge

    def _stationMetadata_to_tuple(self, a_meta):

        list_meta = [None] * len(self._stnmeta_attrs)

        for i, a_attr in enumerate(self._stnmeta_attrs):

            try:
                list_meta[i] = a_meta[a_attr]
            except AttributeError:
                # Doesn't have attribute
                continue

        return tuple(list_meta)

    def download_local(self):
        raise NotImplementedError("NrcsObsIO does not store any local data.")
