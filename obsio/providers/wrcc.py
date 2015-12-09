from .generic import ObsIO
from StringIO import StringIO
from calendar import monthrange
from datetime import datetime
from multiprocessing import Pool
from obsio.util.humidity import convert_rh_to_tdew, calc_pressure, \
    convert_rh_to_vpd, convert_rh_to_vpd_daily
import numpy as np
import os
import pandas as pd
import sys
import urllib
import urllib2

_URL_RAWS_DLY_TIME_SERIES = "http://www.raws.dri.edu/cgi-bin/wea_dysimts.pl?"
_URL_RAWS_DLY_TIME_SERIES2 = 'http://www.raws.dri.edu/cgi-bin/wea_dysimts2.pl'
_URL_RAWS_HRLY_TIME_SERIES = "http://www.raws.dri.edu/cgi-bin/wea_list2.pl"
_URL_RAWS_STN_METADATA = "http://www.raws.dri.edu/cgi-bin/wea_info.pl?"

_HRLY_ELEMS = np.array(['tdew', 'tdewmin', 'tdewmax',
                        'vpd', 'vpdmin', 'vpdmax'])

_f_to_c = lambda f: (f - 32.0) / 1.8

def _parse_raws_metadata(stn_id):
        
    def parse_decdegrees(a_str):
    
        vals = a_str.split("&")
    
        deg = float(vals[0])
    
        vals = vals[1].split()
    
        minute = float(vals[1][0:-1])
        sec = float(vals[2])
    
        return dms2decimal(deg, minute, sec)
    
    def dms2decimal(degrees, minutes, seconds):
        
        decimal = 0.0
        if (degrees >= 0):
            decimal = degrees + float(minutes) / 60 + float(seconds) / 3600
        else:
            decimal = degrees - float(minutes) / 60 - float(seconds) / 3600
            
        return decimal
    
    try:
        stn_id = stn_id.strip()

        response = urllib2.urlopen("".join([_URL_RAWS_DLY_TIME_SERIES,
                                            stn_id]))
        plines = response.readlines()

        start_date = None
        end_date = None
        stn_name = None
        read_stn_name = False

        for pline in plines:

            if start_date is None or end_date is None or stn_name is None:

                if read_stn_name:
                    stn_name = pline.split(">")[1][0:-4]
                    read_stn_name = False

                if "Station:" in pline:
                    read_stn_name = True

                if "Earliest available data" in pline:
                    start_date = datetime.strptime(pline.split(":")[1][1:-2],
                                                   "%B %Y")

                if "Latest available data" in pline:
                    end_date = datetime.strptime(pline.split(":")[1][1:-2],
                                                 "%B %Y")
                    # set to last day of month
                    end_date = end_date.replace(day=monthrange(end_date.year,
                                                               end_date.month)[1])

        response = urllib2.urlopen("".join([_URL_RAWS_STN_METADATA, stn_id]))
        plines = response.readlines()
        lon, lat, elev = [None] * 3

        for x in np.arange(len(plines)):

            pline = plines[x]

            if "Latitude" in pline:
                lat = parse_decdegrees(plines[x + 2])

            if "Longitude" in pline:
                lon = parse_decdegrees(plines[x + 2])

            if "Elevation" in pline:
                elev = float(plines[x + 2].split()[0]) * 0.3048  # convert from feet to meters
        
        stn_meta = (stn_id, stn_name, lon, lat, elev, start_date, end_date)
                
    except:
        
        print "COULD NOT LOAD METADATA FOR: ", stn_id
        stn_meta = tuple([stn_id] + [None] * 6)
        
    return stn_meta

def _build_raws_stn_metadata(nprocs):
    '''Retrieve and build RAWS metadata
    
    Used to update the locally stored RAWS metadata data csv (raws_stns.csv)
    '''
    
    path_root = os.path.dirname(__file__)

    # raws_stnlist_pages.txt has URLs for HTML files that list RAWS stations
    afile = open(os.path.join(path_root, 'data', 'raws_stnlst_pages.txt'))
    stn_ids = []

    for line in afile.readlines():

        req = urllib2.Request(line.strip())
        response = urllib2.urlopen(req)
        plines = response.readlines()
        for pline in plines:

            if "rawMAIN.pl" in pline:

                stn_id = pline.split("?")[1][0:6]
                stn_ids.append(stn_id)

    stn_ids = np.unique(stn_ids)
        
    # http://stackoverflow.com/questions/24171725/
    # scikit-learn-multicore-attributeerror-stdin-instance-
    # has-no-attribute-close
    if not hasattr(sys.stdin, 'close'):
        
        def dummy_close():
            pass
        sys.stdin.close = dummy_close
    
    pool = Pool(processes=nprocs)

    stn_meta = pool.map(_parse_raws_metadata, list(stn_ids), chunksize=5)
    
    pool.close()
    pool.join()
    
    stn_meta = pd.DataFrame(stn_meta, columns=['station_id', 'station_name',
                                               'longitude', 'latitude',
                                               'elevation', 'start_date',
                                               'end_date'])
    
    stn_meta.dropna(axis=0, how='any', inplace=True)
    stn_meta.reset_index(inplace=True, drop=True)
    stn_meta['longitude'] = -stn_meta.longitude
    
    return stn_meta

def _parse_raws_hrly_webform(stn_id, stn_pres, start_date, end_date, pwd,
                             min_hrly_for_dly):
    
    values = {'smon': start_date.strftime("%m"),
              'sday': start_date.strftime("%d"),
              'syea': start_date.strftime("%y"),
              'emon': end_date.strftime("%m"),
              'eday': end_date.strftime("%d"),
              'eyea': end_date.strftime("%y"),
              'secret' : pwd,
              'pcodes' : ["AVA", "AVR"],
              'lim': "Y",
              'dfor':'01',
              'srce':"W",
              'miss': '08',  # -9999
              'flag':'N',
              'Dfmt':'02',
              'Tfmt':'01',
              'Head':'01',
              'Deli':'01',
              # english units. Hourly form always returns English even if metric
              # is selected. Set to english to be safe and perform conversion
              # locally
              'unit': 'E',
              "Submit Info": "Submit Info",
              'stn': stn_id,
              'WsMon': '01',
              'WsDay': '01',
              'WeMon': '12',
              'WeDay': '31',
              'WsHou': '00',
              'WeHou': '24'}
    
    data = urllib.urlencode(values, doseq=True)
    req = urllib2.Request(_URL_RAWS_HRLY_TIME_SERIES, data)
    response = urllib2.urlopen(req)
    lines = response.readlines()
    
    obs = pd.read_csv(StringIO("".join(lines[16:-14])), delim_whitespace=True)
    
    obs.rename(columns={':YYYYMMDDhhmm':'time', 'Temp':'tair', 'Humidty':'rh'},
               inplace=True)
    obs.replace(-9999, np.nan, inplace=True)
    
    obs['time'] = pd.to_datetime(obs.time, format="%Y%m%d%H%M", errors='coerce')
    
    if obs.time.isnull().any():
        
        # one or more dates had an incorrect format
        # remove observations that have incorrect date format
        obs.dropna(axis=0, how='all', subset=['time'], inplace=True)
    
    obs.set_index('time', inplace=True)
    
    obs['tair'] = _f_to_c(obs.tair)
    
    obs['tdew'] = convert_rh_to_tdew(obs.rh, obs.tair)
    obs['vpd'] = convert_rh_to_vpd(obs.rh, obs.tair, stn_pres)
            
    obs_dly = obs[['tdew', 'vpd']].resample('D',
                                            how=['mean', 'min', 'max', 'count'])
    
    # http://stackoverflow.com/questions/14507794/
    # python-pandas-how-to-flatten-a-hierarchical-index-in-columns
    obs_dly.columns = ['_'.join(col) for col in obs_dly.columns.values]
    
    obs_dly.rename(columns={'tdew_mean':'tdew', 'tdew_min':'tdewmin',
                            'tdew_max':'tdewmax', 'vpd_mean':'vpd',
                            'vpd_min':'vpdmin', 'vpd_max':'vpdmax'},
                   inplace=True)
    
    #Set days that don't have minimum number of hourly obs to missing
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
                    
    return obs_dly
    
def _parse_raws_webform(args):
        
    stn, elems, hrly_pwd, min_hrly_for_dly, start_date, end_date = args
    stn_id = stn.station_id
    
    values = {'smon': start_date.strftime("%m"),
              'sday': start_date.strftime("%d"),
              'syea': start_date.strftime("%y"),
              'emon': end_date.strftime("%m"),
              'eday': end_date.strftime("%d"),
              'eyea': end_date.strftime("%y"),
              'qBasic': "ON",
              'unit': 'M',
              'Ofor': 'A',
              'Datareq': 'C',  # Only complete days: C, Any data: A
              'qc': 'Y',
              'miss': '08',  # -9999
              "Submit Info": "Submit Info",
              'stn': stn_id,
              'WsMon': '01',
              'WsDay': '01',
              'WeMon': '12',
              'WeDay': '31'}

    data = urllib.urlencode(values)
    req = urllib2.Request(_URL_RAWS_DLY_TIME_SERIES2, data)
    response = urllib2.urlopen(req)
    lines = response.readlines()
    
    # skip first 7 lines
    lines = lines[7:]

    obs_ls = []

    for line in lines:

        if "Copyright" in line:
            break  # EOF reached
        else:

            try:
                
                vals = line.split()
                year = int(vals[0][6:])
                month = int(vals[0][0:2])
                day = int(vals[0][3:5])
                a_date = datetime(year, month, day)

            except ValueError:
                
                print "RAWS: Error in parsing a observation for", stn_id
                continue

            srad = float(vals[4])
            wspd = float(vals[5])
            tavg = float(vals[8])
            tmax = float(vals[9])
            tmin = float(vals[10])
            rh_avg = float(vals[11])
            rh_max = float(vals[12])
            rh_min = float(vals[13])
            prcp = float(vals[14])

            obs_ls.append((a_date, srad, wspd, tavg, tmax, tmin, rh_avg, rh_max,
                           rh_min, prcp))

    obs = pd.DataFrame(obs_ls, columns=['time', 'srad', 'wspd', 'tavg', 'tmax',
                                        'tmin', 'rh','rhmax','rhmin', 'prcp'])
    obs.set_index('time', inplace=True)
    obs.replace(-9999, np.nan, inplace=True)
    
    # convert kWhr to average watts for the day
    obs['srad'] = (obs.srad * 3.6e+6) / 86400
    
    # Calculate humidity variables
    stn_pres = calc_pressure(stn.elevation)
    # Tdew
    obs['tdew'] = convert_rh_to_tdew(obs.rh, obs.tavg)
    obs['tdewmin'] = convert_rh_to_tdew(obs.rhmax, obs.tmin)
    obs['tdewmax'] = convert_rh_to_tdew(obs.rhmin, obs.tmax)
    # VPD
    obs['vpd'] = convert_rh_to_vpd_daily(obs.tmin, obs.tmax, stn_pres,
                                         obs.rhmin, obs.rhmax)
    obs['vpdmax'] = convert_rh_to_vpd(obs.rhmin, obs.tmax, stn_pres)
    obs['vpdmin'] = convert_rh_to_vpd(obs.rhmax, obs.tmin, stn_pres)
    
    # Recalculate some humidity variables from hourly observations
    # if hrly_pwd available
    elems = np.array(elems)
    elems_hrly = elems[np.in1d(elems, _HRLY_ELEMS)]
    
    if elems_hrly.size > 0 and hrly_pwd:
        
        try:
            
            obsh = _parse_raws_hrly_webform(stn_id, stn_pres, start_date, end_date,
                                            hrly_pwd, min_hrly_for_dly)
            obs = obs.join(obsh, how='outer', lsuffix='_dly')
            #obs.drop(list(np.char.add(elems_hrly, '_old')), axis=1, inplace=True)
        
        except ValueError:
            
            # No valid hourly observations                
            print ("Warning: Could not access hourly humidity observations "
                   "for station %s. Reverting back estimates from daily temperature "
                   "and humidity variables.") % stn_id
    
    obs = obs[elems].copy()
    
    obs = obs.stack().reset_index()
    obs.rename(columns={'level_1':'elem', 0:'obs_value'},
               inplace=True)
    obs['station_id'] = stn_id

    return obs

class WrccRawsObsIO(ObsIO):

    _avail_elems = ['tmin', 'tmax', 'tdew', 'tdewmin', 'tdewmax', 'vpd',
                    'vpdmin','vpdmax', 'rh', 'rhmin', 'rhmax', 'prcp', 'srad',
                    'wspd']
    _requires_local = False
    _MIN_HRLY_FOR_DLY_DFLT = {'tdew':4, 'tdewmin':18, 'tdewmax':18, 'vpd':18,
                              'vpdmin':18, 'vpdmax':18}
    

    def __init__(self, nprocs=1, hrly_pwd=None, min_hrly_for_dly=None, **kwargs):

        super(WrccRawsObsIO, self).__init__(**kwargs)
        self.nprocs = nprocs
        self.hrly_pwd = hrly_pwd

        self.min_hrly_for_dly = (min_hrly_for_dly if min_hrly_for_dly
                                 else self._MIN_HRLY_FOR_DLY_DFLT)

    def _read_stns(self):
        
        fpath_stns = os.path.join(os.path.dirname(__file__), 'data',
                                  'raws_stns.csv')
        
        stns = pd.read_csv(fpath_stns)
        stns['start_date'] = pd.to_datetime(stns.start_date)
        stns['end_date'] = pd.to_datetime(stns.end_date)
        stns['station_id'] = stns.station_id.str[2:]
        stns['station_name'] = stns.station_name.apply(unicode, errors='ignore')
        stns['provider'] = 'WRCC'
        stns['sub_provider'] = 'RAWS'

        if self.bbox is not None:

            mask_bnds = ((stns.latitude >= self.bbox.south) & 
                         (stns.latitude <= self.bbox.north) & 
                         (stns.longitude >= self.bbox.west) & 
                         (stns.longitude <= self.bbox.east))

            stns = stns[mask_bnds].copy()
        
        # TODO Implement period-of-record checks
        
        stns = stns.reset_index(drop=True)
        stns = stns.set_index('station_id', drop=False)

        return stns

    def read_obs(self, stns_ids=None):

        if stns_ids is None:
            stns_obs = self.stns
        else:
            stns_obs = self.stns.loc[stns_ids]
        
        nstns = len(stns_obs.station_id)
        nprocs = self.nprocs if nstns >= self.nprocs else nstns
        
        def get_start_end(stn_id):
        
            if self.has_start_end_dates:
                start_date = self.start_date
                end_date = self.end_date
            else:
                start_date = self.stns.loc[stn_id].start_date
                #Use current date for end date since end_date metadata
                #might not be up-to-date
                end_date = pd.Timestamp.now()
                
            return start_date, end_date
        
        iter_stns = [(row[1], self.elems, self.hrly_pwd, self.min_hrly_for_dly) + 
                     get_start_end(row[0]) for row in stns_obs.iterrows()]
                
        if nprocs > 1:
            
            # http://stackoverflow.com/questions/24171725/
            # scikit-learn-multicore-attributeerror-stdin-instance-
            # has-no-attribute-close
            if not hasattr(sys.stdin, 'close'):
                def dummy_close():
                    pass
                sys.stdin.close = dummy_close

            pool = Pool(processes=nprocs)

            obs = pool.map(_parse_raws_webform, iter_stns, chunksize=1)
            pool.close()
            pool.join()

        else:

            obs = []
    
            for a_stn in iter_stns:
                
                obs_stn = _parse_raws_webform(a_stn)                
                obs.append(obs_stn)
            
        df_obs = pd.concat(obs, ignore_index=True)
        df_obs = df_obs.set_index(['station_id', 'elem', 'time'])
        df_obs = df_obs.sortlevel(0, sort_remaining=True)

        return df_obs
