from .. import LOCAL_DATA_PATH
from ..util.misc import download_if_new_ftp
from .generic import ObsIO
from ftplib import FTP
import numpy as np
import os
import pandas as pd
import tarfile

_NETWORK_CODE_TO_SUBPROVIDER = {'0': '', '1': 'CoCoRaHS', 'C': 'COOP', 'E': 'ECA&D',
                                'M': 'WMO', 'N': ('National Meteorological or '
                                                  'Hydrological Center'),
                                'R': 'RAWS', 'S': 'SNOTEL', 'W': 'WBAN'}
_MISSING = -9999.0
    
def _convert_units(element, value):

    if value == -9999:
        # NO DATA, no conversion
        return value
    elif element == "prcp":
        # tenths of mm to mm
        return value / 10.0
    elif element == "tmax" or element == "tmin":
        # tenths of degrees C to degrees C
        return value / 10.0
    else:
        raise ValueError("".join(["Unrecognized element type: ", element]))

   
class GhcndBulkObsIO(ObsIO):

    _avail_elems = ['tmin', 'tmax', 'prcp', 'tobs_tmin', 'tobs_tmax',
                    'tobs_prcp']
    
    _requires_local = True

    _convert_units_funcs = {'tmin':lambda x: x / 10.0,  # tenths of degC to degC
                            'tmax':lambda x: x / 10.0,  # tenths of degC to degC
                            'prcp':lambda x: x / 10.0}  # tenths of mm to mm
                            

    def __init__(self, local_data_path=None, download_updates=True, **kwargs):

        super(GhcndBulkObsIO, self).__init__(**kwargs)

        self.local_data_path = (local_data_path if local_data_path
                                else LOCAL_DATA_PATH)
        self.path_ghcnd_data = os.path.join(self.local_data_path, 'GHCND')
        if not os.path.isdir(self.path_ghcnd_data):
            os.mkdir(self.path_ghcnd_data)
            
        self.download_updates = download_updates
        self._download_run = False
        
        # Split out normal elems and time-of-observation elements
        elems = np.array(self.elems)
        mask_tobs = np.char.startswith(elems, 'tobs')
        self._elems = elems[~mask_tobs]
        self._elems_tobs = elems[mask_tobs]  
        self._has_tobs = self._elems_tobs.size > 0
       
        self._a_obs_tarfile = None
        self._a_df_tobs = None
        self._a_colspecs = None
        

    @property
    def _df_tobs(self):

        if self._a_df_tobs is None:

            path_yrly = os.path.join(self.path_ghcnd_data, 'by_year')
            fnames = np.array(os.listdir(path_yrly))
            yrs = np.array([int(a_fname[0:4]) for a_fname in fnames])
            idx = np.argsort(yrs)
            yrs = yrs[idx]
            fnames = fnames[idx]

            if self.has_start_end_dates:
                start_yr = self.start_date.year
                end_yr = self.end_date.year
            else:
                start_yr = np.min(yrs)
                end_yr = np.max(yrs)

            print ("GhcndObsIO: Loading time-of-observation data for years "
                   "%d to %d..." % (start_yr, end_yr))

            mask_yrs = np.logical_and(yrs >= start_yr, yrs <= end_yr)
            fnames = fnames[mask_yrs]

            tobs_all = []

            for a_fname in fnames:

                print "Loading file %s..." % a_fname

                fpath = os.path.join(path_yrly, a_fname)

                a_df = pd.read_csv(fpath, header=None, usecols=[0, 1, 2, 7],
                                   names=[
                                       'station_id', 'time', 'elem', 'tobs'],
                                   dtype={'station_id': np.str, 'time': np.str,
                                          'elem': np.str})
                a_df = a_df[~a_df.tobs.isnull()].copy()
                a_df['elem'] = 'tobs_' + a_df.elem.str.lower()
                a_df = a_df[a_df.elem.isin(self.elems)].copy()
                a_df = a_df[a_df.station_id.isin(self.stns.station_id)].copy()
                a_df['time'] = pd.to_datetime(a_df['time'], format="%Y%m%d")

                if self.has_start_end_dates:
                    mask_time = ((a_df['time'] >= self.start_date) & 
                                 (a_df['time'] <= self.end_date))
                    a_df = a_df[mask_time].copy()

                a_df = a_df.rename(columns={'tobs': 'obs_value'})

                tobs_all.append(a_df)

            self._a_df_tobs = pd.concat(tobs_all, ignore_index=True)

        return self._a_df_tobs

    @property
    def _colspecs(self):
        
        if self._a_colspecs is None:
        
            # ID,YEAR,MONTH,ELEMENT
            colspecs = [(0, 11), (11, 15), (15, 17), (17, 21)]
            colnames = ['station_id', 'year', 'month', 'elem']
            
            offset = 0
            obs_column_size = 8
        
            for day in np.arange(1, 32):
                
                cs_value = (21 + offset, offset + 26)
                cs_mflag = (26 + offset, offset + 27)
                cs_qflag = (27 + offset, offset + 28)
                cs_sflag = (28 + offset, offset + 29)
                
                colspecs.extend([cs_value, cs_mflag, cs_qflag, cs_sflag])
                colnames.extend(['OBSV%.2d' % day, 'MFLG%.2d' % day,
                                 'QFLG%.2d' % day, 'SFLG%.2d' % day])
                
                offset += obs_column_size
                
            self._a_colspecs = colspecs, colnames
        
        return self._a_colspecs

    @property
    def _obs_tarfile(self):

        if self._a_obs_tarfile is None:

            print "GhcndObsIO: Initializing ghcnd_all.tar for reading..."
            fpath = os.path.join(self.path_ghcnd_data, 'ghcnd_all.tar.gz')
            self._a_obs_tarfile = tarfile.open(fpath)
            self._a_obs_tarfile.getnames()

        return self._a_obs_tarfile

    def _read_stns(self):
        
        if self.download_updates and not self._download_run:

            self.download_local()

        stns = pd.read_fwf(os.path.join(self.path_ghcnd_data,
                                        'ghcnd-stations.txt'),
                           colspecs=[(0, 11), (12, 20), (21, 30), (31, 37), (38, 40),
                                     (41, 71), (2, 3)], header=None,
                           names=['station_id', 'latitude', 'longitude',
                                  'elevation', 'state', 'station_name',
                                  'network_code'])
        stns['station_name'] = stns.station_name.apply(
            unicode, errors='ignore')
        stns['provider'] = 'GHCND'
        stns['sub_provider'] = (stns.
                                network_code.
                                apply(lambda x:
                                      _NETWORK_CODE_TO_SUBPROVIDER[x]))

        if self.bbox is not None:

            mask_bnds = ((stns.latitude >= self.bbox.south) & 
                         (stns.latitude <= self.bbox.north) & 
                         (stns.longitude >= self.bbox.west) & 
                         (stns.longitude <= self.bbox.east))

            stns = stns[mask_bnds].copy()

        

        fpath_inv = os.path.join(self.path_ghcnd_data,
                                 'ghcnd-inventory.txt')

        stn_inv = pd.read_fwf(fpath_inv,
                              colspecs=[
                                  (0, 11), (31, 35), (36, 40), (41, 45)],
                              header=None, names=['station_id',
                                                  'elem', 'start_year',
                                                  'end_year'])
        stn_inv['elem'] = stn_inv.elem.str.lower()
        stn_inv = stn_inv[stn_inv.elem.isin(self.elems)]
        stn_inv = stn_inv.groupby('station_id').agg({'end_year': np.max,
                                                     'start_year': np.min})
        stn_inv = stn_inv.reset_index()

        stns = pd.merge(stns, stn_inv, on='station_id')

        if self.has_start_end_dates:

            mask_por = (((self.start_date.year <= stns.start_year) & 
                         (stns.start_year <= self.end_date.year)) | 
                        ((stns.start_year <= self.start_date.year) & 
                         (self.start_date.year <= stns.end_year)))

            stns = stns[mask_por].copy()

        stns = stns.reset_index(drop=True)
        stns = stns.set_index('station_id', drop=False)

        return stns

    def download_local(self):
                
        local_path = self.path_ghcnd_data
        
        ftp = FTP('ftp.ncdc.noaa.gov')
        ftp.login()
        
        download_if_new_ftp(ftp, 'pub/data/ghcn/daily/ghcnd-version.txt',
                            os.path.join(local_path, 'ghcnd-version.txt'))
        
        download_if_new_ftp(ftp, 'pub/data/ghcn/daily/status.txt',
                            os.path.join(local_path, 'status.txt'))

        download_if_new_ftp(ftp, 'pub/data/ghcn/daily/readme.txt',
                            os.path.join(local_path, 'readme.txt'))
        
        download_if_new_ftp(ftp, 'pub/data/ghcn/daily/ghcnd-inventory.txt',
                            os.path.join(local_path, 'ghcnd-inventory.txt'))
        
        download_if_new_ftp(ftp, 'pub/data/ghcn/daily/ghcnd-stations.txt',
                            os.path.join(local_path, 'ghcnd-stations.txt'))
        
        downloaded_tar = download_if_new_ftp(ftp, 'pub/data/ghcn/daily/ghcnd_all.tar.gz',
                                              os.path.join(local_path, 'ghcnd_all.tar.gz'))
        
        if downloaded_tar:
            
            # Gunzip tar file
            print ("Unzipping and extracting files from %s. "
                   "This will take several minutes..." % 
                   os.path.join(local_path, 'ghcnd_all.tar.gz'))
            
            with tarfile.open(os.path.join(local_path, 'ghcnd_all.tar.gz'), 'r') as targhcnd:
            
                targhcnd.extractall(local_path)
                    
        if self._has_tobs:
        
            by_yr_path = os.path.join(local_path, 'by_year')
            if not os.path.isdir(by_yr_path):
                os.mkdir(by_yr_path)
                
            if self.has_start_end_dates:
            
                start_year = self.start_date.year
                end_year = self.end_date.year
                
                yr_fnames = ["%s.csv.gz" % yr for yr in np.arange(start_year,
                                                                end_year + 1)]
                
            else:
                
                yr_fnames = np.array(ftp.nlst('pub/data/ghcn/daily/by_year'))
                yr_fnames = list(yr_fnames[np.char.endswith(yr_fnames, ".csv.gz")])
            
            for fname in yr_fnames:
                
                download_if_new_ftp(ftp, 'pub/data/ghcn/daily/by_year/%s' % fname,
                                    os.path.join(by_yr_path, fname))
        
        self._download_run = True
        
    def _parse_stn_obs(self, stn_id):
        
        fpath = os.path.join(self.path_ghcnd_data, 'ghcnd_all',
                             '%s.dly' % stn_id)
                
        colspecs, colnames = self._colspecs
        
        obs = pd.read_fwf(fpath, colspecs=colspecs, names=colnames, header=None,
                          na_values=_MISSING)

        obs.drop(obs.index[~obs.elem.str.lower().isin(self._elems)], axis=0,
                 inplace=True)
        
        # https://github.com/pydata/pandas/issues/8158
        # http://stackoverflow.com/questions/19350806/
        # how-to-convert-columns-into-one-datetime-column-in-pandas
        y = np.array(obs.year - 1970, dtype='<M8[Y]')
        m = np.array(obs.month - 1, dtype='<m8[M]')
        obs['time'] = pd.to_datetime(y + m)
        obs.drop(['year', 'month'], axis=1, inplace=True)
        
        obs.set_index(['time', 'station_id', 'elem'], inplace=True)
        obs = obs.stack().reset_index()
        
        obs['elem'] = obs.elem + "_" + obs.level_3.str.slice(0, 4)
        obs['time'] = obs.time + np.array(obs.level_3.str.slice(-2).astype(np.int),
                                          dtype='<m8[D]')
                
        obs = obs.pivot(index='time', columns='elem', values=0)
        obs.columns = obs.columns.str.lower()
                
        for elem in self._elems:
             
            cname_qflg = "%s_qflg" % elem
                        
            obs.rename(columns={"%s_obsv" % elem:elem}, inplace=True)
            
            try:
                obs.loc[(~obs[cname_qflg].isnull()).values, elem] = np.nan
            except KeyError:
                # No quality flag columns because no observations were flagged
                pass
            
            obs[elem] = self._convert_units_funcs[elem](obs[elem].astype(np.float))
        
        obs.drop(obs.columns[~obs.columns.isin(self._elems)], axis=1,
                 inplace=True)
        
        obs.columns.name = None
        
        return obs

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

            obs = []

            for a_id in stns_obs.station_id:

                obs_stn = self._parse_stn_obs(a_id)

                if self.has_start_end_dates:
                    
                    obs_stn = obs_stn.loc[self.start_date:self.end_date]
                
                obs_stn = obs_stn.stack().reset_index()
                obs_stn.rename(columns={'level_1':'elem', 0:'obs_value'},
                               inplace=True)
                
                obs_stn['station_id'] = a_id

                obs.append(obs_stn)

            df_obs = pd.concat(obs, ignore_index=True)

            if self._has_tobs:

                df_tobs = self._df_tobs[self._df_tobs.station_id.
                                        isin(stns_obs.station_id)].copy()

                df_obs = pd.concat([df_obs, df_tobs], ignore_index=True)

        finally:

            pd.set_option('mode.chained_assignment', opt_val)

        df_obs = df_obs.set_index(['station_id', 'elem', 'time'])
        df_obs = df_obs.sortlevel(0, sort_remaining=True)

        return df_obs
