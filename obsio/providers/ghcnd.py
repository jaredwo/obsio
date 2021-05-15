from .. import LOCAL_DATA_PATH
from ..util.misc import download_if_new_ftp, open_remote_file
from .generic import ObsIO
from ftplib import FTP
from multiprocessing.pool import Pool
import gc
import numpy as np
import os
import pandas as pd
import sys
import tarfile

_NETWORK_CODE_TO_SUBPROVIDER = {'0': '', '1': 'CoCoRaHS', 'C': 'COOP', 'E': 'ECA&D',
                                'M': 'WMO', 'N': ('National Meteorological or '
                                                  'Hydrological Center'),
                                'R': 'RAWS', 'S': 'SNOTEL', 'W': 'WBAN', 'P': ''}
_MISSING = -9999.0

def _build_ghcnd_colspecs():

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
         
    return colspecs, colnames
    
_COLSPECS_GHCND, _COLNAMES_GHCND = _build_ghcnd_colspecs()
        
_CONVERT_UNITS_FUNCS = {'tmin':lambda x: x / 10.0,  # tenths of degC to degC
                        'tmax':lambda x: x / 10.0,  # tenths of degC to degC
                        'prcp':lambda x: x / 10.0}  # tenths of mm to mm
                            
def _parse_ghcnd_dly(fpath, stn_id, elems, start_end=None):
                        
    obs = pd.read_fwf(fpath, colspecs=_COLSPECS_GHCND,
                      names=_COLNAMES_GHCND, header=None, na_values=_MISSING)

    obs.drop(obs.index[~obs.elem.str.lower().isin(elems)], axis=0,
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
    obs['time'] = obs.time + (np.array(obs.level_3.str.slice(-2).astype(np.int),
                                      dtype='<m8[D]') - 1)
            
    obs = obs.pivot(index='time', columns='elem', values=0)
    obs.columns = obs.columns.str.lower()
            
    for elem in elems:
        
        cname_elem = "%s_obsv" % elem
                
        try:
            
            obs[cname_elem] = obs[cname_elem].astype(np.float)

        except KeyError:
            # no observations for element
            continue
        
        obs.rename(columns={cname_elem:elem}, inplace=True)
         
        cname_qflg = "%s_qflg" % elem
                    
        try:
            # Set QA flagged observations to null
            obs.loc[(~obs[cname_qflg].isnull()).values, elem] = np.nan
        
        except KeyError:
            # No quality flag columns because no observations were flagged
            pass
        
        obs[elem] = _CONVERT_UNITS_FUNCS[elem](obs[elem])
    
    obs.drop(obs.columns[~obs.columns.isin(elems)], axis=1,
             inplace=True)
    
    obs.columns.name = None
    
    if start_end is not None:
    
        obs = obs.loc[start_end[0]:start_end[-1]]
    
    obs = obs.stack().reset_index()
    obs.rename(columns={'level_1':'elem', 0:'obs_value'}, inplace=True)
    
    obs['station_id'] = stn_id
    
    return obs

def _parse_ghcnd_dly_star(args):

    return _parse_ghcnd_dly(*args)

def _parse_ghcnd_dly_star_remote(args):
    
    args = list(args)
    args[0] = open_remote_file('https://www1.ncdc.noaa.gov/'
                               'pub/data/ghcn/daily/all/%s.dly' % args[1])
    args = tuple(args)
    
    return _parse_ghcnd_dly(*args)

def _parse_ghcnd_yrly(fpath, elems):
    
    def parse_chunk(chk):
                
        chk.dropna(subset=['tobs'], inplace=True)
            
        chk['elem'] = 'tobs_' + chk.elem.str.lower()
        chk['elem'] = pd.Categorical(chk['elem'], categories=elems)
        chk.dropna(subset=['elem'], inplace=True)
                
        chk['time'] = pd.to_datetime(chk['time'], format="%Y%m%d")
        
        chk.rename(columns={'tobs': 'obs_value'}, inplace=True)
        chk.reset_index(drop=True, inplace=True)

        return chk
        
    print("Loading file %s..." % fpath)
    
    reader = pd.read_csv(fpath, header=None, usecols=[0, 1, 2, 7], sep=',',
                         names=['station_id', 'time', 'elem', 'tobs'],
                         compression='gzip', dtype={'tobs': np.float32},
                         chunksize=1000000)
    
    obs = pd.concat([parse_chunk(chk) for chk in reader], ignore_index=True)
        
    return obs
    
def _parse_ghcnd_yrly_star(args):
    
    return _parse_ghcnd_yrly(*args)

def _parse_ghcnd_stnmeta(fpath_stns, fpath_stninv, elems, start_end=None, bbox=None):
        
    stns = pd.read_fwf(fpath_stns, colspecs=[(0, 11), (12, 20), (21, 30),
                                             (31, 37), (38, 40), (41, 71),
                                             (2, 3), (76, 79)], header=None,
                       names=['station_id', 'latitude', 'longitude',
                              'elevation', 'state', 'station_name',
                              'network_code', 'hcn_crn_flag'])
    stns['provider'] = 'GHCND'
    stns['sub_provider'] = (stns.network_code.apply(lambda x: _NETWORK_CODE_TO_SUBPROVIDER[x]))

    if bbox is not None:

        mask_bnds = ((stns.latitude >= bbox.south) & 
                     (stns.latitude <= bbox.north) & 
                     (stns.longitude >= bbox.west) & 
                     (stns.longitude <= bbox.east))

        stns = stns[mask_bnds].copy()

    stn_inv = pd.read_fwf(fpath_stninv, colspecs=[(0, 11), (31, 35), (36, 40),
                                          (41, 45)],
                          header=None, names=['station_id', 'elem', 'start_year',
                                              'end_year'])
    stn_inv['elem'] = stn_inv.elem.str.lower()
    stn_inv = stn_inv[stn_inv.elem.isin(elems)]
    stn_inv = stn_inv.groupby('station_id').agg({'end_year': np.max,
                                                 'start_year': np.min})
    stn_inv = stn_inv.reset_index()

    stns = pd.merge(stns, stn_inv, on='station_id')

    if start_end is not None:

        start_date, end_date = start_end

        mask_por = (((start_date.year <= stns.start_year) & 
                     (stns.start_year <= end_date.year)) | 
                    ((stns.start_year <= start_date.year) & 
                     (start_date.year <= stns.end_year)))

        stns = stns[mask_por].copy()

    stns = stns.reset_index(drop=True)
    stns = stns.set_index('station_id', drop=False)

    return stns

def _build_tobs_hdfs(path_out, fpaths_yrly, elems, nprocs=1):
    
    fpaths_yrly = np.array(fpaths_yrly)
    nprocs = nprocs if fpaths_yrly.size >= nprocs else fpaths_yrly.size
        
    stn_nums = pd.DataFrame([(np.nan, np.nan)], columns=['station_id', 'station_num'])
    num_inc = 0 
    
    first_append = {elem:True for elem in elems}
    
    # assume ~1.5 millions rows per year to estimate expected number of rows
    erows = 1500000 * len(fpaths_yrly)
    
    def write_data(df_tobs, num_inc, stn_nums):
    
        hdfs = {elem:pd.HDFStore(os.path.join(path_out, '%s.hdf' % elem), 'a')
                for elem in elems}
                
        df_tobs.set_index('station_id', inplace=True)
        df_tobs['obs_value'] = df_tobs.obs_value.astype(np.int16)
        
        uids = pd.DataFrame(df_tobs.index.unique(), columns=['station_id'])
        uids = uids.merge(stn_nums, how='left', on='station_id')
        mask_nonum = uids.station_num.isnull()
        
        if mask_nonum.any():
            
            nums = np.arange(num_inc, (num_inc + mask_nonum.sum()))
            uids.loc[mask_nonum, 'station_num'] = nums
            num_inc = nums[-1] + 1
            stn_nums = pd.concat([stn_nums, uids[mask_nonum]], ignore_index=True)
        
        uids.set_index('station_id', inplace=True)
        uids['station_num'] = uids.station_num.astype(np.int)
        
        df_tobs = df_tobs.join(uids, how='left').set_index('station_num')
        grped = df_tobs.groupby('elem')
        
        for elem in elems:
            
            try:
                grp = grped.get_group(elem)[['time', 'obs_value']].copy()
            except KeyError:
                # no observation for element
                continue
            
            if first_append[elem]:
                
                hdfs[elem].append('df_tobs', grp, data_columns=['time'],
                                  expectedrows=erows, index=False)
                first_append[elem] = False
            
            else:
            
                hdfs[elem].append('df_tobs', grp, data_columns=['time'], index=False)
        
        for store in hdfs.values():
            store.close()
        
        return num_inc, stn_nums
    
    # Initialize output hdfs
    hdfs = [pd.HDFStore(os.path.join(path_out, '%s.hdf' % elem), 'w')
            for elem in elems]
    
    for store in hdfs:
        store.close()
    
    if nprocs > 1:
        
        # http://stackoverflow.com/questions/24171725/
        # scikit-learn-multicore-attributeerror-stdin-instance-
        # has-no-attribute-close
        if not hasattr(sys.stdin, 'close'):
            def dummy_close():
                pass
            sys.stdin.close = dummy_close
        
        for i in np.arange(fpaths_yrly.size, step=nprocs):
            
            fpaths = fpaths_yrly[i:(i + nprocs)]
            gc.collect()
            pool = Pool(processes=nprocs)                
            iter_files = [(fpath, elems) for fpath in fpaths]
            ls_tobs = pool.map(_parse_ghcnd_yrly_star, iter_files, chunksize=1)
            pool.close()
            pool.join()
            
            for df_tobs in ls_tobs:
            
                num_inc, stn_nums = write_data(df_tobs, num_inc, stn_nums)
                
            del df_tobs
            del ls_tobs
            
                
    else:
        
        for fpath in fpaths_yrly:
            
            df_tobs = _parse_ghcnd_yrly(fpath, elems)
            num_inc, stn_nums = write_data(df_tobs, num_inc, stn_nums)
    
    stn_nums = stn_nums.dropna()
    store_stnnums = pd.HDFStore(os.path.join(path_out, 'stn_nums.hdf'), 'w')
    store_stnnums.put('df_stnnums', stn_nums)
    store_stnnums.close()
    
    # Create indexess
    for elem in elems:
        
        with pd.HDFStore(os.path.join(path_out, '%s.hdf' % elem)) as store:
            
            store.create_table_index('df_tobs', optlevel=9, kind='full')
            # store.create_table_index('df_tobs', columns=['time'], optlevel=9, kind='full')

    
    
class GhcndBulkObsIO(ObsIO):

    _avail_elems = ['tmin', 'tmax', 'prcp', 'tobs_tmin', 'tobs_tmax',
                    'tobs_prcp']
    
    _requires_local = True
    name = 'GHCND'


    def __init__(self, nprocs=1, local_data_path=None, download_updates=True,
                 **kwargs):

        super(GhcndBulkObsIO, self).__init__(**kwargs)

        self.local_data_path = (local_data_path if local_data_path
                                else LOCAL_DATA_PATH)
        self.path_ghcnd_data = os.path.join(self.local_data_path, 'GHCND')
        if not os.path.isdir(self.path_ghcnd_data):
            os.mkdir(self.path_ghcnd_data)
            
        self.download_updates = download_updates
        self._download_run = False
        
        self.nprocs = nprocs
        
        # Split out normal elems and time-of-observation elements
        elems = np.array(self.elems)
        mask_tobs = np.char.startswith(elems, 'tobs')
        self._elems = elems[~mask_tobs]
        self._elems_tobs = elems[mask_tobs]  
        self._has_tobs = self._elems_tobs.size > 0
       
        self._a_df_tobs_stnnums = None
    
    @property
    def _df_tobs_stnnums(self):

        if self._a_df_tobs_stnnums is None:
            
            path_store = os.path.join(self.path_ghcnd_data, 'by_year',
                                      'stn_nums.hdf')
            
            store = pd.HDFStore(path_store)
            
            self._a_df_tobs_stnnums = store.select('df_stnnums',
                                                   auto_close=True).set_index('station_id')
            
        return self._a_df_tobs_stnnums
    
    def _read_stns(self):
        
        if self.download_updates and not self._download_run:

            self.download_local()
            
        fpath_stns = os.path.join(self.path_ghcnd_data, 'ghcnd-stations.txt')
        fpath_stninv = os.path.join(self.path_ghcnd_data, 'ghcnd-inventory.txt')
        
        if self.has_start_end_dates:
            start_end = (self.start_date, self.end_date)
        else:
            start_end = None
            
        stns = _parse_ghcnd_stnmeta(fpath_stns, fpath_stninv, self.elems,
                                    start_end, self.bbox)
        
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
        
        ftp.close()
        
        if downloaded_tar:
            
            # Gunzip tar file
            print ("Unzipping and extracting files from %s. "
                   "This will take several minutes..." % 
                   os.path.join(local_path, 'ghcnd_all.tar.gz'))
            
            with tarfile.open(os.path.join(local_path, 'ghcnd_all.tar.gz'), 'r') as targhcnd:
            
                targhcnd.extractall(local_path)
                    
        if self._has_tobs:
            
            ftp = FTP('ftp.ncdc.noaa.gov')
            ftp.login()
        
            by_yr_path = os.path.join(local_path, 'by_year')
            
            if not os.path.isdir(by_yr_path):
                os.mkdir(by_yr_path)
                
            if self.has_start_end_dates:
            
                start_year = self.start_date.year
                end_year = self.end_date.year
                
                yr_fnames = ["%s.csv.gz" % yr for yr in np.arange(start_year,
                                                                end_year + 1)]
                
            else:
                
                yr_fnames = ftp.nlst('pub/data/ghcn/daily/by_year')
                yr_fnames = [fname.split('/')[-1] for fname in yr_fnames]
                yr_fnames = np.array(yr_fnames)
                yr_fnames = list(yr_fnames[np.char.endswith(yr_fnames, ".csv.gz")])
            
            new_yrly = False
                        
            for fname in yr_fnames:
                
                dwnld = download_if_new_ftp(ftp, 'pub/data/ghcn/daily/by_year/%s' % fname,
                                            os.path.join(by_yr_path, fname))
                
                if dwnld:
                    
                    new_yrly = True
            
            ftp.close()
             
            if new_yrly:
                
                fpaths_yrly = [os.path.join(by_yr_path, fname) for fname in yr_fnames]
                tobs_elems_all = (np.array(self._avail_elems)
                                  [np.char.startswith(self._avail_elems,
                                                      'tobs')])
                
                _build_tobs_hdfs(by_yr_path, fpaths_yrly, tobs_elems_all, self.nprocs)
                
        
        self._download_run = True
        
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
                start_end = (self.start_date, self.end_date)
            else:
                start_end = None
            
            if nprocs > 1:
                
                # http://stackoverflow.com/questions/24171725/
                # scikit-learn-multicore-attributeerror-stdin-instance-
                # has-no-attribute-close
                if not hasattr(sys.stdin, 'close'):
                    def dummy_close():
                        pass
                    sys.stdin.close = dummy_close
                
                iter_stns = [(os.path.join(self.path_ghcnd_data, 'ghcnd_all',
                                           '%s.dly' % a_id), a_id, self._elems,
                              start_end) for a_id in stns_obs.station_id]
                
                pool = Pool(processes=nprocs)                
                obs = pool.map(_parse_ghcnd_dly_star, iter_stns)
                
                pool.close()
                pool.join()
            
            else:
            
                obs = []
    
                for a_id in stns_obs.station_id:
                    
                    fpath = os.path.join(self.path_ghcnd_data, 'ghcnd_all',
                                         '%s.dly' % a_id)
                                       
                    obs_stn = _parse_ghcnd_dly(fpath, a_id, self._elems, start_end)
                    obs.append(obs_stn)

            df_obs = pd.concat(obs, ignore_index=True)

            if self._has_tobs:
                
                stnnums = stns_obs.join(self._df_tobs_stnnums).dropna(subset=['station_num'])
                
                if not stnnums.empty:
                    
                    stnnums = stnnums.reset_index(drop=True).set_index('station_num')
                        
                    select_str = "index = a_num"
                                    
                    df_tobs = []
                    path_yrly = os.path.join(self.path_ghcnd_data, 'by_year')
                    
                    for elem in self._elems_tobs:
                        
                        store = pd.HDFStore(os.path.join(path_yrly, '%s.hdf' % elem))
                        
                        # Perform separate read for each station.
                        # Had this in a single call using "index in stnnums"
                        # but memory usage was too high
                    
                        for a_num in stnnums.index:

                            elem_tobs = store.select('df_tobs', select_str).reset_index()
                            elem_tobs['elem'] = elem
                            elem_tobs['station_id'] = stnnums.station_id.loc[a_num]
                                                        
                            df_tobs.append(elem_tobs[['time', 'elem', 'obs_value',
                                                      'station_id']])
                        store.close()
                        del store
                        gc.collect()
                    
                    df_tobs = pd.concat(df_tobs, ignore_index=True)
                    
                    if self.has_start_end_dates:
                        
                        df_tobs = df_tobs[(df_tobs.time >= self.start_date) & 
                                          (df_tobs.time <= self.end_date)]
                    
                    df_obs = pd.concat([df_obs, df_tobs], ignore_index=True)

        finally:

            pd.set_option('mode.chained_assignment', opt_val)

        df_obs = df_obs.set_index(['station_id', 'elem', 'time'])
        df_obs = df_obs.sort_index(0, sort_remaining=True)

        return df_obs

class GhcndObsIO(ObsIO):

    _avail_elems = ['tmin', 'tmax', 'prcp']
    _requires_local = False
    name = 'GHCND'

    def __init__(self, nprocs=1, **kwargs):

        super(GhcndObsIO, self).__init__(**kwargs)

        self.nprocs = nprocs
        
    def _read_stns(self):
        
        fbuf_stns = open_remote_file('https://www1.ncdc.noaa.gov/'
                                     'pub/data/ghcn/daily/ghcnd-stations.txt')
        fbuf_stninv = open_remote_file('https://www1.ncdc.noaa.gov/'
                                       'pub/data/ghcn/daily/ghcnd-inventory.txt')
        
        if self.has_start_end_dates:
            start_end = (self.start_date, self.end_date)
        else:
            start_end = None
            
        stns = _parse_ghcnd_stnmeta(fbuf_stns, fbuf_stninv, self.elems,
                                    start_end, self.bbox)
        
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
                start_end = (self.start_date, self.end_date)
            else:
                start_end = None
            
            if nprocs > 1:
                
                # http://stackoverflow.com/questions/24171725/
                # scikit-learn-multicore-attributeerror-stdin-instance-
                # has-no-attribute-close
                if not hasattr(sys.stdin, 'close'):
                    def dummy_close():
                        pass
                    sys.stdin.close = dummy_close
                
                iter_stns = [(None, a_id, self.elems, start_end)
                             for a_id in stns_obs.station_id]
                
                pool = Pool(processes=nprocs)                
                obs = pool.map(_parse_ghcnd_dly_star_remote, iter_stns)
                
                pool.close()
                pool.join()
            
            else:
            
                obs = []
    
                for a_id in stns_obs.station_id:
                    
                    abuf = open_remote_file('https://www1.ncdc.noaa.gov/'
                                            'pub/data/ghcn/daily/all/%s.dly' % a_id)
                                       
                    obs_stn = _parse_ghcnd_dly(abuf, a_id, self.elems, start_end)
                    obs.append(obs_stn)

            df_obs = pd.concat(obs, ignore_index=True)

        finally:

            pd.set_option('mode.chained_assignment', opt_val)

        df_obs = df_obs.set_index(['station_id', 'elem', 'time'])
        df_obs = df_obs.sort_index(0, sort_remaining=True)

        return df_obs
