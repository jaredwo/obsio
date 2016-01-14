from .. import LOCAL_DATA_PATH
from .generic import ObsIO
from StringIO import StringIO
import glob
import itertools
import numpy as np
import os
import pandas as pd
import subprocess
import tarfile

_RPATH_USHCN = 'ftp://ftp.ncdc.noaa.gov/pub/data/ushcn/v2.5/*'

_ELEMS_TO_USHCN_DATASET = {'tmin_mth_raw': 'raw', 'tmin_mth_tob': 'tob',
                           'tmin_mth_fls': 'FLs.52j', 'tmax_mth_raw': 'raw',
                           'tmax_mth_tob': 'tob', 'tmax_mth_fls': 'FLs.52j',
                           'tavg_mth_raw': 'raw', 'tavg_mth_tob': 'tob',
                           'tavg_mth_fls': 'FLs.52j', 'prcp_mth_raw': 'raw',
                           'prcp_mth_tob': 'tob', 'prcp_mth_fls': 'FLs.52j'}

_ELEMS_TO_USHCN_VNAME = {'tmin_mth_raw': 'tmin', 'tmin_mth_tob': 'tmin',
                         'tmin_mth_fls': 'tmin', 'tmax_mth_raw': 'tmax',
                         'tmax_mth_tob': 'tmax', 'tmax_mth_fls': 'tmax',
                         'tavg_mth_raw': 'tavg', 'tavg_mth_tob': 'tavg',
                         'tavg_mth_fls': 'tavg', 'prcp_mth_raw': 'prcp',
                         'prcp_mth_tob': 'prcp', 'prcp_mth_fls': 'prcp'}

_to_c = lambda x: x / 100.0
_to_mm = lambda x: x / 10.0

_ELEMS_CONVERT_FUNCT = {'tmin_mth_raw': _to_c, 'tmin_mth_tob': _to_c,
                        'tmin_mth_fls': _to_c, 'tmax_mth_raw': _to_c,
                        'tmax_mth_tob': _to_c, 'tmax_mth_fls': _to_c,
                        'tavg_mth_raw': _to_c, 'tavg_mth_tob': _to_c,
                        'tavg_mth_fls': _to_c, 'prcp_mth_raw': _to_mm,
                        'prcp_mth_tob': _to_mm, 'prcp_mth_fls': _to_mm}


class UshcnObsIO(ObsIO):

    _avail_elems = ['tmin_mth_raw', 'tmin_mth_tob', 'tmin_mth_fls',
                    'tmax_mth_raw', 'tmax_mth_tob', 'tmax_mth_fls',
                    'tavg_mth_raw', 'tavg_mth_tob', 'tavg_mth_fls'
                    'prcp_mth_raw', 'prcp_mth_tob', 'prcp_mth_fls']

    _requires_local = True

    def __init__(self, local_data_path=None, download_updates=True, **kwargs):

        super(UshcnObsIO, self).__init__(**kwargs)

        self.local_data_path = (local_data_path if local_data_path
                                else LOCAL_DATA_PATH)
        self.path_ushcn_data = os.path.join(self.local_data_path, 'USHCN')
        if not os.path.isdir(self.path_ushcn_data):
            os.mkdir(self.path_ushcn_data)
            
        self.download_updates = download_updates
        self._download_run = False

        self._a_obs_prefix_dirs = None
        self._a_obs_tarfiles = None
        self._a_df_tobs = None

    @property
    def _obs_tarfiles(self):

        if self._a_obs_tarfiles is None:

            self._a_obs_tarfiles = {}

            for elem in self.elems:

                fpath = os.path.join(self.path_ushcn_data,
                                     'ushcn.%s.latest.%s.tar' %
                                     (_ELEMS_TO_USHCN_VNAME[elem],
                                      _ELEMS_TO_USHCN_DATASET[elem]))

                tfile = tarfile.open(fpath)

                self._a_obs_tarfiles[elem] = tfile

        return self._a_obs_tarfiles

    def _read_stns(self):
        
        if self.download_updates and not self._download_run:

            self.download_local()
        
        stns = pd.read_fwf(os.path.join(self.path_ushcn_data,
                                        'ushcn-v2.5-stations.txt'),
                           colspecs=[(0, 11), (12, 20), (21, 30), (31, 37),
                                     (38, 40), (41, 71)], header=None,
                           names=['station_id', 'latitude', 'longitude',
                                  'elevation', 'state', 'station_name'])
        stns['station_name'] = stns.station_name.apply(unicode,
                                                       errors='ignore')
        stns['provider'] = 'USHCN'
        stns['sub_provider'] = ''

        if self.bbox is not None:

            mask_bnds = ((stns.latitude >= self.bbox.south) &
                         (stns.latitude <= self.bbox.north) &
                         (stns.longitude >= self.bbox.west) &
                         (stns.longitude <= self.bbox.east))

            stns = stns[mask_bnds].copy()

        stns = stns.set_index('station_id', drop=False)

        return stns

    @property
    def _obs_prefix_dirs(self):

        if self._a_obs_prefix_dirs is None:

            self._a_obs_prefix_dirs = {elem: self._obs_tarfiles[elem].
                                       getnames()[0].split('/')[1] for elem in
                                       self.elems}

        return self._a_obs_prefix_dirs

    def download_local(self):

        local_path = self.path_ushcn_data

        print "Syncing USHCN data to local..."
        subprocess.call(['wget', '-N', '--directory-prefix=' + local_path,
                         _RPATH_USHCN])

        print "Unzipping files..."

        fnames_tars = glob.glob(os.path.join(local_path, '*.gz'))

        for fname in fnames_tars:

            subprocess.call(['gunzip', '-f',
                             os.path.join(local_path, fname)])
            
        self._download_run = True

    def _parse_stn_obs(self, stn_id, elem):

        fname = os.path.join('.', self._obs_prefix_dirs[elem],
                             '%s.%s.%s' % (stn_id, _ELEMS_TO_USHCN_DATASET[elem],
                                           _ELEMS_TO_USHCN_VNAME[elem]))
        obs_file = self._obs_tarfiles[elem].extractfile(fname)

        obs = pd.read_fwf(StringIO(obs_file.read()),
                          colspecs=[(12, 16), (17, 17 + 5), (26, 26 + 5),
                                    (35, 35 + 5), (44, 44 + 5), (53, 53 + 5),
                                    (62, 62 + 5), (71, 71 + 5), (80, 80 + 5),
                                    (89, 89 + 5), (98, 98 + 5), (107, 107 + 5),
                                    (116, 116 + 5)], header=None, index_col=0,
                          names=['year'] + ['%.2d' % mth for
                                            mth in np.arange(1, 13)],
                          na_values='-9999')

        obs_file.close()

        obs = obs.unstack().swaplevel(0, 1).sortlevel(0, sort_remaining=True)
        obs = obs.reset_index()
        obs['time'] = pd.to_datetime(obs.year.astype(np.str) + obs.level_1,
                                     format='%Y%m')

        obs.drop(['year', 'level_1'], axis=1, inplace=True)
        obs.rename(columns={0: 'obs_value'}, inplace=True)
        obs.dropna(axis=0, subset=['obs_value'], inplace=True)

        if self.has_start_end_dates:

            mask_time = ((obs.time >= self.start_date) &
                         (obs.time <= self.end_date))
            obs.drop(obs[~mask_time].index, axis=0, inplace=True)

        obs['obs_value'] = _ELEMS_CONVERT_FUNCT[elem](obs['obs_value'])
        obs['station_id'] = stn_id
        obs['elem'] = elem

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

            obs = [self._parse_stn_obs(a_id, elem) for elem, a_id in
                   itertools.product(self.elems, stns_obs.station_id)]

            obs = pd.concat(obs, ignore_index=True)

        finally:

            pd.set_option('mode.chained_assignment', opt_val)

        obs = obs.set_index(['station_id', 'elem', 'time'])
        obs = obs.sortlevel(0, sort_remaining=True)

        return obs
