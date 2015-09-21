from .. import LOCAL_DATA_PATH
from .generic import ObsIO
from urlparse import urljoin
import datetime
import numpy as np
import os
import pandas as pd
import subprocess
import tarfile


_RPATH_GHCND = 'ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/'

_NETWORK_CODE_TO_SUBPROVIDER = {'0': '', '1': 'CoCoRaHS', 'C': 'COOP', 'E': 'ECA&D',
                                'M': 'WMO', 'N': ('National Meteorological or '
                                                  'Hydrological Center'),
                                'R': 'RAWS', 'S': 'SNOTEL', 'W': 'WBAN'}
_MONTH_DAYS = np.arange(1, 32)

_OBS_COLUMN_SIZE = 8

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


class GhcndObsIO(ObsIO):

    _avail_elems = ['tmin', 'tmax', 'prcp', 'tobs_tmin', 'tobs_tmax',
                    'tobs_prcp']
    _requires_local = True

    def __init__(self, local_data_path=None, **kwargs):

        super(GhcndObsIO, self).__init__(**kwargs)

        self.local_data_path = (local_data_path if local_data_path
                                else LOCAL_DATA_PATH)
    
        self.path_ghcnd_data = os.path.join(self.local_data_path, 'GHCND')
       
        if not os.path.isdir(self.path_ghcnd_data):
            os.mkdir(self.path_ghcnd_data)
       
        self._a_obs_tarfile = None
        self._a_df_tobs = None

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
    def _obs_tarfile(self):

        if self._a_obs_tarfile is None:

            print "GhcndObsIO: Initializing ghcnd_all.tar for reading..."
            fpath = os.path.join(self.path_ghcnd_data, 'ghcnd_all.tar')
            self._a_obs_tarfile = tarfile.open(fpath)
            self._a_obs_tarfile.getnames()

        return self._a_obs_tarfile

    def _read_stns(self):

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

        if self.has_start_end_dates:

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

        subprocess.call(['wget', '-N', '--directory-prefix=' + local_path,
                         urljoin(_RPATH_GHCND, 'ghcnd-version.txt')])

        subprocess.call(['wget', '-N', '--directory-prefix=' + local_path,
                         urljoin(_RPATH_GHCND, 'status.txt')])

        subprocess.call(['wget', '-N', '--directory-prefix=' + local_path,
                         urljoin(_RPATH_GHCND, 'readme.txt')])

        subprocess.call(['wget', '-N', '--directory-prefix=' + local_path,
                         urljoin(_RPATH_GHCND, 'ghcnd-inventory.txt')])

        subprocess.call(['wget', '-N', '--directory-prefix=' + local_path,
                         urljoin(_RPATH_GHCND, 'ghcnd-stations.txt')])

        subprocess.call(['wget', '-N', '--directory-prefix=' + local_path,
                         urljoin(_RPATH_GHCND, 'ghcnd_all.tar.gz')])

        print "Unzipping ghcnd_all.tar.gz..."
        subprocess.call(['gunzip', '-f',
                         os.path.join(local_path, 'ghcnd_all.tar.gz')])

        by_yr_dir = os.path.join(local_path, 'by_year')

        if not os.path.isdir(by_yr_dir):
            os.mkdir(by_yr_dir)

        print "Downloading yearly files..."
        subprocess.call(['wget', '-N', '--directory-prefix=' + by_yr_dir,
                         urljoin(_RPATH_GHCND, 'by_year/*.csv.gz')])

    def _parse_stn_obs(self, stn_id):

        fname = os.path.join('ghcnd_all', '%s.dly' % stn_id)

        obs_file = self._obs_tarfile.extractfile(fname)
        lines = obs_file.readlines()
        obs_file.close()
        del obs_file

        all_obs = []

        for line in lines:

            year = int(line[11:15])
            month = int(line[15:17])
            element = line[17:21].strip().lower()

            if element in self.elems:

                offset = 0

                for day in _MONTH_DAYS:

                    try:
                        # throw error if not valid date
                        a_date = datetime.date(year, month, day)
                    except ValueError:
                        # Indicates invalid date, do not insert a record
                        offset += _OBS_COLUMN_SIZE
                        continue

                    value = _convert_units(element,
                                           float(line[21 + offset:offset + 26]))
                    qflag = line[27 + offset:offset + 28].strip()

                    all_obs.append((element, a_date, value, qflag))

                    offset += _OBS_COLUMN_SIZE

        df_obs = pd.DataFrame(all_obs, columns=['elem', 'time', 'obs_value',
                                                'qa_flag'])

        mask_good = (df_obs.obs_value != _MISSING) & (df_obs.qa_flag == '')

        df_obs = df_obs[mask_good]
        df_obs = df_obs.drop('qa_flag', axis=1)
        df_obs['time'] = pd.DatetimeIndex(df_obs.time)
        df_obs = df_obs.reset_index(drop=True)

        return df_obs

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

                    mask_time = ((obs_stn.time >= self.start_date) &
                                 (obs_stn.time <= self.end_date))
                    obs_stn = obs_stn[mask_time]

                obs_stn['station_id'] = a_id

                obs.append(obs_stn)

            df_obs = pd.concat(obs, ignore_index=True)

            if np.any(np.char.startswith(np.array(self.elems), 'tobs')):

                df_tobs = self._df_tobs[self._df_tobs.station_id.
                                        isin(stns_obs.station_id)].copy()

                df_obs = pd.concat([df_obs, df_tobs], ignore_index=True)

        finally:

            pd.set_option('mode.chained_assignment', opt_val)

        df_obs = df_obs.set_index(['station_id', 'elem', 'time'])
        df_obs = df_obs.sortlevel(0, sort_remaining=True)

        return df_obs
