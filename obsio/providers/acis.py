from .generic import ObsIO
import itertools
import json
import numpy as np
import pandas as pd
import urllib
import urllib2

_URL_STN_META = 'http://data.rcc-acis.org/StnMeta'

_URL_MULTI_STN_DATA = 'http://data.rcc-acis.org/MultiStnData'

_SID_NAMES = ['sid_ghcn', 'sid_coop', 'sid_wban', 'sid_wmo', 'sid_faa',
              'sid_cocorahs', 'sid_icao', 'sid_nwsli', 'sid_rcc',
              'sid_threadex']

_SID_CODE_TO_NAME = {1: 'sid_wban', 2: 'sid_coop', 3: 'sid_faa', 4: 'sid_wmo',
                     5: 'sid_icao', 6: 'sid_ghcn', 7: 'sid_nwsli', 8: 'sid_rcc',
                     9: 'sid_threadex', 10: 'sid_cocorahs'}

_ELEMS_TO_ACIS = {'tmin': 'mint', 'tmax': 'maxt', 'prcp': 'pcpn',
                  'tobs_tmin': 'tobs_tmin', 'tobs_tmax': 'tobs_tmax',
                  'tobs_prcp': 'tobs_prcp'}

_ACIS_TO_ELEMS = {a_value: a_key for a_key, a_value in _ELEMS_TO_ACIS.items()}

_f_to_c = lambda f: (f - 32.0) / 1.8
_in_to_mm = lambda i: i * 25.4
_ft_to_m = lambda ft: ft * 0.3048
_multi_100 = lambda x: x * 100

_convert_funcs = {'tmin': _f_to_c, 'tmax': _f_to_c, 'prcp': _in_to_mm,
                  'tobs_tmin': _multi_100, 'tobs_tmax': _multi_100,
                  'tobs_prcp': _multi_100}


def _get_stnids(stns):

    stn_ids = np.empty(len(stns))
    stn_ids.fill(np.nan)

    stn_ids = pd.Series(stn_ids, index=stns.index, dtype=np.object)

    null_ids = True

    cnames_sids = stns.columns[np.char.
                               startswith(stns.columns.values.astype(np.str),
                                          prefix='sid')].values.astype(np.str)

    cnames_uk_sids = cnames_sids[~np.in1d(cnames_sids, _SID_NAMES,
                                          assume_unique=True)]

    sid_names = np.concatenate([_SID_NAMES, cnames_uk_sids])

    for a_sidname in sid_names:

        stn_ids.fillna(stns[a_sidname], inplace=True)

        if stn_ids.isnull().sum() == 0:

            null_ids = False
            break

    if null_ids:
        raise ValueError("Stations with no valid station id.")

    return stn_ids


class AcisObsIO(ObsIO):

    _avail_elems = ['tmin', 'tmax', 'prcp', 'tobs_tmin', 'tobs_tmax',
                    'tobs_prcp']
    _requires_local = False

    def __init__(self, **kwargs):

        super(AcisObsIO, self).__init__(**kwargs)

        self._elems_acis = np.array([_ELEMS_TO_ACIS[a_elem]
                                     for a_elem in self.elems])
        self._elems_acis = list(self._elems_acis[~np.char.startswith(self._elems_acis,
                                                                     "tobs_")])

        if self.has_start_end_dates:

            if self.start_date == self.end_date:

                raise NotImplementedError("AcisObsIO: Single date retrievals "
                                          "are not yet supported")

    def _read_stns(self):

        print "AcisObsIO: Getting station metadata..."

        input_dict = {'elems': self._elems_acis, "meta": ["name", "state", "sids",
                                                          "ll", "elev", "uid",
                                                          "valid_daterange"]}

        if self.has_start_end_dates:

            input_dict['sdate'] = self.start_date.strftime('%Y-%m-%d')
            input_dict['edate'] = self.end_date.strftime('%Y-%m-%d')

        if self.bbox is not None:

            input_dict['bbox'] = "%.10f,%.10f,%.10f,%.10f" % (self.bbox.west,
                                                              self.bbox.south,
                                                              self.bbox.east,
                                                              self.bbox.north)

        params = urllib.urlencode({'params': json.dumps(input_dict)})
        req = urllib2.Request(
            _URL_STN_META, params, {'Accept': 'application/json'})
        response = urllib2.urlopen(req)
        json_str = response.read()
        json_dict = json.loads(json_str)['meta']
        df_stns = pd.DataFrame(json_dict)

        cnames_startend = ["".join(x) for x in
                           itertools.product(['start_', 'end_'], self.elems)]

        def get_ranges_series(rng_row):

            rngs = pd.Series([np.NaN] * len(cnames_startend),
                             index=cnames_startend)

            for i, a_elem in enumerate(self.elems):

                for x, prfx in enumerate(['start_', 'end_']):

                    try:
                        rngs["".join([prfx, a_elem])] = rng_row[i][x]
                    except IndexError:
                        continue

            return pd.to_datetime(rngs)

        def get_sids_series(a_sids):

            sids = pd.Series([np.NaN] * len(_SID_NAMES), index=_SID_NAMES,
                             dtype=np.object)

            for a_sid in a_sids:

                a_id, a_code = a_sid.split()
                a_code = int(a_code)

                dup_num = 2

                try:
                    while 1:

                        if not pd.isnull(sids[_SID_CODE_TO_NAME[a_code]]):
                            # more than one ID for this code
                            sid_name = _SID_CODE_TO_NAME[
                                a_code] + "%.2d" % dup_num
                            if sid_name in sids.index:
                                dup_num += 1
                                continue
                            else:
                                sids[sid_name] = a_sid
                                break
                        else:
                            sids[_SID_CODE_TO_NAME[a_code]] = a_sid
                            break
                except KeyError:
                    sids['sid_unknown%d' % a_code] = a_sid

            return sids

        df_sids = df_stns.sids.apply(get_sids_series)
        df_rngs = df_stns.valid_daterange.apply(get_ranges_series)

        df_stns = pd.merge(df_stns, df_sids, left_index=True, right_index=True)
        df_stns = pd.merge(df_stns, df_rngs, left_index=True, right_index=True)

        df_ll = pd.DataFrame(list(df_stns.ll.values), index=df_stns.index,
                             columns=['longitude', 'latitude'])

        df_stns = pd.merge(df_stns, df_ll, left_index=True, right_index=True)

        df_stns = df_stns.drop(['sids', 'll', 'valid_daterange'], axis=1)

        df_stns = df_stns.rename(columns={'elev': 'elevation'})
        df_stns['elevation'] = _ft_to_m(df_stns['elevation'])

        df_stns['provider'] = 'ACIS'

        def get_sid_name(code):

            try:
                return _SID_CODE_TO_NAME[code]
            except KeyError:
                return '_'

        stnids = _get_stnids(df_stns)
        sid_codes = (stnids.str.split(expand=True)[1]).astype(np.int)
        sub_provid = (sid_codes.apply(get_sid_name).
                      str.split('_', expand=True)[1].str.upper())
        df_stns['sub_provider'] = sub_provid
        df_stns['station_id'] = df_stns['uid'].astype(np.str)
        df_stns = df_stns.rename(columns={'name': 'station_name'})
        df_stns = df_stns.set_index('station_id', drop=False)

        return df_stns

    def read_obs(self, stns_ids=None):

        if stns_ids is None:
            stns_obs = self.stns
        else:
            stns_obs = self.stns.loc[stns_ids]

        stn_ids = _get_stnids(stns_obs)

        elem_dicts = [{'name': a_name, 'add': 'f,t'} for a_name in
                      self._elems_acis]

        if self.has_start_end_dates:

            start_date = self.start_date.strftime("%Y-%m-%d")
            end_date = self.end_date.strftime("%Y-%m-%d")

        else:

            cnames_startend = ["".join(x) for x in
                               itertools.product(['start_', 'end_'], self.elems)]

            start_date = stns_obs[cnames_startend].min(axis=1).min()
            end_date = stns_obs[cnames_startend].max(axis=1).max()

            try:

                start_date = start_date.strftime("%Y-%m-%d")
                end_date = end_date.strftime("%Y-%m-%d")

            except ValueError:

                start_date = "%d-%02d-%02d" % (start_date.year,
                                               start_date.month, start_date.day)

                end_date = "%d-%02d-%02d" % (end_date.year,
                                             end_date.month, end_date.day)

        input_dict = {'sids': list(stn_ids.values.astype(np.str)),
                      'sdate': start_date, 'edate': end_date,
                      'elems': elem_dicts}

        params = urllib.urlencode({'params': json.dumps(input_dict)})
        req = urllib2.Request(_URL_MULTI_STN_DATA, params,
                              {'Accept': 'application/json'})
        response = urllib2.urlopen(req)
        json_str = response.read()
        json_dict = json.loads(json_str)['data']

        dt_i = pd.DatetimeIndex(pd.date_range(start_date, end_date, freq='D'))

        obs_all = []

        elems_no_tobs = (list(np.array(self.elems)
                              [~np.char.startswith(self.elems,
                                                   prefix='tobs')]))

        for a_data in json_dict:

            obs_stn = pd.DataFrame(a_data['data'], columns=elems_no_tobs)

            for a_elem in elems_no_tobs:

                cname_flag = '%s_flag' % a_elem
                cname_tobs = 'tobs_%s' % a_elem

                obs_elem = pd.DataFrame(list(obs_stn[a_elem]),
                                        columns=[a_elem, cname_flag, cname_tobs])
                obs_elem['time'] = dt_i
                obs_elem.loc[obs_elem[a_elem] == 'M', a_elem] = np.nan
                obs_elem[a_elem] = obs_elem[a_elem].astype(np.float)
                obs_elem[a_elem] = _convert_funcs[a_elem](obs_elem[a_elem])

                obs_elem.loc[obs_elem[cname_tobs] == -1, cname_tobs] = np.nan
                obs_elem[cname_tobs] = (_convert_funcs[cname_tobs]
                                        (obs_elem[cname_tobs]))
                for a_vname in [a_elem, cname_tobs]:

                    obs_var = obs_elem[['time', a_vname]].copy()
                    obs_var['elem'] = a_vname
                    obs_var['uid'] = a_data['meta']['uid']
                    obs_var = obs_var.rename(columns={a_vname: 'obs_value'})

                    obs_all.append(obs_var)

        obs_all = pd.concat(obs_all, ignore_index=True)
        obs_all = obs_all[~obs_all.obs_value.isnull()]
        obs_all = obs_all[obs_all.elem.isin(self.elems)]
        obs_all['uid'] = obs_all['uid'].astype(np.str)
        obs_all = obs_all.rename(columns={'uid': 'station_id'})

        obs_all = obs_all.set_index(['station_id', 'elem', 'time'])
        obs_all = obs_all.sortlevel(0, sort_remaining=True)

        return obs_all
