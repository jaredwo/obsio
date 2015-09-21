from .generic import ObsIO
from suds.client import Client
from time import sleep
import pandas as pd

_URL_AWDB_WSDL = 'http://www.wcc.nrcs.usda.gov/awdbWebService/services?WSDL'

_ELEMS_TO_NRCS = {'tmin': 'TMIN', 'tmax': 'TMAX', 'prcp': 'PRCP',
                  'snwd': 'SNWD', 'swe': 'WTEQ'}

_NRCS_TO_ELEMS = {a_value: a_key for a_key, a_value in _ELEMS_TO_NRCS.items()}

_f_to_c = lambda f: (f - 32.0) / 1.8
_in_to_mm = lambda i: i * 25.4
_ft_to_m = lambda ft: ft * 0.3048

_convert_funcs = {'TMIN': _f_to_c, 'TMAX': _f_to_c, 'PRCP': _in_to_mm,
                  'SNWD': _in_to_mm, 'WTEQ': _in_to_mm}


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


class NrcsObsIO(ObsIO):

    _avail_elems = ['tmin', 'tmax', 'prcp', 'snwd', 'swe']
    _requires_local = False

    def __init__(self, **kwargs):

        super(NrcsObsIO, self).__init__(**kwargs)

        self._elems_nrcs = [_ELEMS_TO_NRCS[a_elem] for a_elem in self.elems]

        self._client = Client(_URL_AWDB_WSDL)
        self._stnmeta_attrs = (self._client.factory.
                               create('stationMetaData').__keylist__)

    def _read_stns(self):

        if self.bbox is None:

            stn_triplets = _execute_awdb_call(self._client.service.getStations,
                                              logicalAnd=True,
                                              networkCds=['SNTL', 'SCAN'],
                                              elementCds=self._elems_nrcs)

        else:

            stn_triplets = _execute_awdb_call(self._client.service.getStations,
                                              logicalAnd=True,
                                              minLatitude=self.bbox.south,
                                              maxLatitude=self.bbox.north,
                                              minLongitude=self.bbox.west,
                                              maxLongitude=self.bbox.east,
                                              networkCds=['SNTL', 'SCAN'],
                                              elementCds=self._elems_nrcs)

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
        stns.beginDate = pd.to_datetime(stns.beginDate)
        stns.endDate = pd.to_datetime(stns.endDate)
        stns.elevation = _ft_to_m(stns.elevation)
        stns['provider'] = 'NRCS'
        stns['sub_provider'] = ''
        stns = stns.sort('station_id')

        if self.has_start_end_dates:

            mask_dates = (((self.start_date <= stns.beginDate) &
                           (stns.beginDate <= self.end_date)) |
                          ((stns.beginDate <= self.start_date) &
                           (self.start_date <= stns.endDate)))

            stns = stns[mask_dates].copy()

        stns = stns.reset_index(drop=True)
        stns = stns.set_index('station_id', drop=False)

        return stns

    def read_obs(self, stns_ids=None):

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

        begin = start_date_obs.strftime("%Y-%m-%d")
        end = end_date_obs.strftime("%Y-%m-%d")

        obs_all = []

        for a_elem in self._elems_nrcs:

            datas = _execute_awdb_call(self._client.service.getData,
                                       stationTriplets=list(stns_obs.
                                                            stationTriplet),
                                       elementCd=[a_elem],
                                       ordinal=1,
                                       duration='DAILY',
                                       getFlags=False,
                                       beginDate=begin,
                                       endDate=end,
                                       alwaysReturnDailyFeb29=False)

            series_ls = []

            for a_data in datas:

                try:
                    a_series = pd.Series(a_data.values,
                                         index=pd.date_range(a_data.beginDate,
                                                             a_data.endDate),
                                         name=a_data.stationTriplet)
                    series_ls.append(a_series)
                except AttributeError:
                    continue

            try:
                obs = pd.concat(series_ls, axis=1)
            except ValueError:
                continue  # no observations

            obs = _convert_funcs[a_elem](obs)

            obs = obs.stack(dropna=True)
            obs.index.set_names(['time', 'stationTriplet'], inplace=True)
            obs = obs.reset_index()
            obs = obs.rename(columns={0: 'obs_value'})
            obs['elem'] = _NRCS_TO_ELEMS[a_elem]
            obs_all.append(obs)

        obs_all = pd.concat(obs_all, ignore_index=True)

        # Replace stationTriplet with station_id
        obs_merge = pd.merge(obs_all, self.stns[['station_id',
                                                 'stationTriplet']],
                             how='left', on='stationTriplet', sort=False)

        if obs_merge.shape[0] != obs_all.shape[0]:
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
