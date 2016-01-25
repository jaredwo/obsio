import numpy as np
import pandas as pd
import xray


class ObsIO(object):
    """Interface to access stations and observations from a data source.

    ObsIOs are obtained from obsio.ObsioFactory.

    Attributes
    ----------
    stns : pandas.Dataframe
        The stations for this ObsIO.
    requires_local : boolean
        Does ObsIO require data to be downloaded and stored locally

    Methods
    ----------
    read_obs
        Access observations for a set of stations.
    download_local
        Download required data and store locally
    """

    def __init__(self, elems, bbox=None, start_date=None, end_date=None):

        self.has_start_end_dates = self._check_start_end(start_date, end_date)
        self.elems = self._check_elems(elems)

        self.bbox = bbox
        self.start_date = start_date
        self.end_date = end_date

        self._stns = None

    def _check_start_end(self, start_date, end_date):

        if start_date is None and end_date is None:
            return False

        if type(start_date) is not pd.Timestamp:
            raise TypeError("Expected pandas.Timestamp for start_date. Got: %s" %
                            str(type(start_date)))

        if type(end_date) is not pd.Timestamp:
            raise TypeError("Expected pandas.Timestamp for end_date. Got: %s" %
                            str(type(end_date)))

        return True

    @classmethod
    def _check_elems(cls, elems):

        elems = np.array(elems)
        mask_avail = np.in1d(elems, cls._avail_elems)
        elems_avail = elems[mask_avail]
        elems_not_avail = elems[~mask_avail]

        if elems_not_avail.size == elems.size:

            raise ValueError('%s does not support any of the requested '
                             'elements: %s. Supported elements are: %s' %
                             (cls.__name__, ",".join(elems),
                              ",".join(cls._avail_elems)))

        elif elems_not_avail.size > 0:

            print ("Warning: %s does not support the following requested "
                   "elements: %s" % (cls.__name__, ",".join(elems_not_avail)))

        return elems_avail

    @property
    def stns(self):
        """The stations for this ObsIO as a pandas DataFrame.

        The station DataFrame is loaded lazily. The exact station DataFrame
        columns can differ by ObsIO, but all ObsIO station DataFrames will 
        contain the following standard columns:
        - station_id : The unique identifier of the station (string). This
                       also serves as the DataFrame index.
        - station_name : The name of the station.
        - latitude : The latitude of the station (decimal degrees).
        - longitude : The longitude of the station (decimal degrees).
        - elevation : The elevation of the station (meters).
        - provider : The data provider (e.g.- GHCND, ACIS, NRCS, etc.).
        - sub_provider : The data sub-provider, if any.         
        """

        if self._stns is None:

            self._stns = self._read_stns()

        return self._stns

    @property
    def requires_local(self):
        """Does ObsIO require data to be downloaded and stored locally
        """
        return self._requires_local

    def download_local(self):
        """Force a download of required data and store locally
        
        Some ObsIO data sources are not web services and may require data to be
        downloaded and stored locally.
        """
        raise NotImplementedError

    def _read_stns(self):
        raise NotImplementedError
    
    def _read_obs(self, stn_ids):
        raise NotImplementedError
    
    def read_obs(self, stn_ids=None, data_structure='stacked'):
        """Access observations for a set of stations.

        Parameters
        ----------
        stn_ids : list of str, optional
            The station ids for which to access observations. If not specified,
            will return observations for all available stations according to
            the parameters specified for the ObsIO (elements, spatial bounding
            box, date range, etc.).
        data_structure : str, optional
            The data structure of the returned observations. One of: 'stacked',
            'tidy', or 'array'. Default: 'stacked'. Data structure definitions:
            - stacked (default) :
                A pandas.DataFrame where observations are indexed by a 3 level
                multiindex: station_id, elem, time. The "obs_value" column
                contains the actual observation values.
            - tidy : 
                A pandas.DataFrame where observations are indexed by a 2 level
                multiindex: station_id, time. Column(s) are values for
                the requested elements. Meets the criteria of tidy data where
                each column is a different variable and each row is a different
                observation (http://www.jstatsoft.org/v59/i10/)
            - array :
                A xray.Dataset with separate 2D arrays of observations for each
                element. The 2D observation arrays are of dimension: (time, station_id).
                Corresponding station metadata are also included as arrays.
                 
        Returns
        ----------
        pandas.DataFrame or xray.Dataset
            The observations as a pandas.DataFrame or xray.Dataset dependent on
            the requested data structure.
        """
        
        obs = self._read_obs(stn_ids)

        if data_structure == 'stacked':
            
            #All providers already return data as stacked.
            return obs
        
        elif data_structure == 'tidy':
            
            obs = obs.unstack(level=1)
            obs.columns = [col[1] for col in obs.columns.values]
            return obs
        
        elif data_structure == 'array':
            
            obs = obs.unstack(level=1)
            obs.columns = [col[1] for col in obs.columns.values]
            obs = xray.Dataset.from_dataframe(obs.swaplevel(0,1))
            
            if stn_ids is None:
                stns = self.stns
            else:
                stns = self.stns.loc[stn_ids]
            
            #include station metadata
            obs.merge(xray.Dataset.from_dataframe(stns), inplace=True)
            
            return obs
        
        else:
            
            raise ValueError("Unrecognized data format. Expected one of: "
                             "'stacked', 'tidy', 'array'")
