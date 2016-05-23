from ..util.misc import StatusCheck
import netCDF4 as nc
import numpy as np
import os
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
    
    def to_hdf(self, fpath, stn_ids, chk_rw, verbose=True, **kwargs):
        """Write observations to a PyTables HDF5 file
        
        Uses pandas.HDFStore
        
        Parameters
        ----------
        fpath : str
            File path for output HDF5 file
        stn_ids : list of str
            The station ids for which to write observations
        chk_rw : int
            The chunk size in number of stations for which to read observations
            into memory and output to the HDF5 file. For example, a chunk size
            of 50 will read and write observations from 50 stations at a time.
        verbose : boolean, optional
            Print out progress messages. Default: True.
        **kwargs
            Additional keyword arguments for initializing pandas.HDFStore
            (e.g. complevel=5, complib='zlib')
        """
        
        store = pd.HDFStore(fpath, 'w', **kwargs) #complevel=5, complib='zlib')
        stns = self.stns.loc[stn_ids]
        # Make sure all object columns are str and not unicode
        stns.loc[:, stns.dtypes == object] = stns.loc[:, stns.dtypes == object].astype(np.str)
        stns.index = stns.index.astype(np.str)
        store.append('stns', stns)
        store.get_storer('stns').attrs.elems = self.elems
        store.get_storer('stns').attrs.start_date = self.start_date
        store.get_storer('stns').attrs.end_date = self.end_date
        store.get_storer('stns').attrs.bbox = self.bbox
        store.get_storer('stns').attrs.name = self.name
        
        first_append = True
        
        if verbose:
            schk = StatusCheck(len(stn_ids),chk_rw)
        
        for i in np.arange(len(stn_ids),step=chk_rw):
                        
            obs = self.read_obs(stn_ids[i:(i+chk_rw)], 'tidy')
            obs = obs.reset_index()
            # Make sure all object columns are str and not unicode
            obs.loc[:, obs.dtypes == object] = obs.loc[:, obs.dtypes == object].astype(np.str)
            
            obs = obs.set_index('station_id')
            obs = obs.reindex(columns=['time']+list(self.elems))
            
            if first_append:
                
                erows = np.int(np.round(len(obs)*
                                        (len(stn_ids)/np.float(chk_rw))))
                store.append('obs', obs, data_columns=['time'], index=False,
                             expectedrows=erows)
                first_append = False
                
            else:
               
                store.append('obs', obs, data_columns=['time'], index=False)
            
            if verbose:
                schk.increment(chk_rw)
        
        if verbose:
            print "Creating index..."          
    
        store.create_table_index('obs', optlevel=9, kind='full')
        store.create_table_index('obs', columns=['time'], optlevel=9, kind='full')
        
        store.close()
    
    
    def to_csv(self, path_out, stn_ids, chk_rw, verbose=True):
        """Write observations to CSV files
        
        Writes out a station metadata CSV (stns.csv) and an observation
        CSV for each station.
        
        Parameters
        ----------
        path_out : str
            Path for output CSVs
        stn_ids : list of str
            The station ids for which to write observations
        chk_rw : int
            The chunk size in number of stations for which to read observations
            into memory and output. For example, a chunk size
            of 50 will read and write observations from 50 stations at a time.
        verbose : boolean, optional
            Print out progress messages. Default: True.
        """
        
        stns = self.stns.loc[stn_ids]
        stns.drop(['station_id'], axis=1).to_csv(os.path.join(path_out, 'stns.csv'))
                
        if verbose:
            schk = StatusCheck(len(stn_ids),chk_rw)
        
        for i in np.arange(len(stn_ids),step=chk_rw):
                        
            obs = self.read_obs(stn_ids[i:(i+chk_rw)], 'tidy')
            
            ids_out = obs.index.levels[0].values
            
            for a_id in ids_out:
                
                obs_stn = obs.xs(a_id, level='station_id')
                obs_stn = obs_stn.dropna(axis=1, how='all')
                obs_stn = obs_stn.reindex(pd.date_range(obs_stn.index.min(),
                                                        obs_stn.index.max()))
                obs_stn.index.name = 'time'
                obs_stn.to_csv(os.path.join(path_out, 'obs_%s.csv'%a_id))
                
            if verbose:
                schk.increment(chk_rw)
                
    
    def to_netcdf(self, fpath, stn_ids, start_date, end_date, chk_rw,
                  verbose=True):
        """Write observations to a netCDF file
        
        Creates a 2D netCDF observation variable for each element. The 
        first axis is time and the second axis is station id. 
        
        Parameters
        ----------
        fpath : str
            File path for output netCDF file
        stn_ids : list of str
            The station ids for which to write observations
        start_date : pandas.Timestamp
            Start date of desired date range for output observations. Used in
            combination with end_date to set the size of the time dimension.
        end_date : pandas.Timestamp, optional
            End date of desired date range for output observations. Used in
            combination with start_date to set the size of the time dimension.
        chk_rw : int
            The chunk size in number of stations for which to read observations
            into memory and output to the netCDF file. For example, a chunk size
            of 50 will read and write observations from 50 stations at a time.
        verbose : boolean, optional
            Print out progress messages. Default: True.
        """
        
        stns = self.stns.loc[stn_ids].copy()
        
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Create output netcdf file
        ds_out = nc.Dataset(fpath, 'w')
         
        # Create station id dimension and add station metadata
        ds_out.createDimension('station_id', len(stn_ids))
         
        for acol in stns.columns:
        
            adtype = np.str if stns[acol].dtype == np.dtype('O') else stns[acol].dtype
            avar = ds_out.createVariable(acol, adtype, ('station_id',))
             
            if adtype == np.str:
                avar[:] = stns[acol].astype(np.str).values
            else:
                avar[:] = stns[acol].values
        
        ds_out.sync()   
                
        # Add station index number column    
        stns['station_num'] = np.arange(len(stns))
         
        ds_out.createDimension('time', dates.size)
         
        # Create time dimension and variable
        times = ds_out.createVariable('time', 'f8', ('time',), fill_value=False)
        times.long_name = "time"
        times.units = "days since %d-%02d-%02d 0:0:0"%(dates[0].year, dates[0].month,
                                                       dates[0].day)
        times.standard_name = "time"
        times.calendar = "standard"
        times[:] = nc.date2num(dates.to_pydatetime(), times.units)
         
        # Create main element variables. Optimize chunkshape for single time series slices
        elem_vars = {}
        for a_elem in self.elems:
             
            elem_vars[a_elem] = ds_out.createVariable(a_elem, 'f8',
                                                      ('time', 'station_id'),
                                                      chunksizes=(dates.size,1),
                                                      fill_value=nc.default_fillvals['f8'],
                                                      zlib=True)
            elem_vars[a_elem].missing_value = nc.default_fillvals['f8']
        
        ds_out.sync()
        
        if verbose:
            schk = StatusCheck(len(stn_ids),chk_rw)
        
        for i in np.arange(len(stns),step=chk_rw):
        
            a_stns = stns.iloc[i:(i+chk_rw)]
            
            obs = self.read_obs(a_stns.station_id, 'array')            
            obs = obs.reindex(time=dates)
            
            a_stns_num = stns.loc[obs['station_id'].values, 'station_num'].values
            num_sort = np.argsort(a_stns_num)
            a_stns_num = a_stns_num[num_sort]
            
            for a_elem in self.elems:
     
                try:
                    vals = np.ma.filled(np.ma.masked_invalid(obs[a_elem].values),
                                        nc.default_fillvals['f8'])
                except KeyError:
                    continue
                
                vals = np.take(vals, num_sort, axis=1)
                elem_vars[a_elem][:,a_stns_num] = vals
                
            ds_out.sync()
            
            if verbose:
                schk.increment(chk_rw)
        
        ds_out.close()
        