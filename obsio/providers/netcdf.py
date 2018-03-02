from .generic import ObsIO
import numpy as np
import pandas as pd
import xarray as xr


class NcObsIO(ObsIO):
    
    """ObsIO to read observations from a local NetCDF store created by ObsIO.to_netcdf
    """

    def __init__(self, fpath, elems):
        """
        Parameters
        ----------
        fpath : str
            The local file path of the NetCDF store
        elems : list
            Observation elements to load from NetCDF store when read_obs is
            called
        """
        
        self.ds = xr.open_dataset(fpath)
        self.elems = elems        
        self._stns = None
        
    def _read_stns(self):
        
        vnames = np.array(list(self.ds.variables.keys())) 
        is_stn_var = np.array([self.ds[avar].dims==('station_id',)
                               for avar in vnames])
        vnames = vnames[is_stn_var]
        
        stns = self.ds[list(vnames)].to_dataframe()
        stns['station_id'] = stns.index
        stns['station_index'] = np.arange(len(stns))
        
        # Make sure all object columns are str
        stns.loc[:, stns.dtypes == object] = stns.loc[:, stns.dtypes == object].astype(np.str) 
        stns = stns.set_index('station_id', drop=False)

        return stns
        
    def _read_obs(self, stns_ids=None):
        
        if stns_ids is None:
            
            stns_ids = self.stns.station_id
        
        obs = []
        
        for aelem in self.elems:
            
            obs_df = (pd.DataFrame(self.ds[aelem].loc[:, list(stns_ids)].
                                   to_pandas().stack(dropna=False)))
            obs_df['elem'] = aelem
            obs_df = obs_df.rename(columns={0:'obs_value'})
            obs_df = obs_df.set_index('elem', append=True)
            obs.append(obs_df)
            
        obs = pd.concat(obs)
        
        obs = obs.reorder_levels(['station_id', 'elem',
                                  'time']).sortlevel(0, sort_remaining=True)
        
        return obs
    
    def close(self):
        self.ds.close()
        self.ds = None
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    
    
