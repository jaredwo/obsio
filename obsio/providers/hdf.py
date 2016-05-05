from .generic import ObsIO
import numpy as np
import pandas as pd


class HdfObsIO(ObsIO):
    
    """ObsIO to read observations from a local HDF5 store created by ObsIO.to_hdf
    """

    def __init__(self, fpath):
        """
        Parameters
        ----------
        fpath : str
            The local file path of the HDF5 store
        """
        
        self.store = pd.HDFStore(fpath)
        attrs = self.store.get_storer('stns').attrs
        self.elems = attrs.elems
        self.start_date = attrs.start_date
        self.end_date = attrs.end_date
        self.bbox = attrs.bbox
        self.name = attrs.name
        
        self._stns = None
        
    def _read_stns(self):
        
        return self.store.select('stns')
        
    def _read_obs(self, stns_ids=None):
        
        if stns_ids is None:
            
            obs = self.store.select('obs')
            
        else:
            
            obs = []
            
            # HDFStore can only read in chunks of 31
            stn_chk = 31
            
            for i in np.arange(len(stns_ids), step=stn_chk):
            
                stnids = stns_ids[i:(i+stn_chk)]
                obs_chk = self.store.select('obs', 'index=stnids')
                obs.append(obs_chk)

            obs = pd.concat(obs)

        obs = obs.set_index('time', append=True).stack()
        obs.name = 'obs_value'
        obs.index.rename('elem', level=2, inplace=True)
        obs = obs.reorder_levels(['station_id', 'elem',
                                  'time']).sortlevel(0, sort_remaining=True)
        obs = pd.DataFrame(obs)
        
        return obs
    
    def close(self):
        self.store.close()
        self.store = None
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    
    
