from .generic import ObsIO
import numpy as np
import pandas as pd

class MultiObsIO(ObsIO):
    """ObsIO to combine multiple different provider ObsIOs into a single ObsIO
    
    To avoid duplciate station ids between providers, returned station ids are:
    [ObsIO.name]_[station_id]
    
    """

    def __init__(self, obsios):
        
        self._obsios = obsios
        self._stns = None
        self.elems = np.unique(np.concatenate([a_obsio.elems for 
                                               a_obsio in self._obsios]))
        self.start_date = None
        self.end_date = None
        self.bbox = None
    
    def _read_stns(self):
        
        stns_all = []
        
        for a_obsio in self._obsios:
            
            stns = a_obsio.stns.copy()
            stns['station_id_orig'] = stns.station_id
            stns['station_id'] = a_obsio.name + "_" + stns.station_id
            stns = stns.set_index('station_id', drop=False)
                
            stns_all.append(stns)
            
        stns_all = pd.concat(stns_all, join='inner')
        
        return stns_all
    
    def _read_obs(self, stns_ids=None):

        if stns_ids is None:
            stns_obs = self.stns
        else:
            stns_obs = self.stns.loc[stns_ids]
        
        obs_all = []
        
        for a_obsio in self._obsios:
            
            stns = a_obsio.stns[a_obsio.stns.index.isin(stns_obs.station_id_orig.values)]
            
            if not stns.empty:
                
                obs = a_obsio.read_obs(stns.station_id)
                obs = obs.reset_index()
                obs['station_id'] = a_obsio.name + "_" + obs.station_id
                obs = obs.set_index(['station_id', 'elem', 'time'])
                
                obs_all.append(obs)
        
        obs_all = pd.concat(obs_all)
        
        return obs_all
        
                
