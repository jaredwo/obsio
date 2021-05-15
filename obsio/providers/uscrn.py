from .generic import ObsIO
from ..util.misc import open_remote_file
import pandas as pd
import numpy as np

_COLSPECS = [(20,20+13),(89,89+7)]
_COLNAMES = ['time_local','prcp']

def _parse_uscrn_obs(fpath, stnid):
    
    obs = pd.read_fwf(fpath, colspecs=_COLSPECS, header=None,
                      names=_COLNAMES, na_values=[-9999.0, -99.000])
    obs['time_local'] = pd.to_datetime(obs.time_local)
    obs = obs.rename(columns={'time_local':'time', 'prcp':'obs_value'})
    obs['elem'] = 'prcp'
    obs['station_id'] = stnid
    
    return obs.dropna()
        
class UscrnObsIO(ObsIO):

    _avail_elems = ['prcp']
    _requires_local = False
    name = 'USCRN'

    def __init__(self, **kwargs):

        super(UscrnObsIO, self).__init__(**kwargs)
        
    def _read_stns(self):
        
        stns = pd.read_table(open_remote_file('https://www1.ncdc.noaa.gov/pub/data/uscrn/products/stations.tsv'))
        stns = stns.rename(columns={a[0]:a[1] for a in zip(stns.columns,stns.columns.str.lower())})
        stns['station_id'] = stns.state + "_" + stns.location + '_' + stns.vector
        stns['station_id'] = stns['station_id'].str.replace(' ', '_')
        stns = stns.rename(columns={'name':'station_name'})
        stns = stns.set_index('station_id', drop=False)
        stns['provider'] = 'USCRN'
        stns['sub_provider'] = ''
        stns['end_date'] = pd.to_datetime(stns.closing)
        stns['end_date'] = stns.end_date.fillna(pd.Timestamp.now())
        stns['commissioning'] = pd.to_datetime(stns.commissioning)
        
        # For now, only return stations that are commissioned
        stns = stns[stns.status=='Commissioned'].copy()
            
        if self.bbox is not None:
    
            mask_bnds = ((stns.latitude >= self.bbox.south) & 
                         (stns.latitude <= self.bbox.north) & 
                         (stns.longitude >= self.bbox.west) & 
                         (stns.longitude <= self.bbox.east))

            stns = stns[mask_bnds].copy()

        if self.has_start_end_dates:
        
            mask_por = (((self.start_date <= stns.commissioning) & 
                         (stns.commissioning <= self.end_date)) | 
                        ((stns.commissioning <= self.start_date) & 
                         (self.start_date <= stns.end_date)))
    
            stns = stns[mask_por].copy()

        return stns
    
    def _read_obs(self, stns_ids=None):

        if stns_ids is None:
            stns_obs = self.stns
        else:
            stns_obs = self.stns.loc[stns_ids]
                
        if self.has_start_end_dates:
            yrs = np.arange(self.start_date.year, self.end_date.year+1)
        else:
            yrs = np.arange(2000, pd.Timestamp.now().year+1)
    
        obs_all = []

        for a_id in stns_obs.station_id:
            
            yrs_stn = np.arange(stns_obs.loc[a_id].commissioning.year,
                                stns_obs.loc[a_id].end_date.year+1)
            
            yrs_stn = yrs_stn[np.in1d(yrs_stn, yrs)]
            
            obs_stn  = []
            
            for yr in yrs_stn:
                
                print(a_id,yr)
                
                url = ('https://www1.ncdc.noaa.gov/pub/data/uscrn/products/'
                       'hourly02/%d/CRNH0203-%d-%s.txt')%(yr,yr,a_id)
                abuf = open_remote_file(url)
                obs_stn.append(_parse_uscrn_obs(abuf,a_id))
            
            obs_stn = pd.concat(obs_stn, ignore_index=True)
            
            obs_all.append(obs_stn)
             
        obs_all = pd.concat(obs_all, ignore_index=True)
        obs_all = obs_all.set_index(['station_id', 'elem', 'time'])
        obs_all = obs_all.sort_index(0, sort_remaining=True)

        return obs_all
