import logging
import numpy as np

# tzwhere currently sets log level to debug when imported
# get log level before import and then reset log level to this
# value after tzwhere import
logger = logging.getLogger()
log_level = logger.level

from tzwhere.tzwhere import tzwhere

logger.setLevel(log_level)

_RADIAN_CONVERSION_FACTOR = 0.017453292519943295 #pi/180
_AVG_EARTH_RADIUS_KM = 6371.009 #Mean earth radius as defined by IUGG

class BBox(object):
    """Spatial bounding box
    
    By default, represents a buffered bounding box around the conterminous U.S.    
    """
    
    def __init__(self, west_lon=-126.0, south_lat=22.0, east_lon=-64.0,
                 north_lat=53.0):

        self.west = west_lon
        self.south = south_lat
        self.east = east_lon
        self.north = north_lat
        
    def remove_outbnds_df(self,df):
        
        mask_bnds = ((df.latitude >= self.south) &
                     (df.latitude <= self.north) &
                     (df.longitude >= self.west) &
                     (df.longitude <= self.east))
            
        df = df[mask_bnds].copy()
        
        return df

class TimeZones():
    
    _tzw = None
    
    def __init__(self):
        
        pass
    
    @property
    def tzw(self):

        if TimeZones._tzw is None:
            
            print "Initializing tzwhere for time zone retrieval...",
            TimeZones._tzw = tzwhere(shapely=True, forceTZ=True)
            print 'done.'

        return TimeZones._tzw
    
    def set_tz(self, df_stns, chck_nghs=True):
        
        try:
            mask_tznull = df_stns['time_zone'].isnull()
        except KeyError:
            df_stns['time_zone'] = np.nan
            mask_tznull = df_stns['time_zone'].isnull()
        
        if mask_tznull.any():
        
            df_stns_notz = df_stns[mask_tznull]
            
            print "Getting timezone for %d stations..."%df_stns_notz.shape[0]
            
            tz_for_null = df_stns_notz.apply(lambda x: 
                                          self.tzw.tzNameAt(x.latitude,
                                                            x.longitude,
                                                            forceTZ=True), 1)
            
            df_stns.loc[mask_tznull,'time_zone'] = tz_for_null.values
        
            mask_tznull = df_stns['time_zone'].isnull()
        
            if mask_tznull.any() and chck_nghs:
    
                print ("Could not determine timezone with tzwhere "
                       "for %d stations. Trying nearest neighbors..." % 
                       (mask_tznull.sum(),))
    
                def ngh_tz(a_row):
    
                    df_stns_ngh = df_stns[
                        (df_stns.index != a_row.name) & (~mask_tznull)]
    
                    d = grt_circle_dist(a_row.longitude,
                                        a_row.latitude,
                                        df_stns_ngh.longitude,
                                        df_stns_ngh.latitude)
    
                    ngh_id = d.idxmin()
    
                    return df_stns.loc[ngh_id,'time_zone']
    
                df_stns_tznull = df_stns[mask_tznull]
                tz_for_null = df_stns_tznull.apply(ngh_tz, 1)
                df_stns.loc[mask_tznull,'time_zone'] = tz_for_null.values
                    
def grt_circle_dist(lon1,lat1,lon2,lat2):
    """Calculate great circle distance according to the haversine formula
    
    See http://en.wikipedia.org/wiki/Great-circle_distance
    """
    #convert to radians
    lat1rad = lat1 * _RADIAN_CONVERSION_FACTOR
    lat2rad = lat2 * _RADIAN_CONVERSION_FACTOR
    lon1rad = lon1 * _RADIAN_CONVERSION_FACTOR
    lon2rad = lon2 * _RADIAN_CONVERSION_FACTOR
    deltaLat = lat1rad - lat2rad
    deltaLon = lon1rad - lon2rad
    centralangle = 2 * np.arcsin(np.sqrt((np.sin (deltaLat/2))**2 +
                                         np.cos(lat1rad) * np.cos(lat2rad)
                                         * (np.sin(deltaLon/2))**2))
    #average radius of earth times central angle, result in kilometers
    #distDeg = centralangle/RADIAN_CONVERSION_FACTOR
    distKm = _AVG_EARTH_RADIUS_KM * centralangle 
    return distKm
