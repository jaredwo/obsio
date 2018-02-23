import json
import logging
import numpy as np
import urllib
import pycurl
from io import BytesIO
from gzip import GzipFile
from time import sleep
from datetime import datetime
import os
import sys
import time

# tzwhere currently sets log level to debug when imported
# get log level before import and then reset log level to this
# value after tzwhere import
logger = logging.getLogger()
log_level = logger.level

from tzwhere.tzwhere import tzwhere

logger.setLevel(log_level)

_RADIAN_CONVERSION_FACTOR = 0.017453292519943295  # pi/180
_AVG_EARTH_RADIUS_KM = 6371.009  # Mean earth radius as defined by IUGG


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

    def remove_outbnds_df(self, df):

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

            print("Initializing tzwhere for time zone retrieval..."),
            TimeZones._tzw = tzwhere(shapely=True, forceTZ=True)
            print('done.')

        return TimeZones._tzw

    def set_tz(self, df_stns, chck_nghs=True):

        try:
            mask_tznull = df_stns['time_zone'].isnull()
        except KeyError:
            df_stns['time_zone'] = np.nan
            mask_tznull = df_stns['time_zone'].isnull()

        if mask_tznull.any():

            df_stns_notz = df_stns[mask_tznull]

            print("Getting timezone for %d stations..." % df_stns_notz.shape[0])

            tz_for_null = df_stns_notz.apply(lambda x:
                                             self.tzw.tzNameAt(x.latitude,
                                                               x.longitude,
                                                               forceTZ=True), 1)

            df_stns.loc[mask_tznull, 'time_zone'] = tz_for_null.values

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

                    return df_stns.loc[ngh_id, 'time_zone']

                df_stns_tznull = df_stns[mask_tznull]
                tz_for_null = df_stns_tznull.apply(ngh_tz, 1)
                df_stns.loc[mask_tznull, 'time_zone'] = tz_for_null.values


def grt_circle_dist(lon1, lat1, lon2, lat2):
    """Calculate great circle distance according to the haversine formula

    See http://en.wikipedia.org/wiki/Great-circle_distance
    """
    # convert to radians
    lat1rad = lat1 * _RADIAN_CONVERSION_FACTOR
    lat2rad = lat2 * _RADIAN_CONVERSION_FACTOR
    lon1rad = lon1 * _RADIAN_CONVERSION_FACTOR
    lon2rad = lon2 * _RADIAN_CONVERSION_FACTOR
    deltaLat = lat1rad - lat2rad
    deltaLon = lon1rad - lon2rad
    centralangle = 2 * np.arcsin(np.sqrt((np.sin(deltaLat / 2))**2 +
                                         np.cos(lat1rad) * np.cos(lat2rad)
                                         * (np.sin(deltaLon / 2))**2))
    # average radius of earth times central angle, result in kilometers
    #distDeg = centralangle/RADIAN_CONVERSION_FACTOR
    distKm = _AVG_EARTH_RADIUS_KM * centralangle
    return distKm


def uniquify(items, fudge_start=1, dup_format='_dup%.2d'):
    # http://stackoverflow.com/questions/19071622/
    # automatically-rename-columns-to-ensure-they-are-unique
    seen = set()
    dup_format = "%s" + dup_format

    for item in items:
        fudge = fudge_start - 1
        newitem = item

        while newitem in seen:
            fudge += 1
            newitem = dup_format % (item, fudge)

        yield newitem
        seen.add(newitem)


def get_elevation(lon, lat, usrname_geonames=None):

    def get_elev_usgs(lon, lat):
        """Get elev value from USGS NED 1/3 arc-sec DEM.

        http://ned.usgs.gov/epqs/
        """

        URL_USGS_NED = 'http://ned.usgs.gov/epqs/pqs.php'
        USGS_NED_NODATA = -1000000

        # url GET args
        values = {'x': lon,
                  'y': lat,
                  'units': 'Meters',
                  'output': 'json'}

        data = urllib.urlencode(values)

        req = urllib2.Request(URL_USGS_NED, data)
        response = urllib2.urlopen(req)

        json_response = json.loads(response.read())
        elev = np.float(json_response['USGS_Elevation_Point_Query_Service']
                        ['Elevation_Query']['Elevation'])

        if elev == USGS_NED_NODATA:

            elev = np.nan

        return elev

    def get_elev_geonames(lon, lat):
        """Get elev value from geonames web sevice (SRTM or ASTER)
        """

        URL_GEONAMES_SRTM = 'http://api.geonames.org/srtm3'
        URL_GEONAMES_ASTER = 'http://api.geonames.org/astergdem'

        url = URL_GEONAMES_SRTM

        while 1:
            # ?lat=50.01&lng=10.2&username=demo
            # url GET args
            values = {'lat': lat, 'lng': lon, 'username': usrname_geonames}

            # encode the GET arguments
            data = urllib.urlencode(values)

            # make the URL into a qualified GET statement
            get_url = "".join([url, "?", data])

            req = urllib2.Request(url=get_url)
            response = urllib2.urlopen(req)
            elev = float(response.read().strip())

            if elev == -32768.0 and url == URL_GEONAMES_SRTM:
                # Try ASTER instead
                url = URL_GEONAMES_ASTER
            else:
                break

        if elev == -32768.0 or elev == -9999.0:
            elev = np.nan

        return elev

    elev = get_elev_usgs(lon, lat)

    if np.isnan(elev) and usrname_geonames:

        elev = get_elev_geonames(lon, lat)

    return elev

def open_remote_gz(url, maxtries=3):

    ntries = 0
        
    while 1:
    
        try:
            
            buf = BytesIO()
            c = pycurl.Curl()
            c.setopt(pycurl.URL, url)
            c.setopt(pycurl.WRITEDATA, buf)
            c.setopt(pycurl.FAILONERROR, True)
            c.perform()
            c.close()
            buf.seek(0)
            break
            
        except pycurl.error:
            
            ntries += 1
        
            if ntries >= maxtries:
        
                raise

            sleep(1)
    
    return GzipFile(fileobj=buf, mode='rb')

def open_remote_file(url, maxtries=3):
    
    ntries = 0
    
    while 1:
    
        try:
    
            buf = BytesIO()
            c = pycurl.Curl()
            c.setopt(pycurl.URL, url)
            c.setopt(pycurl.WRITEDATA, buf)
            c.setopt(pycurl.FAILONERROR, True)
            c.perform()
            c.close()
            buf.seek(0)
            break
            
        except pycurl.error:
            
            ntries += 1
        
            if ntries >= maxtries:
        
                raise
        
            sleep(1)
    
    return buf

def download_if_new_ftp(a_ftp, fpath_ftp, fpath_local):
    """Download file from FTP if newer than local copy
    
    Parameters
    ----------
    a_ftp : ftplib.FTP object
    fpath_ftp : str
        Path to file on FTP
    fpath_local : str
        Path to file stored locally
            
    Returns
    ----------
    boolean
        Returns True if a new file was downloaded, otherwise False
    """
    
    downloaded_file = False
    
    mtime_remote = datetime.strptime(a_ftp.sendcmd("MDTM %s"%fpath_ftp),
                                     "213 %Y%m%d%H%M%S")
    #need to set to binary for size cmd
    a_ftp.voidcmd("TYPE I")
    size_remote = a_ftp.size(fpath_ftp)
    
    try:
    
        mtime_local = datetime.utcfromtimestamp(os.path.getmtime(fpath_local))
        size_local = os.path.getsize(fpath_local)
    
    except OSError as e:
        
        if e.args[-1] == 'No such file or directory':
            mtime_local = None
            size_local = None
        else:
            raise
    
    if (size_remote != size_local) or (mtime_remote > mtime_local):
        
        print("Downloading %s to %s..."%(fpath_ftp, fpath_local))
        
        with open(fpath_local, 'wb') as f:
            a_ftp.retrbinary("RETR " + fpath_ftp, f.write)
        
        downloaded_file = True
        
    return downloaded_file


class StatusCheck(object):
    '''
    classdocs
    '''


    def __init__(self, total_cnt, check_cnt):
        '''
        Constructor
        '''
        self.total_cnt = total_cnt
        self.check_cnt = check_cnt
        self.num = 0 
        self.num_last_check = 0
        self.status_time = time.time()
        self.start_time = self.status_time
    
    def increment(self, n=1):
        self.num += n
        if self.num - self.num_last_check >= self.check_cnt:
            currentTime = time.time()
            
            if self.total_cnt != -1:
                print("Total items processed is %d.  Last %d items took %f minutes. %d items to go." % (self.num, self.num - self.num_last_check, (currentTime - self.status_time) / 60.0, self.total_cnt - self.num))
                print("Current total process time: %f minutes" % ((currentTime - self.start_time) / 60.0))
                print("Estimated Time Remaining: %f" % (((self.total_cnt - self.num) / float(self.num)) * ((currentTime - self.start_time) / 60.0)))
            
            else:
                print("Total items processed is %d.  Last %d items took %f minutes" % (self.num, self.num - self.num_last_check, (currentTime - self.status_time) / 60.0))
                print("Current total process time: %f minutes" % ((currentTime - self.start_time) / 60.0))
            sys.stdout.flush()
            self.status_time = time.time()
            self.num_last_check = self.num
