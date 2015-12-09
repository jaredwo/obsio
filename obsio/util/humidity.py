'''
Humidity functions based on those originally developed by Ruben Behnke in R.
'''
import numpy as np

# These constants are needed for calculating enhancement factor for
# vapor pressure and dew point in the following functions
# Taken from Buck (1981) table 3, curves fw5 and fi5
# Discontinuity at 0C for VP is very small, maybe 0.02% or so
Aw = 4.1e-4
Bw = 3.48e-6
Cw = 7.4e-10
Dw = 30.6
Ew = -3.8e-2

Ai = 4.8e-4
Bi = 3.47e-6
Ci = 5.9e-10
Di = 23.8
Ei = -3.1e-2


def calc_pressure(elev):
    '''Calculate standard atmospheric pressure based on elevation.

    Parameters
    ----------
    elev : float
        elevation (m)

    Returns
    -------
    p : float
        atmospheric pressure (Pa)         
    '''

    t1 = 1.0 - (0.0065 * elev) / 288.15
    t2 = 9.80665 / (0.0065 * (8.3143 / 28.9644e-3))
    p = 101325.0 * (t1 ** t2)
    return p

def calc_svp(temp, pressure):
    '''Calculate saturation vapor pressure.
    
    Taken from Buck (1996).

    Parameters
    ----------
    temp : float
        air temperature (C)
    pressure : float
        air temperature (Pa)

    Returns
    -------
    svp : float
        saturation vapor pressure (Pa)      
    '''

    # convert number to ndarray
    try:
        n = len(temp)
    except TypeError:
        temp = np.array([temp])
        n = 1
        
    # convert pandas series to ndarray
    try:
        temp = temp.values
    except AttributeError:
        pass
    

    pressure = pressure / 100.0
    warm = temp > 0
    cold = temp <= 0
    f = np.empty(n)

    f[warm] = 1 + Aw + pressure * \
        (Bw + Cw * (temp[warm] + Dw + Ew * pressure) ** 2)
    f[cold] = 1 + Ai + pressure * \
        (Bi + Ci * (temp[cold] + Di + Ei * pressure) ** 2)

    svp = 6.1121 * \
        np.exp(1) ** (((18.678 - (temp / 234.5)) * temp) / (257.14 + temp)) * 100

    if len(svp) == 1:
        svp = svp[0]

    return svp


def calc_abshum(vp, temp):
    '''Calculate absolute humidity.

    Parameters
    ----------
    vp : float
        vapor pressure (Pa) 
    temp : float
        air temperature (C)

    Returns
    -------
    abs_hum : float
        absolute humidity (g m-3) 
    '''
    
    temp = temp + 273.15
    abs_hum = (vp / (temp * 461.5)) * 1000
    return abs_hum


def calc_shum(vp, pressure):
    '''Calculate specific humidity.

    Parameters
    ----------
    vp : float
        vapor pressure (Pa) 
    pressure : float
        atmospheric pressure (Pa)

    Returns
    -------
    shum : float
        specific humidity (g H20 kg-1 air) 
    '''

    shum = 621.97 * (vp / pressure)
    return shum


def calc_mixratio(vp, pressure):
    '''Calculate mixing ratio

    Parameters
    ----------
    vp : float
        vapor pressure (Pa) 
    pressure : float
        atmospheric pressure (Pa)

    Returns
    -------
    mixratio : float
        absolute humidity (g H20 kg-1 dry air) 
    '''

    mixratio = 621.97 * (vp / (pressure - vp))
    return mixratio


def convert_rh_to_vpd(rh, temp, pressure):
    '''Convert relative humidity to vapor pressure deficit
    
    Parameters
    ----------
    rh : float
        relative humidity between 0.01 and 100
    temp : float
        air temperature (C)
    pressure : float
        atmospheric pressure (Pa)
        
    Returns
    -------
    vpd : float
        vapor pressure deficit (Pa) 
    '''
    
    svp = calc_svp(temp, pressure)
    vp = svp * (rh / 100.0)
    vpd = svp - vp
    
    return vpd

def convert_rh_to_vp(rh, temp, pressure):
    '''
    Parameters
    ----------
    rh : float
        relative humidity between 0.01 and 100
    temp : float
        air temperature (C)
    pressure : float
        atmospheric pressure (Pa)
        
    Returns
    -------
    vp : float
        vapor pressure (Pa) 
    '''
    
    svp = calc_svp(temp, pressure)
    vp = svp * (rh / 100.0)
    
    return vp

def convert_rh_to_tdew(rh, temp):
    '''Convert relative humidity to dewpoint temperature.
    
    Buck's updated 1996 formula for dewpoint from relative humidity
    and temperature 'good' between -80C and +50C. Major advantage is no
    discontinuity at 0C. Matches Vaisala's online calculator very closely. 

    Parameters
    ----------
    rh : float
        relative humidity between 0.01 and 100
    temp : float
        air temperature (C)

    Returns
    -------
    dp : float
        dewpoint temperature (C)         
    '''

    rh = rh / 100.0
    alpha = ((18.678 - (temp / 234.5)) * temp) / (257.14 + temp)
    beta = np.log(rh) + alpha
    a = 0.008528785
    b = 18.678 - beta
    c = -257.14 * beta

    dp = (1.0 / a) * (b - np.sqrt((b ** 2) + (2 * a * c)))

    return dp

def convert_rh_to_vpd_daily(tmin, tmax, pressure, rhmin=None, rhmax=None,
                            rhavg=None):
    '''Calculate daily average vpd from daily rh measurements
    
    Provides best estimate of daily average vapor pressure deficit when only
    daily relative humidity and temperature values are available. Uses hierarchy
    of methods from Allen et al. 1998:
    
    http://www.fao.org/docrep/x0490e/x0490e07.htm#air humidity
    
    Requires minimum and maximum temperature, atmospheric pressure, and at least
    one of 3 possible relative humidity measures: daily minimum and maximum 
    relative humidity, daily maximum relative humidity, or daily average
    relative humidity
    
    
    Parameters
    ----------
    tmin : float
        daily minimum air temperature (C)
    tmax : float
        daily maximum air temperature (C)
    pressure : float
        atmospheric pressure (Pa)
    rhmin : float, optional
        daily minimum relative humidity between 0.01 and 100
    rhmax : float, optional
        daily maximum relative humidity between 0.01 and 100
    rhavg : float, optional
        daily average relative humidity between 0.01 and 100


    Returns
    -------
    vpd : float
        daily average vapor pressure deficit (Pa)    
    '''
    
    svp_tmin = calc_svp(tmin, pressure)
    svp_tmax = calc_svp(tmax, pressure)
    svp_avg = (svp_tmin + svp_tmax) / 2.0
    
    if rhmin is not None and rhmax is not None:
    
        vp_tmin = svp_tmin * (rhmax / 100.0)
        vp_tmax = svp_tmax * (rhmin / 100.0)
        vp_avg = (vp_tmin + vp_tmax) / 2.0
        
    elif rhmax is not None:
        
        vp_avg = svp_tmin * (rhmax / 100.0)
        
    elif rhavg is not None:
        
        vp_avg = svp_avg * (rhavg / 100)
        
    else:
        
        raise Exception("Invalid relative humidity parameters. Need at least 1 "
                        "of 3 possible relative humidity measures: rhmin and "
                        "rhmax, rhmax, or rhavg.")
        
        
    return svp_avg - vp_avg
