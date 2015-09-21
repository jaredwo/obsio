'''
Humidity functions originally developed by Ruben Behnke in R.
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
    '''
    Calculate standard atmospheric pressure based on elevation

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
    p = 101325.0 * (t1**t2)
    return p


def calc_dew(rh, temp):
    '''
    Calculate dewpoint temperature. Matches Vaisala's online calculator very
    closely. Buck's updated 1996 formula for dewpoint from relative humidity
    and temperature 'good' between -80C and +50C. Major advantage is no
    discontinuity at 0C.

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

    dp = (1.0 / a) * (b - np.sqrt((b**2) + (2 * a * c)))

    return dp


def calc_svp(temp, pressure):
    '''
    Calculate saturation vapor pressure from temperature and atmospheric
    pressure. Taken from Buck (1996).

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

    try:
        n = len(temp)
    except TypeError:
        temp = np.array([temp])
        n = 1

    pressure = pressure / 100.0
    warm = temp > 0
    cold = temp <= 0
    f = np.empty(n)

    f[warm] = 1 + Aw + pressure * \
        (Bw + Cw * (temp[warm] + Dw + Ew * pressure)**2)
    f[cold] = 1 + Ai + pressure * \
        (Bi + Ci * (temp[cold] + Di + Ei * pressure)**2)

    svp = 6.1121 * \
        np.exp(1)**(((18.678 - (temp / 234.5)) * temp) / (257.14 + temp)) * 100

    if len(svp) == 1:
        svp = svp[0]

    return svp


def calc_abshum(vp, temp):
    '''
    Calculate absolute humidity from vapor pressure and temperature

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
    '''
    Calculate specific humidity from vapor pressure and atmospheric pressure

    Parameters
    ----------
    vp : float
        vapor pressure (Pa) 
    pressure : float
        atmospheric pressure (Pa)

    Returns
    -------
    shum : float
        absolute humidity (g H20 kg-1 moist air) 
    '''

    shum = 621.97 * (vp / pressure)
    return shum


def calc_mixratio(vp, pressure):
    '''
    Calculating mixing ratio from vapor pressure and atmospheric pressure

    Parameters
    ----------
    vp : float
        vapor pressure (Pa) 
    pressure : float
        atmospheric pressure (Pa)

    Returns
    -------
    mixratio : float
        absolute humidity (g H20 kg-1 moist air) 
    '''

    mixratio = 621.97 * (vp / (pressure - vp))
    return mixratio
