# Adapted from The Salish Sea MEOPAR contributors and The University of British Columbia
# https://salishsea-meopar-docs.readthedocs.io/en/latest/tidalcurrents/

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A collection of tools for dealing with tidal results from the
Salish Sea NEMO model
"""
import collections
from math import radians, cos, sqrt, pi
import datetime
from dateutil import tz
import angles
import numpy as np
import pandas as pdz
from scipy.optimize import curve_fit

# Tide correction for amplitude and phase set to April 19, 2024 from:
# https://tidesandcurrents.noaa.gov/harcon.html?id=8574680
# The freq parameter it the frequency of the tidal consituent in degrees/hour.
CorrTides = {
    'reftime': datetime.datetime(2024, 4, 19, tzinfo=tz.tzutc()),
    'K1': {
        'freq': 15.041069000,
        'ft': 0.21,
        'uvt': 7.0},
    'O1': {
        'freq': 13.943035,
        'ft': 0.18,
        'uvt': 14.0},
    'Q1': {
        'freq': 13.398661,
        'ft': 0.04,
        'uvt': 358.5},
    'P1': {
        'freq': 14.958931,
        'ft': 0.07,
        'uvt': 2.2},
    'M2': {
        'freq': 28.984104,
        'ft': 0.5,
        'uvt': 336.3},
    'N2': {
        'freq': 28.43973,
        'ft': 0.11,
        'uvt': 313.1},
    'S2': {
        'freq': 30.0,
        'ft': 0.08,
        'uvt': 8.3},
    'K2': {
        'freq': 30.082138,
        'ft': 0.03,
        'uvt': 8.9}
    }

def ellipse_params(uamp, upha, vamp, vpha):
    """Calculates ellipse parameters based on the amplitude
    and phase for a tidal constituent.

    :arg uamp: u fitted amplitude of the chosen constituent
    :type uamp: :py:class:`numpy.ndarray`

    :arg upha: u fitted phase of the chosen constituent
    :type upha: :py:class:`numpy.ndarray`

    :arg vamp: v fitted amplitude of the chosen constituent
    :type vamp: :py:class:`numpy.ndarray`

    :arg vpha: v fitted phase of the chosen constituent
    :type vpha: :py:class:`numpy.ndarray`

    :returns: CX, SX, CY, SY, ap, am, ep, em, major, minor, theta, phase
              The positively and negatively rotating amplitude and phase.
              As well as the major and minor axis and the axis tilt.
   """

    CX = uamp*np.cos(np.pi*upha/180.)
    SX = uamp*np.sin(np.pi*upha/180.)
    CY = vamp*np.cos(np.pi*vpha/180.)
    SY = vamp*np.sin(np.pi*vpha/180.)
    ap = np.sqrt((CX+SY)**2+(CY-SX)**2)/2.
    am = np.sqrt((CX-SY)**2+(CY+SX)**2)/2.
    ep = np.arctan2(CY-SX, CX+SY)
    em = np.arctan2(CY+SX, CX-SY)
    major = ap+am
    minor = ap-am
    theta = (ep+em)/2.*180./np.pi
    phase = (em-ep)/2.*180./np.pi

    # Make angles be between [0,360]
    phase = (phase+360) % 360
    theta = (theta+360) % 360

    ind = np.divide(theta, 180)
    k = np.floor(ind)
    theta = theta - k*180
    phase = phase + k*180
    phase = (phase+360) % 360

    return CX, SX, CY, SY, ap, am, ep, em, major, minor, theta, phase

def convention_pha_amp(fitted_amp, fitted_pha):
    """ This function takes the fitted parameters given for phase and
         amplitude of the tidal analysis and returns them following the
         tidal parameter convention; amplitude is positive and phase
         is between -180 and +180 degrees.

    :arg fitted_amp: The amplitude given by the fitting function.
    :type fitted_amp: float

    :arg fitted_pha: The phase given by the fitting function.
    :type fitted_pha: float

    :return fitted_amp, fitted_pha: The fitted parameters following the
        conventions.
    """

    if fitted_amp < 0:
        fitted_amp = -fitted_amp
        fitted_pha = fitted_pha + 180
    fitted_pha = angles.normalize(fitted_pha, -180, 180)

    return fitted_amp, fitted_pha

def double(x, M2amp, M2pha, K1amp, K1pha, mean):
    """Function for the fit, assuming only M2 and K2 tidal constituents.

    :arg x:
    :type x:

    :arg M2amp: Tidal amplitude of the M2 constituent
    :type M2amp:

    :arg M2pha: Phase lag of M2 constituent
    :type M2pha:

    :arg K1amp: Tidal amplitude of the K1 constituent
    :type K1amp:

    :arg K1pha: Phase lag of K1 constituent
    :type K1pha:

    :returns:(mean + M2amp*np.cos(M2FREQ*x-M2pha*np.pi/180.)
        +K1amp*np.cos(K1FREQ*x-K1pha*np.pi/180.))
    """
    return(
        mean +
        M2amp * np.cos((CorrTides['M2']['freq'] * x - M2pha) * np.pi / 180.) +
        K1amp * np.cos((CorrTides['K1']['freq'] * x - K1pha) * np.pi / 180))


def quadruple(x, M2amp, M2pha, K1amp, K1pha, S2amp, S2pha, O1amp, O1pha, mean):
    """Function for the fit, assuming only 4 constituents of importance are:
        M2, K2, S1 and O1.

    :arg x: Independent variable, time.
    :type x:

    :arg \*amp: Tidal amplitude of the a constituent
    :type \*amp: float

    :arg \*pha: Phase lag of a constituent
    :type \*pha: float

    :returns: function for fitting 4 frequencies
    """
    return(
        mean +
        M2amp * np.cos((CorrTides['M2']['freq'] * x - M2pha) * np.pi / 180) +
        K1amp * np.cos((CorrTides['K1']['freq'] * x - K1pha) * np.pi / 180) +
        S2amp * np.cos((CorrTides['S2']['freq'] * x - S2pha) * np.pi / 180) +
        O1amp * np.cos((CorrTides['O1']['freq'] * x - O1pha) * np.pi / 180))


def sextuple(
        x, M2amp, M2pha, K1amp, K1pha,
        S2amp, S2pha, O1amp, O1pha,
        N2amp, N2pha, P1amp, P1pha, mean):
    """Function for the fit, assuming 6 constituents of importance are:
    M2, K2, S1, O1, N2 and P1.

    :arg x: Independent variable, time.
    :type x:

    :arg \*amp: Tidal amplitude of the a constituent
    :type \*amp: float

    :arg \*pha: Phase lag of a constituent
    :type \*pha: float

    :returns: function for fitting 6 frequencies
    """
    return(
        mean +
        M2amp * np.cos((CorrTides['M2']['freq'] * x - M2pha) * np.pi / 180) +
        K1amp * np.cos((CorrTides['K1']['freq'] * x - K1pha) * np.pi / 180) +
        S2amp * np.cos((CorrTides['S2']['freq'] * x - S2pha) * np.pi / 180) +
        O1amp * np.cos((CorrTides['O1']['freq'] * x - O1pha) * np.pi / 180) +
        N2amp * np.cos((CorrTides['N2']['freq'] * x - N2pha) * np.pi / 180) +
        P1amp * np.cos((CorrTides['P1']['freq'] * x - P1pha) * np.pi / 180))


def octuple(
        x, M2amp, M2pha, K1amp, K1pha,
        S2amp, S2pha, O1amp, O1pha,
        N2amp, N2pha, P1amp, P1pha,
        K2amp, K2pha, Q1amp, Q1pha, mean):
    """Function for the fit, for all the constituents: M2, K2, S1, O1, N2, P1,
    K2 and Q1.

    :arg x: Independent variable, time.
    :type x:

    :arg \*amp: Tidal amplitude of the a constituent
    :type \*amp: float

    :arg \*pha: Phase lag of a constituent
    :type \*pha: float

    :returns: function for fitting 8 frequencies
    """
    return(
        mean +
        M2amp * np.cos((CorrTides['M2']['freq'] * x - M2pha) * np.pi / 180) +
        K1amp * np.cos((CorrTides['K1']['freq'] * x - K1pha) * np.pi / 180) +
        S2amp * np.cos((CorrTides['S2']['freq'] * x - S2pha) * np.pi / 180) +
        O1amp * np.cos((CorrTides['O1']['freq'] * x - O1pha) * np.pi / 180) +
        N2amp * np.cos((CorrTides['N2']['freq'] * x - N2pha) * np.pi / 180) +
        P1amp * np.cos((CorrTides['P1']['freq'] * x - P1pha) * np.pi / 180) +
        K2amp * np.cos((CorrTides['K2']['freq'] * x - K2pha) * np.pi / 180) +
        Q1amp * np.cos((CorrTides['Q1']['freq'] * x - Q1pha) * np.pi / 180))

def fittit(uaus, time, nconst):
    """Function to find tidal components of a time series over the whole
    area given.

    Can be done over depth, or an area.
    Time must be in axis one, depth in axis two if applicable then the
    y, x if an area.
    In order to calculate the tidal components of an area at a single
    depth the time series must only have 3 dimensions. For a depth
    profile it must only have 2 dimensions

    :arg uaus: The time series to be analyzed.
    :type uaus:  :py:class:'np.ndarray' or float

    :arg time: Time over which the time series is taken in hours.
    :type time: :py:class:'np.ndarray'

    :arg nconst: The amount of tidal constituents used for the analysis.
                 They added in pairs and by order of importance,
                 M2, K1, S2, O1, N2, P1, K2, Q1.
    :type nconst: int

    :returns: a dictionary object containing a phase an amplitude for each
              harmonic constituent, for each orthogonal velocity
    """
    # Setting up the dictionary space for the ap-parameters to be stored
    apparam = collections.OrderedDict()

    # The function with which we do the fit is based on the amount of
    # constituents
    fitfunction = double

    # The first two harmonic parameters are always M2 and K1
    apparam['M2'] = {'amp': [], 'phase': []}
    apparam['K1'] = {'amp': [], 'phase': []}
    
    if nconst > 2:
        fitfunction = quadruple
        apparam['S2'] = {'amp': [], 'phase': []}
        apparam['O1'] = {'amp': [], 'phase': []}

    if nconst > 4:
        fitfunction = sextuple
        apparam['N2'] = {'amp': [], 'phase': []}
        apparam['P1'] = {'amp': [], 'phase': []}

    if nconst > 6:
        fitfunction = octuple
        apparam['K2'] = {'amp': [], 'phase': []}
        apparam['Q1'] = {'amp': [], 'phase': []}

    # CASE 1: a time series of velocities with depth at a single location.
    if uaus.ndim == 2:
        # The parameters are the same shape as the velocities without the time
        # dimension.
        thesize = uaus.shape[1]
        # creating the right shape of space for each amplitude and phase for
        # every constituent.
        for const, ap in apparam.items():
            for key2 in ap:
                ap[key2] = np.zeros(thesize)

        # Calculates the parameters for one depth and one location at a time
        # from its time series
        for dep in np.arange(0, len(uaus[1])):
            if uaus[:, dep].any() != 0:

                # performs fit of velocity over time using function that is
                # chosen by the amount of constituents
                fitted, cov = curve_fit(fitfunction, time[:], uaus[:, dep])

                # Rotating to have a positive amplitude and a phase between
                # [-180, 180]
                for k in np.arange(nconst):
                    fitted[2*k], fitted[2*k+1] = convention_pha_amp(
                        fitted[2*k], fitted[2*k+1])
                # Putting the amplitude and phase of each constituent of this
                # particlar depth in the right location within the dictionary.
                for const, k in zip(apparam, np.arange(0, nconst)):
                    apparam[const]['amp'][dep] = fitted[2*k]
                    apparam[const]['phase'][dep] = fitted[2*k+1]

    # CASE 2 : a time series of an area of velocities at a single depth
    elif uaus.ndim == 3:
        thesize = (uaus.shape[1], uaus.shape[2])
        for const, ap in apparam.items():
            for key2 in ap:
                ap[key2] = np.zeros(thesize)

        for i in np.arange(0, uaus.shape[1]):
            for j in np.arange(0, uaus.shape[2]):
                if uaus[:, i, j].any() != 0.:
                    fitted, cov = curve_fit(
                        fitfunction, time[:], uaus[:, i, j])
                    for k in np.arange(nconst):
                        fitted[2*k], fitted[2*k+1] = convention_pha_amp(
                            fitted[2*k], fitted[2*k+1])

                    for const, k in zip(apparam, np.arange(0, nconst)):
                        apparam[const]['amp'][i, j] = fitted[2*k]
                        apparam[const]['phase'][i, j] = fitted[2*k+1]

    # CASE 3: a time series of an area of velocities with depth
    elif uaus.ndim == 4:
        thesize = (uaus.shape[1], uaus.shape[2], uaus.shape[3])
        for const, ap in apparam.items():
            for key2 in ap:
                ap[key2] = np.zeros(thesize)

        for dep in np.arange(0, uaus.shape[1]):
            for i in np.arange(0, uaus.shape[2]):
                for j in np.arange(0, uaus.shape[3]):
                    if uaus[:, dep, i, j].any() != 0.:
                        fitted, cov = curve_fit(
                            fitfunction, time[:], uaus[:, dep, i, j])

                        for k in np.arange(nconst):
                            fitted[2*k], fitted[2*k+1] = convention_pha_amp(
                                fitted[2*k], fitted[2*k+1])

                        for const, k in zip(apparam, np.arange(0, nconst)):
                            apparam[const]['amp'][dep, i, j] = fitted[2*k]
                            apparam[const]['phase'][dep, i, j] = fitted[2*k+1]

    # Case 4: a time series of a single location with a single depth.
    else:
        thesize = (0)
        for const, ap in apparam.items():
            for key2 in ap:
                ap[key2] = np.zeros(thesize)

        if uaus[:].any() != 0.:
            fitted, cov = curve_fit(fitfunction, time[:], uaus[:])
            for k in np.arange(nconst):
                fitted[2*k], fitted[2*k+1] = convention_pha_amp(
                    fitted[2*k], fitted[2*k+1])
            for const, k in zip(apparam, np.arange(0, nconst)):
                apparam[const]['amp'] = fitted[2*k]
                apparam[const]['phase'] = fitted[2*k+1]

    # Mask the zero values
    for const, ap in apparam.items():
        for key in ap.keys():
            ap[key] = np.ma.masked_values(ap[key], 0)

    return apparam


def get_params(u, v, time, nconst, tidecorr):
    """Calculates tidal ellipse parameters from the u and v time series.
    Maintains the shape of the velocities enters only loosing the time
    dimensions.

    :arg u: One of the orthogonal tidal current velocities. Must be already
        prepared for the analysis.
    :type u:  :py:class:'np.ndarray'

    :arg v: One of the orthogonal tidal current velocities. Must be already
        prepared for the analysis.
    :type v:  :py:class:'np.ndarray'

    :arg time: Time over which the velocities were taken; in hours.
    :type time: :py:class:'np.ndarray'

    :arg nconst: The amount of tidal constituents used for the analysis. They
        added in pairs and by order of importance, M2, K1, S2, O1, N2, P1, K2,
        Q1.
    :type nconst: int

    :returns: params a dictionary containing the tidal ellipse parameter of the
        chosen constituents
    """
    params = {}
    # Running fittit to get the amplitude and phase of the velcity time series.
    uapparam = fittit(u, time, nconst)
    vapparam = fittit(v, time, nconst)

    # Cycling through the constituents in the ap-parameter dict given by fittit
    for const in uapparam:
    
        # Converting from u/v amplitude and phase to ellipe parameters. Inputs
        # are the amplitude and phase of both velocities, runs once for each
        # contituent
        CX, SX, CY, SY, ap, am, ep, em, maj, mi, the, pha = ellipse_params(
            uapparam[const]['amp'],
            uapparam[const]['phase'],
            vapparam[const]['amp'],
            vapparam[const]['phase'])
        # Filling the dictionary with ep-parameters given by ellipse_param.
        # Each constituent will be a different key.

        params[const] = {
            'Semi-Major Axis': maj,
            'Semi-Minor Axis': mi,
            'Inclination': the,
            'Phase': pha
            }
    
    return params
if __name__=='__main__':
    """
    This script is specifically for accessing data from a current buoy and was created with 
    buoy `cb1501` as a template.  
    
    buoy_id: string with NOAA buoy ID (default = "cb1501")
    start_date: string with start date "YYYYMMDD" (default = "20240401")
    end_date: string with end date "YYYYMMDD" (default ="20240415")
    """
    args = sys.argv[1:]
    u=args[0]
    v=args[1]
    hours=args[2]
    num_const=args[3]
    CorrTides=args[4]

    get_params(u,v,hours,num_const,CorrTides)
