from __future__ import division, print_function as _, absolute_import as _, unicode_literals as _
import os
import sys
import numpy as np
from scipy.stats import binned_statistic

__all__ = ["random_draw", "planck", "downbin_spec", "construct_lam"]

def random_draw(val, sig):
    """
    Draw fake data points from model `val` with errors `sig`
    """
    if type(val) is np.ndarray:
        return val + np.random.randn(len(val))*sig
    elif (type(val) is float) or (type(val) is int) or (type(val) is np.float64):
        return val + np.random.randn(1)[0]*sig

def planck(temp, wav):
    """
    Planck blackbody function

    Parameters
    ----------
    temp : float or array-like
        Temperature [K]
    wav : float or array-like
        Wavelength [microns]

    Returns
    -------
    B_lambda [W/m^2/um/sr]
    """
    h = 6.62607e-34       # Planck constant (J * s)
    c = 2.998e8           # Speed of light (m / s)
    k = 1.3807e-23        # Boltzmann constant (J / K)
    wav = wav * 1e-6
    # Returns B_lambda [W/m^2/um/sr]
    return 1e-6 * (2. * h * c**2) / (wav**5) / (np.exp(h * c / (wav * k * temp)) - 1.0)

def downbin_spec(specHR, lamHR, lamLR, dlam=None):
    """
    Re-bin spectum to lower resolution using scipy.binned_statistic

    Parameters
    ----------
    specHR : array-like
        Spectrum to be degraded
    lamHR : array-like
        High-res wavelength grid
    lamLR : array-like
        Low-res wavelength grid
    dlam : array-like, optional
        Low-res wavelength width grid

    Returns
    -------
    specLR : ndarray
        Low-res spectrum
    """

    if dlam is None:
        ValueError("Please supply dlam in downbin_spec()")

    # Reverse ordering if wl vector is decreasing with index
    if len(lamLR) > 1:
        if lamHR[0] > lamHR[1]:
            lamHI = np.array(lamHR[::-1])
            spec = np.array(specHR[::-1])
        if lamLR[0] > lamLR[1]:
            lamLO = np.array(lamLR[::-1])
            dlamLO = np.array(dlam[::-1])

    # Calculate bin edges
    LRedges = np.hstack([lamLR - 0.5*dlam, lamLR[-1]+0.5*dlam[-1]])

    # Call scipy.stats.binned_statistic()
    specLR = binned_statistic(lamHR, specHR, statistic="mean", bins=LRedges)[0]

    return specLR

def construct_lam(lammin, lammax, Res=None, dlam=None):
    """
    Construct a wavelength grid by specifying either a resolving power (`Res`)
    or a bandwidth (`dlam`)

    Parameters
    ----------
    lammin : float
        Minimum wavelength [microns]
    lammax : float
        Maximum wavelength [microns]
    Res : float, optional
        Resolving power (lambda / delta-lambda)
    dlam : float, optional
        Spectral element width for evenly spaced grid [microns]

    Returns
    -------
    lam : float or array-like
        Wavelength [um]
    dlam : float or array-like
        Spectral element width [um]
    """

    # Keyword catching logic
    goR = False
    goL = False
    if ((Res is None) and (dlam is None)) or (Res is not None) and (dlam is not None):
        print("Error in construct_lam: Must specify either Res or dlam, but not both")
    elif Res is not None:
        goR = True
    elif dlam is not None:
        goL = True
    else:
        print("Error in construct_lam: Should not enter this else statment! :)")
        return None, None

    # If Res is provided, generate equal resolving power wavelength grid
    if goR:

        # Set wavelength grid
        dlam0 = lammin/Res
        dlam1 = lammax/Res
        lam  = lammin #in [um]
        Nlam = 1
        while (lam < lammax + dlam1):
            lam  = lam + lam/Res
            Nlam = Nlam +1
        lam    = np.zeros(Nlam)
        lam[0] = lammin
        for j in range(1,Nlam):
            lam[j] = lam[j-1] + lam[j-1]/Res
        Nlam = len(lam)
        dlam = np.zeros(Nlam) #grid widths (um)

        # Set wavelength widths
        for j in range(1,Nlam-1):
            dlam[j] = 0.5*(lam[j+1]+lam[j]) - 0.5*(lam[j-1]+lam[j])

        #Set edges to be same as neighbor
        dlam[0] = dlam0#dlam[1]
        dlam[Nlam-1] = dlam1#dlam[Nlam-2]

        lam = lam[:-1]
        dlam = dlam[:-1]

    # If dlam is provided, generate evenly spaced grid
    if goL:
        lam = np.arange(lammin, lammax+dlam, dlam)
        dlam = dlam + np.zeros_like(lam)

    return lam, dlam
