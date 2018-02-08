from __future__ import division, print_function as _, absolute_import as _, unicode_literals as _
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import astropy.units as u

import platform
if platform.system() == 'Darwin':
    # On a Mac: usetex ok
    mpl.rc('font',**{'family':'serif','serif':['Computer Modern']})
    mpl.rcParams['font.size'] = 25.0
    mpl.rc('text', usetex=True)
elif platform.node().startswith("D"):
    # On hyak: usetex not ok, must change backend to 'agg'
    mpl.rc('font',**{'family':'serif','serif':['Computer Modern']})
    mpl.rcParams['font.size'] = 25.0
    mpl.rc('text', usetex=False)
    plt.switch_backend('agg')
else:
    # On astro machine or other linux: usetex not ok
    plt.switch_backend('agg')
    mpl.rc('font', family='Times New Roman')
    mpl.rcParams['font.size'] = 25.0
    mpl.rc('text', usetex=False)

# Import local
from . import functions

__all__ = ["System", "LumosMode", "Telescope", "Observation"]

# The location to *this* file
RELPATH = os.path.dirname(__file__)

class System(object):
    """
    Wrapper class for the planet-star system (default is Earth-Sun)

    Parameters
    ----------
    d : float
        Distance to system [pc]
    a : float
        Semi-major axis of planet [AU]
    Rp : float
        Radius of planet [Earth Radii]
    Rs : float
        Radius of star [Solar Radii]
    tdur : float
        Transit duration [seconds]
    Tstar : float
        Stellar temperature
    """
    def __init__(self, d = 10.0, a = 1.0, Rp = 1.0, Rs = 1.0,
                 tdur = 8. * 3600., Tstar = 5780.):
        self.d = d
        self.a = a
        self.Rp = Rp
        self.Rs = Rs
        self.tdur = tdur
        self.Tstar = Tstar

    @classmethod
    def for_TRAPPIST1b(cls):
        return cls(d = 12.2, Rp = 1.086, Rs = 0.117, a = 0.0111, tdur = 36.40 * 60,
                   Tstar = 2560.)

class LumosMode(object):
    """
    Wrapper object for LUVOIR/LUMOS channel specifications
    """

    def __init__(self, fn):
        self.fn = fn
        self.read_lumos_mode()

    def read_lumos_mode(self):
        """
        """

        # Read in mode file
        self._data = np.genfromtxt(self.fn, skip_header=2)

        # Parse
        self.col_names = ["Wavelength", "Read Noise", "Sky", "Dark", "AEff", "BEF", "DispWidth", "XDispWidth"]
        self.col_units = ["(Angstroms)", "(Cnt/Exposure)", "(Cnt/s/res)", "(Cnt/s/res)", "(cm2)", "(Erg/cm2/s/res)", "(Nresels)", "(Nresels)"]
        self.wl   = self._data[:,0]
        self.rn   = self._data[:,1]
        self.sky  = self._data[:,2]
        self.dark = self._data[:,3]
        self.Aeff = self._data[:,4]
        self.BEF  = self._data[:,5]
        self.DW   = self._data[:,6]
        self.XDW  = self._data[:,7]

        # Calculate wavelength widths (assume first bin has same width as second)
        self.dwl = np.hstack([self.wl[1] - self.wl[0], self.wl[1:] - self.wl[:-1]])

    @classmethod
    def G155L(cls):
        return cls(os.path.join(RELPATH, "LUMOS_data/G155L_ETC.dat"))

    @classmethod
    def G300M(cls):
        return cls(os.path.join(RELPATH, "LUMOS_data/G300M_ETC.dat"))

    def plot_all(self, title = None):
        """
        Plots all quantities associated with mode
        """
        fig, axs = plt.subplots(4,2, figsize = (10,15))
        if title is not None: fig.suptitle(title, y = .91)
        axes = axs.flatten()
        axes[-1].remove()
        axes[5].set_xlabel("Wavelength [\AA]", fontsize = 16)
        axes[6].set_xlabel("Wavelength [\AA]", fontsize = 16)
        for i in range(self._data.shape[1]-1):
            plt.setp(axes[i].get_xticklabels(), fontsize=12, rotation=0)
            if i%2 == 1:
                axes[i].set_yticks([])
                axes[i] = axes[i].twinx()
            axes[i].plot(self.wl, self._data[:,i+1])
            axes[i].set_ylabel(u"%s %s" %(unicode(self.col_names[i+1]), unicode(self.col_units[i+1])), fontsize = 16)
            plt.setp(axes[i].get_yticklabels(), fontsize=12, rotation=0)
        axes[1].set_yscale("log")
        axes[4].set_yscale("log")
        return fig, ax

class Telescope(object):
    """
    Wrapper class for the telescope

    Parameters
    ----------
    A : float
        Collecting area [m^2]
    tput : float
        End-to-end throughput
    mode : `LumosMode`
        Instantiated LUMOS mode object
    """
    def __init__(self, A = 134.8, tput = 0.32, mode = G300M):
        self.A = A
        self.tput = tput
        self.mode = mode

class Observation(object):
    """
    Wrapper class for observations
    """

    def __init__(self, telescope, system, nocc = 1, nout = 1):
        self.telescope = telescope
        self.system = system
        self._computed = False
        self._binned = False

    def observe(self, lam, tdepth, sflux = None, nocc = 1, nout = 1):
        """
        Make an observation

        Parameters
        ----------
        lam : numpy.array
            Wavelength array [um]
        tdepth : numpy.array
            Transit Depth (Rp/Rs)^2
        sflux : numpy.array, optional
            Stellar flux incident at planet TOA [W/m^2/um]
            If not provided, blackbody is assumed at given stellar temperature
        nocc : int
            Number of transits to observe
        nout : int
            Number of transit durations to observe out-of-transit
        """

        h = 6.62607e-34       # Planck constant (J * s)
        c = 2.998e8           # Speed of light (m / s)

        # Set number of transits and out-of-transits observed
        self.nocc = nocc
        self.nout = nout

        # Convert to microns
        wlum = 1e-4 * self.telescope.mode.wl
        dwlum = 1e-4 * self.telescope.mode.dwl

        # Check if arrays need to be reversed
        if lam[0] > lam[1]:
            # Reverse arrays
            lam = lam[::-1]
            tdepth = tdepth[::-1]
            if sflux is not None:
                sflux = sflux[::-1]

        # Calculate delta-wavelength grid
        dlam = lam[1:] - lam[:-1]
        dlam = np.hstack([dlam, dlam[-1]])

        # Calculate intensity for the star [W/m^2/um/sr]
        if sflux is None:
            # Using a blackbody
            Bstar = functions.planck(Tstar, lam)
        else:
            # Using provided TOA stellar flux
            Bstar = sflux / ( np.pi*(system.Rs*u.Rsun.in_units(u.km)/(system.a*u.AU.in_units(u.km)))**2. )

        # solid angle in steradians
        omega_star = np.pi*(system.Rs*u.Rsun.in_units(u.km)/(system.d*u.pc.in_units(u.km)))**2.
        omega_planet = np.pi*(system.Rp*u.Rearth.in_units(u.km)/(system.d*u.pc.in_units(u.km)))**2.

        # fluxes at earth [W/m^2/um]
        Fstar = Bstar * omega_star

        # Exposure time is transit diration times number of occultations observed
        tint = nocc * system.tdur

        # Bin high res transit model to instrument resolution
        RpRs2 = functions.downbin_spec(tdepth, lam, wlum, dwlum)

        # Check for non-finite values (usually from mismatched grids)
        if np.sum(~np.isfinite(RpRs2)) > 0:
            # Just interpolate grids
            RpRs2 = np.interp(wlum, lam, tdepth)

        # Bin high res stellar flux to instrument resolution
        Fslo = functions.downbin_spec(Fstar, lam, wlum, dwlum)

        # Check for non-finite values (usually from mismatched grids)
        if np.sum(~np.isfinite(Fslo)) > 0:
            # Just interpolate grids
            Fslo = np.interp(wlum, lam, Fstar)

        # Stellar photon count rate
        cstar = Fslo*dwlum*(wlum*1e-6)/(h*c)*self.telescope.tput*self.telescope.A

        # Sky background count rate
        csky = self.telescope.mode.sky * self.telescope.mode.XDW

        # Dark current count rate
        cdark = self.telescope.mode.dark * self.telescope.mode.XDW

        # Total background photon count rate
        cback =  csky + cdark

        # Count STELLAR photons
        Nstar = tint * cstar

        # Count BACKGROUND photons
        Nback = tint * cback

        # Calculate SNR on missing stellar photons
        SNR = (Nstar * RpRs2) / np.sqrt((1 + 1./nout - RpRs2) * Nstar+ (1 + 1./nout) * Nback)

        # Generate synthetic observations
        sig = RpRs2/SNR
        obs = functions.random_draw(RpRs2, sig)

        # Save values
        self.wlum = wlum
        self.dwlum = dwlum
        self.SNR = SNR
        self.obs = obs
        self.sig = sig
        self.cstar = cstar
        self.csky = csky
        self.cdark = cdark
        self.cback = cback
        self.Nstar = Nstar
        self.Nback = Nback
        self.tint = tint
        self.RpRs2 = RpRs2
        self.Fslo = Fslo

        self._computed = True

        return

    def plot_SNR(self, use_binned = False):
        """
        Plot the SNR spectrum
        """
        assert self._computed

        if not self._binned:
            use_binned = False

        # Plot
        fig, ax = plt.subplots(figsize = (10,8))
        ax.plot(self.wlum, self.SNR, alpha = 1.0, label = "Native Resolution")
        if use_binned:
            ax.plot(self.wlbin, self.SNRbin, alpha = 1.0, label = r"Binned $\times %i$" %self.bfactor)
        ax.set_yscale("log")
        ax.set_xlabel(r"Wavelength [$\mu$m]")
        ax.set_ylabel("S/N on Transit Depth")
        ax.axhline(1.0, color = "C4", ls = "dashed")
        ax.legend()

        return fig, ax

    def plot_spectrum(self, SNRcut = 1.0, use_binned = False):
        """
        Plot the transmission spectrum
        """
        assert self._computed

        if not self._binned:
            use_binned = False

        # Mask for points above SNR cutoff
        m = [self.SNR > SNRcut]

        # Plot
        fig, ax = plt.subplots(figsize = (10,8))
        ax.plot(self.wlum, self.RpRs2, alpha = 1.0, label = "Model Spectrum")

        if use_binned:
            ax.plot(self.wlbin, self.RpRs2bin, "o")
            ax.errorbar(self.wlbin, self.obsbin, yerr=self.sigbin, fmt="o", color="k", label = r"Binned $\times %i$" %self.bfactor)
        else:
            ax.errorbar(self.wlum[m], self.obs[m], yerr=self.sig[m], fmt=".", color="k", alpha = 0.1, label = "Native Resolution")
        #ax.set_yscale("log")
        ax.set_xlabel(r"Wavelength [$\mu$m]")
        ax.set_ylabel("Transit Depth $(R_p / R_{\star})^2$")
        ax.legend()
        return fig, ax

    def plot_max_binning_spectrum(self):
        """
        Plot the maximally binned transmission spectrum
        """
        assert self._computed

        # Bin to a single point for reference
        bRpRs2 = np.mean(self.RpRs2)
        bwlum = np.mean(self.wlum)

        # Quadrature sum SNR
        bSNR = np.sqrt(np.sum(self.SNR**2))

        # Calculate SNR on missing stellar photons
        #bSNR = (Nstar.sum() * bRpRs2) / np.sqrt((1 + 1./nout - bRpRs2) * Nstar.sum()+ (1 + 1./nout) * Nback.sum())

        # Generate synthetic observations
        sig = bRpRs2/bSNR
        obs = functions.random_draw(bRpRs2, sig)

        # Plot
        fig, ax = plt.subplots(figsize = (10,8))
        ax.plot(self.wlum, self.RpRs2, alpha = 1.0, label = "Native Resolution")
        ax.plot(bwlum, bRpRs2, "o", label = "Max Binned Model")
        ax.errorbar(bwlum, obs, yerr=sig, fmt="o", color="k", label = "Max Binned Obs.")
        #ax.set_yscale("log")
        ax.set_xlabel(r"Wavelength [$\mu$m]")
        ax.set_ylabel("Transit Depth $(R_p / R_{\star})^2$")
        ax.legend()
        return fig, ax

    def rebin_spectrum(self, bfactor = 1e4):
        """
        Rebin the observation to a lower spectral resolution by a
        factor of `bfactor`

        Parameters
        ----------
        bfactor : int or float
            Set the binning factor
        """
        assert self._computed

        # Set new bin width
        dwlbin = np.mean(self.dwlum)*bfactor

        # Create new wavelength grid
        wlbin, dwlbin = functions.construct_lam(self.wlum.min() + 0.5*dwlbin, self.wlum.max() - 0.5*dwlbin, dlam=dwlbin)

        print("Number of binned points : %i" %len(wlbin))

        # Calculate bin edges
        LRedges = np.hstack([wlbin - 0.5*dwlbin, wlbin[-1]+0.5*dwlbin[-1]])

        # Call scipy.stats.binned_statistic()
        SNRbin = np.sqrt(binned_statistic(self.wlum, self.SNR**2, statistic="sum", bins=LRedges)[0])
        RpRs2bin = binned_statistic(self.wlum, self.RpRs2, statistic="mean", bins=LRedges)[0]

        # Generate synthetic observations
        sig = RpRs2bin/SNRbin
        obs = functions.random_draw(RpRs2bin, sig)

        self.bfactor = bfactor
        self.wlbin = wlbin
        self.dwlbin = dwlbin
        self.SNRbin = SNRbin
        self.RpRs2bin = RpRs2bin
        self.obsbin = obs
        self.sigbin = sig

        self._binned = True
