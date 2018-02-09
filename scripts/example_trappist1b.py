import sys, os
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import listen

savefigs = True

# Choose LUMOS mode
mode = listen.LumosMode.G300M()

# Create a star-planet system
system = listen.System.TRAPPIST1b()

# Create a telescope
telescope = listen.Telescope(mode = mode)

# Create an observation
obs = listen.Observation(telescope, system)

# Read in spectra
trn = listen.utils.Trnst("trappist1b_o2.trnst")
rad = listen.utils.Rad("trappist1b_o2_toa.rad")

# Make sure trnst and rad grids are same, account for otherwise
if len(trn.lam) < len(rad.lam):
    mask = (rad.lam >= trn.lam.min()) & (rad.lam <= trn.lam.max())
elif len(trn.lam) > len(rad.lam):
    mask = (trn.lam >= rad.lam.min()) & (trn.lam <= rad.lam.max())
else:
    mask = np.array([True for i in trn.lam])

# Get arrays
lam = trn.lam
tdepth = trn.tdepth
absrad = trn.absrad
sflux = rad.sflux[mask]

# Observe!
obs.observe(lam, tdepth, sflux=sflux, nocc = 1, nout = 1)

# Plot SNR at the native resolution
fig, ax = obs.plot_SNR()
if savefigs:
    fig.savefig("../img/trappist1b_o2_native_SNR.png", bbox_inches = "tight")
else:
    plt.show();

# Plot the spectrum at the native resolution
fig, ax = obs.plot_spectrum(SNRcut = 0.0)
if savefigs:
    fig.savefig("../img/trappist1b_o2_native_spec.png", bbox_inches = "tight")
else:
    plt.show();

# Plot maximally binned spectrum
fig, ax = obs.plot_max_binning_spectrum()
if savefigs:
    fig.savefig("../img/trappist1b_o2_maxbin_spec.png", bbox_inches = "tight")
else:
    plt.show();

# Rebin by a large factor
obs.rebin_spectrum(bfactor=1e4)

# Plot binned SNR
fig, ax = obs.plot_SNR(use_binned = True)
if savefigs:
    fig.savefig("../img/trappist1b_o2_binned_SNR.png", bbox_inches = "tight")
else:
    plt.show();

# Plot binned spectrum
fig, ax = obs.plot_spectrum(use_binned = True)
if savefigs:
    fig.savefig("../img/trappist1b_o2_binned_spec.png", bbox_inches = "tight")
else:
    plt.show();
