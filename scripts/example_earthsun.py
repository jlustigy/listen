import sys, os
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import listen

# Choose LUMOS mode
mode = listen.LumosMode.G300M()

# Create a star-planet system
system = listen.System()

# Create a telescope
telescope = listen.Telescope(mode = mode)

# Create an observation
obs = listen.Observation(telescope, system)

# Read in spectra
trn = listen.utils.Trnst("earth_standard_icrccm.trnst")
rad = listen.utils.Rad("earth_standard_icrccm_toa.rad")

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
plt.show();

# Plot the spectrum at the native resolution
fig, ax = obs.plot_spectrum()
plt.show();

# Plot maximally binned spectrum
fig, ax = obs.plot_max_binning_spectrum()
plt.show();

# Rebin by a large factor
obs.rebin_spectrum(bfactor=1e4)

# Plot binned SNR
fig, ax = obs.plot_SNR(use_binned = True)
plt.show();

# Plot binned spectrum
fig, ax = obs.plot_spectrum(use_binned = True)
plt.show();
