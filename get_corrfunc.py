import sys
sys.path.append('/global/homes/j/jpiat/shared_code/abacus_mocks/')

import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from hodpy.cosmology import CosmologyAbacus
from pycorr import TwoPointCorrelationFunction

cosmo = CosmologyAbacus(0)  #c000 cosmology

file = '/pscratch/sd/a/amjsmith/AbacusSummit/secondgen_new/z0.200/AbacusSummit_base_c000_ph000/BGS_box_S.fits'

sky = fits.open(file)
data = Table(sky[1].data)
sky.close()

z = 0.2
a = 1/(1+z)
H = cosmo.H(z)

data['x_rsd'] = data['x'] + data['vx']/(a*H)

pos = data['x_rsd','y','z']

edges = (np.linspace(0., 2000., 51), np.linspace(-80, 80, 161))
result = TwoPointCorrelationFunction('rppi', edges, data_positions1=pos, engine='corrfunc', nthreads=4)

ax = plt.gca()
sep, wp = result(pimax=80, return_sep=True)
ax.plot(sep, sep * wp)
ax.set_xlabel(r'$r_{p}$')
ax.set_ylabel(r'$r_{p} w_{p}(r_{p})$')
ax.grid(True)
plt.savefig('/global/homes/j/jpiat/Relativistic_effects/figures/corr_func0.2.png')