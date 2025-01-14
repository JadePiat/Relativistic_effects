import sys
sys.path.append('/global/homes/j/jpiat/hodpy/')

import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from hodpy.cosmology import CosmologyAbacus
from pycorr import TwoPointCorrelationFunction,utils
from hodpy.k_correction import GAMA_KCorrection
from hodpy import lookup
import time

from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from scipy.integrate import cumulative_trapezoid

cosmo = CosmologyAbacus(0)  #c000 cosmology
kcorr = GAMA_KCorrection(cosmo, k_corr_file=lookup.kcorr_file, cubic_interpolation=True)


def random_dist(data,xmin,xmax,dx,N):
    
    bins = np.arange(xmin,xmax+dx,dx)
    x = bins[:-1]+dx/2
    n,_ = np.histogram(data,bins=bins)
    
    n_interp = interp1d(x,n,fill_value='extrapolate')
    
    xinterp = np.linspace(xmin,xmax,100)
    pdf = n_interp(xinterp)
    
    cdf = cumulative_trapezoid(pdf,xinterp,initial=0)
    cdf = cdf/trapezoid(pdf,xinterp)
    cdf_inv = interp1d(cdf,xinterp)
    
    plt.plot(xinterp,cdf)
 
    r = np.random.uniform(0,1,N)
    x_rand = cdf_inv(r)
    
    return x_rand


start = time.time()


file = '/pscratch/sd/j/jpiat/Abacus_mocks/z0.200/AbacusSummit_base_c000_ph000/cutsky/N/galaxy_cut_sky_N.fits'

cat = fits.open(file)
box = Table(cat[1].data)
cat.close()

zmax = 0.4
threshold = -20.5

M = box['R_MAG_ABS']

mask = (M<=threshold)*(box['RA']>=0)*(box['Z']<=zmax)
data = box['RA','DEC','Z'][mask]
pos_data = np.array([data['RA'],data['DEC'],cosmo.comoving_distance(data['Z'])])


RA_rand = np.random.uniform(0,360,len(data)*10)
DEC_rand = np.degrees(np.arcsin(np.random.uniform(-1,1,len(data)*10)))
z_rand = random_dist(data['Z'],0,zmax,0.01,len(data)*10)
pos_rand = np.array([RA_rand,DEC_rand,cosmo.comoving_distance(z_rand)])


from pycorr import KMeansSubsampler
subsampler = KMeansSubsampler(mode='angular', positions=pos_data, nsamples=128, nside=512, random_state=42, position_type='rdd')
labels = subsampler.label(pos_data)
subsampler.log_info('Labels from {:d} to {:d}.'.format(labels.min(), labels.max()))


data_samples = subsampler.label(pos_data)
randoms_samples = subsampler.label(pos_rand)
edges = (np.logspace(-2, 2, 25), np.linspace(-80, 80, 161))
result = TwoPointCorrelationFunction('rppi', edges, data_positions1=pos_data, randoms_positions1=pos_rand, 
                                     data_samples1=data_samples, randoms_samples1=randoms_samples, position_type='rdd',
                                     engine='corrfunc', compute_sepsavg=False, nthreads=64)


fn = os.path.join('/global/homes/j/jpiat/Relativistic_effects/', 'my_wp_205_N.npy')
result.save(fn)

end = time.time()

print(end-start)