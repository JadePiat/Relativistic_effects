import sys
sys.path.append('/global/homes/j/jpiat/shared_code/abacus_mocks/')

import numpy as np
import matplotlib.pyplot as plt
from hodpy.cosmology import CosmologyAbacus
cosmo = CosmologyAbacus(0)  #c000 cosmology
from cosmoprimo import *

def R(f,H,Om,r,s,be=0):
    
    return 3-be-3/2*Om-3/2*Om/f-(2-5*s)*(1-1/(H*r))
    
    
def matter_power_spectrum(k,z):
    
    fo_camb = cosmo.cosmo_cosmoprimo.get_fourier(engine='camb') # second initialisation style
    pk = fo_camb.pk_interpolator() # same number of cells are output than in CLASS

    pk_matter = pk(k, z)
    
    return pk_matter
    
    
def model_dipole(k,z,s,b):
    
    """
    
    Leading order of the dipole
    
    """
    
    D = cosmo.growth_factor(z)
    f = cosmo.growth_rate(z)
    H = cosmo.H(z)
    r = cosmo.comoving_distance(z)
    Om = cosmo.cosmo_cosmoprimo.Omega_m(z)
    
    s_b, s_f = s[0], s[1]
    b_b, b_f = b[0], b[1]
    
    R_b = R(f,H,Om,r,s_b)
    R_f = R(f,H,Om,r,s_f)
    
    pk_matter = matter_power_spectrum(k,z)
    
    pk1 = H/k*(f*(b_b*R_f - b_f*R_b) + f**2*3/5*(R_f-R_b)+3/2*(b_b-b_f)*Om)*D**2*pk_matter
    
    return pk1
    
    
    
if __name__ == "__main__":
    
    data = np.loadtxt(f'/global/homes/j/jpiat/data/Pk1_b10_f90.dat')
    
    k = data[:,0]
    s = [0.980,0.413]
    b = [1.28,1.05]
    
    dz = 0.05
    z_bins = np.arange(0,0.5+dz,dz)
    z = z_bins[:-1]+dz/2
    
    plt.figure()
    plt.xlabel('k')
    plt.ylabel(r'P$_1$(k)')
    
    for zi in z:
        
        pk1 = model_dipole(k,zi,s,b)
        
        plt.plot(pk1)
        
    plt.show()
        
        
        
    
    
    
