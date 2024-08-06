import sys
sys.path.append('/global/homes/j/jpiat/shared_code/abacus_mocks/')

import numpy as np
import matplotlib.pyplot as plt
from hodpy.cosmology import CosmologyAbacus
cosmo = CosmologyAbacus(0)  #c000 cosmology
from cosmoprimo import *

def R(f,H,Om,r,s,be=0,c=3e5):
    
    return 3-be-3/2*Om-3/2*Om/f-(2-5*s)*(1-c/(H*r))
    
    
def matter_power_spectrum(k,z):
    
    fo = Fourier(cosmo.cosmo_cosmoprimo, engine='class') 
    pk = fo.pk_interpolator()

    pk_matter = pk(k, z)
    
    return pk_matter

    
    
def model(k,z,s,b):
    
    """
    
    Leading order of the dipole
    
    """
    
    c = 3e5
    a = 1/(1+z)
    D = cosmo.growth_factor(z)
    f = cosmo.growth_rate(z)
    H = a*cosmo.H(z)
    r = cosmo.comoving_distance(z)
    Om = cosmo.cosmo_cosmoprimo.Omega_m(z)
    
    s_b, s_f = s[0], s[1]
    b_b, b_f = b[0], b[1]
    
    R_b = R(f,H,Om,r,s_b)
    R_f = R(f,H,Om,r,s_f)
    
    pk_matter = matter_power_spectrum(k,z)
    
    pk1 = H/(c*k)*(f*(b_b*R_f - b_f*R_b) + f**2*3/5*(R_f-R_b)+3/2*(b_b-b_f)*Om)*D**2*pk_matter
    
    return pk1
        
    
    

def Doppler(k,z,s,b):
        
    
    c = 3e5
    a = 1/(1+z)
    D = cosmo.growth_factor(z)
    f = cosmo.growth_rate(z)
    H = a*cosmo.H(z)
    r = cosmo.comoving_distance(z)
    Om = cosmo.cosmo_cosmoprimo.Omega_m(z)
    
    s_b, s_f = s[0], s[1]
    b_b, b_f = b[0], b[1]
    
    R_b = R(f,H,Om,r,s_b)
    R_f = R(f,H,Om,r,s_f)
    
    pk_matter = matter_power_spectrum(k,z)
    
    pk1 = H/(c*k)*(f*(b_b*R_f - b_f*R_b) + f**2*3/5*(R_f-R_b))*D**2*pk_matter
    
    return pk1


def Grav_redshift(k,z,b):
    
    """
    
    Leading order of the dipole
    
    """
    
    c = 3e5
    a = 1/(1+z)
    D = cosmo.growth_factor(z)
    f = cosmo.growth_rate(z)
    H = a*cosmo.H(z)
    r = cosmo.comoving_distance(z)
    Om = cosmo.cosmo_cosmoprimo.Omega_m(z)

    b_b, b_f = b[0], b[1]
    
    pk_matter = matter_power_spectrum(k,z)
    
    pk1 = H/(c*k)*(3/2*(b_b-b_f)*Om)*D**2*pk_matter
    
    return pk1
    
