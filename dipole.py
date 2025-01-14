import sys
sys.path.append('/global/homes/j/jpiat/my_hodpy/')

import numpy as np
import matplotlib.pyplot as plt
from hodpy.cosmology import CosmologyAbacus
cosmo = CosmologyAbacus(0)  #c000 cosmology
from cosmoprimo import *

#def R(f,H,Om,r,s,be=0,c=3e5):
    
#    return 3-be-3/2*Om-3/2*Om/f-(2-5*s)*(1-c/(H*r))

def R(f,H,Om,r,s,be=0,c=3e5):
    
    #kappa = (c/(r*H)-1)/c
    #R1 = (1/(1-kappa)**2)**(2.5*s)-1-2/(r*H)
    
    #R = -2/(H*r) - 5*s/c + 5*s/(r*H)
    R = -(3-be-3/2*Om-3/2*Om/f-(2-5*s)*(1-c/(H*r)))/c
    
    return R
    
    
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
    D = cosmo.growth_factor(z)
    f = cosmo.growth_rate(z)
    H = 1/(1+z)*cosmo.H(z)
    Om = cosmo.cosmo_cosmoprimo.Omega_m(z)
    r = cosmo.comoving_distance(z)
    
    s_x, s_y = s[0], s[1]
    b_x, b_y = b[0], b[1]
    
    R_x = R(f,H,Om,r,s_x)
    R_y = R(f,H,Om,r,s_y)
    
    pk_matter = matter_power_spectrum(k,z=0)
    
    pk1 = H/k*(f*(b_y*R_x - b_x*R_y) + f**2*3/5*(R_x-R_y))*D**2*pk_matter
    
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
    
