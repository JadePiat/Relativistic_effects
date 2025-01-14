import sys
sys.path.append('/global/homes/j/jpiat/my_hodpy/')

import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from hodpy.k_correction import DESI_KCorrection
from hodpy import lookup
from hodpy.cosmology import CosmologyAbacus
from scipy.interpolate import CubicSpline
from gravitational_redshift import z_grav
#from split_mocks_eff import magnitude_split

cosmo = CosmologyAbacus(0)  #c000 cosmology


def apparent_mag(data,z):
    
    N = data['DEC']>32.375
    S = data['DEC']<=32.375
    
    m = np.zeros(len(z))
    
    for photsys in ["N","S"]:
        
        if photsys == "N":
            keep = N
        else:
            keep = S
    
        # get apparent magnitude
        kcorr_r = DESI_KCorrection(band='r', photsys=photsys, cosmology=cosmo) 
        m[keep] = kcorr_r.apparent_magnitude(data['R_MAG_ABS'][keep], z[keep], data['G_R_REST'][keep])
    
    return m
    


def magnification_bias(input_file, m_lim, z_max, n_bins, cut_bright, cut_faint, output_path):
    
    cat = fits.open(input_file)
    data = Table(cat[1].data)
    cat.close()
    
    #phi = data['Pot A Soft 7']
    #phi0 = phi[z_cos==np.min(z_cos)]
    
    z_cos = data['Z_COSMO']
    z_tot = data['Z']
    #z_tot = z_grav(z_cos, z, phi, phi0)
    
    #data['Z_RSD+GR'] = z_tot
    
    #m = apparent_mag(M,z_tot,g_r)
    m_tot = data['R_MAG_APP']
    
    cond = (np.isnan(m_tot)==False)*(m_tot<=m_lim)*(z_tot<=z_max)
    m = m_tot[cond]
    z = z_tot[cond]
    
    z_bins = np.linspace(0., z_max, n_bins+1)
        
    z_means = np.zeros(n_bins)
    m_cuts_b = np.zeros(n_bins)
    m_cuts_f = np.zeros(n_bins)
    
    for i in range(n_bins):
        if i==(n_bins-1):
            keep = (z>=z_bins[i])*(z<=z_bins[i+1])
        else:
            keep = (z>=z_bins[i])*(z<z_bins[i+1])
        
        m_bin = m[keep]
        z_means[i] = np.mean(z[keep])
        
        m_cuts_b[i] = np.percentile(m_bin,cut_bright)
        m_cuts_f[i] = np.percentile(m_bin,100-cut_faint)
    
    m_interp_b = CubicSpline(z_means, m_cuts_b, extrapolate=True)
    m_interp_f = CubicSpline(z_means, m_cuts_f, extrapolate=True)
    
    data_b = data[(m<=m_interp_b(z))]
    data_f = data[(m>m_interp_f(z))]
    
        
    #### magnification bias ####

    #fig, axs = plt.subplots(1, 2, figsize = (9,5), layout='tight')
    #fig.suptitle(f'{cut_bright}% / {cut_faint}%')
    #
    #axs[0].set_ylabel(r's$\rm _B$')
    #axs[1].set_ylabel(r's$\rm _F$')
    #axs[0].set_xlabel(r'$\mu$-1')
    #axs[1].set_xlabel(r'$\mu$-1')
    #
    #mu = np.arange(1.005,1.02,0.001)
    #s_b = np.zeros(len(mu))
    #s_f = np.zeros(len(mu))
    #
    ##s_eff = split_mag_eff(input_file, space, m_lim, z_max, n_bins, cut_bright, cut_faint, output_path)
    ##sb_eff = s_eff[0]
    ##sf_eff = s_eff[1]
    ##axs[1].axhline(sf_eff,ls='--',color='k',label='effective value')
    ##axs[0].axhline(sb_eff,ls='--',color='k',label='effective value')
    ##axs[1].axhspan(-0.02*sf_eff+sf_eff, 0.02*sf_eff+sf_eff, color='gray', alpha=0.5,label='2%')
    ##axs[0].axhspan(-0.01*sb_eff+sb_eff, 0.01*sb_eff+sb_eff, color='gray', alpha=0.5,label='1%')
    #
    #for j,mu_j in enumerate(mu):
    #    
    #    m_mag = m_tot - 2.5*np.log10(mu_j)
    #    
    #    cond_mag = (np.isnan(m_tot)==False)*(m_mag<=m_lim)*(z_tot<=z_max)
    #    m_mag = m_mag[cond_mag]
    #    z_mag = z_tot[cond_mag]
    #
    #    m_b_mag = m_mag[(m_mag<=m_interp_b(z_mag))]
    #    m_f_mag = m_mag[(m_mag>m_interp_f(z_mag))]
    #    
    #    dm_b = (len(m_b_mag) - len(m_b))/len(m_b)
    #    s_b[j] = 0.4*dm_b*1/(mu_j-1)
    #    
    #    dm_f = (len(m_f_mag) - len(m_f))/len(m_f)
    #    s_f[j] = 0.4*dm_f*1/(mu_j-1)
    #
    #axs[0].plot(mu-1,s_b,c='orange',marker='o')
    #axs[1].plot(mu-1,s_f,c='lightseagreen',marker='o')
    ##axs[0].legend(loc='lower right')
    ##axs[1].legend(loc='lower right')
    #
    #plt.show()
    
    print(cut_bright, '/', cut_faint, f'mean magnification biases: s_b = {s_b}, s_f = {s_f}, ds =', s_b-s_f)
        
        
        

        
        
