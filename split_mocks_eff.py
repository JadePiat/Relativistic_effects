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


def distances(data):

    data['DIST_COM'] = cosmo.comoving_distance(data['Z_COSMO'])
    data['DIST_RSD'] = cosmo.comoving_distance(data['Z'])
    #data['DIST_RSD+GR'] = cosmo.comoving_distance(data['Z_RSD+GR'])


def write_file(data,output_file):
    
    t = data['RA','DEC','DIST_COM','DIST_RSD']  #'DIST_RSD+GR'
    t.write(output_file,format='fits',overwrite=True)


def magnitude_split(input_file, m_lim, z_max, n_bins, cut_bright, cut_faint, output_path):
    
    cat = fits.open(input_file)
    data = Table(cat[1].data)
    cat.close()
    
    #M = data['R_MAG_ABS']
    #g_r = data['G_R_REST']
    
    #phi = data['Pot A Soft 7']
    #phi0 = phi[z_cos==np.min(z_cos)]
    
    z_cos = data['Z_COSMO']
    z_tot = data['Z']
    #z_tot = z_grav(z_cos, z, phi, phi0)
    
    #data['Z_RSD+GR'] = z_tot
    
    #m = apparent_mag(M,z_tot,g_r)
    m_tot = data['R_MAG_APP']
    
    cond = (np.isnan(m)==False)*(m_tot<=m_lim)*(z_tot<=z_max)
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
    
    cond_b = (m<=m_interp_b(z_tot))
    cond_f = (m>m_interp_f(z_tot))
    
    data_b = data[cond][cond_b]
    data_f = data[cond][cond_f]
    
    distances(data_b)
    distances(data_f)
    
    output_file = output_path+f'cutsky_zmax{z_max:.1f}_m{m_lim:.1f}'
    output_b = output_file+f'_bright_{cut_bright:d}.fits'
    output_f = output_file+f'_faint_{cut_faint:d}.fits'
    
    write_file(data_b,output_b)
    write_file(data_f,output_f)
    
    
   
    
def magnification_bias(m):

    n,bins = np.histogram(m,100)
    N = np.cumsum(n)
    m_bin = 0.5*(bins[1:]+bins[:-1])
    
    #N = len(m)
    #m_dist = np.arange(1,N+1,1)
    
    #plt.figure()
    #plt.plot(m_sort[:-1000], m_dist[:-1000])
    #plt.show()
    
    s = (np.log10(N[-1])-np.log10(N[-2]))/(m_bin[-1]-m_bin[-2])
    
    return s



##### magnification bias #####
#    
#    z_bins = np.linspace(0., z_max, 11)
#    s_b = np.zeros(len(z_bins)-1)
#    s_f = np.zeros(len(z_bins)-1)
#    s_b_eff = 0
#    s_f_eff = 0
#    N_b = 0
#    N_f = 0
#    
#    for i in range(len(z_bins)-1):
#        
#        if i==(n_bins-1):
#            keep = (z_tot>=z_bins[i])*(z_tot<=z_bins[i+1])
#            keep_b = (data_b['Z']>=z_bins[i])*(data_b['Z']<=z_bins[i+1])
#            keep_f = (data_f['Z']>=z_bins[i])*(data_f['Z']<=z_bins[i+1])
#        else:
#            keep = (z_tot>=z_bins[i])*(z_tot<z_bins[i+1])
#            keep_b = (data_b['Z']>=z_bins[i])*(data_b['Z']<=z_bins[i+1])
#            keep_f = (data_f['Z']>=z_bins[i])*(data_f['Z']<=z_bins[i+1])
#            
#        m_b = data_b['R_MAG_APP'][keep_b]
#        m_f = data_f['R_MAG_APP'][keep_f]
#        m_bin = m[keep]
#        
#        si = magnification_bias(m_bin)
#        si_b = magnification_bias(m_b)
#        
#        if (cut_bright + cut_faint) != 100:
#            
#            print('cut_b + cut_f should be 100')
#            
#            #m_ = m_bin[m_bin<=m_cuts_f[i]]
#            #s_ = magnification_bias(m_)
#            #s_f = s*len(m_bin)/len(m_f) - s_*len(m_)/len(m_f)     
#
#        else:
#            
#            si_f = si*len(m_bin)/len(m_f) - si_b*len(m_b)/len(m_f)
#        
#        s_b[i] = si_b
#        s_f[i] = si_f
#        s_b_eff += len(m_b)*si_b
#        s_f_eff += len(m_f)*si_f
#        N_b += len(m_b)
#        N_f += len(m_f)
#        
#    s_b_eff /= N_b
#    s_f_eff /= N_f
#    
#    print(cut_bright, '/', cut_faint)
#    print(f'effective magnification biases: s_b = {s_b_eff}, s_f = {s_f_eff}, ds =', s_b_eff-s_f_eff)
#    print(f'magnification biases: s_b(z) = {s_b}, s_f(z) = {s_f}, ds(z) =', s_b-s_f)


