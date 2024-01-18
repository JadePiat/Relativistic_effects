import sys
sys.path.append('/global/homes/j/jpiat/shared_code/abacus_mocks/')

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from hodpy.k_correction import GAMA_KCorrection
from hodpy import lookup
from hodpy.cosmology import CosmologyAbacus
from scipy.interpolate import CubicSpline

cosmo = CosmologyAbacus(0)  #c000 cosmology
kcorr_r = GAMA_KCorrection(cosmo, k_corr_file=lookup.kcorr_file, cubic_interpolation=True)


def write_file(ra,dec,dist,output_file):
    
    c1 = fits.Column(name='RA', format='E', array=ra)
    c2 = fits.Column(name='DEC', format='E', array=dec)
    c3 = fits.Column(name='dist', format='E', array=dist)
    
    t = fits.BinTableHDU.from_columns([c1, c2, c3])
    t.writeto(output_file,overwrite=True)


def split_magnitudes_randoms(path, n_randoms, m_lim, z_cut, n_bins, cut_bright, cut_faint, output_path):

    output_file_b = output_path+'random_zobs0.5_m19.5_bright_'+str(cut_bright)+'.fits'
    output_file_f = output_path+'random_zobs0.5_m19.5_bright_'+str(cut_faint)+'.fits'
    
    randoms_ra_b, randoms_ra_f = np.array([]), np.array([]) 
    randoms_dec_b, randoms_dec_f =  np.array([]), np.array([]) 
    randoms_dist_b, random_dist_f = np.array([]), np.array([]) 
    
    for i in range(1,n_randoms+1):
    
        sky = fits.open(path+'random_S'+str(i)+'00_1X.fits')
        data = sky[1].data
        sky.close()
    
        z = data['Z']
        M = data['R_MAG_ABS']
        g_r_rest = data['G_R_REST']
        ra = data['RA']
        dec = data['DEC']
    
        m = kcorr_r.apparent_magnitude(M, z, g_r)
    
        cond = (np.isnan(m)==False)*(m<=m_lim)*(z<=z_cut)
        m = m[cond]
        z = z[cond]
    
        z_bins = np.linspace(0, z_cut, n_bins+1)
    
        z_means = np.zeros(n_bins)
        m_cuts_b = np.zeros(n_bins)
        m_cuts_f = np.zeros(n_bins)  
    
        for i in range(n_bins):
            if i==(n_bins-1):
                bin = (z>=z_bins[i])*(z<=z_bins[i+1])
            else:
                bin = (z>=z_bins[i])*(z<z_bins[i+1])
    
            mi = m[bin]
            zi = z[bin]
        
            z_means[i] = np.mean(zi)
        
            m_cut_b = np.percentile(mi,cut_bright)
            m_cut_f = np.percentile(mi,100-cut_faint)
            m_cuts_b[i] = m_cut_b
            m_cuts_f[i] = m_cut_f
    
        m_interp_b = CubicSpline(z_means, m_cuts_b, extrapolate=True)
        m_interp_f = CubicSpline(z_means, m_cuts_f, extrapolate=True)
    
        cond_b, cond_f = (m<=m_interp_b(z)), (m>m_interp_f(z))
        
        z_b, z_f = z[cond_b], z[cond_f]
        ra_b, ra_f = ra[cond][cond_b], ra[cond][cond_f]
        dec_b, dec_f = dec[cond][cond_b], dec[cond][cond_f]
        
        dist_b = cosmo.comoving_distance(z_b)
        dist_f = cosmo.comoving_distance(z_f)
        
        randoms_ra_b = np.concatenate((randoms_ra_b, ra_b), axis=None)
        randoms_ra_f = np.concatenate((randoms_ra_f, ra_f), axis=None)
        randoms_dec_b = np.concatenate((randoms_dec_b, dec_b), axis=None)
        randoms_dec_f = np.concatenate((randoms_dec_f, dec_f), axis=None)
        randoms_dist_b = np.concatenate((random_dist_b, dist_b), axis=None)
        randoms_dist_f = np.concatenate((random_dist_f, dist_f), axis=None)
        
    
    write_file(randoms_ra_b,randoms_dec_b,randoms_dist_b,output_file_b)
    write_file(randoms_ra_f,randoms_dec_f,randoms_dist_f,output_file_f)