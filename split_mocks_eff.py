import sys
sys.path.append('/global/homes/j/jpiat/shared_code/abacus_mocks/')

import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from hodpy.k_correction import GAMA_KCorrection
from hodpy import lookup
from hodpy.cosmology import CosmologyAbacus
from scipy.interpolate import CubicSpline
from Relativistic_effects.gravitational_redshift import z_grav

cosmo = CosmologyAbacus(0)  #c000 cosmology
kcorr_r = GAMA_KCorrection(cosmo, k_corr_file=lookup.kcorr_file, cubic_interpolation=True)


def apparent_mag(M,z,g_r,Q=0.8,z0=0.1):
    
    K = kcorr_r.apparent_magnitude(M, z, g_r)
    EC = Q*(z - z0)
    
    return K - EC


def distances(data):

    data['DIST_COM'] = cosmo.comoving_distance(data['Z_COSMO'])
    data['DIST_RSD'] = cosmo.comoving_distance(data['Z_RSD'])
    data['DIST_RSD+GR'] = cosmo.comoving_distance(data['Z_RSD+GR'])


def write_file(data,output_file):
    
    t = data['RA','DEC','DIST_COM','DIST_RSD','DIST_RSD+GR']
    t.write(output_file,format='fits',overwrite=True)


def magnitude_split(input_file, m_lim, z_max, n_bins, cut_bright, cut_faint, output_path):
    
    cat = fits.open(input_file)
    data = Table(cat[1].data)
    cat.close()
    
    M = data['R_MAG_ABS']
    g_r = data['G_R_REST']
    z_cos = data['Z_COSMO']    
    
    phi = data['Pot A Soft 7']
    phi0 = phi[z_cos==np.min(z_cos)]
    
    z = data['Z']
    z_tot = z_grav(z_cos, z, phi, phi0)
    
    data.columns['Z'].name = 'Z_RSD'
    data['Z_RSD+GR'] = z_tot
    
    m = apparent_mag(M,z_tot,g_r)
    
    cond = (np.isnan(m)==False)*(m<=m_lim)*(z_tot<=z_max)
    m = m[cond]
    z_tot = z_tot[cond]
    
    z_bins = np.linspace(0, z_max, n_bins+1)
    
    z_means = np.zeros(n_bins)
    m_cuts_b = np.zeros(n_bins)
    m_cuts_f = np.zeros(n_bins)
    
    for i in range(n_bins):
        if i==(n_bins-1):
            inside = (z_tot>=z_bins[i])*(z_tot<=z_bins[i+1])
        else:
            inside = (z_tot>=z_bins[i])*(z_tot<z_bins[i+1])
        
        z_means[i] = np.mean(z_tot[inside])
        
        m_cuts_b[i] = np.percentile(m[inside],cut_bright)
        m_cuts_f[i] = np.percentile(m[inside],100-cut_faint)
    
    m_interp_b = CubicSpline(z_means, m_cuts_b, extrapolate=True)
    m_interp_f = CubicSpline(z_means, m_cuts_f, extrapolate=True)
    
    cond_b = (m<m_interp_b(z_tot))
    cond_f = (m>=m_interp_f(z_tot))
    
    data_b = data[cond][cond_b]
    data_f = data[cond][cond_f]
    
    distances(data_b)
    distances(data_f)
    
    output_file = output_path+f'cutsky_zmax{z_max:.1f}_m{m_lim:.1f}'
    output_b = output_file+f'_bright_{cut_bright:d}.fits'
    output_f = output_file+f'_faint_{cut_faint:d}.fits'
    
    write_file(data_b,output_b)
    write_file(data_f,output_f)
    
    




def eff_magnification_bias(input_file, space, m_lim, z_max, n_bins, cut_bright, cut_faint, output_path):
    
    cat = fits.open(input_file)
    data = cat[1].data
    cat.close()
    
    M = data['R_MAG_ABS']
    g_r = data['G_R_REST']
    
    ra = data['RA']
    dec = data['DEC']
    
    phi = data['Pot A Soft 7']
    phi0 = phi[z_cos==np.min(z_cos)]
    
    z_cos = data['Z_COSMO']    
    z = data['Z']
    z_tot = z_grav(z_cos, z, phi, phi0)
    
    m = apparent_mag(M,z_tot,g_r)
    
    cond = (np.isnan(m)==False)*(m<=m_lim)*(z_tot<=z_max)
    m = m[cond]
    M = M[cond]
    z_g = z_tot[cond]
    z_cos = z_cos[cond]
    
    z_bins = np.linspace(0, np.max(z_g), n_bins+1)
    
    #hm, hbm = np.histogram(m,bins=50,density=True)
    #binsm = 0.5*(hbm[:-1]+hbm[1:])
    #chm = np.cumsum(hm)
    
    z_means = np.zeros(n_bins)
    m_cuts_b = np.zeros(n_bins)
    m_cuts_f = np.zeros(n_bins)
    s_b = np.zeros(n_bins)
    s_f = np.zeros(n_bins)
    s_b_eff = 0
    s_f_eff = 0
    s_eff = 0
    N_b = 0
    N_f = 0
    
    
    for i in range(n_bins):
        if i==(n_bins-1):
            bin = (z_g>=z_bins[i])*(z_g<=z_bins[i+1])
        else:
            bin = (z_g>=z_bins[i])*(z_g<z_bins[i+1])
    
        mi = m[bin]
        Mi = M[bin]
        zi = z_g[bin]
        
        z_means[i] = np.mean(zi)
        
        m_cut_b = np.percentile(mi,cut_bright)
        m_cut_f = np.percentile(mi,100-cut_faint)
        m_cuts_b[i] = m_cut_b
        m_cuts_f[i] = m_cut_f
        
        # compute magnification bias
        
        mi_b = mi[mi<=m_cut_b]
        mi_f = mi[mi>m_cut_f]
        
        Mi_b = Mi[mi<=m_cut_b]
        Mi_f = Mi[mi>m_cut_f]
        
        n_b,bins_b = np.histogram(mi_b,bins=50)
        cn_b = np.cumsum(n_b)
        centres_b = (bins_b[1:]+bins_b[:-1])/2
        si_b = (np.log10(cn_b[-1])-np.log10(cn_b[-2]))/(centres_b[-1]-centres_b[-2])
        s_b[i] = si_b
        
        n,bins = np.histogram(mi,bins=50)
        cn = np.cumsum(n)
        centres = (bins[1:]+bins[:-1])/2
        si = (np.log10(cn[-1])-np.log10(cn[-2]))/(centres[-1]-centres[-2])
        
        
        if (cut_bright + cut_faint) != 100:
       
            mi_ =  mi[mi<=m_cut_f]
            n_,bins_ = np.histogram(mi_,bins=50)
            cn_ = np.cumsum(n_)
            centres_ = (bins_[1:]+bins_[:-1])/2
            si_ = (np.log10(cn_[-1])-np.log10(cn_[-2]))/(centres_[-1]-centres_[-2])
            
            si_f = si*len(mi)/len(mi_f)-si_*len(mi_)/len(mi_f)
            s_f[i] = si_f
            
       
        else:
            si_f = si*len(mi)/len(mi_f)-si_b*len(mi_b)/len(mi_f)
            s_f[i] = si_f
              
        #s_eff += len(mi)*si
        s_b_eff += len(mi_b)*si_b
        s_f_eff += len(mi_f)*si_f
        N_b += len(mi_b)
        N_f += len(mi_f)
            

    s_b_eff /= N_b
    s_f_eff /= N_f
    #s_eff /= len(m)
    
    
    print(cut_bright, '/', cut_faint, f'\neffective magnification biases: s_bright = {s_b_eff}, s_faint = {s_f_eff}, ds =', s_b_eff-s_f_eff)
    
    
    m_interp_b = CubicSpline(z_means, m_cuts_b, extrapolate=True)
    m_interp_f = CubicSpline(z_means, m_cuts_f, extrapolate=True)
    
    cond_b = (m<m_interp_b(z_g))
    cond_f = (m>=m_interp_f(z_g))    
    
    m_b = m[cond_b]
    m_f = m[cond_f]
    
    n,bins = np.histogram(m,bins=50)
    cn = np.cumsum(n)
    centres = (bins[1:]+bins[:-1])/2
    s = (np.log10(cn[-1])-np.log10(cn[-2]))/(centres[-1]-centres[-2])
    
    test = s*len(m)/len(m_f)-s_b_eff*len(m_b)/len(m_f)
    
    print('test:', test)
    
    #h, bins = np.histogram(m,bins=50)
    #bins = (bins[1:]+bins[:-1])/2
    #ch = np.cumsum(h)
    
    #hb, binsb = np.histogram(m_b,bins=50)
    #binsb = (binsb[1:]+binsb[:-1])/2
    #chb = np.cumsum(hb)
    
    #hf, binsf = np.histogram(m_f,bins=50)
    #binsf = (binsf[1:]+binsf[:-1])/2
    #chf = np.cumsum(hf)
       
    #s = (np.log10(ch[-1])-np.log10(ch[-2]))/(bins[-1]-bins[-2])
    #sb = (np.log10(chb[-1])-np.log10(chb[-2]))/(binsb[-1]-binsb[-2])
    #sf = (np.log10(chf[-1])-np.log10(chf[-2]))/(binsf[-1]-binsf[-2])
    
    #print(s, sb, sf)

    
    #if space == 'real':
    #
    #    z_b = z_cos[cond_b]
    #    z_f = z_cos[cond_f]
    #    
    #if space == 'grav':
    #    
    #    z_b = z_g[cond_b]
    #    z_f = z_g[cond_f]
    #
    #ra_b = ra[cond][cond_b]
    #ra_f = ra[cond][cond_f]
    #dec_b = dec[cond][cond_b]
    #dec_f = dec[cond][cond_f]
    #dist_b = cosmo.comoving_distance(z_b)
    #dist_f = cosmo.comoving_distance(z_f)
    #
    #output_file = output_path+'cutsky_'+space+'_zmax'+str(z_max)+'_m'+str(m_lim)
    #output_b = output_file+'_bright_'+str(cut_bright)+'.fits'
    #output_f = output_file+'_faint_'+str(cut_faint)+'.fits'
    #
    #
    #write_file(ra_b,dec_b,dist_b,output_b)
    #write_file(ra_f,dec_f,dist_f,output_f)
    
    return np.array([s_b_eff, s_f_eff, s_b_eff-s_f_eff])