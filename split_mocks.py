import sys
sys.path.append('/global/homes/j/jpiat/shared_code/abacus_mocks/')

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from hodpy.k_correction import GAMA_KCorrection
from hodpy import lookup
from hodpy.cosmology import CosmologyAbacus
from scipy.interpolate import CubicSpline
from Relativistic_effects.gravitational_redshift import z_grav

cosmo = CosmologyAbacus(0)  #c000 cosmology
kcorr_r = GAMA_KCorrection(cosmo, k_corr_file=lookup.kcorr_file, cubic_interpolation=True)


def write_file(ra,dec,dist,output_file):
    
    c1 = fits.Column(name='RA', format='E', array=ra)
    c2 = fits.Column(name='DEC', format='E', array=dec)
    c3 = fits.Column(name='dist', format='E', array=dist)
    
    t = fits.BinTableHDU.from_columns([c1, c2, c3])
    t.writeto(output_file,overwrite=True)
    

def split_magnitudes(input_file, space, m_lim, z_max, n_bins, cut_bright, cut_faint, output_path):
    
    sky = fits.open(input_file)
    data = sky[1].data
    sky.close()
    
    M = data['R_MAG_ABS']
    g_r = data['G_R_REST']
    ra = data['RA']
    dec = data['DEC']
    phi = data['Pot A Soft 7']
    phi0 = 0
    
    if space == 'real':
        
        z = data['Z_COSMO']
        
    elif space == 'rsd':
    
        z = data['Z']
        
    elif space == 'obs':
        
        z_rsd = data['Z']
        z_cosmo = data['Z_COSMO']
        z = z_grav(z_cosmo, z_rsd, phi, phi0)
        
    m = kcorr_r.apparent_magnitude(M, z, g_r)
    
    cond = (np.isnan(m)==False)*(m<=m_lim)*(z<=z_max)
    m = m[cond]
    z = z[cond]
    
    z_bins = np.linspace(0, z_max, n_bins+1)
    
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
    N = 0
    
    
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
        
        # compute magnification bias
        
        mi_b = mi[mi<=m_cut_b]
        
        if cut_bright == cut_faint:
            mi_f = mi[mi>=m_cut_f]
        else:
            mi_f = mi[mi>m_cut_f]
        
        n_b,bins_b = np.histogram(mi_b,bins=50)
        cn_b = np.cumsum(n_b)
        centres_b = (bins_b[1:]+bins_b[:-1])/2
        si_b = (np.log10(cn_b[-1])-np.log10(cn_b[-2]))/(centres_b[-1]-centres_b[-2])
        s_b[i] = si_b
        
        n,bins = np.histogram(mi,bins=50)
        cn = np.cumsum(n)
        centres = (bins[1:]+bins[:-1])/2
        si = (np.log10(cn[-1])-np.log10(cn[-2]))/(centres[-1]-centres[-2])
        
        if cut_bright == cut_faint:
       
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
      
        s_eff += len(mi)*si
        s_b_eff += len(mi_b)*si_b
        s_f_eff += len(mi_f)*si_f
        N_b += len(mi_b)
        N_f += len(mi_f)
        N += len(mi)
    
    s_b_eff /= N_b
    s_f_eff /= N_f
    s_eff /= N
    
    print(f'effective magnification biases: s_bright = {s_b_eff}, s_faint = {s_f_eff}, $\Delta$s =', s_b_eff-s_f_eff)
    
    
    m_interp_b = CubicSpline(z_means, m_cuts_b, extrapolate=True)
    m_interp_f = CubicSpline(z_means, m_cuts_f, extrapolate=True)
    
    cond_b = (m<=m_interp_b(z))
    cond_f = (m>m_interp_f(z))
    z_b = z[cond_b]
    z_f = z[cond_f]
    m_b = m[cond_b]
    m_f = m[cond_f]
    ra_b = ra[cond][cond_b]
    ra_f = ra[cond][cond_f]
    dec_b = dec[cond][cond_b]
    dec_f = dec[cond][cond_f]
    dist_b = cosmo.comoving_distance(z_b)
    dist_f = cosmo.comoving_distance(z_f)
    
    output_file = output_path+'cutsky_'+space+'_zmax'+str(z_max)+'_m'+str(m_lim)
    output_b = output_file+'_bright_'+str(cut_bright)+'.fits'
    output_f = output_file+'_faint_'+str(cut_faint)+'.fits'
    
    #plt.figure()
    #plt.plot(np.linspace(0, z_max, 50),m_interp_b(np.linspace(0, z_max, 50)),label='magnitude cut bright')
    #plt.plot(np.linspace(0, z_max, 50),m_interp_f(np.linspace(0, z_max, 50)),label='magnitude cut faint')
    #plt.xlabel('z')
    #plt.ylabel('m')
    #plt.legend()
    #plt.show()
    #
    #plt.figure()
    #plt.hist((z_f,z_b),bins=np.linspace(0, z_max, 50),label=('faint','bright'),density=True)
    #plt.xlabel('z')
    #plt.ylabel('n(z)')
    #plt.legend()
    #plt.show()
    #
    #plt.figure()
    #plt.hist(m_f,bins=50,label='faint',histtype='step')
    #plt.hist(m_b,bins=50,label='bright',histtype='step')
    #plt.xlabel('m')
    #plt.ylabel('n(m)')
    #plt.legend()
    #plt.show()
    
    
    write_file(ra_b,dec_b,dist_b,output_b)
    write_file(ra_f,dec_f,dist_f,output_f)
    
    return s_b_eff, s_f_eff, s_b_eff-s_f_eff