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


def apparent_mag(M,z,g_r,Q=0.8):
    
    K = kcorr_r.apparent_magnitude(M, z, g_r)
    EC = Q*(z - 0.1)
    
    return K - EC


def write_file(ra,dec,dist,output_file):
    
    c1 = fits.Column(name='RA', format='E', array=ra)
    c2 = fits.Column(name='DEC', format='E', array=dec)
    c3 = fits.Column(name='COM_DIST', format='E', array=dist)
    
    t = fits.BinTableHDU.from_columns([c1, c2, c3])
    t.writeto(output_file,overwrite=True)
    

def split_magnitudes(input_file, space, m_lim, z_max, n_bins, cut_bright, cut_faint, mu, output_path):
    
    sky = fits.open(input_file)
    data = sky[1].data
    sky.close()
    
    M = data['R_MAG_ABS']
    g_r = data['G_R_REST']
    ra = data['RA']
    dec = data['DEC']
    
    z_cos = data['Z_COSMO']    
    z = data['Z']
    phi = data['Pot A Soft 7']
    phi0 = phi[z_cos==np.min(z_cos)]
    z_tot = z_grav(z_cos, z, phi, phi0)
    
    m = apparent_mag(M,z_tot,g_r)
    m_mag = m - 2.5*np.log10(mu)
    
    cond = (np.isnan(m)==False)*(m<=m_lim)*(z<=z_max)
    cond_mag = (np.isnan(m)==False)*(m_mag<=m_lim)*(z<=z_max)
    m = m[cond]
    m_mag = m_mag[cond_mag]
    z_g = z_tot[cond]
    z_g_mag = z_tot[cond_mag]
    z_cos = z_cos[cond]
    
    z_bins = np.linspace(0, np.max(z_g), n_bins+1)
    
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
            bin = (z_g>=z_bins[i])*(z_g<=z_bins[i+1])
        else:
            bin = (z_g>=z_bins[i])*(z_g<z_bins[i+1])
    
        mi = m[bin]
        zi = z_g[bin]
        
        z_means[i] = np.mean(zi)
        
        m_cut_b = np.percentile(mi,cut_bright)
        m_cut_f = np.percentile(mi,100-cut_faint)
        m_cuts_b[i] = m_cut_b
        m_cuts_f[i] = m_cut_f
        
        # compute magnification bias
        
        mi_b = mi[mi<=m_cut_b]
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
    
    print(cut_bright, '/', cut_faint, f'\neffective magnification biases: s_bright = {s_b_eff}, s_faint = {s_f_eff}, ds =', s_b_eff-s_f_eff)
    
    
    m_interp_b = CubicSpline(z_means, m_cuts_b, extrapolate=True)
    m_interp_f = CubicSpline(z_means, m_cuts_f, extrapolate=True)
    
    cond_b = (m<=m_interp_b(z_g))
    cond_f = (m>m_interp_f(z_g))
    
    cond_b_mag = (m_mag<=m_interp_b(z_g_mag))
    cond_f_mag = (m_mag>m_interp_f(z_g_mag))
    
    
    m_b = m[cond_b]
    m_f = m[cond_f]
    
    m_b_mag = m_mag[cond_b_mag]
    m_f_mag = m_mag[cond_f_mag]
    
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
    
    dm_b = abs(len(m_b_mag) - len(m_b))/len(m_b)
    s_b = 0.4*dm_b*1/(mu-1)
    
    dm_f = abs(len(m_f_mag) - len(m_f))/len(m_f)
    s_f = 0.4*dm_f*1/(mu-1)
    
    print(f'magnification method: s_bright = {s_b}, s_faint = {s_f}, ds =', s_b-s_f)
    
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
    #plt.figure()
    #plt.plot(np.linspace(0, z_max, 50),m_interp_b(np.linspace(0, z_max, 50)),label='magnitude cut bright')
    #plt.plot(np.linspace(0, z_max, 50),m_interp_f(np.linspace(0, z_max, 50)),label='magnitude cut faint')
    #plt.xlabel('z')
    #plt.ylabel('m')
    #plt.legend()
    #plt.show()
    #
    #plt.figure()
    #plt.hist((z_f,z_b),bins=np.linspace(0, 0.6, 50),label=('faint','bright'),density=True)
    #plt.hist(z, bins=np.linspace(0, 0.6, 50),label='full mock',density=True,alpha=0.5)
    #plt.hist(z_cos, bins=np.linspace(0, 0.6, 50),label='full mock',density=True,alpha=0.5)
    #plt.xlabel('z')
    #plt.ylabel('n(z)')
    #plt.legend()
    #plt.show()
    #
    #
    #plt.figure()
    #plt.hist(m_f,bins=50,label='faint',histtype='step')
    #plt.hist(m_b,bins=50,label='bright',histtype='step')
    #plt.xlabel('m')
    #plt.ylabel('n(m)')
    #plt.legend()
    #plt.show()
    
    #plt.figure()
    #plt.plot(bins, np.log10(ch), label = 'mock')
    #plt.plot(binsb, np.log10(chb), label = 'bright')
    #plt.plot(binsf, np.log10(chf), label = 'faint')
    #plt.ylabel('log(N<m)')
    #plt.xlabel('m')
    #plt.grid()
    #plt.legend()
    #plt.show()

    
    #write_file(ra_b,dec_b,dist_b,output_b)
    #write_file(ra_f,dec_f,dist_f,output_f)
    
    return np.array([s_b, s_f, s_b-s_f]), np.array([s_b_eff, s_f_eff, s_b_eff-s_f_eff])