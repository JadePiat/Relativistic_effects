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
from Relativistic_effects.split_mocks_eff import split_mag_eff

cosmo = CosmologyAbacus(0)  #c000 cosmology
kcorr_r = GAMA_KCorrection(cosmo, k_corr_file=lookup.kcorr_file, cubic_interpolation=True)


def magnitude_bias(file_name):
    
    sky = fits.open(file_name)
    data = sky[1].data
    sky.close()
    
    ...
    


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
    

def split_mag(input_file, space, m_lim, z_max, n_bins, cut_bright, cut_faint, output_path, bias = False):
    
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
    
    m_tot = apparent_mag(M,z_tot,g_r)
    m_cos = apparent_mag(M,z_cos,g_r)
    mu = 10**(0.4*(m_cos - m_tot))
    
    print(np.mean(mu))
    
    cond = (np.isnan(m_tot)==False)*(m_tot<=m_lim)*(z<=z_max)
    m = m_tot[cond]
    z_g = z_tot[cond]
    
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
    
    
    m_interp_b = CubicSpline(z_means, m_cuts_b, extrapolate=True)
    m_interp_f = CubicSpline(z_means, m_cuts_f, extrapolate=True)
    
    cond_b = (m<=m_interp_b(z_g))
    cond_f = (m>m_interp_f(z_g))
    
    m_b = m[cond_b]
    m_f = m[cond_f]
    
    
    if bias:
    
        fig, axs = plt.subplots(1, 2, figsize = (9,5), layout='tight')
        fig.suptitle(f'{cut_faint}% / {cut_bright}%')
        #axs[0,0].set_title('Bright')
        #axs[0,1].set_title('Faint')
        axs[1].set_ylabel(r's$\rm _B$')
        axs[0].set_ylabel(r's$\rm _F$')
        axs[0].set_xlabel(r'$\mu$-1')
        axs[1].set_xlabel(r'$\mu$-1')
        
        mu = np.arange(1.0001,1.011,0.001)
        sb = np.zeros(len(mu))
        sf = np.zeros(len(mu))
        
        s_eff = split_mag_eff(input_file, space, m_lim, z_max, n_bins, cut_bright, cut_faint, output_path)
        
        sb_eff = s_eff[0]
        sf_eff = s_eff[1]
        
        for j,mu_j in enumerate(mu):
            
            m_mag = m_tot - 2.5*np.log10(mu_j)
            
            cond_mag = (np.isnan(m_tot)==False)*(m_mag<=m_lim)*(z<=z_max)
            m_mag = m_mag[cond_mag]
            z_g_mag = z_tot[cond_mag]
            
            cond_b_mag = (m_mag<=m_interp_b(z_g_mag))
            cond_f_mag = (m_mag>m_interp_f(z_g_mag))
        
            m_b_mag = m_mag[cond_b_mag]
            m_f_mag = m_mag[cond_f_mag]
            
            dm_b = abs(len(m_b_mag) - len(m_b))/len(m_b)
            s_b = 0.4*dm_b*1/(mu_j-1)
            
            dm_f = abs(len(m_f_mag) - len(m_f))/len(m_f)
            s_f = 0.4*dm_f*1/(mu_j-1)
            
            sb[j] = s_b
            sf[j] = s_f
            
        axs[0].axhline(sf_eff,ls='--',color='k',label='effective value')
        axs[1].axhline(sb_eff,ls='--',color='k',label='effective value')
        
        axs[0].plot(mu-1,sf,c='lightseagreen',marker='o')
        axs[1].plot(mu-1,sb,c='orange',marker='o')
        
        axs[0].axhspan(-0.02*sf_eff+sf_eff, 0.02*sf_eff+sf_eff, color='gray', alpha=0.5,label='2%')
        axs[1].axhspan(-0.01*sb_eff+sb_eff, 0.01*sb_eff+sb_eff, color='gray', alpha=0.5,label='1%')
        
        axs[0].legend(loc='lower right')
        axs[1].legend(loc='lower right')
        plt.show()
        
    
    
    if space == 'real':
    
        z_b = z_cos[cond][cond_b]
        z_f = z_cos[cond][cond_f]
        
    if space == 'grav':
        
        z_b = z_g[cond_b]
        z_f = z_g[cond_f]
    
    #ra_b = ra[cond][cond_b]
    #ra_f = ra[cond][cond_f]
    #dec_b = dec[cond][cond_b]
    #dec_f = dec[cond][cond_f]
    #dist_b = cosmo.comoving_distance(z_b)
    #dist_f = cosmo.comoving_distance(z_f)
    
    #output_file = output_path+'cutsky_'+space+'_zmax'+str(z_max)+'_m'+str(m_lim)
    #output_b = output_file+'_bright_'+str(cut_bright)+'.fits'
    #output_f = output_file+'_faint_'+str(cut_faint)+'.fits'
    
    #write_file(ra_b,dec_b,dist_b,output_b)
    #write_file(ra_f,dec_f,dist_f,output_f)
    
        
        
        
    ###### figures    
        
    
    #plt.figure()
    #plt.plot(np.linspace(0, z_max, 50),m_interp_b(np.linspace(0, z_max, 50)),label='magnitude cut bright')
    #plt.plot(np.linspace(0, z_max, 50),m_interp_f(np.linspace(0, z_max, 50)),label='magnitude cut faint')
    #plt.xlabel('z')
    #plt.ylabel('m')
    #plt.legend()
    #plt.show()
    
    #plt.figure()
    #plt.scatter(ra[cond],dec[cond],s=2,alpha=0.5)
    #plt.xlabel('RA [deg]',fontsize=13)
    #plt.ylabel('Dec [deg]',fontsize=13)
    #plt.show()
    #
    #plt.figure()
    #plt.title(f'{cut_faint}% / {cut_bright}%',fontsize=13)
    #plt.hist(z_g, bins=np.linspace(0, 0.5, 51),color='green',label='full mock',histtype='step',density=True,alpha=0.5,zorder=1)
    #plt.hist((z_f,z_b),bins=np.linspace(0, 0.5, 51),label=('faint','bright'),histtype='step',density=True,zorder=0)
    #plt.xlabel('z',fontsize=13)
    #plt.ylabel('n(z)',fontsize=13)
    #plt.legend(fontsize=10)
    #plt.show()
    #
    #plt.figure()
    #plt.hist(z_g, bins=np.linspace(0, 0.55, 51),alpha=0.5)
    #plt.xlabel('z',fontsize=13)
    #plt.ylabel('n(z)',fontsize=13)
    #plt.show()
    #
    #plt.figure()
    #plt.hist(m, bins=np.linspace(10,19.5,51),alpha=0.5)
    #plt.xlabel('m',fontsize=13)
    #plt.ylabel('n(m)',fontsize=13)
    #plt.show()
    #
    #bins = np.linspace(12,np.max(m_f),51)
    #plt.figure()
    #plt.title('10/90')
    #plt.hist(m,bins=bins,label='full',color='green',alpha=0.5,lw=1.5,histtype='step')
    #plt.hist(m_f,bins=bins,label='faint',alpha=0.5,lw=1.5,histtype='step')
    #plt.hist(m_b,bins=bins,label='bright',alpha=0.5,lw=1.5,histtype='step')
    #plt.xlabel('m',fontsize=13)
    #plt.ylabel('dN/dm',fontsize=13)
    #plt.legend(fontsize=10)
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
    
    
    #return m_interp_b, m_interp_f
