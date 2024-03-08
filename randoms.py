import sys
sys.path.append('/global/homes/j/jpiat/')
sys.path.append('/global/homes/j/jpiat/shared_code/abacus_mocks/')

import numpy as np
from astropy.io import fits
from astropy.table import vstack,Table
from Relativistic_effects.gravitational_redshift import z_grav
from hodpy.cosmology import CosmologyAbacus
from hodpy.k_correction import GAMA_KCorrection
from hodpy import lookup
cosmo = CosmologyAbacus(0)  #c000 cosmology
kcorr_r = GAMA_KCorrection(cosmo, k_corr_file=lookup.kcorr_file, cubic_interpolation=True)


def main(path_in, path_out, n_randoms, space, z_max, m_lim):

    for i in range(1,n_randoms+1):
        
        input_file = path_in+'random_S'+str(i)+'00_1X.fits'
        
        sky = fits.open(input_file)
        data = Table(sky[1].data)
        sky.close()
    
        cond = (data['R_MAG_APP']<=m_lim) & (data['Z']<=z_max)
                           
        data = data[cond]
        
        if space == 'real':
        
            data['COM_DIST'] = cosmo.comoving_distance(data['Z_COSMO'])
            
        else:
            
            data['COM_DIST'] = cosmo.comoving_distance(data['Z'])
        
        if i == 1:
            
            table = data['RA','DEC','COM_DIST']
            
            continue
        
        table = vstack([table,data['RA','DEC','COM_DIST']])
        
    output_file = path_out+'randoms_'+str(n_randoms)+'_cutsky_'+space+'_zmax'+str(z_max)+'.fits'
        
    table.write(output_file,format='fits',overwrite=True)



if __name__ == "__main__":
    
    path_in = '/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/CutSky/BGS/v0.1/z0.200/'
    path_out = '/pscratch/sd/j/jpiat/Abacus/Ab_c000_ph006/z0.200/Split/BGS/'

    n_randoms = 10
    
    main(path_in, path_out, n_randoms, 'grav', 0.5, 19.5)
    main(path_in, path_out, n_randoms, 'real', 0.5, 19.5)
    
    
    