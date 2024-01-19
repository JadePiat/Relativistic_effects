import sys
sys.path.append('/global/homes/j/jpiat/shared_code/abacus_mocks/')

from astropy.io import fits
from astropy.table import vstack,Table
from hodpy.cosmology import CosmologyAbacus
cosmo = CosmologyAbacus(0)  #c000 cosmology

path_in = '/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/CutSky/BGS/v0.1/z0.200/'
path_out = 'pscratch/sd/j/jpiat/Abacus/Ab_c000_ph006/z0.200/Split/BGS/'

n_randoms = 10

def main(path_in, path_out, z_max):

    for i in range(1,n_randoms+1):
        
        input_file = path_in+'random_S'+str(i)+'00_1X.fits'
        
        print(1)
        
        sky = fits.open(input_file)
        data = Table(sky[1].data)
        sky.close()
                           
        data = data[data['Z_COSMO']<=z_max]
        
        data['DIST'] = cosmo.comoving_distance(data['Z_COSMO'])
        
        if i == 1:
            
            table = data['RA','DEC','DIST']
            
            print(2)
            
            continue
        
        table = vstack([table,data['RA','DEC','DIST']])
        
    output_file = path_out+'randoms_'+str(n_randoms)+'_cutsky_zmax'+str(z_max)+'.fits'
        
    table.write(output_file,format='fits')



if __name__ == "__main__":
    
    main(path_in, path_out, 0.5)
    
    
    