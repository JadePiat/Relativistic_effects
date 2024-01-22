import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
import scipy.constants as const
import matplotlib.pyplot as plt
from pypower import CatalogMesh, MeshFFTPower, CatalogFFTPower, PowerSpectrumStatistics, utils, setup_logging


def read_file(file):
    
    sky = fits.open(file)
    data = sky[1].data
    sky.close()
    
    return np.array(data['RA','DEC','COM_DIST'])



def Pk_check(Pk, ell):
    
    if ell%2 == 0:
           return = Pk.real
    else:
           return = Pk.imag
            


def get_Pk(data1, data2, rand1, rand2, output_file, ells, kedges = np.arange(0,0.5,0.005):
    
    
    data_pos1 = read_file(data1)
    data_pos2 = read_file(data2)
    rand_pos1 = read_file(rand1)
    rand_pos2 = read_file(rand2)
    
    
    result = CatalogFFTPower(data_positions1=data_pos1, data_positions2=data_pos2,
                               randoms_positions1=rand_pos1, randoms_positions2=rand_pos2,
                               edges=kedges, ells=ells, nmesh=1024, resampler='tsc', position_type='rdd')
    
           
    k, Pk = result.poles(ell = ells[0])
    
    data = np.zeros((len(k),len(ells)+1))
    data[:,0] = k
    data[:,1] = Pk_check(Pk, ells[0])
    
    if len(ells) > 1:
           
           for i,ell in enumerate(ells[1:]):
                   
                   Pk = result.poles.power(ell = ell)
                   data[:,i+2] = Pk_check(Pk, ell)
                   
    
    np.savetxt(output_file, data)
           
        
        

if __name__ == "__main__" :
           
        path = '/pscratch/sd/j/jpiat/Abacus/Ab_c000_ph006/z0.200/Split/BGS/'
        file_random = path+'randoms_10_cutsky_zmax0.5.fits'

        cuts = [10,20,30,40,50,60,70,80,90]
           
        for cut in cuts:
           
           file_bright = path+f'cutsky_real_zmax0.5_m19.5_bright_{cut}.fits'
           file_faint = path+f'cutsky_real_zmax0.5_m19.5_faint_{cut}.fits'
           output_bright = f'/global/homes/j/jpiat/data/Pk_bright_{cut}.dat'
           output_faint = f'/global/homes/j/jpiat/data/Pk_faint_{cut}.dat'
           
           get_Pk(file_bright, file_bright, file_random, file_random, output_bright, ells=(0))
           get_Pk(file_faint, file_faint, file_random, file_random, output_faint, ells=(0))
           
           