import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
import scipy.constants as const
import matplotlib.pyplot as plt
from pypower import CatalogMesh, MeshFFTPower, CatalogFFTPower, PowerSpectrumStatistics, utils, setup_logging


def read_file(file):
    
    sky = Table.read(file)
    
    data = np.zeros((3,len(sky)))
    data[0] = sky['RA']
    data[1] = sky['DEC']
    data[2] = sky['COM_DIST']
    
    return data



def Pk_check(Pk, ell):
    
    if ell%2 == 0:
           return Pk.real
    else:
           return Pk.imag
            


def get_Pk(data1, data2, rand1, rand2, output_file, ells, kedges = np.arange(0,0.5,0.005)):
    
    
    data_pos1 = read_file(data1)
    data_pos2 = read_file(data2)
    rand_pos1 = read_file(rand1)
    rand_pos2 = read_file(rand2)
    
    
    result = CatalogFFTPower(data_positions1=data_pos1, data_positions2=data_pos2, 
                             randoms_positions1=rand_pos1, randoms_positions2=rand_pos2,
                             edges=kedges, ells=ells, nmesh=1024, resampler='tsc', position_type='rdd')
      
    
    k, Pk = result.poles(ell = ells[0], return_k = True)
    
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
    
    
    cuts_b = [10,10,50]
    cuts_f = [10,90,50]
    
    for cut_b, cut_f in zip(cuts_b, cuts_f):
    
        file_b_grav = path+f'cutsky_grav_zmax0.5_m19.5_bright_{cut_b}.fits'
        file_f_grav = path+f'cutsky_grav_zmax0.5_m19.5_faint_{cut_f}.fits'
        file_r_grav = path+'randoms_10_cutsky_grav_zmax0.5.fits'
        
        file_b_real = path+f'cutsky_real_zmax0.5_m19.5_bright_{cut_b}.fits'
        file_f_real = path+f'cutsky_real_zmax0.5_m19.5_faint_{cut_f}.fits'
        file_r_real = path+'randoms_10_cutsky_real_zmax0.5.fits'
        
        output_grav = f'/global/homes/j/jpiat/data/Pk1_f{cut_f}_b{cut_b}.dat'
        output_real = f'/global/homes/j/jpiat/data/Pk1_real_f{cut_f}_b{cut_b}.dat'
        
        print(1)
        
        get_Pk(file_f_grav, file_b_grav, file_r_grav, file_r_grav, output_grav, ells=[1])
        get_Pk(file_f_real, file_b_real, file_r_real, file_r_real, output_real, ells=[1])
        
        output_grav = f'/global/homes/j/jpiat/data/Pk1_b{cut_b}_f{cut_f}.dat'
        output_real = f'/global/homes/j/jpiat/data/Pk1_real_b{cut_b}_f{cut_f}.dat'
        
        print(2)
        
        get_Pk(file_b_grav, file_f_grav, file_r_grav, file_r_grav, output_grav, ells=[1])
        get_Pk(file_b_real, file_f_real, file_r_real, file_r_real, output_real, ells=[1])
    
    
    #cuts = [20,30,40,50,60,70,80,90]
    #       
    #for cut in cuts:
    #    
    #    file_bright = path+f'cutsky_real_zmax0.5_m19.5_bright_{cut}.fits'
    #    file_faint = path+f'cutsky_real_zmax0.5_m19.5_faint_{cut}.fits'
    #    output_bright = f'/global/homes/j/jpiat/data/Pk_bright_{cut}.dat'
    #    output_faint = f'/global/homes/j/jpiat/data/Pk_faint_{cut}.dat'
    #    
    #    get_Pk(file_bright, file_bright, file_random, file_random, output_bright, ells=[0])
    #    get_Pk(file_faint, file_faint, file_random, file_random, output_faint, ells=[0])
           
           