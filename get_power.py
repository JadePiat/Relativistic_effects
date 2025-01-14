import sys, os
sys.path.append('/global/homes/j/jpiat/my_hodpy/')
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
import scipy.constants as const
import matplotlib.pyplot as plt
import time
from pypower import CatalogMesh, MeshFFTPower, CatalogFFTPower, PowerSpectrumStatistics, utils, setup_logging
from scipy.interpolate import CubicSpline
from hodpy.cosmology import CosmologyAbacus

cosmo = CosmologyAbacus(0)  #c000 cosmology


def read_file(file):
    
    sky = Table.read(file)
    
    data, data_real = np.zeros((3,len(sky))), np.zeros((3,len(sky)))
    
    data[0] = sky['RA']
    data[1] = sky['DEC']
    data[2] = sky['DIST_RSD']
    
    data_real[0] = sky['RA']
    data_real[1] = sky['DEC']
    data_real[2] = sky['DIST_COM']
    
    return data, data_real


def get_random(data):
    
    N = len(data[0])
    
    ra_rand = np.random.uniform(0,360,N*10)
    dec_rand = np.degrees(np.arcsin(np.random.uniform(-1,1,N*10)))
    
    dist_sort = np.sort(data[2])
    cdf = np.arange(1,N+1,1)/N
    cdf_inv = CubicSpline(cdf,dist_sort)
 
    r = np.random.uniform(0,1,N*10)
    dist_rand = cdf_inv(r)
    
    rand = np.array([ra_rand,dec_rand,dist_rand])
    
    return rand
    


def Pk_check(Pk, ell):
    
    if ell%2 == 0:
           return Pk.real
    else:
           return Pk.imag
            


def get_Pk(data1, data2, rand1, rand2, output_file, ells, gridsize=512, kedges = np.arange(0,0.5,0.005)):
    
    start = time.time()
    
    print('start power spectrum computation')
    
    result = CatalogFFTPower(data_positions1=data1, data_positions2=data2, 
                             randoms_positions1=rand1, randoms_positions2=rand2,
                             edges=kedges, ells=ells, nmesh=gridsize, resampler='tsc', position_type='rdd')     #nmesh=1024
    
    print('end power spectrum computation')
    
    end = time.time()
    print(end-start)
    
    k, Pk = result.poles(ell = ells[0], return_k = True)
    
    data = np.zeros((len(k),len(ells)+1))
    data[:,0] = k
    data[:,1] = Pk_check(Pk, ells[0])
    
    if len(ells) > 1:
        
        for i,ell in enumerate(ells[1:]):
            
            Pk = result.poles.power(ell = ell)
            data[:,i+2] = Pk_check(Pk, ell)
                   
    
    np.savetxt(output_file, data)
           
        
        

def get_dipole(file1, file2, output_path):
    
    data1, data1_real = read_file(file1)
    data2, data2_real = read_file(file2)
    
    rand1, rand1_real = get_random(data1), get_random(data1_real)
    rand2, rand2_real = get_random(data2), get_random(data2_real)
    
    output = output_path+'.dat'
    output_real = output_path+'_real.dat'
    
    print('data loaded')
        
    get_Pk(data1, data2, rand1, rand2, output, ells=[1])
    get_Pk(data1_real, data2_real, rand1_real, rand2_real, output_real, ells=[1])
        
     
    
def get_monopole(input_file, output_path):
    
    data_real = read_file(input_file)[1]
    
    z_bins = np.arange(0,0.55,0.05)
    dist_bins = cosmo.comoving_distance(z_bins)
    
    for i in range(len(z_bins)-1):
    
        if i==(len(z_bins)-1):
            keep = (data_real[2]>=dist_bins[i])*(data_real[2]<=dist_bins[i+1])
        else:
            keep = (data_real[2]>=dist_bins[i])*(data_real[2]<dist_bins[i+1])
        
        data_bin = (data_real.T[keep]).T
    
        rand_bin = get_random(data_bin)
    
        output_real = output_path+'_real_'+f'{z_bins[i]:.2f}_z_{z_bins[i+1]:.2f}.dat'
    
        get_Pk(data_bin, data_bin, rand_bin, rand_bin, output_real, ells=[0])
        
        
        
if __name__ == "__main__":
    
    path = '/pscratch/sd/j/jpiat/Abacus_mocks/z0.200/AbacusSummit_base_c000_ph000/cutsky/magnitude_splits/'
    
    cuts_b = [10]
    cuts_f = [90]
    
    for cut_b, cut_f in zip(cuts_b, cuts_f):
    
        file_b = path+f'cutsky_zmax0.5_m19.5_bright_{cut_b}.fits'
        file_f = path+f'cutsky_zmax0.5_m19.5_faint_{cut_f}.fits'
    
        output_path_bf = path+f'Pk1_bright{cut_b}_faint{cut_f}'
        output_path_fb = path+f'Pk1_faint{cut_f}_bright{cut_b}'
        
        output_path_b = path+f'Pk0_bright{cut_b}'
        output_path_f = path+f'Pk0_faint{cut_f}'
    
        #for n in [256,512,1024]:
        
        #get_dipole(file_b, file_f, output_path_bf)    
        #get_dipole(file_f, file_b, output_path_fb)
        
        get_monopole(file_b,output_path_b)
        get_monopole(file_f,output_path_f)
           