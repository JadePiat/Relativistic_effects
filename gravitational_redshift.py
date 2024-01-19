def z_grav(z_cosmo, z, phi0, phi):
    
    c = 299792.458  # c in km/s

    # get the LOS velocity from Z and Z_COSMO
    v_los = c * ((1 + z)/(1 + z_cosmo) - 1)

    # redshift including only gravitational redshifts
    z_g = (1 + z_cosmo)*(1 + phi0 - phi) - 1
    
    # redshift including both RSD and gravitational redshifts
    z_g_rsd = (1 + z_cosmo)*(1 + phi0 - phi + v_los/c) - 1
    
    return z_g_rsd