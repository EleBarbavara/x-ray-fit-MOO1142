'''
general functions used in the main code 
'''

import numpy as np
import os
import astropy
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian2DKernel, convolve_fft
from numba import njit


def coord_to_pixel (wcs, ra, dec, units):
    '''
    Parameters
    ----------
    wcs : astropy.wcs.wcs.WCS
        The WCS transformation to use
        
    ra : float
        Right ascension
        
    dec : float
        Declination

    units : str
            Units of (ra,dec)

    Returns
    -------
    x,y : np.ndarray
        Pixel position of input coordinates     
    '''
        
    coord=SkyCoord(ra, dec, unit=units)  
    x,y = astropy.wcs.utils.skycoord_to_pixel(coord, wcs, origin=0, mode='all')
    
    if x==0.5:
        x=np.ceil(x)
    if y==0.5:
        y=np.ceil(y)
    
    else: 
        x=np.round(x)
        y=np.round(y)
    
    x=x.astype(int)
    y=y.astype(int)    
    
    return x,y


def pixel_to_coord (wcs, xp, yp):
    '''
    Parameters
    ----------
    wcs : astropy.wcs.wcs.WCS
        The WCS transformation to use
        
    xp : int
        Pixel coordinate x
        
    yp : int
        Pixel coordinate y

    Returns
    -------
    (ra, dec) : Returns the sky coordinates in degrees.            
    '''
    
    return wcs.all_pix2world(xp, yp, 1)


@njit
def gnomonic_coords_pix (pix_size, ra1, dec1, ra2, dec2, mean_ra, mean_dec):
    # this function compute the pixel position of (ra1, dec1) and (ra2, dec2) after gnomonic reprojection  
    # the centre of the reprojection is (mean_ra, mean_dec)
    
    scale=1/(pix_size)  #pix_size in deg
    alpha1, delta1 =  ra1, dec1  #coords in rad
    alpha2, delta2 =  ra2, dec2  #coords in rad
    alpha0, delta0 = mean_ra, mean_dec  #coords in rad
    
    A1 = np.cos(delta1)*np.cos(alpha1 - alpha0)
    F1 = scale * (180/np.pi)/(np.sin(delta0)*np.sin(delta1) + A1*np.cos(delta0))
    A2 = np.cos(delta2)*np.cos(alpha2 - alpha0)
    F2 = scale * (180/np.pi)/(np.sin(delta0)*np.sin(delta2) + A2*np.cos(delta0))
    
    # these are shifts with respect to the centre of the map (the point in which the reprojection is done)
    # signs are fixed in order to have 'gnomonic' pixel_position=centre_of_map + dy (or +dx)
    dy1 = F1 * (np.cos(delta0) * np.sin(delta1) - A1 * np.sin(delta0))
    dx1 = -F1 * np.cos(delta1) * np.sin(alpha1 - alpha0)
    dy2 = F2 * (np.cos(delta0) * np.sin(delta2) - A2 * np.sin(delta0))
    dx2 = -F2 * np.cos(delta2) * np.sin(alpha2 - alpha0)
    
    return dy1, dx1, dy2, dx2


def convolve (img, PSF):
    '''
    Parameters
    ----------
    img : np.ndarray
        The map to be smoothed.
        
    res : float
        FWHM (resolution) of the smoothing (arcmin).
        
    dim_pixel : float
        Pixel size in the map (arcmin).

    Returns
    -------
    img_convolved: np.ndarray
        Same map but smoothed.
    '''

    img_convolved=convolve_fft(img, PSF)

    return img_convolved


@njit
def safe_inv (a, tol=1e-10):
    '''
    Parameters
    ----------
    a : np.ndarray
        Matrix to invert.
        
    tol : float
        The default is 1e-10.

    Returns
    -------
    1/a : np.ndarray
        Inverse of input matrix 
    '''
    
    return 1/np.maximum(a, np.max(a)*tol)


def save_fits (img, path, fits_name, header):
    '''
    Parameters
    ----------
    img : np.ndarray
        Map to be saved.
        
    path : str
        Path to save the map to.

    fits_name : str
                Name of fits file.
                
    header : header for the map.

    Returns
    -------
    None.
    '''
    
    myfile = path + fits_name + '.fits'
    if os.path.isfile(myfile):
        os.remove(myfile)
    hdu = fits.PrimaryHDU(img)
    hdu.header=header
    hdul = fits.HDUList([hdu])
    hdul.writeto(myfile)


# circular aperture of radius R centred on (x,y)
@njit
def circular_aperture (size_map, R, x, y): 
    '''
    Parameters
    ----------
    maps : 2D array. 
        Map on which to apply the circular aperture.
    R : int.
        Radius of the aperture.
    x,y : int.
        Coordinates of the centre of the aperture.
    
    Returns
    -------
    aperture : 2D array. 
        Same map as in input but with a circular aperture.
    '''
    
    aperture=np.ones((size_map,size_map))
    
    for i in range(size_map):
        for j in range(size_map):
            r=np.sqrt((i-y)**2+(j-x)**2)
            if r<=R:
                aperture[i,j]=np.nan
    return aperture
