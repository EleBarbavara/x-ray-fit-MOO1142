'''
physical models for clusters and bridge
'''

import numpy as np
from itertools import repeat
import scipy.integrate as integrate
import scipy
from numba import cfunc, carray, jit, njit
from numba.types import intc, CPointer, float64
from scipy import LowLevelCallable

import tools


'''
these are functions used to define the grids for the models computation
'''
@njit
def grid_rotated (x, y, xfil, yfil, fil_angle, lfil):
    # grid used for the cylindrical beta model for the bridge
    
    grid=(y-yfil)*np.cos(fil_angle)-(x-xfil)*np.sin(fil_angle)
    grid=np.abs(grid)
    
    for i in range (grid.shape[0]):
        for j in range (grid.shape[1]):
            if (i-yfil)*np.sin(fil_angle) + (j-xfil)*np.cos(fil_angle) > lfil/2 or (i-yfil)*np.sin(fil_angle) + (j-xfil)*np.cos(fil_angle) < -lfil/2:
                grid[i,j]=np.inf
    return grid


@njit
def grid_circ (mappa, xc, yc):
    # grid used for the gnfw (circular) model for the clusters

    h  = mappa.shape[0]  #square maps
    
    # create an array of integer radial distances from the point (xc,yc)
    Y=np.arange(0,h,1)
    Y=Y.reshape((h,1))
    X=Y.T
    r = np.hypot(X - xc, Y - yc)
    return r


@njit
def grid_ell (x, y, xc, yc, angle, ecc):
    # grid used for the gnfw (elliptical) model for the clusters

    ra=(x-xc)*np.cos(yc)*np.cos(angle) - (y-yc)*np.sin(angle)
    rb=(x-xc)*np.cos(yc)*np.sin(angle) + (y-yc)*np.cos(angle)
    r=np.sqrt(ra*ra + (1/(1-ecc*ecc))*rb*rb)
    return r

'''
@njit
def apodization (distances, radius):
    # apodization for the gnfw models as suggested in Bonjean et al. 2018, A&A 609, A49 and Arnaud et al. 2010, A&A 517, A92
    theta_lim=5*radius
    diff_theta=distances - theta_lim
    diff_theta[diff_theta <= 0]=0
    apod_factor=np.exp(-(diff_theta/2.)**2)  
    return apod_factor
'''

def jit_integrand_function(integrand_function):
    jitted_function = jit(integrand_function, nopython=True)

    @cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):
        values = carray(xx, n)
        return jitted_function(values)
    return LowLevelCallable(wrapped.ctypes)


'''
---------------------------------- CLUSTERS -----------------------------------
'''
def exp(a, map):
    
    return a*map

def bkg(a, map):
    
    return a*map

#@jit(forceobj=True) #@njit 
def beta_model_cluster_circ (x, y, xc, yc, rc, beta, A):
    '''
    Parameters
    ----------
    x : np.ndarray (from meshgrid)
        x coordinate of the map model.
        
    y : np.ndarray (from meshgrid)
        y coordinate of the map model.
        
    xc : int
        x coordinate (PIXEL) of the cluster centre.
    
    yc : int
        y coordinate (PIXEL) of the cluster centre.

    rc : int
        Core radius (in PIXEL).
    
    beta : float
        Beta parameter.
    
    A : float
        Amplitude.

    Returns
    -------
    np.ndarray : Projected beta-profile with circular symmetry
    '''
    #scipy.special.gamma(3*beta - 0.5)/scipy.special.gamma(3*beta))
    return A * (1 + ((x-xc)/rc)**2 + ((y-yc)/rc)**2 )**(-3*beta + 0.5)



#@jit(forceobj=True) #@njit
def beta_model_cluster_ell(x, y, xc, yc, rc, beta, A, theta_C, ecc):
    '''
    Parameters
    ----------
    x : np.ndarray (from meshgrid)
        x coordinate of the map model.
        
    y : np.ndarray (from meshgrid)
        y coordinate of the map model.
        
    xc : int
        x coordinate (PIXEL) of the cluster centre.
    
    yc : int
        y coordinate (PIXEL) of the cluster centre.

    rc : int
        Core radius (in PIXEL).
    
    beta : float
        Beta parameter.
    
    A : float
        Amplitude.

    theta_C : float 
        Angle (in RAD) of the axis of revolution of the cluster.
    
    ecc : float
        Eccentricity (ratio of the minor to the major axis of the cluster).

    Returns
    -------
    np.ndarray : Projected beta-profile with elliptical symmetry
    '''
    
    X=(x-xc)*np.cos(theta_C) + (y-yc)*np.sin(theta_C)
    Y=(y-yc)*np.cos(theta_C) - (x-xc)*np.sin(theta_C)
    return A * (scipy.special.gamma(3*beta - 0.5)/scipy.special.gamma(3*beta)) * ( 1 + (X/rc)**2 + (Y/(ecc*rc))**2 )**(- 3 * beta + 0.5)



def GNFW_model_cluster_circ (x, y, xc, yc, rs, beta, alpha, gamma, A):
    '''
    Parameters
    ----------
    x : np.ndarray (from meshgrid)
        x coordinate of the map model.
        
    y : np.ndarray (from meshgrid)
        y coordinate of the map model.
    
    xc : int
        x coordinate (PIXEL) of the cluster centre.
    
    yc : int
        y coordinate (PIXEL) of the cluster centre.
    
    rs : int
        characteristic radius (in PIXEL). It is fixed
    
    beta : float
        Beta parameter.

    A : float 
        Amplitude.

    Returns
    -------
    np.ndarray : Projected spherical Generalized Navarro Frenk & White pressure profile, converted to y-parameter
    '''
    
    grid=grid_circ(x, xc, yc)    
    distances, inverse=np.unique(grid, return_index=False, return_inverse=True)
    '''
    # apodization
    apod_factor=apodization(distances, rs*1.81)  #c500=1.81 from universal profile fitted by Planck V (2013)
    '''
    # integration   
    inputs=zip(distances, repeat(rs), repeat(beta), repeat(alpha), repeat(gamma))
    sph_proj=list(map(do_integrate_cls, inputs)) 
    #sph_proj=sph_proj*apod_factor
    sph_proj=np.asanyarray(sph_proj)
    
    # map
    sph_map = sph_proj[inverse]
    sph_map.shape = (grid.shape[0], grid.shape[1]) 
    sph_map /= np.max(sph_map)
    
    return A * sph_map

@jit_integrand_function
def integrand_cls (args):
    r=args[0]
    theta_RAD=args[1]
    rs=args[2]
    beta=args[3]
    alpha=args[4]
    gamma=args[5]

    #gamma=0.3081  #from universal profile fitted by Planck V (2013)
    #alpha=1.0510
    return (r/rs)**(-gamma) * (1+ (r/rs)**alpha)**(-(beta-gamma) / alpha) * 2*r * (r**2-theta_RAD**2)**(-0.5)

def do_integrate_cls (inputs):
    (theta_RAD, rs, beta, alpha, gamma)=inputs
    epsrel=1.00e-05
    value_r=integrate.quad(integrand_cls, theta_RAD, np.inf, args=(theta_RAD, rs, beta, alpha, gamma), epsrel=epsrel)
    return value_r[0]




def GNFW_model_cluster_ell (x, y, xc, yc, rs, angle, ecc, beta, alpha, gamma, A):
    '''
    Parameters
    ----------
    x : np.ndarray (from meshgrid)
        x coordinate of the map model.
        
    y : np.ndarray (from meshgrid)
        y coordinate of the map model.
    
    xc : int
        x coordinate (PIXEL) of the cluster centre.
    
    yc : int
        y coordinate (PIXEL) of the cluster centre.
    
    rs : int
        characteristic radius (in PIXEL). It is fixed
    
    angle : float
        position angle (RAD) of the plane-of-sky major axis.
    
    ecc : float
        Eccentricity.
    
    beta : float
        Beta parameter.
    
    A : float 
        Amplitude.

    Returns
    -------
    np.ndarray : Projected elliptical Generalized Navarro Frenk & White pressure profile, converted to y-parameter
    '''
    
    grid=grid_ell(x, y, xc, yc, angle, ecc)    
    distances, inverse=np.unique(grid, return_index=False, return_inverse=True)
    
    # apodization
    #apod_factor=apodization(distances, rs*1.81)  #c500=1.81 from universal profile fitted by Planck V (2013)
        
    # integration   
    inputs=zip(distances, repeat(rs), repeat(beta), repeat(alpha), repeat(gamma))
    ell_proj=list(map(do_integrate_cls_ell, inputs)) 
    #ell_proj=ell_proj*apod_factor
    ell_proj=np.asanyarray(ell_proj)
    
    # map
    ell_map = ell_proj[inverse]
    ell_map.shape = (grid.shape[0], grid.shape[1]) 
    ell_map /= np.max(ell_map)
    
    return A * ell_map

@jit_integrand_function
def integrand_cls_ell (args):
    r=args[0]
    theta_RAD=args[1]
    rs=args[2]
    beta=args[3]
    alpha=args[4]
    gamma=args[5]

    #gamma=0.31  #from universal profile fitted by Planck V (2013)
    #alpha=1.33
    return (r/rs)**(-gamma) * (1+ (r/rs)**alpha)**(-(beta-gamma) / alpha) * 2*r * (r**2-theta_RAD**2)**(-0.5)

def do_integrate_cls_ell (inputs):
    (theta_RAD, rs, beta, alpha, gamma)=inputs
    epsrel=1.00e-05 #relative error tolerance
    value_r=integrate.quad(integrand_cls_ell, theta_RAD, np.inf, args=(theta_RAD, rs, beta, alpha, gamma), epsrel=epsrel)
    return value_r[0]


'''
---------------------------------- BRIDGE -----------------------------------
'''

@njit
def mesa_model_fil (x, y, fil_angle, xfil, yfil, l0_mesa, w0_mesa, A):
    '''
    Parameters
    ----------
    x : np.ndarray (from meshgrid)
        x coordinate of the map model.
        
    y : np.ndarray (from meshgrid)
        y coordinate of the map model.

    fil_angle : float
                Filament angle (in rad).
    
    xfil : int
        x coordinate (PIXEL) of the filament centre.
    
    yfil : int
        y coordinate (PIXEL) of the filament centre.

    l0_mesa : int
        characteristic length of the mesa model, i.e. lfil/2 (in PIXEL).

    w0_mesa : int
        characteristic width of the mesa model, i.e. wfil/2 (in PIXEL).
    
    A : float 
        Amplitude.

    Returns
    -------
    np.ndarray : Projected mesa-model for the filament
    '''
    
    X=(x-xfil)*np.cos(fil_angle) + (y-yfil)*np.sin(fil_angle)
    Y=(y-yfil)*np.cos(fil_angle) - (x-xfil)*np.sin(fil_angle)
    return 0.5 * A * (1 + (1 - (X**8/l0_mesa**8 + Y**8/w0_mesa**8)) / (1 + (X**8/l0_mesa**8 + Y**8/w0_mesa**8)) )



def cyl_beta_model_fil_integrate (x, y, fil_angle, xfil, yfil, lfil, rc, A):   
    '''
    Parameters
    ----------
    x : np.ndarray (from meshgrid)
        x coordinate of the map model.
        
    y : np.ndarray (from meshgrid)
        y coordinate of the map model.

    fil_angle : float
                Filament angle (in rad).
    
    xfil : int
        x coordinate (PIXEL) of the filament centre.
    
    yfil : int
        y coordinate (PIXEL) of the filament centre. 

    lfil : int
        length (in PIXEL) of the filament.

    rc : int
        Core radius in the direction perpendicular to the length of the filament (in PIXEL).
    
    A : float 
        Amplitude.

    Returns
    -------
    np.ndarray : Projected cylindrical beta-model for the filament
    '''
    
    grid=grid_rotated(x, y, xfil, yfil, fil_angle, lfil)                
    distances, inverse=np.unique(grid, return_index=False, return_inverse=True)  
    
    # exclude np.inf from apodization and integration
    distances=distances[distances != np.inf]

    # apodization
    #apod_factor=apodization(distances, rc)
    
    # integration    
    inputs=zip(distances, repeat(rc))
    cyl_proj=list(map(do_integrate_fil, inputs))
    #cyl_proj=cyl_proj*apod_factor
    cyl_proj=np.insert(cyl_proj, len(cyl_proj), np.inf)
    
    # map
    cyl_map = cyl_proj[inverse]
    cyl_map.shape = (grid.shape[0], grid.shape[1]) 
    cyl_map[cyl_map == np.inf]=0
    cyl_map /= np.max(cyl_map)
    return A * cyl_map

@jit_integrand_function
def integrand_fil (args):
    radius=args[0]
    theta_RAD=args[1]
    rc=args[2]    
    return (1+radius**2/rc**2)**(-2) * 2*radius * (radius**2-theta_RAD**2)**(-0.5)

def do_integrate_fil (inputs):
    theta_RAD,rc = inputs
    value_r=integrate.quad(integrand_fil, theta_RAD, np.inf, args=(theta_RAD, rc))
    return value_r[0]



def cyl_beta_model_fil_analytic (x, y, fil_angle, xfil, yfil, lfil, rc, A):
    '''
    Parameters
    ----------
    x : np.ndarray (from meshgrid)
        x coordinate of the map model.
        
    y : np.ndarray (from meshgrid)
        y coordinate of the map model.

    fil_angle : float
        Filament angle (in rad).
    
    xfil : int
        x coordinate (PIXEL) of the filament centre.
    
    yfil : int
        y coordinate (PIXEL) of the filament centre. 

    lfil : int
        length (in PIXEL) of the filament.

    rc : int
        Core radius in the direction perpendicular to the length of the filament (in PIXEL).
    
    A : float 
        Amplitude.

    Returns
    -------
    np.ndarray : Projected cylindrical beta-model for the filament
    '''
    
    grid=grid_rotated(x, y, xfil, yfil, fil_angle, lfil)    
    distances, inverse=np.unique(grid, return_index=False, return_inverse=True)
    
    # exclude np.inf from apodization and integration
    distances=distances[distances != np.inf]
    
    # apodization
    #apod_factor=apodization(distances, rc)
    
    # analytic integration
    cyl_proj=loop_integration(distances, rc)
    #cyl_proj=cyl_proj*apod_factor
    cyl_proj=np.insert(cyl_proj, len(cyl_proj), np.inf)     

    # map
    cyl_map = cyl_proj[inverse]
    cyl_map.shape = (grid.shape[0], grid.shape[1]) 
    cyl_map[cyl_map == np.inf]=0
    cyl_map /= np.max(cyl_map)
    return A * cyl_map

@njit
def loop_integration (distances, rc):
    cyl_proj=[]
    for i in distances:
        cyl_proj.append(analytic_integration(i, rc))
    return cyl_proj

@njit
def analytic_integration (distances,rc):
    return (np.pi * rc**4) / (2* (distances**2+rc**2)**(3.5)) 

def cyl_beta_model_fil_craig (x, y, fil_angle, xfil, yfil, lfil, rc, A, beta):
    '''
    Parameters
    ----------
    x : np.ndarray (from meshgrid)
        x coordinate of the map model.
        
    y : np.ndarray (from meshgrid)
        y coordinate of the map model.

    fil_angle : float
        Filament angle (in rad).
    
    xfil : int
        x coordinate (PIXEL) of the filament centre.
    
    yfil : int
        y coordinate (PIXEL) of the filament centre. 

    lfil : int
        length (in PIXEL) of the filament.

    rc : int
        Core radius in the direction perpendicular to the length of the filament (in PIXEL).
    
    A : float 
        Amplitude.
    
    beta : float 
        Beta parameter.

    Returns
    -------
    np.ndarray : Projected cylindrical beta-model for the filament
    '''
    
    grid=grid_rotated(x, y, xfil, yfil, fil_angle, lfil)    
    distances, inverse=np.unique(grid, return_index=False, return_inverse=True)
    
    # exclude np.inf from apodization and integration
    distances=distances[distances != np.inf]
    
    # apodization
    #apod_factor=apodization(distances, rc)
    
    # analytic integration
    cyl_proj=loop_integration_craig(distances, rc, beta)
    #cyl_proj=cyl_proj*apod_factor
    cyl_proj=np.insert(cyl_proj, len(cyl_proj), np.inf)     

    # map
    cyl_map = cyl_proj[inverse]
    cyl_map.shape = (grid.shape[0], grid.shape[1]) 
    cyl_map[cyl_map == np.inf]=0
    cyl_map /= np.max(cyl_map)
    return A * cyl_map

@njit
def loop_integration_craig(distances, rc, beta):
    cyl_proj=[]
    for i in distances:
        cyl_proj.append(analytic_integration_craig(i, rc, beta))
    return cyl_proj

@njit
def analytic_integration_craig(distances, rc, beta):
    return (1 + (distances/rc)**2 )**(-3*beta + 0.5) #(np.pi * rc**4) / (2* (distances**2+rc**2)**(1.5)) 

'''
---------------------------------- BACKGROUND -----------------------------------
'''

@njit
def plane (x, y, a, b, c):
    '''
    Parameters
    ----------
    x : np.ndarray (from meshgrid)
        x coordinate of the map model.
        
    y : np.ndarray (from meshgrid)
        y coordinate of the map model.

    a : float
    
    b : float
    
    c : float

    Returns
    -------
    np.ndarray : Plane model for the background.
    '''
    
    return a + b*x + c*y