import numpy as np
import scipy.stats

from fitter import data_cls_ps
from fitter import models
from fitter import tools

import dill
import glob
import time
import os

os.environ['OMP_NUM_THREADS'] = '1'

from fitter.parameters import get_pars, get_prof, get_point
from pixell import enmap, bunch

import corner
import dynesty
import nautilus
import emcee

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u

from fitter.pool import ncpu

import pocomc

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--nlive',type=int,default=1000)
parser.add_argument('--dlogz',type=float,default=1.00)
parser.add_argument('--model',type=str)
parser.add_argument('--point',type=str,default='tophat')
parser.add_argument('--length',type=int,default=1)
parser.add_argument('--maxiter',type=int,default=np.inf)
parser.add_argument('--method',type=str)
parser.add_argument('--cib',type=str,default='wt')
args = parser.parse_args()

# Initialize modelling parameters
# =============================================================================
method  = args.method
resume = True

nlive  = nwalk = args.nlive
ncores = int(ncpu)-2

maxiter = args.maxiter
dlogz   = args.dlogz

if method in ['emcee','pocomc']:
    from multiprocess import Pool
    
    if not np.isfinite(maxiter) and method=='emcee':
        raise ValueError('Please define a finite number of iterations')
else:
    from fitter.pool import Pool

# Select the values
# -----------------------------------------------------------------------------
cls_pair = 'A3391_A3395'
cls_1 = 'A3391'
cls_2 = 'A3395'

# Select the fitting method
# -----------------------------------------------------------------------------
fit_method = args.model

if   args.cib=='wt': cib = 'wtCIB'
elif args.cib=='de': cib = 'deCIB'

main_path_maps = 'A3391_A3395/input/maps/{0}'.format(cib)
main_path_file = 'A3391_A3395/input/info'
main_path_res  = 'A3391_A3395/output'

print(main_path_maps)

point_model = args.point
if point_model in ['tophat','1gauss','2gauss']:
    point_model = f'{point_model}_sph'

prova_num = f'{point_model}_{args.length}_{cib}'

# KS test for fields
# -----------------------------------------------------------------------------
if main_path_maps[-1]=='/': main_path_maps = main_path_maps[:-1]
if main_path_file[-1]=='/': main_path_file = main_path_file[:-1]

pair_name=cls_1 + '_' + cls_2

fields = glob.glob(f'{main_path_maps}/field*.fits')
fields = sorted(fields)

fields_for_cov = []

for file in fields:
    
    field_name = file.split('/')[-1]
    field_name = field_name.replace('.fits','')

    hdu = fits.open(file)
    map_flat = hdu[0].data.flatten()   

    loc, scale = scipy.stats.norm.fit(map_flat)
    n = scipy.stats.norm(loc=loc,scale=scale)
    
    stat, p_value = scipy.stats.kstest(map_flat, n.cdf)

    if p_value>0.05:
        fields_for_cov.append(field_name)
    hdu.close(); del hdu

fields = np.array(fields_for_cov)
num_fields = len(fields)

if num_fields<5:
    raise ValueError('Not enough fields for covariance matrix')

# Load data 
# -----------------------------------------------------------------------------
map_name = cls_pair + '.fits'
map_file = fits.open(f'{main_path_maps}/{map_name}')

wcs = WCS(map_file[0].header)
data = map_file[0].data
Xdim = map_file[0].header['NAXIS1']
Ydim = map_file[0].header['NAXIS2']
dim_pix = 0.50 # pixel size in arcmin
map_res = 1.65 # map resolution in arcmin


# Define cluster geometry
# -----------------------------------------------------------------------------
p_cls = data_cls_ps.data_cls(cls_1, cls_2, main_path_maps = main_path_maps, main_path_file = main_path_file)
x_C1, y_C1, x_C2, y_C2, theta500_C1_arcmin, theta500_C2_arcmin, x_ps, y_ps = p_cls
theta500_C1_pix = theta500_C1_arcmin/dim_pix
theta500_C2_pix = theta500_C2_arcmin/dim_pix

print(theta500_C1_arcmin)
print(theta500_C2_arcmin)

# For the GNFW model
c500 = 1.81  #from Planck Collaboration V 2013 
rs_C1 = (theta500_C1_arcmin/c500)/dim_pix
rs_C2 = (theta500_C2_arcmin/c500)/dim_pix

# bridge position
xfil_m = (Xdim/2)
yfil_m = (Ydim/2)
    
# Create the map for the model
x = np.arange(0, Xdim, 1)
y = np.arange(0, Ydim, 1)
x,y = np.meshgrid(x,y)

# Estimate noise covariance matrix and its inverse
# -----------------------------------------------------------------------------

d = bunch.Bunch(noise = enmap.enmap([enmap.read_map(f'{main_path_maps}/{fields[i]}.fits') for i in range(num_fields)] ),
               models = [bunch.Bunch(resid = enmap.ndmap(data*0, wcs)),])

pix_apod=int(np.round(Xdim/100*11))  # 11% of the map side-length
apod = (d.models[0].resid*0.00+1.00).apod(pix_apod)
d.N = np.mean(np.abs(enmap.fft(d.noise[:,:,:]*apod))**2, 0) / np.mean(apod**2)

for i in range(3):
    d.N += np.roll(d.N,1,axis=0) + np.roll(d.N,-1,axis=0) + np.roll(d.N,1,axis=1) + np.roll(d.N,-1,axis=1)
    d.N /= 5.0

d.iN = tools.safe_inv(d.N)
chinorm = np.sum(np.log(2.00*np.pi*d.N))

# Define range of values for the parameters
# -----------------------------------------------------------------------------
pvar, pfix = get_pars(fit_method)

if '2gauss' in point_model:
    for p in ['x_S1b','y_S1b','A_S1b','r_S1b','theta_S1b','ecc_S1b']:
        pvar.append(p)

if '_sph' in point_model:
    for c in ['S1a','S1b','S2']:
        for p in ['theta','ecc']:
            if f'{p}_{c}' in pvar:
                pvar.remove(f'{p}_{c}'); pfix.append(f'{p}_{c}')

if point_model=='none':
    for s in ['S1a','S1b','S2']:
        for p in ['A','x','y','r','theta','ecc']:
            if f'{p}_{s}' in pvar:
                pvar.remove(f'{p}_{s}'); pfix.append(f'{p}_{s}')

    x_S1a_fix = x_ps; y_S1a_fix = y_ps; r_S1a_fix = 1.00E-04; theta_S1a_fix = 0.00; ecc_S1a_fix = 0.00; A_S1a_fix = 0.00
    x_S1b_fix = x_ps; y_S1b_fix = y_ps; r_S1b_fix = 1.00E-04; theta_S1b_fix = 0.00; ecc_S1b_fix = 0.00; A_S1b_fix = 0.00
    x_S2_fix  = x_ps; y_S2_fix  = y_ps; r_S2_fix  = 1.00E-04; theta_S2_fix  = 0.00; ecc_S2_fix  = 0.00; A_S2_fix  = 0.00

pnum = len(pvar)

x_C1_fix = x_C1; y_C1_fix = y_C1; rs_C1_fix = rs_C1
x_C2_fix = x_C2; y_C2_fix = y_C2; rs_C2_fix = rs_C2
x_C3_fix = x_C2; y_C3_fix = y_C2; rs_C3_fix = rs_C2

# point source
# ------------
delta_ps = int(2/dim_pix) #2arcmin, freedom of the center
x_S1a_in = [ x_ps-delta_ps-8, x_ps+delta_ps  ]; x_S1a_fix = x_ps
y_S1a_in = [ y_ps-delta_ps  , y_ps+delta_ps+8]; y_S1a_fix = y_ps
x_S1b_in = [ x_ps-delta_ps  , x_ps+delta_ps  ]; x_S1b_fix = x_ps
y_S1b_in = [ y_ps-delta_ps  , y_ps+delta_ps  ]; y_S1b_fix = y_ps

x_S2_in = [ (x_ps+15)-delta_ps, (x_ps+15)+delta_ps]; x_S2_fix = x_ps
y_S2_in = [ (y_ps-13)-delta_ps, (y_ps-13)+delta_ps]; y_S2_fix = y_ps

A_S1a_in = [1.00E-06,1.00E-02]; r_S1a_in = np.array([1.00,5.00])/dim_pix  #1-5arcmin
A_S1b_in = [1.00E-06,1.00E-02]; r_S1b_in = np.array([1.00,5.00])/dim_pix  #1-5arcmin
A_S2_in  = [1.00E-06,1.00E-02]; r_S2_in  = np.array([1.00,5.00])/dim_pix  #1-5arcmin

theta_S1a_fix = 0.00; theta_S1a_in = [0.00,np.pi]; ecc_S1a_fix = 0.00; ecc_S1a_in = [0.00, 0.90]
theta_S1b_fix = 0.00; theta_S1b_in = [0.00,np.pi]; ecc_S1b_fix = 0.00; ecc_S1b_in = [0.00, 0.90]
theta_S2_fix  = 0.00; theta_S2_in  = [0.00,np.pi]; ecc_S2_fix  = 0.00; ecc_S2_in  = [0.00, 0.90]

# filament
# ------------
delta_fil  = 2*max(theta500_C1_arcmin,theta500_C2_arcmin)  #in arcmin, freedom of the center
delta_fil  = int(delta_fil/dim_pix)
xfil_m_in  = [xfil_m-delta_fil, xfil_m+delta_fil]
yfil_m_in = [yfil_m-delta_fil, yfil_m+delta_fil]
yfil_m_fix = yfil_m

rl_fil_fix = 0.00

A_fil_in  = data_cls_ps.init_cond_fil()
rc_fil_in = np.array([0.10,2.00*max(theta500_C1_arcmin, theta500_C2_arcmin)])/dim_pix
if   args.length==1: l_fil_in  = np.array([20.00, 60.00])/dim_pix  #in pixels
elif args.length==2: l_fil_in  = np.array([ 5.00,500.00])/dim_pix

fil_angle_fix = np.arctan((y_C1-y_C2)/(x_C1-x_C2))

# cluster
# ------------
if fit_method == '2gnfw_circ_no_plane':
    A_C1_in, A_C2_in, beta_C1_in, beta_C2_in = data_cls_ps.init_cond_cls('gnfw_circ')

elif fit_method == '2gnfw_circ2_no_plane':
    A_C1_in, A_C2_in, beta_C1_in, beta_C2_in = data_cls_ps.init_cond_cls('gnfw_circ')
    gamma_C1_in, gamma_C2_in = [0.0001, 0.8], [0.0001, 0.8]

elif fit_method == '2gnfw_ell_no_plane':
    A_C1_in, A_C2_in, beta_C1_in, beta_C2_in, ecc_C1_in, ecc_C2_in, theta_C1_in, theta_C2_in = data_cls_ps.init_cond_cls('gnfw_ell')

elif fit_method in ['2gnfw_ell_freepos_no_plane','2gnfw_circ_freepos_no_plane']:
    A_C1_in, A_C2_in, beta_C1_in, beta_C2_in, ecc_C1_in, ecc_C2_in, theta_C1_in, theta_C2_in = data_cls_ps.init_cond_cls('gnfw_ell')
    del x_C1_fix, y_C1_fix; x_C1_in = [x_C1-theta500_C1_pix, x_C1+theta500_C1_pix]; y_C1_in = [y_C1-theta500_C1_pix, y_C1+theta500_C1_pix]
    del x_C2_fix, y_C2_fix; x_C2_in = [x_C2-theta500_C2_pix, x_C2+theta500_C2_pix]; y_C2_in = [y_C2-theta500_C2_pix, y_C2+theta500_C2_pix]

    if 'circ' in fit_method:
        del theta_C1_in, ecc_C1_in; theta_C1_fix = ecc_C1_fix = 0.00
        del theta_C2_in, ecc_C2_in; theta_C2_fix = ecc_C2_fix = 0.00

elif fit_method in ['2gnfw_ell_freepos_freers_no_plane','2gnfw_circ_freepos_freers_no_plane']:
    A_C1_in, A_C2_in, beta_C1_in, beta_C2_in, ecc_C1_in, ecc_C2_in, theta_C1_in, theta_C2_in = data_cls_ps.init_cond_cls('gnfw_ell')
    del x_C1_fix, y_C1_fix; x_C1_in = [x_C1-theta500_C1_pix, x_C1+theta500_C1_pix]; y_C1_in = [y_C1-theta500_C1_pix, y_C1+theta500_C1_pix]
    del x_C2_fix, y_C2_fix; x_C2_in = [x_C2-theta500_C2_pix, x_C2+theta500_C2_pix]; y_C2_in = [y_C2-theta500_C2_pix, y_C2+theta500_C2_pix]
    
    del rs_C1_fix; rs_C1_in = [1.00E-02,1.00E+04]
    del rs_C2_fix; rs_C2_in = [1.00E-02,1.00E+04]

    if 'circ' in fit_method:
        del theta_C1_in, ecc_C1_in; theta_C1_fix = ecc_C1_fix = 0.00
        del theta_C2_in, ecc_C2_in; theta_C2_fix = ecc_C2_fix = 0.00

elif fit_method == '2beta_circ':
    A_C1_in, A_C2_in, beta_C1_in, beta_C2_in, rc_C1_in, rc_C2_in = data_cls_ps.init_cond_cls('beta_circ')
    rc_C1_in = np.array(rc_C1_in)/dim_pix
    rc_C2_in = np.array(rc_C2_in)/dim_pix

    a_in, b_in, c_in = data_cls_ps.init_cond_background()
    
elif fit_method == '2beta_ell':
    A_C1_in, A_C2_in, beta_C1_in, beta_C2_in, rc_C1_in, rc_C2_in, ecc_C1_in, ecc_C2_in, theta_C1_in, theta_C2_in = data_cls_ps.init_cond_cls('beta_ell')
    rc_C1_in = np.array(rc_C1_in)/dim_pix
    rc_C2_in = np.array(rc_C2_in)/dim_pix

    a_in, b_in, c_in = data_cls_ps.init_cond_background()

    del x_S1a_in; x_S1a_in = [x_C2-theta500_C2_pix, x_C2+theta500_C2_pix]
    del y_S1a_in; y_S1a_in = [y_C2-theta500_C2_pix, y_C2+theta500_C2_pix]

elif fit_method == '2gnfw_circ+cyl_beta_no_plane':
    A_C1_in, A_C2_in, beta_C1_in, beta_C2_in = data_cls_ps.init_cond_cls('gnfw_circ')

elif fit_method == '2gnfw_ell+cyl_beta_no_plane':
    A_C1_in, A_C2_in, beta_C1_in, beta_C2_in, ecc_C1_in, ecc_C2_in, theta_C1_in, theta_C2_in = data_cls_ps.init_cond_cls('gnfw_ell')

elif fit_method == '2gnfw_ell+cyl_beta_no_plane2':
    A_C1_in, A_C2_in, beta_C1_in, beta_C2_in, ecc_C1_in, ecc_C2_in, theta_C1_in, theta_C2_in = data_cls_ps.init_cond_cls('gnfw_ell')

elif fit_method in ['2gnfw_ell_freepos+cyl_beta_no_plane','2gnfw_ell_freepos+cyl_beta_no_plane2',]:
    A_C1_in, A_C2_in, beta_C1_in, beta_C2_in, ecc_C1_in, ecc_C2_in, theta_C1_in, theta_C2_in = data_cls_ps.init_cond_cls('gnfw_ell')
    del x_C1_fix, y_C1_fix; x_C1_in = [x_C1-theta500_C1_pix, x_C1+theta500_C1_pix]; y_C1_in = [y_C1-theta500_C1_pix, y_C1+theta500_C1_pix]
    del x_C2_fix, y_C2_fix; x_C2_in = [x_C2-theta500_C2_pix, x_C2+theta500_C2_pix]; y_C2_in = [y_C2-theta500_C2_pix, y_C2+theta500_C2_pix]

elif fit_method in ['2gnfw_circ_freepos_freers+cyl_beta_no_plane', '2gnfw_ell_freepos_freers+cyl_beta_no_plane',
                    '2gnfw_circ_freepos_freers+cyl_beta_no_plane2','2gnfw_ell_freepos_freers+cyl_beta_no_plane2']:
    A_C1_in, A_C2_in, beta_C1_in, beta_C2_in, ecc_C1_in, ecc_C2_in, theta_C1_in, theta_C2_in = data_cls_ps.init_cond_cls('gnfw_ell')
    del x_C1_fix, y_C1_fix; x_C1_in = [x_C1-theta500_C1_pix, x_C1+theta500_C1_pix]; y_C1_in = [y_C1-theta500_C1_pix, y_C1+theta500_C1_pix]
    del x_C2_fix, y_C2_fix; x_C2_in = [x_C2-theta500_C2_pix, x_C2+theta500_C2_pix]; y_C2_in = [y_C2-theta500_C2_pix, y_C2+theta500_C2_pix]
    
    del rs_C1_fix; rs_C1_in = [1.00E-02,1.00E+04]
    del rs_C2_fix; rs_C2_in = [1.00E-02,1.00E+04]

    if 'circ' in fit_method:
        del theta_C1_in, ecc_C1_in; theta_C1_fix = ecc_C1_fix = 0.00
        del theta_C2_in, ecc_C2_in; theta_C2_fix = ecc_C2_fix = 0.00

elif fit_method == '3gnfw_ell_freepos_no_plane':
    A_C1_in, A_C2_in, beta_C1_in, beta_C2_in, ecc_C1_in, ecc_C2_in, theta_C1_in, theta_C2_in = data_cls_ps.init_cond_cls('gnfw_ell')
    del x_C1_fix, y_C1_fix; x_C1_in = [x_C1-theta500_C1_pix, x_C1+theta500_C1_pix]; y_C1_in = [y_C1-theta500_C1_pix, y_C1+theta500_C1_pix]
    del x_C2_fix, y_C2_fix; x_C2_in = [x_C2-theta500_C2_pix, x_C2+theta500_C2_pix]; y_C2_in = [y_C2-theta500_C2_pix, y_C2+theta500_C2_pix]

    A_C3_in = A_C2_in; beta_C3_in = beta_C2_in; ecc_C3_in = ecc_C2_in; theta_C3_in = theta_C2_in
    x_C3_in = [-theta500_C2_pix, theta500_C2_pix]
    y_C3_in = [-theta500_C2_pix, theta500_C2_pix]

# ------------------------

elif 'mesa' in fit_method:
    l0_fil_in = l_fil_in
    w0_fil_in = np.array([5.00,60.00])/dim_pix

    if  '2gnfw_circ' in fit_method:
        A_C1_in, A_C2_in, beta_C1_in, beta_C2_in = data_cls_ps.init_cond_cls('gnfw_circ')

    elif '2gnfw_ell' in fit_method:
        A_C1_in, A_C2_in, beta_C1_in, beta_C2_in, ecc_C1_in, ecc_C2_in, theta_C1_in, theta_C2_in = data_cls_ps.init_cond_cls('gnfw_ell')

    if 'freepos' in fit_method:
        del x_C1_fix; x_C1_in = [x_C1-theta500_C1_pix, x_C1+theta500_C1_pix]
        del y_C1_fix; y_C1_in = [y_C1-theta500_C1_pix, y_C1+theta500_C1_pix]

        del x_C2_fix; x_C2_in = [x_C2-theta500_C2_pix, x_C2+theta500_C2_pix]
        del y_C2_fix; y_C2_in = [y_C2-theta500_C2_pix, y_C2+theta500_C2_pix]

    if 'freers' in fit_method:
        del rs_C1_fix; rs_C1_in = [1.00E-02,1.00E+04]
        del rs_C2_fix; rs_C2_in = [1.00E-02,1.00E+04]


fixed = [eval(f'{key}_fix') for key in pfix]

periodic = np.zeros(pnum,dtype=bool)
for pi, p in enumerate(pvar):
    if 'theta' in p and eval(f'{p}_in[0]')+np.pi==eval(f'{p}_in[1]'):
        periodic[pi] = True

pdist = []
for pi, par in enumerate(pvar):
    loc   = eval(f'{par}_in[0]')
    scale = eval(f'{par}_in[1]')-loc
    if par[:2] in ['A_','rs_']:
        print(par,par[:2])
        pdist.append(scipy.stats.loguniform(a=loc,b=scale))
    else:
        # [x0,x1]
        # loc = x0
        # scale = x1-x0
        pdist.append(scipy.stats.uniform(loc=loc,scale=scale))
    
# Funcions for the nested sampling
# =============================================================================

def get_model(theta):
    p = dict(zip(pvar,theta))
    p.update(dict(zip(pfix,fixed)))

    if '3gnfw' in fit_method:
        p['x_C3'] = p['x_C2']+p['x_C3']
        p['y_C3'] = p['y_C2']+p['y_C3']

    map_model = get_prof(x,y,p,fit_method)
    if point_model!='none': map_model += get_point(x,y,p,point_model)
    return tools.convolve(map_model, map_res, dim_pix)

# Prior hypercube transformation
# -----------------------------------------------------------------------------
def prior_transform(theta):
    prior = []

    xold = y_S1b_in[0]
    pordr = {'y_S1a': None,'y_S1b': None}

    for ki, key in enumerate(pordr.keys()):
        if key in pvar:
            xold = xold+(y_S1a_in[1]-xold)*theta[pvar.index(key)]**(1.00/(len(pordr.keys())-ki))
            pordr[key] = xold

    for pi in range(pnum):
        if pvar[pi] not in ['y_S1a','y_S1b']:
            prior.append(pdist[pi].ppf(theta[pi]))
        else:
            prior.append(pordr[pvar[pi]])

    return np.array(prior)
   #return np.array([pdist[pi].ppf(theta[pi]) for pi in range(pnum)])

# Log-prior function
# -----------------------------------------------------------------------------
def log_prior(theta):
    prior = np.array([pdist[pi].logpdf(theta[pi]) for pi in range(pnum)])
    return -np.inf if np.any(~np.isfinite(prior)) else np.sum(prior)

# Log-likehood function
# -----------------------------------------------------------------------------
def log_likelihood(theta):   
    map_model = get_model(theta)
    map_resid = data - map_model

    chisq = np.abs(enmap.fft(map_resid*apod))**2
    chisq = d.iN*chisq
    return -0.50*(np.sum(chisq)+chinorm)

# Log-posterior function
# -----------------------------------------------------------------------------
def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


# Main sampling routine 
# =============================================================================

if prova_num[ 0]=='_': prova_num = prova_num[1:]
if prova_num[-1]=='_': prova_num = prova_num[:-1]

os.system(f'mkdir -p {main_path_res}/{fit_method}_{prova_num}')

path_res = f'{main_path_res}/{fit_method}_{prova_num}/{fit_method}_{prova_num}_{method}_nlive={nlive}'
checkpoint_file = f'{path_res}_checkpoint'

occ = None

# Nautilus sampler
# -----------------------------------------------------------------------------
if method=='emcee':
    nsteps = 100
    ncheck = int(maxiter/nsteps)

    coeff_burn = 10
    chain_length = 20

    pool = None if ncores==1 else Pool(ncores)

    backend = emcee.backends.HDFBackend(f'{checkpoint_file}.hdf5')
    sampler = emcee.EnsembleSampler(nwalk,pnum,log_posterior,pool=pool,backend=backend)
   
    if sampler.iteration<maxiter:

        if sampler.iteration==0:
            state = np.array([pdist[pi].rvs(size=nwalk) for pi in range(pnum)]).T
            sampler.run_mcmc(state,nsteps,progress=True)

        autocorr_min = np.empty(ncheck)
        autocorr_max = np.empty(ncheck)
        autocorr_avg = np.empty(ncheck)

        old_tau = np.inf
    #for k in range(ncheck):
        converged, k = False, 0
        while k<ncheck and sampler.iteration<maxiter and not converged:
            sampler.run_mcmc(None,nsteps,progress=True)

            tau = sampler.get_autocorr_time(tol=0)
            autocorr_min[k] = np.min(tau)
            autocorr_max[k] = np.max(tau)
            autocorr_avg[k] = np.mean(tau)
            
            converged =  np.all(tau*chain_length<sampler.iteration)
            converged &= np.all(np.abs(old_tau-tau)/tau<0.01)

            old_tau = tau
            k += 1

    if pool is not None:
        pool.close()

    tau = sampler.get_autocorr_time(tol=0)
    nburns = coeff_burn*int(np.max(tau))

    if nburns<sampler.iteration and not np.isnan(np.max(tau)):
        nburns = coeff_burn*int(np.max(tau))
        samples = sampler.get_chain(discard=nburns,flat=True)
    else:
        samples = sampler.get_chain(discard=0,flat=True)

    print(samples.shape)
    weights = None
    logl = logz = None

elif method=='pocomc':
    os.system(f'mkdir -p {checkpoint_file}')
    pool = None if ncores==1 else Pool(ncores)
    
    prior = pocomc.Prior(pdist)
    sampler = pocomc.Sampler(likelihood = log_likelihood,
                               periodic = np.arange(pnum)[periodic],
                                  prior = prior,
                            n_effective = nlive,
                               n_active = nlive//2,
                              vectorize = False,
                                   pool = pool,
                             output_dir = f'{checkpoint_file}',
                           random_state = 0)
    states_ = sorted(glob.glob(f'{checkpoint_file}/*.state'))
    sampler.run(save_every=10,resume_state_path=states_[-1] if len(states_) else None,progress=True)

    if pool is not None: pool.close()

    samples, weights, logl, _ = sampler.posterior()
    logz, _ = sampler.evidence()
else:
    if method=='nautilus':
        toc = time.time()
        if ncores==1:
            sampler = nautilus.Sampler(prior = prior_transform,
                                  likelihood = log_likelihood,
                                    periodic = np.arange(pnum)[periodic],
                                       n_dim = pnum, 
                                      n_live = nlive,
                                   pass_dict = False,
                                    filepath = f'{checkpoint_file}.hdf5',
                                      resume = resume,
                                        pool = None)
            sampler.run(n_like_max=maxiter,verbose=True,discard_exploration=True)
            samples, weights, logl = sampler.posterior()
            occ = sampler.shell_bound_occupation()
        else:
            with Pool(ncores,log_likelihood,prior_transform) as pool:
                sampler = nautilus.Sampler(prior = pool.prior_transform,
                                      likelihood = pool.loglike,
                                        periodic = np.arange(pnum)[periodic],
                                           n_dim = pnum, 
                                          n_live = nlive,
                                       pass_dict = False,
                                        filepath = f'{checkpoint_file}.hdf5',
                                          resume = resume,
                                            pool = pool)
            
                sampler.run(n_like_max=maxiter,verbose=True,discard_exploration=True)
                samples, weights, logl = sampler.posterior()
                occ = sampler.shell_bound_occupation()
                
        tic = time.time(); print(f'Elapsed time: {tic-toc:.2f} s')

        logz = sampler.log_z
        
        weights = np.exp(weights)
        weights[np.isnan(weights)] = 0.00
        weights = weights/np.sum(weights)
    elif method in ['static','dynamic']:

    # Dyntesty nested sampler (static|dynamic)
    # ---------------------------------------------------------------------------
        if method=='static':
            _sampler = dynesty.NestedSampler
        elif method=='dynamic':
            _sampler = dynesty.DynamicNestedSampler

        with Pool(ncores,log_likelihood,prior_transform) as pool:

            if resume and os.path.exists(checkpoint_file):
                sampler = _sampler.restore(checkpoint_file,pool=pool)
            else:
                sampler = _sampler(loglikelihood = pool.loglike,
                                prior_transform = pool.prior_transform,
                                        periodic = periodic,
                                        nlive = nlive,
                                            ndim = pnum,
                                    queue_size = ncores,
                                            pool = pool)
            kwargs_sampler = dict(checkpoint_file=checkpoint_file,resume=resume)

            sampler.run_nested(maxiter=maxiter,**kwargs_sampler)

        results = sampler.results
        results.summary()
        
        samples = results['samples']
        weights = results.importance_weights()
        logl = results['logl']
        logz = results.logz[-1]

    print(f'logz = {logz:.2f}')

# Plotting results
# =============================================================================

# for key in ['theta_C2']:
#     if key in pvar:
#         pp = samples[:,pvar.index(key)]
#         pp[pp<0.00] = pp[pp<0.00]+np.pi
#         samples[:,pvar.index(key)] = pp

pout = np.empty((3,pnum))
for pi in range(pout.shape[1]):
    qi = corner.quantile(samples[:,pi],[0.16,0.50,0.84],weights=weights)
    pout[:,pi] = np.array([qi[1],*np.diff(qi)])

if True:
    mout = [get_model(pout[0])]
    rout = [data-mout]
else:
    nsamp = dynesty.utils.resample_equal(samples,weights)
    ksamp = scipy.stats.gaussian_kde(nsamp.T)
    nsamp = ksamp.resample(200).T

    msamp = []
    for si, samp in enumerate(nsamp):
        msamp.append(get_model(samp))
    
    mout = np.quantile(msamp,0.50,axis=0)
    rout = data-mout

# tools.save_fits(mout,f'{path_res}_','map_model',map_file[0].header)
# tools.save_fits(rout,f'{path_res}_','map_residual',map_file[0].header)

for component in ['C1','C2','C3','S1a','S1b','S2','fil']:
    pone = pout[0].copy()
    for ci, c in enumerate(['C1','C2','C3','S1a','S1b','S2','fil']):
        if component!=c and f'A_{c}' in pvar:
            pone[pvar.index(f'A_{c}')] = 0.00
                    
    mone = get_model(pone)
    rone = data-mone

    mout.append(mone)
    rout.append(rone)

mout = np.array(mout)
rout = np.array(rout)[:,0,:,:]

fits.writeto(f'{path_res}_map_model.fits',mout,header=map_file[0].header,overwrite=True)
fits.writeto(f'{path_res}_map_residual.fits',rout,header=map_file[0].header,overwrite=True)

for key in ['r_s','delta_fil','rc_fil','l_fil','rc_C1','rc_C2']:
    if key in pvar:
        samples[:,pvar.index(key)] *= dim_pix

for key in ['rs_C1','rs_C2','rs_C3','A_C1','A_C2','A_C3','A_S1a','A_S1b','A_S2','A_fil']:
    if key in pvar:
        samples[:,pvar.index(key)] = np.log10(samples[:,pvar.index(key)])

for key in ['x_S1b','y_S1b']:
    if key in pvar:
        samples[:,pvar.index(key)] = samples[:,pvar.index(key)]+samples[:,pvar.index(key.replace('2','1'))]

with open(f'{path_res}_samples','wb') as out_file:
    dill.dump(dict(samples=samples,weights=weights,logl=logl,logz=logz,occ=occ),out_file,dill.HIGHEST_PROTOCOL)

with open(f'{path_res}_fit_results','w') as out_file:
    out_file.write('#Parameter   BestFit   -1sigma   +1sigma\n')

    for pi, p in enumerate(pout.T):
        out_file.write(f'{pvar[pi]}      {p[0]:.4e}      {p[1]:.4e}      {p[2]:.4e}\n')

# edges = []
# for pi in range(pnum):
#     if 'theta' in pvar[pi]:
#         edge = (f'{pvar[pi]}_in[0]',f'{pvar[pi]}_in[0]')
#     else:
#         edge = (np.maximum(eval(f'{pvar[pi]}_in[0]'),pout[0,pi]-5.00*np.abs(pout[1,pi])), \
#                 np.minimum(eval(f'{pvar[pi]}_in[1]'),pout[0,pi]+5.00*np.abs(pout[2,pi])))
#     
#     edges.append(edge)

for component in ['C1','C2','C3','S1a','S1b','S2','fil']:
    subsamp, subpvar = [], []
    for pi, p in enumerate(pvar):
        if component in p:
            subsamp.append(samples[:,pi])
            subpvar.append(p)
    subsamp = np.array(subsamp).T

    if len(subpvar)>0:
        fig = corner.corner(subsamp,weights=weights,labels=subpvar,quantiles=[0.16,0.5,0.84],show_titles=True,title_fmt='.2E',plot_datapoints=True)
        fig.savefig(f'{path_res}_corner_plots_{component}.pdf',format='pdf',dpi=300)

subsamp, subpvar = [], []
for component in ['C1','C2','C3','fil']:
    for pi, p in enumerate(pvar):
        if component in p:
            subsamp.append(samples[:,pi])
            subpvar.append(p)
subsamp = np.array(subsamp).T

fig = corner.corner(subsamp,weights=weights,labels=subpvar,quantiles=[0.16,0.5,0.84],show_titles=True,title_fmt='.2E',plot_datapoints=True)
fig.savefig(f'{path_res}_corner_plots_total.pdf',format='pdf',dpi=300)

subsamp, subpvar = [], []

subsamp, subpvar = [], []
for component in ['C2','S1a','S1b','S2']:
    for pi, p in enumerate(pvar):
        if component in p:
            subsamp.append(samples[:,pi])
            subpvar.append(p)
subsamp = np.array(subsamp).T

fig = corner.corner(subsamp,weights=weights,labels=subpvar,quantiles=[0.16,0.5,0.84],show_titles=True,title_fmt='.2E',plot_datapoints=True)
fig.savefig(f'{path_res}_corner_plots_total_C2+S1+S2.pdf',format='pdf',dpi=300)
