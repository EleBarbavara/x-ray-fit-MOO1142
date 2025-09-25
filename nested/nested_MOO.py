'''
This code run the first iteration of the mcmc to initialize the chain. Then, use restart_mcmc.py
The fit is adapted for cases like A3391-95, i.e. 2cls + bridge + point source
'''
import os
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from math import factorial
import matplotlib.pyplot as plt
import corner
import matplotlib as mpl
import scipy
import yaml

import tools
import models_ac_fixed as models
import initial_params_ac_fixed as initial_params

import pocomc
import glob
#from pixell import enmap, bunch
import multiprocess
from multiprocess import Pool



'''
----------------------------- FUNCTIONS FOR MCMC ------------------------------
'''
# functions for implementing the MCMC are structured following emcee manual (https://emcee.readthedocs.io)

def log_prior (var):
        
    if fit_method == 'gnfw_ell':      
        x_main, y_main, rs_main, angle_main, ecc_main, beta_main, A_main, a_bkg = var 
        
        if x_main_in[0] < x_main < x_main_in[1] and \
        y_main_in[0] < y_main < y_main_in[1] and \
        rs_main_in[0] < rs_main < rs_main_in[1] and \
        angle_main_in[0] < angle_main < angle_main_in[1] and \
        ecc_main_in[0] < ecc_main < ecc_main_in[1] and \
        beta_main_in[0] < beta_main < beta_main_in[1] and \
        A_main_in[0] < A_main < A_main_in[1] and \
        a_bkg_in[0] < a_bkg < a_bkg_in[1]: 
            return 0.0
        else:
            return -np.inf
        
def log_likelihood (var):            
    
    if fit_method == 'gnfw_ell':      
        x_main, y_main, rs_main, angle_main, ecc_main, beta_main, A_main, a_bkg = var 
        
        mod_main = models.GNFW_model_cluster_ell(x, y, x_main, y_main, rs_main, angle_main, ecc_main, beta_main, alpha_main_fix, gamma_main_fix, A_main)
        
        map_model = models.exp(1, exp_main_map)*(tools.convolve(mod_main, psf_main_map)) +\
                    models.bkg(a_bkg, bkg_map)
    
    
    c = np.nansum(map_model - map_data*np.log(map_model)) + c0
    #print('c = ', c)
    '''
    def cstat(md, mp):
        if md == np.nan or md<0:
            return np.nan
        else:
            return md - mp*np.log(md)

    cstat = np.vectorize(cstat)

    c1 = np.nansum(cstat(map_model, map_data)) +c0
    print('c1 = ', c1)
    '''
    
    return 0.5*c

                
def log_posterior (var):
    
    lp = log_prior(var)
    
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(var)

def get_config_file(config_file):
    with open(config_file, 'r') as stream:
        try:
            config_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config_data

#def get_priors(cfg):
#    return cfg['priors']['x_C1'], cfg['priors']['y_C1'], cfg['priors']['x_C2'], cfg['priors']['y_C2'], cfg['priors']['c500'], cfg['priors']['rs_C1'], cfg['priors']['rs_C2'], cfg['priors']['delta_fil'], cfg['priors']['rc_fil'], cfg['priors']['l_fil']


def get_n0(I0, I_cap, beta, rc):
    gamma_func_ratio = scipy.special.gamma(3*beta - 0.5)/scipy.special.gamma(3*beta)
    rc_norm = rc #trasformare in Mpc????????
    n0 = 1e-3 * (np.pi)**(-0.25) * np.sqrt( (I0/I_cap) * gamma_func_ratio / rc_norm )
    return n0


'''
-------------------------------------------------------------------------------
'''
    
###############################################################################
############################# SELECT THESE VALUES #############################
cfg = get_config_file('./config_MOO1142.yaml')

# Select the fitting method (here some examples of combinations of models for cls and bridge, see models.py)
fit_method = cfg['fit_method']
# Folder with the map
path = cfg['data']['gen_path']
prova_num= cfg['save_fit']['prova_num']  # Number/Name of this check

# Important numbers to run the fit
Nwalkers= int(cfg['Nwalkers'])  # Number of point you want to generate for each step. It must be 4-5 times the number of parameters
Nsamples=int(cfg['Nsamples']) # Number of iterations you want to run
Nsteps=int(cfg['Nsteps'])  # Check the autocorrelation time every Nsteps
chain_length=int(cfg['chain_length'])  # the chain must be longer 'chain_length' times the autocorrelation time
N_check=int( Nsamples/Nsteps)  # Number of checks that will be performed
coeff_burn=int(cfg['coeff_burn'])  # Burn the first coeff_burn*np.max(tau) iterations
ncores=int(multiprocess.cpu_count()/2) - 10 # Number of cores used for parallelization

###############################################################################
###############################################################################

# Create the folders for the results 
path_res = path + 'fit/' + fit_method + prova_num + '/'
if os.path.isdir(path_res)==False:
	os.mkdir(path_res)
print(path_res)
print('Num samples: ', Nsamples)
# Open the map 
map_name = cfg['data']['map_file']
psf_main_name = cfg['data']['psf_main_file']
psf_sub_name = cfg['data']['psf_sub_file']
exp_main_name = cfg['data']['exp_main_file']
exp_sub_name = cfg['data']['exp_sub_file']
bkg_name = cfg['data']['bkg_file']

map_file=fits.open(path + map_name)
psf_main_file=fits.open(path + psf_main_name)
psf_sub_file=fits.open(path + psf_sub_name)
exp_main_file=fits.open(path + exp_main_name)
#wcs_north = WCS(exp_north_file[0].header)
exp_sub_file=fits.open(path + exp_sub_name)
#wcs_south = WCS(exp_south_file[0].header)
bkg_file=fits.open(path + bkg_name)


header=map_file[0].header
wcs=WCS(header)
map_data = map_file[0].data
map_data = map_data.astype(float)
psf_main_map=psf_main_file[0].data
psf_sub_map=psf_sub_file[0].data
exp_main_map_raw=exp_main_file[0].data
#Xdim_exp=map_file[0].header['NAXIS1']  
#Ydim_exp=map_file[0].header['NAXIS2']
exp_main_map_norm = exp_main_map_raw /  np.max(exp_main_map_raw)
store_exp_main_norm_factor = np.max(exp_main_map_raw)
exp_sub_map_raw=exp_sub_file[0].data
#Xdim_exp=map_file[0].header['NAXIS1']  
#Ydim_exp=map_file[0].header['NAXIS2']
exp_sub_map_norm = exp_sub_map_raw /  np.max(exp_sub_map_raw)
store_exp_sub_norm_factor = np.max(exp_sub_map_raw)
#Xdim_exp=map_file[0].header['NAXIS1']  
#Ydim_exp=map_file[0].header['NAXIS2']

#set the pixel with zero exposure to nan
e_min = cfg['min_exposure']
exp_main_map = np.where(exp_main_map_norm>e_min, exp_main_map_norm, np.nan)
exp_main_map = exp_main_map.astype(float)
exp_sub_map = np.where(exp_sub_map_norm>e_min, exp_sub_map_norm, np.nan)
exp_sub_map = exp_sub_map.astype(float)

bkg_map=bkg_file[0].data
#Xdim_bkg=map_file[0].header['NAXIS1']  
#Ydim_bkg=map_file[0].header['NAXIS2']
bkg_map = bkg_map.astype(float)

print(fit_method)
if fit_method=='gnfw_ell':
    '''
    min_x = min(np.shape(map_data)[0], np.shape(bkg_map)[0])
    min_y = min(np.shape(map_data)[1], np.shape(bkg_map)[1])
    min_dim = min(min_x, min_y)
    map_data = map_data[:min_dim, :min_dim]
    exp_main_map = exp_main_map[:min_dim, :min_dim]
    exp_sub_map = exp_sub_map[:min_dim, :min_dim]
    bkg_map = bkg_map[:min_dim, :min_dim]
    ''' 
    map_data=map_data[1:85,16:100]
    exp_main_map=exp_main_map[1:85,16:100]
    exp_sub_map=exp_sub_map[1:85,16:100]
    bkg_map=bkg_map[1:85,16:100]
    Xdim=np.shape(map_data)[1] 
    Ydim=np.shape(map_data)[0]
else:
    Xdim=map_file[0].header['NAXIS1']  
    Ydim=map_file[0].header['NAXIS2']

print(Xdim)
print(Ydim)
print(np.shape(map_data))
print(np.shape(exp_main_map))
print(np.shape(exp_sub_map))
print(np.shape(psf_main_map))
print(np.shape(psf_sub_map))
print(np.shape(bkg_map))


dim_pix=cfg['dim_pixel']*u.deg.to('arcmin')  # pixel size in arcmin = 0.04167 arcmin

def stirling(n):
    if n == 0:
        return 1
    else:
        return n*np.log(n)-n

stirling = np.vectorize(stirling)

c0 = 2*np.nansum(stirling(map_data))
print('c0 = ', c0)

# clusters position (in pixels) and R500
#x_C1, y_C1, x_C2, y_C2, c500, rs_C1, rs_C2, delta_fil_factor, rc_fil, l_fil = get_priors(cfg) #read this data from a file or define them
#theta500_C1_arcmin=c500*rs_C1
#theta500_C2_arcmin=c500*rs_C2
'''
# For the GNFW model
theta500_arcmin = 0
theta500_pix=theta500_arcmin/dim_pix
c500=1.81  #from Planck Collaboration V 2013 

c_vir = 6.8 #+2.1, -1.8 from https://arxiv.org/pdf/1204.3630.pdf
theta_vir_arcmin = 7 #arcmic -> sarebbe r_vir in angolo (theta=r/D_l)
theta_vir_pix = theta_vir_arcmin/dim_pix
#rs=(theta_vir_arcmin/c_vir)/dim_pix

# For the GNFW model
c500=1.81  #from Planck Collaboration V 2013 
rs_C1=(theta500_C1_arcmin/c500)/dim_pix
rs_C2=(theta500_C2_arcmin/c500)/dim_pix
'''

# Create the map for the model
x = np.arange(0, Xdim, 1)
y = np.arange(0, Ydim, 1)
x,y = np.meshgrid(x,y)

# Define range of values for the parameters
if fit_method == 'gnfw_ell':
    # Clusters
    '''
    x_C1_fix=x_C1
    y_C1_fix=y_C1
    x_C2_fix=x_C2
    y_C2_fix=y_C2
    
    x_main_fix = 39 #53.2 for rectangular image
    y_main_fix = 41 #40.31 for rectangular image
    '''
    alpha_main_fix = 2.26 #1.0510
    gamma_main_fix = 0.93 #2*0.3081
    #beta_main_fix = 6 #2*3
    
    
    x_main_in, y_main_in, rs_main_in, angle_main_in, ecc_main_in, beta_main_in, \
    A_main_in = initial_params.init_cond_cls('gnfw_ell') 
    a_bkg_in = initial_params.init_cond_cal()
    rs_main_in = np.array(rs_main_in)/dim_pix
    
    print('A main: ', A_main_in)
    print('x_main_in: ', x_main_in)
    print('y_main_in: ', y_main_in)
    print('rs_main_in: ', rs_main_in)
    print('angle_main_in: ', angle_main_in)
    print('ecc_main_in: ', ecc_main_in)
    print('beta_main_in: ', beta_main_in)
    print('-------------------------------------------')
    
    # Write a file with the Initial Parameters used 
    out_file = open(path_res+'initial_parameters', 'w')
    out_file.write('#Parameter  min_value  max_value \n')
    out_file.write('x_main             %.4e      %.4e      \n' % (x_main_in[0], x_main_in[1]) )
    out_file.write('y_main             %.4e      %.4e      \n' % (y_main_in[0], y_main_in[1]) )
    out_file.write('rs_main(arcmin)    %.4e      %.4e      \n' % (rs_main_in[0]*dim_pix, rs_main_in[1]*dim_pix) )
    out_file.write('angle_main          %.4e      %.4e      \n' % (angle_main_in[0], angle_main_in[1]) )
    out_file.write('ecc_main          %.4e      %.4e      \n' % (ecc_main_in[0], ecc_main_in[1]) )
    out_file.write('alpha_main         %.4e                \n' % (alpha_main_fix) )#(alpha_main_in[0], alpha_main_in[1]) )
    out_file.write('beta_main         %.4e     %.4e    \n' % (beta_main_in[0], beta_main_in[1]) )
    out_file.write('gamma_main         %.4e                \n' % (gamma_main_fix) )#(gamma_main_in[0], gamma_main_in[1]) )
    out_file.write('A_main             %.4e      %.4e      \n' % (A_main_in[0], A_main_in[1]) )
    out_file.write('a_bkg            %.4e      %.4e      \n' % (a_bkg_in[0], a_bkg_in[1]) )
    out_file.close()

# Generate the initial chain using an uniform distribution
# NB: use the parameters in the same order as in var in the mcmc functions
if fit_method == 'gnfw_ell':
    x_main_first=np.random.uniform(x_main_in[0], x_main_in[1], Nwalkers)
    y_main_first=np.random.uniform(y_main_in[0], y_main_in[1], Nwalkers)
    rs_main_first=np.random.uniform(rs_main_in[0], rs_main_in[1], Nwalkers)
    angle_main_first=np.random.uniform(angle_main_in[0], angle_main_in[1], Nwalkers)
    ecc_main_first=np.random.uniform(ecc_main_in[0], ecc_main_in[1], Nwalkers)
    beta_main_first=np.random.uniform(beta_main_in[0], beta_main_in[1], Nwalkers)
    A_main_first=np.random.uniform(A_main_in[0], A_main_in[1], Nwalkers)
    a_bkg_first=np.random.uniform(a_bkg_in[0], a_bkg_in[1], Nwalkers)
    
    init_sample = np.array ([x_main_first, y_main_first, rs_main_first, \
                            angle_main_first, ecc_main_first, beta_main_first, A_main_first, a_bkg_first]).T 

ndim = init_sample.shape[1]  # number of parameters

pool = None if ncores==1 else Pool(ncores)

pdist = np.copy(init_sample)
pnum = np.copy(ndim)
nlive = 1000

prior = pocomc.Prior(pdist)
sampler = pocomc.Sampler(likelihood = log_likelihood,
                            periodic = np.arange(pnum),
                                prior = prior,
                        n_effective = nlive,
                            n_active = nlive//2,
                            vectorize = False,
                                pool = pool,
                            output_dir = f'{path_res}',
                        random_state = 0)
states_ = sorted(glob.glob(f'{path_res}/*.state'))
sampler.run(save_every=10,resume_state_path=states_[-1] if len(states_) else None,progress=True)

if pool is not None: pool.close()

samples, weights, logl, _ = sampler.posterior()
logz, _ = sampler.evidence()



tau=sampler.get_autocorr_time(tol=0)
print ('Number of iterations: {0}'.format(sampler.iteration))
print ('Autocorrelation for each parameter: {0}'.format(tau))
print ('Mean acceptance fraction: {0:.3f}'.format(np.mean(sampler.acceptance_fraction))) # output must be between 0.2 & 0.5, otherwise you must include moves=[(emcee.moves.StretchMove)] in emcee.EnsembleSampler

results = np.empty((3,pnum))
for pi in range(results.shape[1]):
    qi = corner.quantile(samples[:,pi],[0.16,0.50,0.84],weights=weights)
    results[:,pi] = np.array([qi[1],*np.diff(qi)])
    print('len postsamples: ', len(results))
    print(results)

print('Fit finito.')
'''
------------------------------------- PLOTS ----------------------------------
generate and save:
- corner plots for posteriors 
- .fits files for best-fitting models and residuals
- convergence plots for the free parameters
'''

# Convergence plot - min, max, mean
fig=plt.figure()
n = Nsteps * np.arange(1, index + 1)
y_min = autocorr_min[:index]
y_max = autocorr_max[:index]
y_mean = autocorr_mean[:index]

plt.plot(n, n / chain_length, "--k")
plt.plot(n, y_min, label=r'$\tau_{min}$')
plt.plot(n, y_max, label=r'$\tau_{max}$')
plt.plot(n, y_mean, label=r'$\tau_{mean}$')
plt.xlabel("number of iterations")
plt.ylabel(r"$\tau$")
plt.legend()
plt.savefig(path_res + 'convergence_plot2.png')


###########################################################################
if fit_method == 'gnfw_ell':
    
    # Save best fit parameters    
    
    out_file = open(path_res+'Fit_Results', 'w')
    out_file.write('#Parameter  BestFit -1sigma +1sigma\n')
    out_file.write('x_main\t%.4e\t%.4e\t%.4e\n' % (results[0,0], results[0,1], results[0,2]))
    out_file.write('y_main\t%.4e\t%.4e\t%.4e\n' % (results[1,0], results[1,1], results[1,2]))
    out_file.write('rs_main(arcmin)\t%.4e\t%.4e\t%.4e\n' % (results[2,0]*dim_pix, results[2,1]*dim_pix, results[2,2]*dim_pix))
    out_file.write('angle_main(arcmin)\t%.4e\t%.4e\t%.4e\n' % (results[3,0]*dim_pix, results[3,1]*dim_pix, results[3,2]*dim_pix))
    out_file.write('ecc_main\t%.4e\t%.4e\t%.4e\n' % (results[4,0], results[4,1], results[4,2]))
    out_file.write('beta_main\t%.4e\t%.4e\t%.4e\n' % (results[5,0], results[5,1], results[5,2]))
    out_file.write('A_sub\t%.4e\t%.4e\t%.4e\n' % (results[6,0], results[6,1], results[6,2]))
    out_file.write('a_bkg\t%.4e\t%.4e\t%.4e\n' % (results[7,0], results[7,1], results[7,2]))
    out_file.write('#Mean acceptance fraction    %.3f\n'       % (np.mean(sampler.acceptance_fraction)))
    out_file.write('#Normalization factor exp map main cluster    %.3f\n'       % (store_exp_main_norm_factor))
    out_file.write('#Normalization factor exp map sub cluster    %.3f\n'       % (store_exp_sub_norm_factor))
    store_exp_sub_norm_factor
    out_file.close()
    
    x_main_mcmc = results[0,0]
    y_main_mcmc = results[1,0]
    rs_main_mcmc = results[2,0]
    angle_main_mcmc = results[3,0]
    ecc_main_mcmc = results[4,0]
    beta_main_mcmc = results[5,0]
    A_main_mcmc = results[6,0]
    a_bkg_mcmc = results[7,0]
    
    postsamples[:,2]=postsamples[:,2]*dim_pix  #pix to arcmin
    '''
    out_file = open(path_res+'Fit_Results', 'w')
    out_file.write('#Parameter  BestFit -1sigma +1sigma\n')
    out_file.write('x_main\t%.4e\t%.4e\t%.4e\n' % (results[0,0], results[0,1], results[0,2]))
    out_file.write('y_main\t%.4e\t%.4e\t%.4e\n' % (results[1,0], results[1,1], results[1,2]))
    out_file.write('x_sub\t%.4e\t%.4e\t%.4e\n' % (results[2,0], results[2,1], results[2,2]))
    out_file.write('y_sub\t%.4e\t%.4e\t%.4e\n' % (results[3,0], results[3,1], results[3,2]))
    out_file.write('rs_main(arcmin)\t%.4e\t%.4e\t%.4e\n' % (results[4,0]*dim_pix, results[4,1]*dim_pix, results[4,2]*dim_pix))
    out_file.write('rc_sub(arcmin)\t%.4e\t%.4e\t%.4e\n' % (results[5,0]*dim_pix, results[5,1]*dim_pix, results[5,2]*dim_pix))
    out_file.write('beta_main\t%.4e\t%.4e\t%.4e\n' % (results[6,0], results[6,1], results[6,2]))
    out_file.write('beta_sub\t%.4e\t%.4e\t%.4e\n' % (results[7,0], results[7,1], results[7,2]))
    out_file.write('A_main\t%.4e\t%.4e\t%.4e\n' % (results[8,0], results[8,1], results[8,2]))
    out_file.write('A_sub\t%.4e\t%.4e\t%.4e\n' % (results[9,0], results[9,1], results[9,2]))
    out_file.write('a_bkg\t%.4e\t%.4e\t%.4e\n' % (results[10,0], results[10,1], results[10,2]))
    out_file.write('#Mean acceptance fraction    %.3f\n'       % (np.mean(sampler.acceptance_fraction)))
    out_file.close()
    
    x_main_mcmc = results[0,0]
    y_main_mcmc = results[1,0]
    x_sub_mcmc = results[2,0]
    y_sub_mcmc = results[3,0]
    rs_main_mcmc = results[4,0]
    rc_sub_mcmc = results[5,0]
    beta_main_mcmc = results[6,0]
    beta_sub_mcmc = results[7,0]
    A_main_mcmc = results[8,0]
    A_sub_mcmc = results[9,0]
    a_bkg_mcmc = results[10,0]
    
    postsamples[:,4]=postsamples[:,4]*dim_pix  #pix to arcmin
    postsamples[:,5]=postsamples[:,5]*dim_pix  #pix to arcmin
    '''
    # Create & save the best fit model
    mod_main = models.GNFW_model_cluster_ell(x, y, x_main_mcmc, y_main_mcmc, rs_main_mcmc, angle_main_mcmc, ecc_main_mcmc, beta_main_mcmc, alpha_main_fix, gamma_main_fix, A_main_mcmc)

    md = models.exp(1, exp_main_map)*(tools.convolve(mod_main, psf_main_map)) +\
        models.bkg(a_bkg_mcmc, bkg_map)

    tools.save_fits(md, path_res, 'FITTING_MODEL', header)
    tools.save_fits(map_data-md, path_res, 'RESIDUAL', header)

    # Corner plots
    mpl.use("Agg")

    

    fig = corner.corner(postsamples[:, :], labels=[r'$x_{main}$', r'$y_{main}$', r'$rs_{main}$', r'$angle_{main}$', r'$ecc_{main}$', r'$\beta_{main}$', r'$A_{main}$', r'$a_{bkg}$'], quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='.2E')
                #r'$x_{main}$', r'$y_{main}$', 
    fig.savefig(path_res + 'corner_plots.png')
    plt.close()

    # Convergence plot - all
    fig=plt.figure()
    n = Nsteps * np.arange(1, index + 1)

    y_x_main = autocorr_x_main[:index]
    y_y_main = autocorr_y_main[:index]
    y_rs_main = autocorr_rs_main[:index]
    y_angle_main = autocorr_angle_main[:index]
    y_ecc_main = autocorr_ecc_main[:index]
    y_beta_main = autocorr_beta_main[:index]
    y_A_main = autocorr_A_main[:index]
    y_a_bkg = autocorr_a_bkg[:index]


    plt.plot(n, n / chain_length, "--k")
    plt.plot(n, y_x_main, label=r'$x_{main}$')
    plt.plot(n, y_y_main, label=r'$y_{main}$')
    plt.plot(n, y_rs_main, label=r'$rs_{main}$')
    plt.plot(n, y_angle_main, label=r'$angle_{main}$')
    plt.plot(n, y_ecc_main, label=r'$ecc_{main}$')
    plt.plot(n, y_beta_main, label=r'$\beta_{main}$')
    plt.plot(n, y_A_main, label=r'$A_{main}$')
    plt.plot(n, y_a_bkg, linestyle='dashed', label=r'$a_{bkg}$')
    plt.xlabel("number of iterations")
    plt.ylabel(r"$\tau$")
    plt.legend(loc=2)
    plt.savefig(path_res + 'convergence_plot_all.png')
    plt.close()

    plt.plot(n, likelihood)
    plt.savefig(path_res + 'likelihood_plot_walkersmean.png')
    plt.close()
