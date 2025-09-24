import numpy as np

def init_cond_cal ():
    #a_exp = [0.7, 3] #coefficient of exposure map
    a_bkg = [0.7, 1.3] #coefficient of background map
    return a_bkg #a_exp,

def init_cond_cls (model): 
    if model == 'gnfw_circ+beta_circ':
        #x_main = [51, 55] #main cluster 53.58, 40.18 from x-ray chandra (?)
        #y_main = [38, 42]
        x_sub = [63, 67] #sub cluster 64.87, 44.03 from x-ray chandra (?)
        y_sub = [42, 46]
        rs_main = [0.05, 0.3] #Cluster characteristic scale radius (0.3 arcmin) #rs = r_vir/c_vir = 1.6 Mpc hˆ-1 / 6.9
        rc_sub = [0.05, 0.4] #Cluster characteristic scale radius (0.1 arcmin) #rs = r_vir/c_vir = 1.6 Mpc hˆ-1 / 6.9
        beta_main = [4, 7]   # Beta parameter , outer slope
        beta_sub = [2, 6]   # Beta parameter, outer slope 
        A_main = [50,1000] #[70, 90] #[50,1000]    # First Cluster Amplitude (counts)
        A_sub = [10,300] #[10, 30] #[10,300]     # Second Cluster Amplitude (counts)
        return x_sub, y_sub, rs_main, rc_sub, beta_main, beta_sub, A_main, A_sub #x_main, y_main, 
    
    #--------
    if model == 'beta_circ':
        x_fit = [70, 80]
        y_fit = [70, 80]
        A = [450, 550]    # First Cluster Amplitude (Compton-y)
        beta = [0.3, 0.7]     # Beta parameter
        rc = [0.05, 0.1]        # Core radius (arcmin)
        return x_fit, y_fit, A, beta, rc
    
    if model == 'beta_ell':
        x_fit = [75, 82]
        y_fit = [75, 82]
        A = [50, 1000]    # First Cluster Amplitude (Compton-y)
        beta = [0.3, 0.8]     # Beta parameter
        rc = [0.1, 1]        # Core radius (arcmin)
        ecc = [0.7, 0.9]      # Cluster eccentricity
        theta = [0, np.pi/2]   # Cluster angle (rad)
        return x_fit, y_fit, A, beta, rc, ecc, theta
    
    
    if model == '2gnfw_ell':
        x_C1 = [128, 132] #Higher cluster
        y_C1 = [138, 142]
        x_C2 = [73, 81] #Lower cluster
        y_C2 = [68, 76]
        rs_C1 = [5, 20] #Cluster characteristic scale radius (arcmin) #rs = r_vir/c_vir = 1.6 Mpc hˆ-1 / 6.9
        rs_C2 = [5, 20] #Cluster characteristic scale radius (arcmin) #rs = r_vir/c_vir = 1.6 Mpc hˆ-1 / 6.9
        theta_C1 = [0, np.pi/2]   # Cluster angle (rad)
        theta_C2 = [0, np.pi]    # Cluster angle (rad)
        ecc_C1 = [0.1, 1.0]      # Cluster eccentricity
        ecc_C2 = [0.1, 1.0]      # Cluster eccentricity
        beta_C1 = [2, 6]  # Beta parameter , outer slope
        beta_C2 = [2, 6]  # Beta parameter, outer slope 
        A_C1 = [5, 50]    # First Cluster Amplitude (Compton-y)
        A_C2 = [5, 50]    # Second Cluster Amplitude (Compton-y)
        return x_C1, y_C1, x_C2, y_C2, rs_C1, rs_C2, theta_C1, theta_C2, ecc_C1, ecc_C2, beta_C1, beta_C2, A_C1, A_C2


