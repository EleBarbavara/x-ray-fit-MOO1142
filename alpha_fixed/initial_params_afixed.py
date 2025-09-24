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
        rs_main = [0.05, 0.3]#[0.2, 0.5] #Cluster characteristic scale radius (0.3 arcmin) #rs = r_vir/c_vir = 1.6 Mpc hˆ-1 / 6.9
        rc_sub = [0.05, 0.4] #[0.3, 0.7] #Cluster characteristic scale radius (0.1 arcmin) #rs = r_vir/c_vir = 1.6 Mpc hˆ-1 / 6.9
        beta_main = [5, 7] #[4, 7]   # Beta parameter , outer slope of GNFW model
        gamma_main = [0.2,0.4]
        beta_sub = [2, 6]   # Beta parameter of isobeta model  
        A_main = [120, 400] #[70, 90] #[50,1000]     # First Cluster Amplitude (counts)
        A_sub = [40, 100] #[10,30] #[10, 300]      # Second Cluster Amplitude (counts)
        return x_sub, y_sub, rs_main, rc_sub, beta_main, gamma_main, beta_sub, A_main, A_sub #x_main, y_main, 
    

    
    