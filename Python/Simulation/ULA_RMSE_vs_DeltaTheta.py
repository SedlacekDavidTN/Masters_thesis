# -*- coding: utf-8 -*-
"""
Created on Mon May 20 08:43:53 2024

@author: sedla
"""

import DSP_Lib as dsp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)
matplotlib.rc('font', weight = 'bold')
matplotlib.rcParams['xtick.major.pad']='7'
matplotlib.rcParams['ytick.major.pad']='3'

sample_rate = 1e6
num_samples = 100000
num_signals = 2
freq = [20e3, 22e3]
theta_deg = [10, 40]
snr_dB = 20

d_theta = np.linspace(theta_deg[0], theta_deg[1], 20)

num_elem = 5
delta = 0.5

rmse_d_theta = np.zeros((len(d_theta),6))

theta_doa = np.deg2rad(theta_deg)


for i in range(len(d_theta)):
    ref_data = dsp.generate_data_ULA(freq, [theta_deg[0], d_theta[i]], num_elem, delta, sample_rate, num_samples)
    data = dsp.add_noise(ref_data, snr_dB)
    
    R_xx = dsp.covariance_matrix(data)
    
    results, theta_scan = dsp.doa_bartlett_ULA(R_xx, delta, 1000)
    max_val_arg = dsp.peaks_doa(results, num_signals)
    theta_result = theta_scan[max_val_arg]
    theta_result_sorted = theta_result[np.argsort(theta_result)]
    rmse_d_theta[i,0] = dsp.calculate_rmse(theta_result_sorted, [theta_doa[0], np.deg2rad(d_theta[i])])
    
    results, theta_scan = dsp.doa_MVDR_ULA(R_xx, delta, 1000)
    max_val_arg = dsp.peaks_doa(results, num_signals)
    theta_result = theta_scan[max_val_arg]
    theta_result_sorted = theta_result[np.argsort(theta_result)]
    rmse_d_theta[i,1] = dsp.calculate_rmse(theta_result_sorted, [theta_doa[0], np.deg2rad(d_theta[i])])
    
    results, theta_scan = dsp.doa_linpred_ULA(R_xx, delta, 0, 1000)
    max_val_arg = dsp.peaks_doa(results, num_signals)
    theta_result = theta_scan[max_val_arg]
    theta_result_sorted = theta_result[np.argsort(theta_result)]
    rmse_d_theta[i,2] = dsp.calculate_rmse(theta_result_sorted, [theta_doa[0], np.deg2rad(d_theta[i])])
    
    results, theta_scan = dsp.doa_MUSIC_ULA(R_xx, delta, num_signals, 1000)
    max_val_arg = dsp.peaks_doa(results, num_signals)
    theta_result = theta_scan[max_val_arg]
    theta_result_sorted = theta_result[np.argsort(theta_result)]
    rmse_d_theta[i,3] = dsp.calculate_rmse(theta_result_sorted, [theta_doa[0], np.deg2rad(d_theta[i])])
    
    results, theta_scan = dsp.doa_minnorm_ULA(R_xx, delta, num_signals, 1000)
    max_val_arg = dsp.peaks_doa(results, num_signals)
    theta_result = theta_scan[max_val_arg]
    theta_result_sorted = theta_result[np.argsort(theta_result)]
    rmse_d_theta[i,4] = dsp.calculate_rmse(theta_result_sorted, [theta_doa[0], np.deg2rad(d_theta[i])])
    
    theta_result = dsp.doa_rootMUSIC_ULA(R_xx, delta, num_signals)
    theta_result_sorted = theta_result[np.argsort(theta_result)]
    rmse_d_theta[i,5] = dsp.calculate_rmse(theta_result_sorted, [theta_doa[0], np.deg2rad(d_theta[i])])
    
plt.figure()
plt.plot(d_theta - theta_deg[0], rmse_d_theta[:,0], '--x', linewidth=2, markersize=6, alpha=1, label="Bartlett")
plt.plot(d_theta - theta_deg[0], rmse_d_theta[:,1], '-.s', linewidth=2, markersize=6, alpha=1, label="MVDR")
plt.plot(d_theta - theta_deg[0], rmse_d_theta[:,2], ':^', linewidth=2,markersize=6, alpha=1, label="Lin. Pred.")

plt.legend(loc = "upper right", fontsize=15)
plt.yscale("log")
plt.xlabel("$\Delta$$\Theta$ [°]", fontsize=18)
plt.ylabel("RMSE", fontsize=18)
plt.tight_layout()
plt.grid()

plt.figure()
plt.plot(d_theta - theta_deg[0], rmse_d_theta[:,3], '-.x', linewidth=2, markersize=6, alpha=1, label="MUSIC")
plt.plot(d_theta - theta_deg[0], rmse_d_theta[:,4], '-.^', linewidth=2, markersize=6, alpha=1, label="Min. Norm")
plt.plot(d_theta - theta_deg[0], rmse_d_theta[:,5], '-.s', linewidth=2, markersize=6, alpha=1, label="Root-MUSIC")

plt.legend(loc = "upper right", fontsize=15)
plt.yscale("log")
plt.xlabel("$\Delta$$\Theta$ [°]", fontsize=18)
plt.ylabel("RMSE", fontsize=18)
plt.tight_layout()
plt.grid()