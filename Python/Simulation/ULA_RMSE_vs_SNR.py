# -*- coding: utf-8 -*-
"""
Created on Mon May 20 08:48:36 2024

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
num_signals = 1
freq = [20e3]
theta_deg = [15]
snr_step = np.linspace(-25,25,20)

num_elem = 5
delta = 0.5

sample_rate = 1e6
num_samples = 100000

rmse_doa = np.zeros((len(snr_step),6))
theta_doa = np.deg2rad(theta_deg)
theta_doa_sorted = theta_doa[np.argsort(theta_doa)]

ref_data = dsp.generate_data_ULA(freq, theta_deg, num_elem, delta, sample_rate, num_samples)

# Accuracy based on SNR
for i in range(len(snr_step)):
    data = dsp.add_noise(ref_data, snr_step[i])
    
    R_xx = dsp.covariance_matrix(data)
    # R_xx = dsp.forward_backward_avg(data)
    # R_xx = dsp.spatial_smoothing(data, 4, "forward-backward")
    
    results, theta_scan = dsp.doa_bartlett_ULA(R_xx, delta, 10000)
    max_val_arg = dsp.peaks_doa(results, num_signals)
    theta_result = theta_scan[max_val_arg]
    theta_result_sorted = theta_result[np.argsort(theta_result)]
    rmse_doa[i,0] = dsp.calculate_rmse(theta_result_sorted, theta_doa_sorted)
    
    results, theta_scan = dsp.doa_MVDR_ULA(R_xx, delta, 10000)
    max_val_arg = dsp.peaks_doa(results, num_signals)
    theta_result = theta_scan[max_val_arg]
    theta_result_sorted = theta_result[np.argsort(theta_result)]
    rmse_doa[i,1] = dsp.calculate_rmse(theta_result_sorted, theta_doa_sorted)
    
    results, theta_scan = dsp.doa_linpred_ULA(R_xx, delta, 0, 10000)
    max_val_arg = dsp.peaks_doa(results, num_signals)
    theta_result = theta_scan[max_val_arg]
    theta_result_sorted = theta_result[np.argsort(theta_result)]
    rmse_doa[i,2] = dsp.calculate_rmse(theta_result_sorted, theta_doa_sorted)
    
    results, theta_scan = dsp.doa_MUSIC_ULA(R_xx, delta, num_signals, 10000)
    max_val_arg = dsp.peaks_doa(results, num_signals)
    theta_result = theta_scan[max_val_arg]
    theta_result_sorted = theta_result[np.argsort(theta_result)]
    rmse_doa[i,3] = dsp.calculate_rmse(theta_result_sorted, theta_doa_sorted)
    
    results, theta_scan = dsp.doa_minnorm_ULA(R_xx, delta, num_signals, 10000)
    max_val_arg = dsp.peaks_doa(results, num_signals)
    theta_result = theta_scan[max_val_arg]
    theta_result_sorted = theta_result[np.argsort(theta_result)]
    rmse_doa[i,4] = dsp.calculate_rmse(theta_result_sorted, theta_doa_sorted)
    
    theta_result = dsp.doa_rootMUSIC_ULA(R_xx, delta, num_signals)
    theta_result_sorted = theta_result[np.argsort(theta_result)]
    rmse_doa[i,5] = dsp.calculate_rmse(theta_result_sorted, theta_doa_sorted)

    
plt.figure()
plt.plot(np.array(snr_step), rmse_doa[:,0], '--x', linewidth=2, markersize = 6, alpha=1, label="Bartlett")
plt.plot(np.array(snr_step), rmse_doa[:,1], '-.s', linewidth=2, markersize = 6, alpha=1, label="MVDR")
plt.plot(np.array(snr_step), rmse_doa[:,2], ':^', linewidth=2, markersize = 6, alpha=1, label="LinPred")

plt.legend(loc = "upper right", fontsize=15)
plt.yscale("log")
plt.xlabel("SNR [dB]", fontsize=18)
plt.ylabel("RMSE", fontsize=18)
plt.tight_layout()
plt.grid()    

plt.figure()
plt.plot(np.array(snr_step), rmse_doa[:,3], '--x', linewidth=2, markersize = 6, alpha=1, label="MUSIC")
plt.plot(np.array(snr_step), rmse_doa[:,4], '-.s', linewidth=2, markersize = 6, alpha=1, label="Min-norm")
plt.plot(np.array(snr_step), rmse_doa[:,5], ':^', linewidth=2, markersize = 6, alpha=1, label="Root-MUSIC")

plt.legend(loc = "upper right", fontsize=15)
plt.yscale("log")
plt.xlabel("SNR [dB]", fontsize=18)
plt.ylabel("RMSE", fontsize=18)
plt.tight_layout()
plt.grid()    