# -*- coding: utf-8 -*-
"""
Created on Mon May 20 08:53:04 2024

@author: sedla
"""

import DSP_Lib as dsp
import numpy as np

sample_rate = 1e6
num_samples = 100000
num_signals = 2
freq = [20e3, 20e3]
theta_deg = [10, 25]

num_elem = 5
delta = 0.5

snr_dB = 20

rmse_doa = np.zeros((1,6))
theta_doa = np.deg2rad(theta_deg)
theta_doa_sorted = theta_doa[np.argsort(theta_doa)]

ref_data = dsp.generate_data_ULA(freq, theta_deg, num_elem, delta, sample_rate, num_samples)
data = dsp.add_noise(ref_data, snr_dB)
    
R_xx = dsp.covariance_matrix(data)
# R_xx = dsp.forward_backward_avg(data)
# R_xx = dsp.spatial_smoothing(data, 4, "forward-backward")
    
results, theta_scan = dsp.doa_bartlett_ULA(R_xx, delta, 1000)
max_val_arg = dsp.peaks_doa(results, num_signals)
theta_result = theta_scan[max_val_arg]
theta_result_sorted = theta_result[np.argsort(theta_result)]
rmse_doa[0, 0] = dsp.calculate_rmse(theta_result_sorted, theta_doa_sorted)
    
results, theta_scan = dsp.doa_MVDR_ULA(R_xx, delta, 1000)
max_val_arg = dsp.peaks_doa(results, num_signals)
theta_result = theta_scan[max_val_arg]
theta_result_sorted = theta_result[np.argsort(theta_result)]
rmse_doa[0, 1] = dsp.calculate_rmse(theta_result_sorted, theta_doa_sorted)

results, theta_scan = dsp.doa_linpred_ULA(R_xx, delta, 0, 1000)
max_val_arg = dsp.peaks_doa(results, num_signals)
theta_result = theta_scan[max_val_arg]
theta_result_sorted = theta_result[np.argsort(theta_result)]
rmse_doa[0, 2] = dsp.calculate_rmse(theta_result_sorted, theta_doa_sorted)

results, theta_scan = dsp.doa_MUSIC_ULA(R_xx, delta, num_signals, 1000)
max_val_arg = dsp.peaks_doa(results, num_signals)
theta_result = theta_scan[max_val_arg]
theta_result_sorted = theta_result[np.argsort(theta_result)]
rmse_doa[0, 3] = dsp.calculate_rmse(theta_result_sorted, theta_doa_sorted)

results, theta_scan = dsp.doa_minnorm_ULA(R_xx, delta, num_signals, 1000)
max_val_arg = dsp.peaks_doa(results, num_signals)
theta_result = theta_scan[max_val_arg]
theta_result_sorted = theta_result[np.argsort(theta_result)]
rmse_doa[0, 4] = dsp.calculate_rmse(theta_result_sorted, theta_doa_sorted)

theta_result = dsp.doa_rootMUSIC_ULA(R_xx, delta, num_signals)
theta_result_sorted = theta_result[np.argsort(theta_result)]
rmse_doa[0, 5] = dsp.calculate_rmse(theta_result_sorted, theta_doa_sorted)

print("Bartlett: ", rmse_doa[0,0], "RMSE")
print("MVDR: ", rmse_doa[0,1], "RMSE")
print("Lin. Pred.: ", rmse_doa[0,2], "RMSE")
print("MUSIC: ", rmse_doa[0,3], "RMSE")
print("Min-norm: ", rmse_doa[0,4], "RMSE")
print("Root-MUSIC: ", rmse_doa[0,5], "RMSE")