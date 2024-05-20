# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 20:00:03 2024

@author: sedla
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import doaLib as dsp

matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)
matplotlib.rc('font', weight = 'bold')
matplotlib.rcParams['xtick.major.pad']='7'
matplotlib.rcParams['ytick.major.pad']='3'

## Input parameters
num_elem = 5
# n_elem_sub = 3 # Number of elements in the subarray (For spatial smoothing)
# smoothing = "fb" # "fb" - Forward-Backward, "f" - Forward only, "b" - Backward only (For spatial smoothing)
num_signals = 2
delta = 0.5
sample_size = 2**20

theta_deg = np.array(([5, -20],[-5, -20], [-10, -20], [-15, -20]))
theta_delta = [25, 15, 10, 5]

doa_rmse = np.zeros((6,4))
for i in range(4):
    data = np.zeros((num_elem, sample_size), dtype=np.complex64)
    # Load data from files
    data[0,:] = np.fromfile(open(f"Recordings\Task2\Step{i+1}\Ch_0"), dtype=np.complex64)[:sample_size]
    data[1,:] = np.fromfile(open(f"Recordings\Task2\Step{i+1}\Ch_1"), dtype=np.complex64)[:sample_size]
    data[2,:] = np.fromfile(open(f"Recordings\Task2\Step{i+1}\Ch_2"), dtype=np.complex64)[:sample_size]
    data[3,:] = np.fromfile(open(f"Recordings\Task2\Step{i+1}\Ch_3"), dtype=np.complex64)[:sample_size]
    data[4,:] = np.fromfile(open(f"Recordings\Task2\Step{i+1}\Ch_4"), dtype=np.complex64)[:sample_size]
    
    spatial_samples = 1000
    results = np.zeros((6,spatial_samples))
    doa_peaks = np.zeros((6,num_signals))

    R_xx = data @ data.conj().T / sample_size
    # R_xx = dsp.forward_backward_avg(data)
    # R_xx = dsp.spatial_smoothing(data, 4)

    results[0,:], theta_scan = dsp.doa_bartlett_ULA(R_xx, delta, spatial_samples)
    doa_peaks = theta_scan[dsp.peaks_doa(results[0,:], num_signals)]
    doa_rmse[0,i] = dsp.calculate_rmse(doa_peaks, np.deg2rad(theta_deg[i,:]))
    
    results[1,:], _ = dsp.doa_MVDR_ULA(R_xx, delta, spatial_samples)
    doa_peaks = theta_scan[dsp.peaks_doa(results[1,:], num_signals)]
    doa_rmse[1,i] = dsp.calculate_rmse(doa_peaks, np.deg2rad(theta_deg[i,:]))

    results[2,:], _ = dsp.doa_linpred_ULA(R_xx, delta, 0, spatial_samples)
    doa_peaks = theta_scan[dsp.peaks_doa(results[2,:], num_signals)]
    doa_rmse[2,i] = dsp.calculate_rmse(doa_peaks, np.deg2rad(theta_deg[i,:]))

    results[3, :], _ = dsp.doa_MUSIC_ULA(R_xx, delta, num_signals, 1000)
    doa_peaks = theta_scan[dsp.peaks_doa(results[3,:], num_signals)]
    doa_rmse[3,i] = dsp.calculate_rmse(doa_peaks, np.deg2rad(theta_deg[i,:]))
    
    results[4, :], _ = dsp.doa_minnorm_ULA(R_xx, delta, num_signals, 1000)
    doa_peaks = theta_scan[dsp.peaks_doa(results[4,:], num_signals)]
    doa_rmse[4,i] = dsp.calculate_rmse(doa_peaks, np.deg2rad(theta_deg[i,:]))
    
    theta_doa = dsp.doa_rootMUSIC_ULA(R_xx, delta, num_signals)
    theta_doa = theta_doa[np.argsort(-theta_doa)]
    doa_rmse[5,i] = dsp.calculate_rmse(doa_peaks, np.deg2rad(theta_deg[i,:]))
    
plt.figure()
plt.plot(theta_delta, doa_rmse[0,:], '--x', linewidth=2, label="Bartlett")
plt.plot(theta_delta, doa_rmse[1,:], '-.s', linewidth=2, label="MVDR")
plt.plot(theta_delta, doa_rmse[2,:], ':^', linewidth=2, label="Lin. pred.")

plt.xlabel("$\Delta\Theta$ [°]", fontsize=18)
plt.ylabel("RMSE", fontsize=18)
plt.legend(loc = "lower left", fontsize=15)
plt.yscale('log')
plt.tight_layout()
plt.grid()
plt.draw()

plt.figure()
plt.plot(theta_delta, doa_rmse[3,:], '--x', linewidth=2, label="MUSIC")
plt.plot(theta_delta, doa_rmse[4,:], '-.s', linewidth=2, label="Min-norm")
plt.plot(theta_delta, doa_rmse[5,:], ':^', linewidth=2, label="Root-MUSIC")

plt.xlabel("$\Delta\Theta$ [°]", fontsize=18)
plt.ylabel("RMSE", fontsize=18)
plt.legend(loc = "lower left", fontsize=15)
plt.yscale('log')
plt.tight_layout()
plt.grid()
plt.draw()
