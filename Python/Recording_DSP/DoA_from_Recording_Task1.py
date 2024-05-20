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

# Input parameters
num_elem = 5
num_signals = 1
delta = 0.5
sample_size = 2**20

data = np.zeros((num_elem, sample_size), dtype=np.complex64)

# Load data from files
# Directory "Recordings\Task1\Angle0\..." for theta 0 degrees
# Directory "Recordings\Task1\Angle20\..." for theta 20 degrees
for i in range(num_elem):
    data[4,:] = np.fromfile(open("Recordings\Task1\Angle0\Ch_0"), dtype=np.complex64)[:sample_size]
    data[3,:] = np.fromfile(open("Recordings\Task1\Angle0\Ch_1"), dtype=np.complex64)[:sample_size]
    data[2,:] = np.fromfile(open("Recordings\Task1\Angle0\Ch_2"), dtype=np.complex64)[:sample_size]
    data[1,:] = np.fromfile(open("Recordings\Task1\Angle0\Ch_3"), dtype=np.complex64)[:sample_size]
    data[0,:] = np.fromfile(open("Recordings\Task1\Angle0\Ch_4"), dtype=np.complex64)[:sample_size]
    
spatial_samples = 1000
results = np.zeros((6,spatial_samples))

R_xx = data @ data.conj().T / sample_size

# num_signals = dsp.MDL_test(data)

results[0,:], theta_scan = dsp.doa_bartlett_ULA(R_xx, delta, spatial_samples)
doa_peaks = dsp.peaks_doa(results[0,:], num_signals)
print("Bartlett: ", np.rad2deg(theta_scan[doa_peaks]))
results[1,:], _ = dsp.doa_MVDR_ULA(R_xx, delta, spatial_samples)
doa_peaks = dsp.peaks_doa(results[1,:], num_signals)
print("MVDR: ", np.rad2deg(theta_scan[doa_peaks]))
results[2,:], _ = dsp.doa_linpred_ULA(R_xx, delta, 0, spatial_samples)
doa_peaks = dsp.peaks_doa(results[2,:], num_signals)
print("Lin. pred.: ", np.rad2deg(theta_scan[doa_peaks]))

plt.figure()
plt.plot(theta_scan*180/np.pi, results[0,:], linewidth=2, label="Bartlett")
plt.plot(theta_scan*180/np.pi, results[1,:], linewidth=2, label="MVDR")
plt.plot(theta_scan*180/np.pi, results[2,:], linewidth=2, label="Lin. pred.")
plt.xlabel("$\Theta$ [°]", fontsize=18)
plt.ylabel("DOA Metric", fontsize=18)
plt.legend(loc = "upper left", fontsize=15)
plt.xlim(-90, 90)
plt.tight_layout()
plt.grid()
plt.draw()

results[3, :], _ = dsp.doa_MUSIC_ULA(R_xx, delta, num_signals, 1000)
doa_peaks = dsp.peaks_doa(results[3,:], num_signals)
print("MUSIC: ", np.rad2deg(theta_scan[doa_peaks]))
results[4, :], _ = dsp.doa_minnorm_ULA(R_xx, delta, num_signals, 1000)
doa_peaks = dsp.peaks_doa(results[4,:], num_signals)
print("Min-norm: ", np.rad2deg(theta_scan[doa_peaks]))
theta_doa = dsp.doa_rootMUSIC_ULA(R_xx, delta, num_signals)
print("Root-MUSIC: ", np.rad2deg(theta_doa))

plt.figure()
plt.plot(theta_scan*180/np.pi, results[3,:], linewidth=2, label="MUSIC")
plt.plot(theta_scan*180/np.pi, results[4,:], linewidth=2, label="Min-norm")
plt.plot(np.rad2deg(theta_doa), np.ones((len(theta_doa))), 'rx', markersize=10, label="Root-MUSIC")
plt.xlabel("$\Theta$ [°]", fontsize=18)
plt.ylabel("DOA Metric", fontsize=18)
plt.legend(loc = "upper left", fontsize=15)
plt.xlim(-90, 90)
plt.tight_layout()
plt.grid()
plt.draw()
