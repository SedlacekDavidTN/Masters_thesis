# -*- coding: utf-8 -*-
"""
Created on Mon May 20 08:40:51 2024

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

num_signals = 1
freq = [20e3]
theta_deg = [15]
snr_dB = 20

num_elem = 5
delta = 0.5

sample_rate = 1e6
num_samples = 100000


data = dsp.generate_data_ULA(freq, theta_deg, num_elem, delta, sample_rate, num_samples)
data = dsp.add_noise(data, snr_dB)
    
R_xx = dsp.covariance_matrix(data)
# R_xx = dsp.forward_backward_avg(data)
# R_xx = dsp.spatial_smoothing(data, 4, "forward-backward")

# Plot spatial spectra of Beamforming methods
plt.figure()
results, theta_scan = dsp.doa_bartlett_ULA(R_xx, delta, 1000)
dsp.plot_doa(results, theta_scan, num_signals, "cartesian")

results, theta_scan = dsp.doa_MVDR_ULA(R_xx, delta, 1000)
dsp.plot_doa(results, theta_scan, num_signals, "cartesian")

results, theta_scan = dsp.doa_linpred_ULA(R_xx, delta, 0, 1000)
dsp.plot_doa(results, theta_scan, num_signals, "cartesian")

plt.legend(["Bartlett", "MVDR", "Lin. pred."], loc = "upper left", fontsize=15)

# Plot spatial spectra of Subspace-based methods
plt.figure()
results, theta_scan = dsp.doa_MUSIC_ULA(R_xx, delta, num_signals, 1000)
dsp.plot_doa(results, theta_scan, num_signals, "cartesian")

results, theta_scan = dsp.doa_minnorm_ULA(R_xx, delta, num_signals, 1000)
dsp.plot_doa(results, theta_scan, num_signals, "cartesian")

theta_doa = dsp.doa_rootMUSIC_ULA(R_xx, delta, num_signals)
print(np.rad2deg(theta_doa))
plt.plot(np.rad2deg(theta_doa), np.ones((len(theta_doa))),'rx', markersize=10)
# for i in range(len(theta_doa)):
#     plt.annotate(text=str(np.rad2deg(theta_doa[i]))[:5], xy=(np.rad2deg(theta_doa[i]) + 0.2, 1.03))
plt.xlabel("$\Theta$ [°]", fontsize=18)
plt.ylabel("DOA Metric", fontsize=18)
plt.tight_layout()
plt.ylim([-0.1, 1.1])
plt.grid()

plt.legend(["MUSIC", "Min-norm", "Root-MUSIC"], loc = "upper left", fontsize=15)