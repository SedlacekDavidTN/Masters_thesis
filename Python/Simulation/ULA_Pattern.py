# -*- coding: utf-8 -*-
"""
Created on Fri May 10 20:32:15 2024

@author: sedla
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)
matplotlib.rc('font', weight = 'bold')
matplotlib.rcParams['xtick.major.pad']='7'
matplotlib.rcParams['ytick.major.pad']='3'

## Array pattern
def plot_antenna_pattern(delta, theta_deg, num_elem=3, num_samples=1024):
    theta_rad = np.deg2rad(theta_deg)
    
    elem_pos = np.arange(num_elem) * delta
    theta_samp = np.linspace(-np.pi/2, np.pi/2, num_samples)
    
    spatial_taper = np.ones(num_elem)
    # spatial_taper = np.hamming(num_elem)
    # spatial_taper = np.hanning(num_elem)
    # spatial_taper = np.bartlett(num_elem)
    # spatial_taper = np.kaiser(num_elem, 2)
    
    AF = np.zeros(num_samples, dtype=complex)
    for i in range(len(elem_pos)):
        phase_shift = -2j * np.pi * elem_pos[i] * np.sin(theta_rad)
        AF += spatial_taper[i] * np.exp(2j * np.pi * elem_pos[i] * np.sin(theta_samp) + phase_shift)
    
    AF = np.abs(AF)**2
    AF_dB = 10 * np.log10(AF)
    AF_dB -= np.max(AF_dB)
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta_samp, AF_dB, linewidth=2)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(55)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_ylim([-30, 1])
    ax.set_yticks(ax.get_yticks()[::2])
    ax.set_xlabel('dB', fontsize=18, weight='bold')
    ax.xaxis.set_label_coords(0.6, 0.233)
    # ax.xaxis.set_label_coords(0.27, 0.17)
    plt.tight_layout(pad = -2)
    
def plot_window_function(num_elem, func_type=""):
    sample_count = 1000
    
    elem_points = np.arange(num_elem)
    elem_range = np.linspace(0, num_elem-1, sample_count)
    
    if func_type == "Hamming":
        w_points = np.hamming(num_elem)
        w_func = np.hamming(sample_count)
    elif func_type == "Bartlett":
        w_points = np.bartlett(num_elem)
        w_func = np.bartlett(sample_count)
    elif func_type == "Kaiser":
        w_points = np.kaiser(num_elem, 2)
        w_func = np.kaiser(sample_count, 2)
    else:
        w_points = np.ones(num_elem)
        w_func = np.ones(sample_count)
        w_func[0] = 0
        w_func[-1] = 0
    
    plt.figure()
    plt.plot(elem_range, w_func, '--', linewidth=2)
    plt.plot(elem_points, w_points, 'rx', markersize=12)
    plt.xticks(np.arange(0, num_elem, 1.0))
    plt.xlabel("Element", fontsize=18)
    plt.ylabel("Applied weight", fontsize=18)
    plt.xlim([-0.5, num_elem-0.5])
    plt.ylim([-0.1, 1.1])
    plt.tight_layout()
    plt.grid()
    plt.show()

elem_values = [8]
delta_values = [0.5]
theta_values = [0]

for num_elem in elem_values:
    for delta in delta_values:
        for theta_deg in theta_values:
            plot_antenna_pattern(delta, theta_deg, num_elem)
            # plot_window_function(num_elem, func_type="")
        
## Antenna pattern via Bartlett beamformer
## To make this work, disable the normalization in the Bartlett beamformer function
# data = dsp.generate_data_ULA([20e3], [20], 3, 0.5, 1e6, 1000)
# R_xx = dsp.covariance_matrix(data)

# results, theta = dsp.doa_bartlett_ULA(R_xx, 0.5, 1000)
# results -= np.max(results) 

# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# ax.plot(theta, results)
# ax.set_title("Normalized Array Pattern for d = 0.5")
# ax.set_theta_zero_location('N')
# ax.set_theta_direction(-1)
# ax.set_rlabel_position(55)
# ax.set_ylim([-30, 1])
# ax.set_thetamin(-90)
# ax.set_thetamax(90)
# plt.show()
