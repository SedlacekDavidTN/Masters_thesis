# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:03:12 2024

@author: sedla
"""

import numpy as np
import matplotlib.pyplot as plt
import DSP_Lib as dsp
import matplotlib

matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)
matplotlib.rc('font', weight = 'bold')
matplotlib.rcParams['xtick.major.pad']='7'
matplotlib.rcParams['ytick.major.pad']='3'

## Set simulation parameters
sample_rate = 1e6
n = 10000 # Samples
freq = 20e3 # Frequency
n_signals = 1

# radius = 0.5 # Array radius normalized to wavelength
n_elem = 5 # Number of array elements
phi_deg = 15 # Elevation in degrees
theta_deg = 25 # Azimuth in degrees

radius = (0.5 / np.sqrt(2 * (1 - np.cos(2 * np.pi / n_elem)))) # Calculate radius [m]

phi_step = 2 * np.pi / n_elem # Angular distance between elements
elem_step = np.zeros((1,n_elem)) # Angular element position matrix [rad]
for i in range(5):
    elem_step[0,i] = phi_step * i

## Create the signal
t = np.arange(n) / sample_rate # Time vector
signal = np.exp(-2j * np.pi * freq * t)
signal = signal.reshape(1,-1)

## Simulate the signal on UCA
phi = phi_deg * np.pi / 180
theta = theta_deg * np.pi / 180
pred_doa = np.array([theta, phi])

a = np.exp(-2j * np.pi * radius * np.sin(phi) * np.cos(theta-elem_step)) # Calculate the array factor 
a = a.reshape(-1,1)

snr_step = np.linspace(-25,25,25)
rmse_doa = np.zeros((len(snr_step),3))
for j in range(len(snr_step)):

    data = a @ signal
    data = dsp.add_noise(data, snr_step[j])
    
    ## Estimate the spatial covariance matrix
    R_xx = data @ data.conj().T / n
    
    ## Simulate the conventional DoA
    theta_samp = 360
    phi_samp = 90
    theta_scan = np.linspace(-1*np.pi, np.pi, theta_samp)
    phi_scan = np.linspace(0, np.pi/2, phi_samp)
    results = np.zeros((phi_samp,theta_samp))
    
    ## Bartlett
    for theta_i in range(theta_samp):
        for phi_i in range(phi_samp):
            a_i = np.exp(-2j * np.pi * radius * np.sin(phi_scan[phi_i]) * np.cos(theta_scan[theta_i]-elem_step)).reshape(-1,1)
            
            w = a_i / np.sqrt(np.dot(a_i.conj().T,a_i))
            r_weighted = np.abs(w.conj().T @ R_xx @ w)
            
            results[phi_i, theta_i] = r_weighted
    
    results /= np.max(results)
    results_db = 10*np.log10(results)
    azi_max = np.argmax(results[int(phi_samp/2),:])
    elev_max = np.argmax(results[:,azi_max])
    
    angles_doa = np.array([theta_scan[azi_max], phi_scan[elev_max]])
    rmse_doa[j,0] = dsp.calculate_rmse(angles_doa, pred_doa)
    
    ## MVDR
    R_xx_inv = np.linalg.inv(R_xx)
    
    for theta_i in range(theta_samp):
        for phi_i in range(phi_samp):
            a_i = np.exp(-2j * np.pi * radius * np.sin(phi_scan[phi_i]) * np.cos(theta_scan[theta_i]-elem_step))
            a_i = a_i.reshape(-1,1)
            
            w = (R_xx_inv @ a_i) / (a_i.conj().T @ R_xx_inv @ a_i)
        
            r_weighted = np.abs(w.conj().T @ R_xx @ w)
            
            results[phi_i, theta_i] = r_weighted
        
    results_db = 10*np.log10(results)
    results_db -= np.max(results_db)
    azi_max = np.argmax(results_db[int(phi_samp/2),:])
    elev_max = np.argmax(results_db[:,azi_max])
    
    angles_doa = np.array([theta_scan[azi_max], phi_scan[elev_max]])
    rmse_doa[j,1] = dsp.calculate_rmse(angles_doa, pred_doa)
    
    ## MUSIC
    w,v = np.linalg.eig(R_xx)
    eig_val_order = np.argsort(np.abs(w))
    v_sorted = v[:, eig_val_order]

    Vn = np.zeros((n_elem,n_elem-n_signals), dtype=np.complex64)
    for i in range(n_elem-n_signals):
        Vn[:,i] = v_sorted[:,i] 

    for theta_i in range(theta_samp):
        for phi_i in range(phi_samp):
            a_i = np.exp(-2j * np.pi * radius * np.sin(phi_scan[phi_i]) * np.cos(theta_scan[theta_i]-elem_step))
            a_i = a_i.reshape(-1,1)
            
            p_theta = 1 / (a_i.conj().T @ Vn @ Vn.conj().T @ a_i)
            
            results[phi_i, theta_i] = 10*np.log10(np.abs(p_theta))

    results /= np.max(results)

    azi_max = np.argmax(results[int(phi_samp/2),:])
    elev_max = np.argmax(results[:,azi_max])
    
    angles_doa = np.array([theta_scan[azi_max], phi_scan[elev_max]])
    rmse_doa[j,2] = dsp.calculate_rmse(angles_doa, pred_doa)  

plt.figure()
plt.plot(np.array(snr_step), rmse_doa[:,0], '--x', linewidth=2, markersize = 6, alpha=1, label="Bartlett")
plt.plot(np.array(snr_step), rmse_doa[:,1], '-.s', linewidth=2, markersize = 6, alpha=1, label="MVDR")
plt.plot(np.array(snr_step), rmse_doa[:,2], ':^', linewidth=2, markersize = 6, alpha=1, label="MUSIC")

plt.legend(loc = "upper right", fontsize=15)
plt.yscale("log")
plt.xlabel("SNR [dB]", fontsize=18)
plt.ylabel("RMSE", fontsize=18)
plt.tight_layout()
plt.grid()   