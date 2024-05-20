# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:30:27 2024

@author: sedla
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)
matplotlib.rc('font', weight = 'bold')

## Set simulation parameters
sample_rate = 1e6
n = 10000 # Samples
freq = 20e3 # Frequency

radius = 0.5 # Array element spacing
n_elem = 5 # Number of array elements
n_signals = 1 # Number of estimated signals
phi_deg = 15 # Elevation in degrees
theta_deg = 25 # Angle of incidence in degrees

phi_step = 2 * np.pi / n_elem # Angular distance between elements
elem_step = np.zeros((1,n_elem)) # Angular element position matrix [rad]
for i in range(5):
    elem_step[0,i] = phi_step * i

## Create the signal
t = np.arange(n) / sample_rate # Time vector
signal = np.exp(-2j * np.pi * freq * t) # Signal vector
signal = signal.reshape(1,-1)

## Simulate the array
phi = phi_deg * np.pi / 180
theta = theta_deg * np.pi / 180
a = np.exp(-2j * np.pi * radius * np.sin(phi) * np.cos(theta-elem_step)) # Calculate the array factor 
a = a.reshape(-1,1)
noise = np.random.randn(n_elem, n) + 1j*np.random.randn(n_elem, n) # Generate noise

## Get signals at array elements
r = a @ signal + 0.1*noise

## Estimate the spatial covariance matrix
R_xx = r @ r.conj().T / n

## Obtain eigenvectors and their eigenvalues
w,v = np.linalg.eig(R_xx)

# Sort eigenvectors to obtain the noise subsopace
eig_val_order = np.argsort(np.abs(w))
v_sorted = v[:, eig_val_order]

Vn = np.zeros((n_elem,n_elem-n_signals), dtype=np.complex64)
for i in range(n_elem-n_signals):
    Vn[:,i] = v_sorted[:,i] 

## Simulate MUSIC DoA estimation
theta_samp = 1000
phi_samp = 500
theta_scan = np.linspace(-np.pi, np.pi, theta_samp)
phi_scan = np.linspace(0, np.pi/2, phi_samp)
results = np.zeros((phi_samp,theta_samp))

for theta_i in range(theta_samp):
    for phi_i in range(phi_samp):
        a_i = np.exp(-2j * np.pi * radius * np.sin(phi_scan[phi_i]) * np.cos(theta_scan[theta_i]-elem_step))
        a_i = a_i.reshape(-1,1)
        
        p_theta = 1 / (a_i.conj().T @ Vn @ Vn.conj().T @ a_i)
        
        results[phi_i, theta_i] = 10*np.log10(np.abs(p_theta))

results /= np.max(results) # Normalize results

azi_max = np.argmax(results[int(phi_samp/2),:])
elev_max = np.argmax(results[:,azi_max])
print("Azimuth: ", theta_scan[azi_max] * 180 / np.pi)
print("Elevation: ", phi_scan[elev_max] * 180 / np.pi)

# theta_grid, phi_grid = np.meshgrid(theta_scan, phi_scan)
# x = results * np.sin(phi_grid) * np.cos(theta_grid)
# y = results * np.sin(phi_grid) * np.sin(theta_grid)
# z = results * np.cos(phi_grid)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(x, y, z, cmap='viridis')
# fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([0, 1])

# plt.show()

theta_deg_scan = theta_scan * 180 / np.pi
phi_deg_scan = phi_scan * 180 / np.pi
theta_grid, phi_grid = np.meshgrid(theta_deg_scan, phi_deg_scan)

azival = str(theta_scan[azi_max] * 180 / np.pi)
eleval = str(phi_scan[elev_max] * 180 / np.pi)
textlabel = "Az: " + azival[:6] + "\nEl: " + eleval[:6]
# Plotting the results
plt.figure()
plt.pcolormesh(theta_grid, phi_grid, results, shading='auto', cmap='jet')
plt.plot(theta_scan[azi_max] * 180 / np.pi, phi_scan[elev_max] * 180 / np.pi, 'bx', markersize = 6)
plt.text(theta_scan[azi_max] * 180 / np.pi + 8, phi_scan[elev_max] * 180 / np.pi - 7, textlabel, fontsize = 16, c='w')
cbar = plt.colorbar()
cbar.labelsize = 12
plt.xlabel('$\Theta$ [°]', fontsize=18, weight='bold')
plt.ylabel('$\psi$ [°]', fontsize=18, weight='bold')
plt.tight_layout()
plt.show()