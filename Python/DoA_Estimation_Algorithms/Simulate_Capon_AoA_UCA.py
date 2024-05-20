# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:30:27 2024

@author: sedla
"""

import numpy as np
import matplotlib.pyplot as plt


## Set simulation parameters
sample_rate = 1e6
n = 1000 # Samples
freq = 400e6 # Frequency

radius = 0.5 # Array radius normalized to wavelength
n_elem = 5 # Number of array elements
theta_deg = 17 # Angle of incidence in degrees

phi_step = 2 * np.pi / n_elem # Angular distance between elements
elem_step = np.zeros((1,n_elem)) # Angular element position matrix [rad]
for i in range(5):
    elem_step[0,i] = phi_step * i

## Create the signal
t = np.arange(n) / sample_rate # Time vector
signal = np.exp(-2j * np.pi * freq * t)
signal = signal.reshape(1,-1)

## Simulate the array
theta = theta_deg * np.pi / 180
a = np.exp(-2j * np.pi * radius * np.cos(theta-elem_step)) # Calculate the array factor 
a = a.reshape(-1,1)
noise = np.random.randn(n_elem, n) + 1j*np.random.randn(n_elem, n) # Generate noise

## Get signals at array elements
r = a @ signal + 0.1*noise

## Estimate the spatial covariance matrix
R_xx = r @ r.conj().T / n
R_xx_inv = np.linalg.inv(R_xx)

## Simulate the conventional DoA
theta_scan = np.linspace(-1*np.pi, np.pi, 360)
results = []

for theta_i in theta_scan:
    a_i = np.exp(-2j * np.pi * radius * np.cos(theta_i-elem_step))
    a_i = a_i.reshape(-1,1)
    
    w = (R_xx_inv @ a_i) / (a_i.conj().T @ R_xx_inv @ a_i)

    r_weighted = np.abs(w.conj().T @ R_xx @ w)
    
    results.append(r_weighted.squeeze())
    
results_db = 10*np.log10(results)
print(theta_scan[np.argmax(results_db)] * 180 / np.pi)

## Plot the output in a cartesian plot
plt.figure(num=1)
plt.plot(theta_scan*180/np.pi, results_db)
plt.plot(theta_scan[np.argmax(results_db)] * 180 / np.pi, np.max(results_db),'rx')
plt.xlabel("Theta [Degrees]")
plt.ylabel("DOA Metric")
plt.xlim(-180,180)
plt.grid()

## Plot the output in a polar plot
plt.figure(num=2)
ax = plt.subplot(polar=True)
ax.plot(theta_scan, results_db)
ax.set_theta_zero_location('N')
ax.axes.set_theta_direction(-1) # Increase CW
ax.set_rlabel_position(55) # Shift grid labels