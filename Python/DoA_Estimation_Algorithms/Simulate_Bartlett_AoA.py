# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:03:12 2024

@author: sedla
"""

import numpy as np
import matplotlib.pyplot as plt


## Set simulation parameters
sample_rate = 1e6
n = 1000 # Samples
freq = 20e3 # Frequency

delta = 0.5 # Array element spacing
n_elem = 3 # Number of array elements
theta_deg = 25 # Angle of incidence in degrees

## Create the signal
t = np.arange(n) / sample_rate # Time vector
signal = np.exp(-2j * np.pi * freq * t)
signal = signal.reshape(1,-1)

## Simulate the signal
theta = theta_deg * np.pi / 180
a = np.exp(-2j * np.pi * delta * np.arange(n_elem) * np.sin(theta)) # Calculate the array factor 
a = a.reshape(-1,1)
noise = np.random.randn(n_elem, n) + 1j*np.random.randn(n_elem, n) # Generate noise

## Get signals at array elements
r = a @ signal + 0.1*noise

## Estimate the spatial covariance matrix
R_xx = r @ r.conj().T / n

## Simulate the conventional DoA
theta_scan = np.linspace(-1*np.pi/2, np.pi/2, 1000)
results = []

for theta_i in theta_scan:
    a_i = np.exp(-2j * np.pi * delta * np.arange(n_elem) * np.sin(theta_i))
    a_i = a_i.reshape(-1,1)
    
    w = a_i
    # w_nr = np.sqrt(np.dot(a_i.conj().T,a_i))
    # w = a_i / np.sqrt(np.dot(a_i.conj().T,a_i)) # Normalize weights
    # w = a_i / np.linalg.norm(a_i) # Normalize weights alternatively (same outcome)

    r_weighted = abs(w.conj().T @ R_xx @ w)
    # r_weighted = (a_i.conj().T @ R_xx @ a_i) / (a_i.conj().T @ a_i) # Normalize
    
    results.append(r_weighted.squeeze())
    
results_db = 10*np.log10(results)

# results /= np.max(results) # Normalize results

print(theta_scan[np.argmax(results_db)] * 180 / np.pi)

## Plot the output in a cartesian plot
plt.figure(num=1)
plt.plot(theta_scan*180/np.pi, results_db)
plt.plot(theta_scan[np.argmax(results_db)] * 180 / np.pi, np.max(results_db),'rx')
plt.xlabel("Theta [Degrees]")
plt.ylabel("DOA Metric")
plt.xlim(-100,100)
plt.grid()

## Plot the output in a polar plot
plt.figure(num=2)
ax = plt.subplot(polar=True)
ax.plot(theta_scan, results)
ax.set_theta_zero_location('N')
ax.axes.set_theta_direction(-1) # Increase CW
ax.set_rlabel_position(55) # Shift grid labels



    
    