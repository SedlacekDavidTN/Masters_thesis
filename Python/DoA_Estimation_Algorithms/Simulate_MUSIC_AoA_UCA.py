# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:30:27 2024

@author: sedla
"""

import numpy as np
import matplotlib.pyplot as plt

## Set simulation parameters
sample_rate = 1e6
n = 10000 # Samples
freq = 20e3 # Frequency

radius = 0.5
n_elem = 5 # Number of array elements
n_signals = 1 # Number of estimated signals
theta_deg = 180 # Angle of incidence in degrees

phi_step = 2 * np.pi / n_elem # Angular distance between elements
elem_step = np.zeros((1,n_elem)) # Angular element position matrix [rad]
for i in range(5):
    elem_step[0,i] = phi_step * i

## Create the signal
t = np.arange(n) / sample_rate # Time vector
signal = np.exp(-2j * np.pi * freq * t) # Signal vector
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

## Obtain eigenvectors and their eigenvalues
w,v = np.linalg.eig(R_xx)

# Sort eigenvectors to obtain the noise subsopace
eig_val_order = np.argsort(np.abs(w))
v_sorted = v[:, eig_val_order]

Vn = np.zeros((n_elem,n_elem-n_signals), dtype=np.complex64)
for i in range(n_elem-n_signals):
    Vn[:,i] = v_sorted[:,i] 

## Simulate MUSIC DoA estimation
theta_scan = np.linspace(-1*np.pi, np.pi, 1000)
results = []

for theta_i in theta_scan:
    a_i = np.exp(-2j * np.pi * radius * np.cos(theta_i-elem_step))
    a_i = a_i.reshape(-1,1)
    
    # p_theta = (a_i.conj().T @ a_i) / (a_i.conj().T @ Vn @ Vn.conj().T @ a_i)
    p_theta = 1 / (a_i.conj().T @ Vn @ Vn.conj().T @ a_i)
    
    # results.append(p_theta.squeeze())
    results.append(10*np.log10(np.abs(p_theta.squeeze())))
    
results /= np.max(results) # Normalize results

print(theta_scan[np.argmax(results)] * 180 / np.pi)

## Plot the output in a cartesian plot
plt.figure(num=1)
plt.plot(theta_scan*180/np.pi, results)
plt.plot(theta_scan[np.argmax(results)] * 180 / np.pi, np.max(results),'rx')
plt.xlabel("Theta [Degrees]")
plt.ylabel("DOA Metric")
plt.xlim(-180,180)
plt.grid()

## Plot the output in a polar plot
plt.figure(num=2)
ax = plt.subplot(polar=True)
ax.plot(theta_scan, results)
ax.set_theta_zero_location('N')
ax.axes.set_theta_direction(-1) # Increase CW
ax.set_rlabel_position(55) # Shift grid labels