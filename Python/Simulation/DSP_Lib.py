"""
Created on Wed May  8 21:15:00 2024

@author: sedla
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def generate_data_ULA(freq, theta_deg, num_elem, delta, sample_rate, num_samples):
    freq = np.array(freq)
    theta_deg = np.array(theta_deg)
    num_signals = len(freq)
    
    time = np.arange(num_samples) / sample_rate
    data = np.zeros((num_elem, num_samples), dtype = np.complex128)
    
    theta_rad = np.deg2rad(theta_deg)
    for i in range(num_signals):
        s = np.exp(-2j * np.pi * freq[i] * time).reshape(1,-1)
        a = generate_steering_vectors_ULA(num_elem, delta, theta_rad[i])
        data += a @ s
        
    return data

def generate_steering_vectors_ULA(num_elem, delta, theta_rad):
    a = np.exp(-2j * np.pi * delta * np.arange(num_elem) * np.sin(theta_rad)).reshape(-1,1)
    
    return a

def add_noise(data, snr_db):
    sig_pwr = np.abs(data[0,:]) ** 2
    
    epsilon = 1e-10
    sig_pwr = np.where(sig_pwr == 0, epsilon, sig_pwr)
    
    sig_avg_db = np.mean(10 * np.log10(sig_pwr))
    noise_avg_db = sig_avg_db - snr_db
    noise_avg = 10 ** (noise_avg_db / 10)
    
    ## Generate the white noise
    noise = np.random.normal(0, np.sqrt(noise_avg / 2), data.shape) + 1j * np.random.normal(0, np.sqrt(noise_avg / 2), data.shape)
    noisy_data = data + noise
    
    return noisy_data

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def covariance_matrix(data):
    R_xx = data @ data.conj().T / np.size(data,1)
    
    return R_xx

def forward_backward_avg(data):
    R_xx = data @ data.conj().T / np.size(data,1)
    Pi_x = np.fliplr(np.eye(np.size(R_xx,0)))
    R_xx = (R_xx + (Pi_x @ R_xx.conj() @ Pi_x)) / (2*np.size(data,1))
    
    return R_xx

def spatial_smoothing(data, num_elem_sub, smoothing_type="forward"):
    num_elem = np.size(data,0)
    
    num_sub_arr = num_elem - num_elem_sub + 1
    R_xx = np.zeros((num_elem_sub,num_elem_sub), dtype=np.complex128)

    if (smoothing_type == "forward" or smoothing_type == "forward-backward"):
        for i in range(num_sub_arr):
            R_ss = np.zeros((num_elem_sub,num_elem_sub), dtype=np.complex128)
            R_ss = data[i:i+num_elem_sub,:] @ data[i:i+num_elem_sub,:].conj().T / np.size(data,1)
            R_xx += R_ss
            
    if (smoothing_type == "backward" or smoothing_type == "forward-backward"):
        for i in range(num_sub_arr):
            R_ss = np.zeros((num_elem_sub,num_elem_sub), dtype=np.complex128)
            R_ss = data[(num_elem-num_elem_sub-i):(num_elem-i),:] @ data[(num_elem-num_elem_sub-i):(num_elem-i),:].conj().T / np.size(data,1)
            R_xx += R_ss
            
    if smoothing_type == "forward-backward":
        R_xx = R_xx / (2*num_sub_arr)
    else:
        R_xx = R_xx / (num_sub_arr)
        
    return R_xx

def MDL_test(data):
    num_samples = data.shape[1]
    num_elem = data.shape[0]
    
    R_xx = covariance_matrix(data)
    w,_ = np.linalg.eig(R_xx)

    eig_val_order = np.argsort(-w)
    w = np.abs(w[eig_val_order])
    
    mdl_metric = np.zeros((num_elem,1))
    
    for d in range(num_elem):
        Ld = num_samples * (num_elem - d) * np.log(np.mean(w[d:num_elem]) / np.prod(w[d:num_elem] ** (1 / (num_elem - d))))
        
        mdl_metric[d,0] = Ld + d*(2*num_elem-d+1)*np.log(num_samples)/4
        
    num_signals = np.argmin(mdl_metric)
    print("Estimated number of signals is: ", num_signals)
    
    return num_signals

def estimate_SNR(R_xx):
    w,_ = np.linalg.eig(R_xx)
    w.sort()
    noise_power = np.abs(w[0])
    signal_plus_noise_power = np.abs(w[-1])
    power_ratio = (signal_plus_noise_power - noise_power) / noise_power
    snr = 10.0 * np.log10(power_ratio)
    print("Estimated SNR is: ", snr, " dB")
    
    return snr

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
def doa_bartlett_ULA(R_xx, delta, resolution):
    num_elem = R_xx.shape[0]
    
    theta_scan = np.linspace(-1*np.pi/2, np.pi/2, resolution)
    results = np.zeros((resolution,1))

    for i in range(resolution):
        a_i = generate_steering_vectors_ULA(num_elem, delta, theta_scan[i])
        w = a_i / np.sqrt(np.dot(a_i.conj().T,a_i))
        # w = a_i 

        p_theta = np.abs(w.conj().T @ R_xx @ w)
        results[i,0] = p_theta
        
    results = results.squeeze()
    results_db = 10*np.log10(results)
    results_db -= np.min(results_db)
    results_db /= np.max(results_db)
    
    return results_db, theta_scan

def doa_MVDR_ULA(R_xx, delta, resolution):
    R_xx_inv = np.linalg.inv(R_xx)
    num_elem = R_xx.shape[0]

    theta_scan = np.linspace(-1*np.pi/2, np.pi/2, resolution)
    results = np.zeros((resolution,1))

    for i in range(resolution):
        a_i = generate_steering_vectors_ULA(num_elem, delta, theta_scan[i])
        w = (R_xx_inv @ a_i) / (a_i.conj().T @ R_xx_inv @ a_i)

        p_theta = np.abs(w.conj().T @ R_xx @ w)
        results[i,0] = p_theta
        
    results = results.squeeze()
    results_db = 10*np.log10(results)
    results_db -= np.min(results_db)
    results_db /= np.max(results_db)
    
    return results_db, theta_scan

def doa_linpred_ULA(R_xx, delta, unity_pos, resolution):
    R_inv = np.linalg.inv(R_xx)
    num_elem = R_xx.shape[0]

    u = np.zeros((num_elem,1))
    u[unity_pos,0] = 1

    theta_scan = np.linspace(-1*np.pi/2, np.pi/2, resolution)
    results = np.zeros((resolution,1))

    for i in range(resolution):
        a_i = generate_steering_vectors_ULA(num_elem, delta, theta_scan[i])
        
        p_theta = np.abs((u.conj().T @ R_inv @ u) / np.power(np.abs(u.conj().T @ R_inv @ a_i), 2))
        
        results[i,0] = p_theta
        
    results = results.squeeze()
    results_db = 10*np.log10(results)
    results_db -= np.min(results_db)
    results_db /= np.max(results_db)
    
    return results_db, theta_scan

def doa_MUSIC_ULA(R_xx, delta, num_signals, resolution):
    num_elem = R_xx.shape[0]

    w,v = np.linalg.eig(R_xx)
    eig_val_order = np.argsort(np.abs(w))
    v_sorted = v[:, eig_val_order]

    Vn = np.zeros((num_elem,num_elem-num_signals), dtype=np.complex64)
    for i in range(num_elem-num_signals):
        Vn[:,i] = v_sorted[:,i] 

    theta_scan = np.linspace(-1*np.pi/2, np.pi/2, resolution)
    results = np.zeros((resolution,1))

    for i in range(resolution):
        a_i = generate_steering_vectors_ULA(num_elem, delta, theta_scan[i])
        
        p_theta = np.abs((a_i.conj().T @ a_i) / (a_i.conj().T @ Vn @ Vn.conj().T @ a_i))
        
        results[i,0] = p_theta
    
    results = results.squeeze()
    results_db = 10*np.log10(results)
    results_db /= np.max(results_db)

    return results_db, theta_scan

def doa_minnorm_ULA(R_xx, delta, num_signals, resolution):
    num_elem = R_xx.shape[0]

    w,v = np.linalg.eig(R_xx)
    eig_val_order = np.argsort(np.abs(w))
    v_sorted = v[:, eig_val_order]

    Vn = np.zeros((num_elem,num_elem-num_signals), dtype=np.complex64)
    for i in range(num_elem-num_signals):
        Vn[:,i] = v_sorted[:,i] 
        
    u = np.zeros((num_elem,1))
    u[0,0] = 1
    w = u @ u.T

    theta_scan = np.linspace(-1*np.pi/2, np.pi/2, resolution)
    results = np.zeros((resolution,1))

    for i in range(resolution):
        a_i = generate_steering_vectors_ULA(num_elem, delta, theta_scan[i])
        
        p_theta = np.abs(1 / (a_i.conj().T @ Vn @ Vn.conj().T @ w @ Vn @ Vn.conj().T @ a_i))
        
        results[i,0] = p_theta
    
    results = results.squeeze()
    results_db = 10*np.log10(results)
    results_db /= np.max(results_db)

    return results_db, theta_scan

def doa_rootMUSIC_ULA(R_xx, delta, num_signals):
    num_elem = R_xx.shape[0]

    w,v = np.linalg.eig(R_xx)
    eig_val_order = np.argsort(np.abs(w))
    v_sorted = v[:, eig_val_order]

    Vn = np.zeros((num_elem,num_elem-num_signals), dtype=np.complex64)
    for i in range(num_elem-num_signals):
        Vn[:,i] = v_sorted[:,i]
        
    ct = Vn @ Vn.conj().T
    p_coeff = np.zeros(2 * num_elem - 1, dtype=np.complex64)

    for i in range(-num_elem + 1, num_elem):
        p_coeff[i + (num_elem - 1)] = np.trace(ct, i)
        
    all_roots = np.roots(p_coeff)

    candidate_roots_abs = np.abs(all_roots)
    sorted_idx = candidate_roots_abs.argsort()[(num_elem - 1 - num_signals) : (num_elem - 1)]

    valid_roots = all_roots[sorted_idx]
    args = np.angle(valid_roots)

    theta_doa = np.arcsin(args / (delta * 2.0 * np.pi))
    
    return theta_doa

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def peaks_doa(results, num_signals, dist=1):
    peaks, _ = find_peaks(results, distance=dist)
    
    peak_values = results[peaks]
    peaks_argsort = np.argsort(-peak_values)
    peaks_sorted = peaks[peaks_argsort]
    
    peaks_doa = peaks_sorted[0:num_signals]
    
    return peaks_doa

def calculate_rmse(estimate, prediction):
    diff_rad = np.angle(np.exp(1j * (prediction - estimate)))
    diff_squared = np.rad2deg(diff_rad) ** 2
    diff_mean = np.mean(diff_squared)
    
    rmse = np.sqrt(diff_mean)
    
    return rmse
    
def plot_doa(results, theta_scan, num_signals, plot_type=""):
    max_val_arg = peaks_doa(results, num_signals)
    
    theta_doa = theta_scan[max_val_arg]
    peak_doa = results[max_val_arg]
    
    if plot_type == "polar":
        plt.figure()
        ax = plt.subplot(polar=True)
        ax.plot(theta_scan, results)
        ax.plot(theta_doa, peak_doa,'rx')
        ax.set_theta_zero_location('N')
        ax.axes.set_theta_direction(-1) # Increase CW
        ax.set_rlabel_position(55) # Shift grid labels
        
    elif plot_type == "cartesian":
        plt.plot(np.rad2deg(theta_scan), results, linewidth=2)
        # plt.plot(np.rad2deg(theta_doa), peak_doa,'rx')
        plt.xlabel("$\Theta$ [°]", fontsize=18)
        plt.ylabel("DOA Metric", fontsize=18)
        plt.ylim([-0.1, 1.1])
        plt.tight_layout()
        plt.grid()
        # for i in range(len(theta_doa)):
        #     plt.annotate(text=str(np.rad2deg(theta_doa[i]))[:5], xy=(np.rad2deg(theta_doa[i]) + 0.2, peak_doa[i] + 0.03))
        
    print(np.rad2deg(theta_scan[max_val_arg]))
        
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""    
    
    
    
    
    
    
    
