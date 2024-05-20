#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2022 KrakenRF Inc.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Edited by: David Sedlacek, xsedla1n@vutbr.cz, Brno University of Technology, 2024.

import numpy as np
import socket
import _thread
import queue
from threading import Thread
from threading import Lock
import matplotlib.pyplot as plt

from struct import pack,unpack
import sys

class krakensdr_dsp:
    def __init__(self,
                 # Input SDR Config parameters
                 ipAddr="192.168.1.6", port=5000, ctrlPort = 5001, numChannels=5, freq=416.588, gain=[20.7,20.7,20.7,20.7,20.7], debug=False,
                 doaAlgortihm="MUSIC", sampleSize=2**20, numSignals=1, interElemSpacing=0.5, steerVectSamples=1000, preprocessing="None", numSubChannels=4):

        # SDR Configuration
        self.ipAddr = ipAddr
        self.port = port
        self.ctrlPort = ctrlPort
        self.numChannels = numChannels
        self.freq = int(freq*10**6)
        self.gain = gain
        self.debug = debug
        self.iq_header = IQHeader()
        
        self.valid_gains = [0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4, 15.7, 16.6, 19.7, 20.7, 22.9, 25.4, 28.0, 29.7, 32.8, 33.8, 36.4, 37.2, 38.6, 40.2, 42.1, 43.4, 43.9, 44.5, 48.0, 49.6]
        
        # DoA Configuration
        self.doaAlgortihm = doaAlgortihm
        self.sampleSize = sampleSize
        self.numSignals = numSignals
        self.interElemSpacing = interElemSpacing
        self.steerVectSamples = steerVectSamples
        self.preprocessing = preprocessing
        self.numSubChannels = numSubChannels

        # Data Interface Setup
        self.socket_inst = socket.socket()
        self.receiver_connection_status = False
        self.receiverBufferSize = 2 ** 18

        # Control interface Setup
        self.ctr_iface_socket = socket.socket()
        self.ctr_iface_port = self.ctrlPort
        self.ctr_iface_thread_lock = Lock() # Used to synchronize the operation of the ctr_iface thread

        # Init cpi_len from heimdall header 
        # Sometimes cpi_len is initially zero. If so, loop until we get a non-zero value.
        self.get_iq_online()
        while self.iq_header.cpi_length == 0: 
            self.get_iq_online()
            
        self.cpi_len = self.iq_header.cpi_length
        self.total_fetched = self.iq_header.cpi_length

        self.iq_samples = self.get_iq_online()
        self.iq_sample_queue = queue.Queue(10)
        
        self.stop_threads = False
        self.buffer_thread = Thread(target = self.buffer_iq_samples)
        self.buffer_thread.start()
        

    '''
        Continuously receive sample frames from heimdall and put into a buffer.
        Drops frames if buffer is full because downstream DSP was too slow.
    '''
    
    def buffer_iq_samples(self):
        while(True):

            if self.debug:
                self.iq_header.dump_header()
                
            if self.stop_threads: # Stop thread on close
                return
                
            iq_samples = self.get_iq_online()

            try:
                if self.iq_header.frame_type == self.iq_header.FRAME_TYPE_DATA: # Only output DATA frames, not calibration frames
                    self.iq_sample_queue.put_nowait(iq_samples)
            except:
                print("[Buffer overflow] Failed to put IQ Samples into the Queue.")
                print("Skipping frames.")
    
    
    '''
        Gets IQ samples of an array from the buffer Queue, and processes them via requested DoA estimation algortihm.
        The processing function repeats until all IQ samples have processed. Only after all samples have been processed new IQ samples are obtained from the buffer.
    '''
    def process_iq_samples(self):

        # Acquire IQ samples from the buffer    
        if self.total_fetched == self.cpi_len:   
            try:
                self.iq_samples = self.iq_sample_queue.get(True, 3) # Wait until samples are ready
            except:
                print("[Buffer empty] Failed to obtain IQ samples from the Queue.")

            self.total_fetched = 0

        fetch_left = self.cpi_len - self.total_fetched  # Ammount of IQ samples left to process
        req_samples = self.sampleSize
        rem_samples = min(req_samples, fetch_left)  # Ammount of IQ samples to be processed in the next loop
        
        x = self.iq_samples[:,self.total_fetched:self.total_fetched + rem_samples] # Obtain the next batch of IQ samples
        
        self.estimate_doa(x, rem_samples) # Process samples

        self.total_fetched = self.total_fetched + rem_samples
        
    '''
        This is the custom DSP solution enabling the DoA estimation. Some lines can be uncommented to perform slight adjustments to the estimation methods.
        It is a replacement for the work function that reads inputs, processes and writes outputs for other GNU radio blocks. The work function needs GNU Radios
        scheduler to sucessfully forward samples to other blocks.
    '''
    def estimate_doa(self, x, rem_samples):
        try:            
            if self.preprocessing == "None":
                R_xx = x @ x.conj().T / rem_samples # Estimate the spatial covariance matrix
                
            elif self.preprocessing == "FBA":
                R_xx = x @ x.conj().T
                Pi_m = np.fliplr(np.eye(np.size(R_xx,0)))
                R_xx = (R_xx + (Pi_m @ R_xx.conj() @ Pi_m)) / (2*rem_samples)
                
            elif self.preprocessing == "SS-F" or self.preprocessing == "SS-B" or self.preprocessing == "SS-FB":
                numSubElem = self.numChannels - self.numSubChannels + 1 # Number of subarrays
                R_xx = np.zeros((numSubElem,numSubElem), dtype=np.complex128) # Initialize the smoothing matrix

                if self.preprocessing == "SS-F" or self.preprocessing == "SS-FB":
                    for i in range(numSubElem):
                        R_ss = x[i:i+numSubElem,:] @ x[i:i+numSubElem,:].conj().T / rem_samples
                        R_xx += R_ss
                        
                if self.preprocessing == "SS-B" or self.preprocessing == "SS-FB":
                    for i in range(numSubElem):
                        R_ss = x[self.numChannels-numSubElem-i:self.numChannels-i,:] @ x[self.numChannels-numSubElem-i:self.numChannels-i,:].conj().T / rem_samples
                        R_xx += R_ss
                        
                if self.preprocessing == "SS-FB":
                    R_xx = R_xx / (numSubElem*2)
                    
                else:
                    R_xx = R_xx / numSubElem
            
            if (self.doaAlgortihm == "Bartlett"):
                    
                theta_scan = np.linspace(-1*np.pi/2, np.pi/2, self.steerVectSamples) # Generate angles for steering vectors
                results = []

                for theta_i in theta_scan:
                    
                    a_i = np.exp(-2j * np.pi * self.interElemSpacing * np.arange(self.numChannels) * np.sin(theta_i)).reshape(-1,1)
                    w = a_i / np.sqrt(np.dot(a_i.conj().T,a_i)) # Normalize weights
                    # w = a_i # Continue without normalization
                    
                    r_weighted = abs(w.conj().T @ R_xx @ w)   
                    results.append(10*np.log10(r_weighted.squeeze()))
                        
                print(theta_scan[np.argmax(results)] * 180 / np.pi)

                # Plot the output in a cartesian plot
                plt.clf()
                plt.plot(theta_scan*180/np.pi, results)
                plt.plot(theta_scan[np.argmax(results)] * 180 / np.pi, np.max(results),'rx')
                plt.xlabel("Theta [Degrees]")
                plt.ylabel("Magnitude [dB]")
                plt.xlim(-100,100)
                plt.grid()
                plt.draw()
                plt.pause(0.1)
                
            elif (self.doaAlgortihm == "MVDR"):
                
                R_xx_inv = np.linalg.inv(R_xx)

                theta_scan = np.linspace(-1*np.pi/2, np.pi/2, self.steerVectSamples) # Generate angles for steering vectors
                results = []

                for theta_i in theta_scan:
                    a_i = np.exp(-2j * np.pi * self.interElemSpacing * np.arange(self.numChannels) * np.sin(theta_i)).reshape(-1,1)            
                    w = (R_xx_inv @ a_i) / (a_i.conj().T @ R_xx_inv @ a_i)

                    r_weighted = np.real(w.conj().T @ R_xx @ w)
                    results.append(10*np.log10(r_weighted.squeeze()))
                    
                print(theta_scan[np.argmax(results)] * 180 / np.pi)

                ## Plot the output in a cartesian plot
                plt.clf()
                plt.plot(theta_scan*180/np.pi, results)
                plt.plot(theta_scan[np.argmax(results)] * 180 / np.pi, np.max(results),'rx')
                plt.xlabel("Theta [Degrees]")
                plt.ylabel("Magnitude [dB]")
                plt.xlim(-100,100)
                plt.grid()
                plt.draw()
                plt.pause(0.1)
                
            elif (self.doaAlgortihm == "MUSIC"):
                
                if self.preprocessing == "SS-F" or self.preprocessing == "SS-B" or self.preprocessing == "SS-FB":
                    w,v = np.linalg.eig(R_xx) # Obtain eigenvectors and their eigenvalues
    
                    # Sort eigenvectors to obtain the noise subsopace
                    eig_val_order = np.argsort(np.abs(w))
                    v_sorted = v[:, eig_val_order]
    
                    Vn = np.zeros((numSubElem,numSubElem-self.numSignals), dtype=np.complex64)
                    for i in range(numSubElem-self.numSignals):
                        Vn[:,i] = v_sorted[:,i] 
    
                    ## Simulate MUSIC DoA estimation
                    theta_scan = np.linspace(-1*np.pi/2, np.pi/2, self.steerVectSamples)
                    results = []
    
                    for theta_i in theta_scan:
                        a_i = np.exp(-2j * np.pi * self.interElemSpacing * np.arange(numSubElem) * np.sin(theta_i)).reshape(-1,1)
                        p_theta = 1 / (a_i.conj().T @ Vn @ Vn.conj().T @ a_i)
                        # p_theta = (a_i.conj().T @ a_i) / (a_i.conj().T @ Vn @ Vn.conj().T @ a_i) # Alternative spatial spectrum
                        
                        # results.append(p_theta.squeeze())
                        results.append(10*np.log10(np.abs(p_theta.squeeze())))
                        
                    results /= np.max(results) # Normalize results
                    print(theta_scan[np.argmax(results)] * 180 / np.pi)
                
                else:
                    w,v = np.linalg.eig(R_xx) # Obtain eigenvectors and their eigenvalues
                    
                    w.sort()
                    noise_power = np.abs(w[0])
                    signal_plus_noise_power = np.abs(w[-1])
                    power_ratio = (signal_plus_noise_power - noise_power) / noise_power
                    snr = 10.0 * np.log10(power_ratio)
                    print("Estimated SNR is: ", snr, " dB")
    
                    # Sort eigenvectors to obtain the noise subsopace
                    eig_val_order = np.argsort(np.abs(w))
                    v_sorted = v[:, eig_val_order]
    
                    Vn = np.zeros((self.numChannels,self.numChannels-self.numSignals), dtype=np.complex64)
                    for i in range(self.numChannels-self.numSignals):
                        Vn[:,i] = v_sorted[:,i] 
    
                    ## Simulate MUSIC DoA estimation
                    theta_scan = np.linspace(-1*np.pi/2, np.pi/2, self.steerVectSamples)
                    results = []
    
                    for theta_i in theta_scan:
                        a_i = np.exp(-2j * np.pi * self.interElemSpacing * np.arange(self.numChannels) * np.sin(theta_i)).reshape(-1,1)
                        p_theta = 1 / (a_i.conj().T @ Vn @ Vn.conj().T @ a_i)
                        # p_theta = (a_i.conj().T @ a_i) / (a_i.conj().T @ Vn @ Vn.conj().T @ a_i) # Alternative spatial spectrum
                        
                        # results.append(p_theta.squeeze())
                        results.append(10*np.log10(np.abs(p_theta.squeeze())))
                        
                    results /= np.max(results) # Normalize results
                    print(theta_scan[np.argmax(results)] * 180 / np.pi)

                ## Plot the output in a cartesian plot
                plt.clf()
                plt.plot(theta_scan*180/np.pi, results)
                plt.plot(theta_scan[np.argmax(results)] * 180 / np.pi, np.max(results),'rx')
                plt.xlabel("Theta [Degrees]")
                plt.ylabel("Magnitude [dB]")
                plt.xlim(-100,100)
                plt.grid()
                plt.draw()
                plt.pause(0.1)
                
            elif (self.doaAlgortihm == "Root-MUSIC"):
                
                if self.preprocessing == "SS-F" or self.preprocessing == "SS-B" or self.preprocessing == "SS-FB":
                    w,v = np.linalg.eig(R_xx) # Obtain eigenvectors and their eigenvalues
    
                    # Sort eigenvectors to obtain the noise subsopace
                    eig_val_order = np.argsort(np.abs(w))
                    v_sorted = v[:, eig_val_order]
    
                    Vn = np.zeros((numSubElem,numSubElem-self.numSignals), dtype=np.complex64)
                    for i in range(numSubElem-self.numSignals):
                        Vn[:,i] = v_sorted[:,i] 
    
                    ct = Vn @ Vn.conj().T
                    M = R_xx.shape[0]

                    ## Define the polynomial using the noise subspace
                    p_coeff = np.zeros(2 * M - 1, dtype=np.complex64)

                    for i in range(-M + 1, M):
                        p_coeff[i + (M - 1)] = np.trace(ct, i)
                        
                    all_roots = np.roots(p_coeff)

                    candidate_roots_abs = np.abs(all_roots)
                    sorted_idx = candidate_roots_abs.argsort()[(M - 1 - self.numSignals) : (M - 1)]

                    valid_roots = all_roots[sorted_idx]
                    args = np.angle(valid_roots)

                    doas = np.arcsin(args / (self.interElemSpacing * 2.0 * np.pi))

                    doas_deg = np.rad2deg(doas)
                    
                    print(doas_deg)
                
                else:
                    w,v = np.linalg.eig(R_xx) # Obtain eigenvectors and their eigenvalues
    
                    # Sort eigenvectors to obtain the noise subsopace
                    eig_val_order = np.argsort(np.abs(w))
                    v_sorted = v[:, eig_val_order]
    
                    Vn = np.zeros((self.numChannels,self.numChannels-self.numSignals), dtype=np.complex64)
                    for i in range(self.numChannels-self.numSignals):
                        Vn[:,i] = v_sorted[:,i] 
    
                    ct = Vn @ Vn.conj().T
                    M = R_xx.shape[0]

                    ## Define the polynomial using the noise subspace
                    p_coeff = np.zeros(2 * M - 1, dtype=np.complex64)

                    for i in range(-M + 1, M):
                        p_coeff[i + (M - 1)] = np.trace(ct, i)
                        
                    all_roots = np.roots(p_coeff)

                    candidate_roots_abs = np.abs(all_roots)
                    sorted_idx = candidate_roots_abs.argsort()[(M - 1 - self.numSignals) : (M - 1)]

                    valid_roots = all_roots[sorted_idx]
                    args = np.angle(valid_roots)

                    doas = np.arcsin(args / (self.interElemSpacing * 2.0 * np.pi))

                    doas_deg = np.rad2deg(doas)
                    
                    print(doas_deg)
                
            else:
                raise Exception("The selected algorithm is not supported.")
            
        except Exception as e:
            print("Failed to estimate DoA.")
            print("Exception: " + str(e))
            return 0
        

    def stop(self):
        self.stop_threads = True
        self.buffer_thread.join()
        self.eth_close()
        return True

    def eth_connect(self):
        """
            Compatible only with DAQ firmwares that has the IQ streaming mode.
            HeIMDALL DAQ Firmware version: 1.0 or later
        """
        try:
            if not self.receiver_connection_status:
                # Establish IQ data interface connection
                self.socket_inst.connect((self.ipAddr, self.port))
                self.socket_inst.sendall(str.encode('streaming'))
                test_iq = self.receive_iq_frame()
                print(test_iq)

                # Establish control interface connection
                self.ctr_iface_socket.connect((self.ipAddr, self.ctr_iface_port))
                self.receiver_connection_status = True
                self.ctr_iface_init()

                self.set_center_freq(self.freq)
                self.set_if_gain(self.gain)
        except Exception as e:
            print(e)
            errorMsg = sys.exc_info()[0]
            self.receiver_connection_status = False
            print("Ethernet Connection Failed, Error: " + str(errorMsg))
        return -1


    def ctr_iface_init(self):
        """
            Initialize connection with the DAQ FW through the control interface
        """
        if self.receiver_connection_status: # Check connection
            # Assembling message
            cmd="INIT"
            msg_bytes=(cmd.encode()+bytearray(124))
            try:
                _thread.start_new_thread(self.ctr_iface_communication, (msg_bytes,))
            except:
                errorMsg = sys.exc_info()[0]
                print("Unable to start communication thread")
                print("Error message: {:s}".format(errorMsg))


    def ctr_iface_communication(self, msg_bytes):
        """
            Handles communication on the control interface with the DAQ FW

            Parameters:
            -----------

                :param: msg: Message bytes, that will be sent ont the control interface
                :type:  msg: Byte array
        """
        self.ctr_iface_thread_lock.acquire()
        print("Sending control message")
        self.ctr_iface_socket.send(msg_bytes)

        # Waiting for the command to take effect
        reply_msg_bytes = self.ctr_iface_socket.recv(128)

        print("Control interface communication finished")
        self.ctr_iface_thread_lock.release()

        status = reply_msg_bytes[0:4].decode()
        if status == "FNSD":
            print("Reconfiguration succesfully finished")

        else:
            print("Failed to set the requested parameter, reply: {0}".format(status))

    def set_center_freq(self, center_freq):
        """
            Configures the RF center frequency of the receiver through the control interface

            Paramters:
            ----------
                :param: center_freq: Required center frequency to set [Hz]
                :type:  center_freq: float
        """
        if self.receiver_connection_status: # Check connection
            self.freq = int(center_freq)
            # Set center frequency
            cmd="FREQ"
            freq_bytes=pack("Q",int(center_freq))
            msg_bytes=(cmd.encode()+freq_bytes+bytearray(116))
            try:
                _thread.start_new_thread(self.ctr_iface_communication, (msg_bytes,))
            except:
                errorMsg = sys.exc_info()[0]
                print("Unable to start communication thread")
                print("Error message: {:s}".format(errorMsg))

    def set_if_gain(self, gain):
        """
            Configures the IF gain of the receiver through the control interface

            Paramters:
            ----------
                :param: gain: IF gain value [dB]
                :type:  gain: int
        """
        if self.receiver_connection_status: # Check connection
            cmd="GAIN"

            # Find the closest valid gain to the input gain value
            for i in range(len(gain)):
                gain[i] = min(self.valid_gains, key=lambda x:abs(x-gain[i]))

            gain_list= [int(i * 10) for i in gain]

            gain_bytes=pack("I"*self.numChannels, *gain_list)
            msg_bytes=(cmd.encode()+gain_bytes+bytearray(128-(self.numChannels+1)*4))
            try:
                _thread.start_new_thread(self.ctr_iface_communication, (msg_bytes,))
            except:
                errorMsg = sys.exc_info()[0]
                print("Unable to start communication thread")
                print("Error message: {:s}".format(errorMsg))


    def get_iq_online(self):
        """
            This function obtains a new IQ data frame through the Ethernet IQ data or the shared memory interface
        """

        # Check connection
        if not self.receiver_connection_status:
            fail = self.eth_connect()
            if fail:
                return -1

        self.socket_inst.sendall(str.encode("IQDownload")) # Send iq request command
        #self.iq_samples = self.receive_iq_frame()
        return self.receive_iq_frame()

    def receive_iq_frame(self):
        """
            Called by the get_iq_online function. Receives IQ samples over the establed Ethernet connection
        """
        total_received_bytes = 0
        recv_bytes_count = 0
        iq_header_bytes = bytearray(self.iq_header.header_size)  # allocate array
        view = memoryview(iq_header_bytes)  # Get buffer

        while total_received_bytes < self.iq_header.header_size:
            # Receive into buffer
            recv_bytes_count = self.socket_inst.recv_into(view, self.iq_header.header_size-total_received_bytes)
            view = view[recv_bytes_count:]  # reset memory region
            total_received_bytes += recv_bytes_count

        self.iq_header.decode_header(iq_header_bytes)
        # Uncomment to check the content of the IQ header
        #self.iq_header.dump_header()

        incoming_payload_size = self.iq_header.cpi_length*self.iq_header.active_ant_chs*2*int(self.iq_header.sample_bit_depth/8)
        if incoming_payload_size > 0:
            # Calculate total bytes to receive from the iq header data
            total_bytes_to_receive = incoming_payload_size
            receiver_buffer_size = 2**18

            total_received_bytes = 0
            recv_bytes_count = 0
            iq_data_bytes = bytearray(total_bytes_to_receive + receiver_buffer_size)  # allocate array
            view = memoryview(iq_data_bytes)  # Get buffer

            while total_received_bytes < total_bytes_to_receive:
                # Receive into buffer
                recv_bytes_count = self.socket_inst.recv_into(view, receiver_buffer_size)
                view = view[recv_bytes_count:]  # reset memory region
                total_received_bytes += recv_bytes_count

            # Convert raw bytes to Complex float64 IQ samples
            self.iq_samples = np.frombuffer(iq_data_bytes[0:total_bytes_to_receive], dtype=np.complex64).reshape(self.iq_header.active_ant_chs, self.iq_header.cpi_length)

            self.iq_frame_bytes =  bytearray()+iq_header_bytes+iq_data_bytes
            return self.iq_samples
        else:
              return 0

    def eth_close(self):
        """
            Close Ethernet conenctions including the IQ data and the control interfaces
        """
        try:
            if self.receiver_connection_status:
                self.socket_inst.sendall(str.encode('q')) # Send exit message
                self.socket_inst.close()
                self.socket_inst = socket.socket() # Re-instantiating socket

                # Close control interface connection
                exit_message_bytes=("EXIT".encode()+bytearray(124))
                self.ctr_iface_socket.send(exit_message_bytes)
                self.ctr_iface_socket.close()
                self.ctr_iface_socket = socket.socket()

            self.receiver_connection_status = False
        except:
            errorMsg = sys.exc_info()[0]
            print("Error message: {0}".format(errorMsg))
            return -1

        return 0



"""
    Desctiption: IQ Frame header definition
    For header field description check the corresponding documentation
    Total length: 1024 byte
    Author: Tamás Pető
"""
class IQHeader():

    FRAME_TYPE_DATA  = 0
    FRAME_TYPE_DUMMY = 1
    FRAME_TYPE_RAMP  = 2
    FRAME_TYPE_CAL   = 3
    FRAME_TYPE_TRIGW = 4

    SYNC_WORD = 0x2bf7b95a

    def __init__(self):

        #self.logger = logging.getLogger(__name__)
        self.header_size = 1024 # size in bytes
        self.reserved_bytes = 192

        self.sync_word=self.SYNC_WORD        # uint32_t
        self.frame_type=0                    # uint32_t
        self.hardware_id=""                  # char [16]
        self.unit_id=0                       # uint32_t
        self.active_ant_chs=0                # uint32_t
        self.ioo_type=0                      # uint32_t
        self.rf_center_freq=0                # uint64_t
        self.adc_sampling_freq=0             # uint64_t
        self.sampling_freq=0                 # uint64_t
        self.cpi_length=0                    # uint32_t
        self.time_stamp=0                    # uint64_t
        self.daq_block_index=0               # uint32_t
        self.cpi_index=0                     # uint32_t
        self.ext_integration_cntr=0          # uint64_t
        self.data_type=0                     # uint32_t
        self.sample_bit_depth=0              # uint32_t
        self.adc_overdrive_flags=0           # uint32_t
        self.if_gains=[0]*32                 # uint32_t x 32
        self.delay_sync_flag=0               # uint32_t
        self.iq_sync_flag=0                  # uint32_t
        self.sync_state=0                    # uint32_t
        self.noise_source_state=0            # uint32_t
        self.reserved=[0]*self.reserved_bytes # uint32_t x reserverd_bytes
        self.header_version=0                # uint32_t

    def decode_header(self, iq_header_byte_array):
        """
            Unpack,decode and store the content of the iq header
        """
        iq_header_list = unpack("II16sIIIQQQIQIIQIII"+"I"*32+"IIII"+"I"*self.reserved_bytes+"I", iq_header_byte_array)

        self.sync_word            = iq_header_list[0]
        self.frame_type           = iq_header_list[1]
        self.hardware_id          = iq_header_list[2].decode()
        self.unit_id              = iq_header_list[3]
        self.active_ant_chs       = iq_header_list[4]
        self.ioo_type             = iq_header_list[5]
        self.rf_center_freq       = iq_header_list[6]
        self.adc_sampling_freq    = iq_header_list[7]
        self.sampling_freq        = iq_header_list[8]
        self.cpi_length           = iq_header_list[9]
        self.time_stamp           = iq_header_list[10]
        self.daq_block_index      = iq_header_list[11]
        self.cpi_index            = iq_header_list[12]
        self.ext_integration_cntr = iq_header_list[13]
        self.data_type            = iq_header_list[14]
        self.sample_bit_depth     = iq_header_list[15]
        self.adc_overdrive_flags  = iq_header_list[16]
        self.if_gains             = iq_header_list[17:49]
        self.delay_sync_flag      = iq_header_list[49]
        self.iq_sync_flag         = iq_header_list[50]
        self.sync_state           = iq_header_list[51]
        self.noise_source_state   = iq_header_list[52]
        self.header_version       = iq_header_list[52+self.reserved_bytes+1]

    def encode_header(self):
        """
            Pack the iq header information into a byte array
        """
        iq_header_byte_array=pack("II", self.sync_word, self.frame_type)
        iq_header_byte_array+=self.hardware_id.encode()+bytearray(16-len(self.hardware_id.encode()))
        iq_header_byte_array+=pack("IIIQQQIQIIQIII",
                                self.unit_id, self.active_ant_chs, self.ioo_type, self.rf_center_freq, self.adc_sampling_freq,
                                self.sampling_freq, self.cpi_length, self.time_stamp, self.daq_block_index, self.cpi_index,
                                self.ext_integration_cntr, self.data_type, self.sample_bit_depth, self.adc_overdrive_flags)
        for m in range(32):
            iq_header_byte_array+=pack("I", self.if_gains[m])

        iq_header_byte_array+=pack("I", self.delay_sync_flag)
        iq_header_byte_array+=pack("I", self.iq_sync_flag)
        iq_header_byte_array+=pack("I", self.sync_state)
        iq_header_byte_array+=pack("I", self.noise_source_state)

        for m in range(self.reserved_bytes):
            iq_header_byte_array+=pack("I",0)

        iq_header_byte_array+=pack("I", self.header_version)
        return iq_header_byte_array

    def dump_header(self):
        """
            Prints out the content of the header in human readable format
        """
        print("Sync word: {:d}".format(self.sync_word))
        print("Header version: {:d}".format(self.header_version))
        print("Frame type: {:d}".format(self.frame_type))
        print("Hardware ID: {:16}".format(self.hardware_id))
        print("Unit ID: {:d}".format(self.unit_id))
        print("Active antenna channels: {:d}".format(self.active_ant_chs))
        print("Illuminator type: {:d}".format(self.ioo_type))
        print("RF center frequency: {:.2f} MHz".format(self.rf_center_freq/10**6))
        print("ADC sampling frequency: {:.2f} MHz".format(self.adc_sampling_freq/10**6))
        print("IQ sampling frequency {:.2f} MHz".format(self.sampling_freq/10**6))
        print("CPI length: {:d}".format(self.cpi_length))
        print("Unix Epoch timestamp: {:d}".format(self.time_stamp))
        print("DAQ block index: {:d}".format(self.daq_block_index))
        print("CPI index: {:d}".format(self.cpi_index))
        print("Extended integration counter {:d}".format(self.ext_integration_cntr))
        print("Data type: {:d}".format(self.data_type))
        print("Sample bit depth: {:d}".format(self.sample_bit_depth))
        print("ADC overdrive flags: {:d}".format(self.adc_overdrive_flags))
        for m in range(32):
            print("Ch: {:d} IF gain: {:.1f} dB".format(m, self.if_gains[m]/10))
        print("Delay sync  flag: {:d}".format(self.delay_sync_flag))
        print("IQ sync  flag: {:d}".format(self.iq_sync_flag))
        print("Sync state: {:d}".format(self.sync_state))
        print("Noise source state: {:d}".format(self.noise_source_state))

    def check_sync_word(self):
        """
            Check the sync word of the header
        """
        if self.sync_word != self.SYNC_WORD:
            return -1
        else:
            return 0
