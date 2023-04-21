# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 09:07:06 2023

@author: Takahiro Iwami
"""

#################################################################
#                      Experiment C
#################################################################

#================================================================
#                      Import modules
#================================================================
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jn, spherical_jn
from scipy import signal as sg
from tqdm import tqdm

#================================================================
#                  Definition of Functions
#================================================================
def RK(dim, k_max, r, r_p):
    dis = np.linalg.norm(r - r_p, axis=-1)
    dis[dis==0] += 1e-15
    return (k_max/(2*np.pi*dis))**(dim/2) * jn(dim/2, k_max*dis)

def K(dim, k_max, r1, r2):
    r = np.transpose(np.tile(r1, (r2.shape[0],1,1)), axes=(1,0,2))
    r_p = np.tile(r2, (r1.shape[0],1,1))
    return RK(dim, k_max, r, r_p)

def get_matrix_for_arrival_power_estimation(k_max, vartheta_vec, r, dim):
    r = np.transpose(np.tile(r, (r.shape[0],1,1)), axes=(1,0,2))
    r_p = np.transpose(r, axes=(1,0,2))
    diff = r - r_p
    return (2*k_max/(2*np.pi)**dim)*np.sinc(k_max*np.matmul(diff,vartheta_vec.T)/np.pi)

def add_noise(p, SNR=30):
    noise = np.random.rand(p.size)-0.5
    gain = np.sum(np.abs(p))/(10**(SNR/20)*np.sum(np.abs(noise)))
    tmp = noise.reshape(p.shape)*gain
    return p+tmp

def sinc_interpolation(d, t, fs):
    t_n = np.arange(d.shape[-1]) / fs
    Tn, T = np.meshgrid(t_n, t)
    return fs*np.matmul(spherical_jn(0,np.pi*fs*(Tn - T)), d)

#================================================================
#                       Main routine
#================================================================

if __name__ ==  "__main__":
    np.random.seed(10) # fix seed
    dim = 2
    rho = 1.293
    kappa = 142.0e3
    c = np.sqrt(kappa/rho)
    f_c = 2000
    f_low = f_c / np.sqrt(2)
    f_high = f_c * np.sqrt(2)
    k_max = 2*np.pi*f_high / c
    sl = 0.6 # length of each side
    fs = 8000
    
    # observation region
    div = 50
    int_num = div**2
    width = 0.02
    x_o, y_o= (np.mgrid[0:div:1, 0:div:1]+0.5) * width
    x_o = x_o - width*(div/2)
    y_o = y_o - width*(div/2)
    r_o = np.concatenate((x_o.reshape(div**2,1), y_o.reshape(div**2,1)), axis=1)
    
    # sensor placement
    U = 100
    r_u = (np.random.rand(U, 2) - 0.5)*sl
    
    # directions of sound sources
    phi_deg = np.array([[30, 90, 160], [40, 140], [60, 160]])
    
#    # input signals
    T_loc = int(0.2 * fs)
    W_loc = T_loc//2 + 1
    k = np.linspace(0, 2*np.pi*(fs/2)/c, num=W_loc)
    high_index = int(2*f_high*W_loc/fs)
    low_index = int(2*f_low*W_loc/fs)
    T = 3 * T_loc
    
    # driving signals
    L = 0 # number of DOA
    for i in range(phi_deg.shape[0]):
        L += len(phi_deg[i])
    D = np.zeros([L,W_loc], dtype=np.complex128)
    amplitude = np.ones(L)
    for i in range(low_index, high_index):
        phase = np.random.rand(L) * 2*np.pi
        D[:,i] = amplitude * np.exp(1j*phase)
    d = np.fft.irfft(D, axis=-1)#*window

    # input signal
    delay_margin = (np.ceil(np.matmul(r_o, np.array([np.cos(np.pi/4), np.sin(np.pi/4)]))/c*fs)*2).astype(np.int).max()
    p = np.zeros([U, T+delay_margin], dtype=np.float64)
    l = -1
    for i in tqdm(range(phi_deg.shape[0])):
        theta = np.array(phi_deg[i]) * np.pi / 180
        e_theta = -np.array([np.cos(theta), np.sin(theta)]).T
        for j in range(theta.shape[0]):
            l += 1
            delay = np.matmul(r_u, e_theta[j]) / c
            delay -= delay.min()
            delay_tap = np.ceil(delay*fs).astype(np.int)
            delay_adjust = delay_tap/fs - delay
            delay_adjust[delay_adjust<1e-15] = 0
            for k in range(U):
                p[k, T_loc*i+delay_tap[k]:T_loc*i+delay_tap[k]+T_loc] += sinc_interpolation(d[l], np.arange(T_loc)/fs+delay_adjust[k],fs)
    for t in range(T):
        p[:,t] = add_noise(p[:,t], SNR=30)

    # preprocessing
    lam = 0.1
    vartheta_num = 180
    vartheta_deg = np.linspace(0, 180, num=vartheta_num)
    vartheta = np.deg2rad(vartheta_deg)
    vartheta_vec = np.concatenate((np.cos(vartheta).reshape(vartheta_num,1), np.sin(vartheta).reshape(vartheta_num,1)), axis=1)

    # proposed method
    K_inv = np.linalg.inv(K(dim, k_max, r_u, r_u) + lam*np.eye(r_u.shape[0]))
    C = np.einsum("ijk,jl->ilk", np.einsum("ij,jkl->ikl", K_inv.T, get_matrix_for_arrival_power_estimation(k_max, vartheta_vec, r_u, dim)), K_inv)
    P_p = np.empty([T, vartheta_num], dtype=np.float64)
    for t in range(T):
        tmp = np.matmul(np.matmul(C.T, p[:,t]), p[:,t]) # proposed method
        tmp -= np.min(tmp)
        tmp /= np.max(tmp)
        P_p[t] = tmp
        
    # MUSIC algorithm
    win = "hann"
    N = 2**8
    freq_num = N//2+1
    index_range = np.array([int(np.ceil(2*f_low*freq_num/fs)), int(np.floor(2*f_high*freq_num/fs))])
    index_num = index_range[1] - index_range[0]
    num_frames = 4
    len_loc = N * num_frames
    P_MU = np.zeros([T, vartheta_num], dtype=np.float64)
    for t in tqdm(range(T-len_loc)):
        source_num = len(phi_deg[(t+len_loc)//T_loc])
        beta = np.zeros([index_num], dtype=np.float64)
        _,_,P = sg.stft(p[:, t:t+len_loc], fs, window=win, nperseg=N)
        Rxx = np.mean(np.einsum("ijk,ljk->iljk", P[:,index_range[0]:index_range[1]], np.conjugate(P[:,index_range[0]:index_range[1]])), axis=-1)
        tmp = np.empty([index_num, vartheta_num], dtype=np.float64)
        for w in range(index_num):
            k = 2*np.pi*(fs*(w+index_range[0])/(2*freq_num))/c
            u,s,v = np.linalg.svd(Rxx[:,:,w]) # singular value decomposition
            En = u[:,source_num:]
            beta[w] = np.sum(s[:source_num])
            for i in range(vartheta_num):
                a = np.exp(1j*k*np.matmul(r_u, vartheta_vec[i]))
                tmp[w,i] = np.real(np.matmul(np.conjugate(a), a)/(np.matmul(np.matmul(np.matmul(np.conjugate(a), En), np.conjugate(En.T)),a)))
        weighted_tmp = np.sum(tmp*np.tile(beta.reshape(-1,1), (1, vartheta_num)), axis=0)
        weighted_tmp -= np.min(weighted_tmp)
        weighted_tmp /= np.max(weighted_tmp)
        P_MU[t+len_loc//2] = weighted_tmp
    
    # drawing
    fig1, ax1 = plt.subplots(figsize=(16,9))
    ax1.tick_params(labelsize=20)
    ax1.set_xticks(np.arange(7)*100)
    ax1.set_yticks(np.arange(7)*30)
    ax1.vlines([200, 400], 0, 180, color="black", linestyles='--')
    im1 = ax1.imshow(P_p.T, cmap=plt.cm.Greys, vmin=0, vmax=1, interpolation='None', origin='lower', extent=np.array([0, T/fs*1000, 0, 180]))
    cb1 = fig1.colorbar(im1, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb1.ax.tick_params(labelsize=20)
    ax1.set_xlabel("Time [ms]", fontsize=20)
    ax1.set_ylabel("Angle [deg]", fontsize=20)
    
    fig2, ax2 = plt.subplots(figsize=(16,9))
    ax2.tick_params(labelsize=20)
    ax2.set_xticks(np.arange(7)*100)
    ax2.set_yticks(np.arange(7)*30)
    ax2.vlines([200, 400], 0, 180, color="black", linestyles='--')
    im2 = ax2.imshow(P_MU.T, cmap=plt.cm.Greys, vmin=0, vmax=1, interpolation='None', origin='lower', extent=np.array([0, T/fs*1000, 0, 180]))
    cb2 = fig2.colorbar(im2, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb2.ax.tick_params(labelsize=20)
    ax2.set_xlabel("Time [ms]", fontsize=20)
    ax2.set_ylabel("Angle [deg]", fontsize=20)
    
    plt.show()