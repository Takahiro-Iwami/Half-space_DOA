# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 08:55:30 2023

@author: Takahiro Iwami
"""

#################################################################
#                      Experiment A
#################################################################

#================================================================
#                      Import modules
#================================================================
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from cycler import cycler
import numpy as np
from scipy.special import jn
from scipy import signal as sg

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
#================================================================
#                       Main routine
#================================================================

if __name__ ==  "__main__":
    color_cycle = cycler("color", ['lightgray', 'darkgray', 'black', 'lightgray', 'darkgray', 'black'])
    ls_cycle = cycler("linestyle", ['-', '-', '-', '-.', '-.', '-.'])
    ms_cycle = cycler("markersize", np.ones(6)*8)
    plt.rc('axes', prop_cycle=(color_cycle + ls_cycle + ms_cycle))
    
    np.random.seed(10) # fix seed
    dim = 2
    rho = 1.293
    kappa = 142.0e3
    c = np.sqrt(kappa/rho)
    f_c = np.array([250,1000,2000])
    f_low = f_c / np.sqrt(2)
    f_high = f_c * np.sqrt(2)
    k_min = 2*np.pi*f_low / c
    k_max = 2*np.pi*f_high / c
    k_lim = np.array([k_min, k_max]).T
    sl = 0.6 # length of each side
    # for MUSIC
    win = "hann"
    T = 2**11
    W = T//2+1
    N = 2**8
    fs = 8000
    low_index = (2*f_low*W/fs).astype(int)
    high_index = (2*f_high*W/fs).astype(int)
    k = np.linspace(0, 2*np.pi*(fs/2)/c, num=W)
    
    # observation points
    div = 50
    int_num = div**2
    width = 0.02
    x_o, y_o= (np.mgrid[0:div:1, 0:div:1]+0.5) * width
    x_o = x_o - width*(div/2)
    y_o = y_o - width*(div/2)
    r_o = np.concatenate((x_o.reshape(div**2,1), y_o.reshape(div**2,1)), axis=1)
    disp_lim = np.array([np.min(x_o), np.max(x_o), np.min(y_o), np.max(y_o)])
    
    # sensor placement
    U = 100
    r_u = (np.random.rand(U, 2) - 0.5)*sl
    
    # directions of sound sources
    phi_deg = np.array([40, 110, 150])
    phi = np.deg2rad(phi_deg)
    phi_vec = -np.array([np.cos(phi), np.sin(phi)]).T

    P = np.zeros([f_c.shape[0], U, W], dtype=np.complex128)
    P_pri = np.zeros([f_c.shape[0], int_num, W], dtype=np.complex128)
    inner_prod = np.matmul(phi_vec, r_u.T)
    for i in range(f_c.shape[0]):
        for j in range(phi.shape[0]):
            phase = 2*np.pi*np.random.rand(W)
            for u in range(U):
                P[i,u] += np.exp(-1j*(k*np.matmul(phi_vec[j], r_u[u]) + phase))
            for l in range(int_num):
                P_pri[i,l] += np.exp(-1j*(k*np.matmul(phi_vec[j], r_o[l]) + phase))
        P[i,:,:low_index[i]] = 0
        P[i,:,high_index[i]:] = 0
        P_pri[i,:,:low_index[i]] = 0
        P_pri[i,:,high_index[i]:] = 0
    p = np.fft.irfft(P, axis=-1)
    p_pri = np.fft.irfft(P_pri, axis=-1)
    for i in range(f_c.shape[0]):
        for t in range(T):
            p[i,:,t] = add_noise(p[i,:,t], SNR=30)
    
    # preprocessing
    lam = 0.1
    vartheta_num = 400
    vartheta_deg = np.linspace(0, 180, num=vartheta_num)
    vartheta = np.deg2rad(vartheta_deg)
    vartheta_vec = np.concatenate((np.cos(vartheta).reshape(vartheta_num,1), np.sin(vartheta).reshape(vartheta_num,1)), axis=1)

    # drawing
    fig1, ax1 = plt.subplots()
    ax1.tick_params(labelsize=16)
    ax1.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax1.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
    tmp = p_pri[2,:,T//2] / np.max(np.abs(p_pri[2,:,T//2]))
    im1 = ax1.imshow(tmp.reshape(div,div).T, cmap=plt.cm.seismic, vmin=-1, vmax=1, interpolation='bicubic', origin='lower', extent=disp_lim)
    cb1 = fig1.colorbar(im1, ticks=[-1.0, -0.5, 0, 0.5, 1.0])
    cb1.ax.tick_params(labelsize=16)
    cb1.set_label("Sound pressure [Pa]", fontname="Arial", fontsize=16)    
    ax1.scatter(r_u[:,0], r_u[:,1], s=10, lw=0.2, c="black", edgecolor="white")
    rec = pat.Rectangle(xy = (-sl/2, -sl/2), width=sl, height=sl, angle=0, ec="gray", fill=False, ls="--", lw=1)
    ax1.add_patch(rec)
    ax1.set_xlabel("x [m]", fontsize=16)
    ax1.set_ylabel("y [m]", fontsize=16)
    
    fig2, ax2 = plt.subplots(figsize=(16,9))
    ax2.vlines(phi_deg, 0, 1.05, color="dimgrey", linestyles='-', label="True DOAs")
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks([0, 40, 90, 110, 150, 180])
    ax2.tick_params(labelsize=16)
    ax2.set_xlabel("Angle [deg]", fontsize=16)
    ax2.set_ylabel("Normalized value", fontsize=16)
    ax2.grid(ls="--")

    # proposed method
    for f in range(f_c.shape[0]):
        K_inv = np.linalg.inv(K(dim, k_max[f], r_u, r_u) + lam*np.eye(r_u.shape[0]))
        C = np.einsum("ijk,jl->ilk", np.einsum("ij,jkl->ikl", K_inv.T, get_matrix_for_arrival_power_estimation(k_max[f], vartheta_vec, r_u, dim)), K_inv)
        P_p = np.matmul(np.matmul(C.T, p[f,:,T//2]), p[f,:,T//2]) # proposed method
        P_p -= np.min(P_p)
        P_p /= np.max(P_p)
        ax2.plot(vartheta_deg, P_p, lw=2, markevery=5, label="Proposed({} Hz)".format(f_c[f]))
        
    # MUSIC algorithm
    freq_num = N//2+1
    for f in range(f_c.shape[0]):
        index_range = np.array([int(np.ceil(2*f_low[f]*freq_num/fs)), int(np.floor(2*f_high[f]*freq_num/fs))])
        index_num = index_range[1] - index_range[0]
        _,_,P = sg.stft(p[f], fs, window=win, nperseg=N)
        beta = np.zeros([index_num], dtype=np.float64)
        Rxx = np.mean(np.einsum("ijk,ljk->iljk", P[:,index_range[0]:index_range[1]], np.conjugate(P[:,index_range[0]:index_range[1]])), axis=-1)
        P_MU = np.empty([index_num, vartheta_num], dtype=np.float64)
        for w in range(index_num):
            k = 2*np.pi*(fs*(w+index_range[0])/(2*freq_num))/c
            u,s,v = np.linalg.svd(Rxx[:,:,w]) # singular value decomposition
            En = u[:,phi.shape[0]:]
            beta[w] = np.sum(s[:phi.shape[0]])
            for i in range(vartheta_num):
                a = np.exp(1j*k*np.matmul(r_u, vartheta_vec[i]))
                P_MU[w,i] = np.real(np.matmul(np.conjugate(a), a)/(np.matmul(np.matmul(np.matmul(np.conjugate(a), En), np.conjugate(En.T)),a)))
        P_MU = np.sum(P_MU*np.tile(beta.reshape(-1,1), (1, vartheta_num)), axis=0)
        P_MU -= np.min(P_MU)
        P_MU /= np.max(P_MU)
        ax2.plot(vartheta_deg, P_MU, lw=2, markevery=5, label="MUSIC({} Hz)".format(f_c[f]))

    ax2.legend(loc='upper left', fontsize=15)

    plt.show()
