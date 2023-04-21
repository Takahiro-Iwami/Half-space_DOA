# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 08:40:07 2023

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
import time

#================================================================
#                  Definition of Functions
#================================================================
def get_primary_field(c, k_lim, t, phi_vec, r):
    inner_prod = np.matmul(phi_vec, r.T)
    p = np.zeros([r.shape[0]], dtype=np.float64)
    for i in range(phi_vec.shape[0]):
        p += (k_lim[1]/np.pi)*np.sinc(k_lim[1]*(inner_prod[i]-c*t)/np.pi) - (k_lim[0]/np.pi)*np.sinc(k_lim[0]*(inner_prod[i]-c*t)/np.pi)
    return p

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
    T = 2**8
    W = T//2+1
    fs = 8000
    t = np.arange(T)/fs
    t -= np.mean(t)
    
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
    
    # preprocessing
    lam = 0.1
    vartheta_num = 400
    vartheta_deg = np.linspace(0, 180, num=vartheta_num)
    vartheta = np.deg2rad(vartheta_deg)
    vartheta_vec = np.concatenate((np.cos(vartheta).reshape(vartheta_num,1), np.sin(vartheta).reshape(vartheta_num,1)), axis=1)

    # drawing
    p_pri = get_primary_field(c, k_lim[0], 0, phi_vec, r_o).reshape(div,div)
    p_pri /= np.max(np.abs(p_pri))
    fig1, ax1 = plt.subplots()
    ax1.tick_params(labelsize=24)
    ax1.set_xticks([-0.4, 0, 0.4])
    ax1.set_yticks([-0.4, 0, 0.4])
    im1 = ax1.imshow(p_pri.T, cmap=plt.cm.seismic, vmin=-1, vmax=1, interpolation='bicubic', origin='lower', extent=disp_lim)
    ax1.scatter(r_u[:,0], r_u[:,1], s=10, lw=0.2, c="black", edgecolor="white")
    rec = pat.Rectangle(xy = (-sl/2, -sl/2), width=sl, height=sl, angle=0, ec="gray", fill=False, ls="--", lw=1)
    ax1.add_patch(rec)
    ax1.set_xlabel("x [m]", fontsize=24)
    ax1.set_ylabel("y [m]", fontsize=24)
    
    p_pri = get_primary_field(c, k_lim[2], 0, phi_vec, r_o).reshape(div,div)
    p_pri /= np.max(np.abs(p_pri))
    fig2, ax2 = plt.subplots()
    ax2.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    ax2.tick_params(bottom=False, left=False, right=False, top=False)
    im2 = ax2.imshow(p_pri.T, cmap=plt.cm.seismic, vmin=-1, vmax=1, interpolation='bicubic', origin='lower', extent=disp_lim)
    cb2 = fig2.colorbar(im2, ticks=[-1.0, -0.5, 0, 0.5, 1.0])
    cb2.ax.tick_params(labelsize=24)
    cb2.set_label("Sound pressure [Pa]", fontname="Arial", fontsize=24)    
    ax2.scatter(r_u[:,0], r_u[:,1], s=10, lw=0.2, c="black", edgecolor="white")
    rec = pat.Rectangle(xy = (-sl/2, -sl/2), width=sl, height=sl, angle=0, ec="gray", fill=False, ls="--", lw=1)
    ax2.add_patch(rec)
    
    fig3, ax3 = plt.subplots(figsize=(16,9))
    ax3.vlines(phi_deg, 0, 1.05, color="dimgrey", linestyles='-', label="True DOAs")
    ax3.set_ylim(0, 1.05)
    ax3.set_xticks([0, 40, 90, 110, 150, 180])
    ax3.tick_params(labelsize=20)
    ax3.set_xlabel("Angle [deg]", fontsize=20)
    ax3.set_ylabel("Normalized value", fontsize=20)
    ax3.grid(ls="--")

    # proposed method
    for f in range(f_c.shape[0]):
        K_inv = np.linalg.inv(K(dim, k_max[f], r_u, r_u) + lam*np.eye(r_u.shape[0]))
        C = np.einsum("ijk,jl->ilk", np.einsum("ij,jkl->ikl", K_inv.T, get_matrix_for_arrival_power_estimation(k_max[f], vartheta_vec, r_u, dim)), K_inv)
        p = get_primary_field(c, k_lim[f], 0, phi_vec, r_u)
        p = add_noise(p, SNR=30)       
        start = time.time()
        P_p = np.matmul(np.matmul(C.T, p), p) # proposed method
        print("Proposed method", time.time() - start)
        P_p -= np.min(P_p)
        P_p /= np.max(P_p)
        ax3.plot(vartheta_deg, P_p, lw=2, markevery=5, label="Proposed ({} Hz)".format(f_c[f]))
        
    # MUSIC algorithm
    p = np.empty([T, U], dtype=np.float64)
    window = np.hanning(T)
    for f in range(f_c.shape[0]):
        index_range = np.array([int(np.ceil(2*f_low[f]*W/fs)), int(np.floor(2*f_high[f]*W/fs))])
        index_num = index_range[1] - index_range[0]
        for i in range(T):
            p[i] = get_primary_field(c, k_lim[f], t[i], phi_vec, r_u)
            p[i] = add_noise(p[i], SNR=30)
        start = time.time()
        p = p * np.tile(window.reshape(-1,1), (1,U))
        P = np.fft.rfft(p, axis=0)
        beta = np.zeros([index_num], dtype=np.float64)
        P_MU = np.empty([index_num, vartheta_num], dtype=np.float64)
        Rxx = np.einsum("ijk,ikl->ijl", P[index_range[0]:index_range[1]].reshape(index_num,U,1), np.conjugate(P[index_range[0]:index_range[1]]).reshape(index_num,1,U)) / index_num
        for w in range(index_num):
            k = 2*np.pi*(fs*(w+index_range[0])/(2*W))/c
            u,s,v = np.linalg.svd(Rxx[w]) # singular value decomposition
            En = u[:,phi_deg.shape[0]:]
            beta[w] = np.sum(s[:phi.shape[0]])
            for i in range(vartheta_num):
                a = np.exp(1j*k*np.matmul(r_u, vartheta_vec[i]))
                P_MU[w,i] = np.real(np.matmul(np.conjugate(a), a)/(np.matmul(np.matmul(np.matmul(np.conjugate(a), En), np.conjugate(En.T)),a)))
        P_MU = np.sum(P_MU*np.tile(beta.reshape(-1,1), (1, vartheta_num)), axis=0)
        print("MUSIC", time.time() - start)
        P_MU -= np.min(P_MU)
        P_MU /= np.max(P_MU)
        ax3.plot(vartheta_deg, P_MU, lw=2, markevery=5, label="MUSIC ({} Hz)".format(f_c[f]))
        
    ax3.legend(loc='upper left', fontsize=15)
    
    plt.show()
