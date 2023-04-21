# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 17:08:02 2023

@author: Takahiro Iwami
"""

#################################################################
#                      Experiment D
#################################################################

#================================================================
#                      Import modules
#================================================================
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jn

#================================================================
#                  Definition of Functions
#================================================================
def get_primary_field(k_max, phi_vec, r):
    p = np.zeros([r.shape[0]], dtype=np.float64)
    for i in range(phi_vec.shape[0]):
        p += (k_max/np.pi)*np.sinc(k_max*np.matmul(r,phi_vec[i])/np.pi)
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

def get_matrix_for_weighted_arrival_power_estimation(k_max, vartheta_vec, r, dim):
    r = np.transpose(np.tile(r, (r.shape[0],1,1)), axes=(1,0,2))
    r_p = np.transpose(r, axes=(1,0,2))
    diff = r - r_p
    return (k_max**2/(2*np.pi)**dim)*(2*np.sinc(k_max*np.matmul(diff,vartheta_vec.T)/np.pi)-np.sinc(k_max*np.matmul(diff,vartheta_vec.T)/(2*np.pi))**2)

def add_noise(p, SNR=30):
    noise = np.random.rand(p.size)-0.5
    gain = np.sum(np.abs(p))/(10**(SNR/20)*np.sum(np.abs(noise)))
    tmp = noise.reshape(p.shape)*gain
    return p+tmp

#================================================================
#                       Main routine
#================================================================

if __name__ ==  "__main__":
    np.random.seed(10) # fix seed
    dim = 2
    rho = 1.293
    kappa = 142.0e3
    c = np.sqrt(kappa/rho)
    f_high = 2500
    k_max = 2*np.pi*f_high / c
    sl = 0.6 # length of each side
    
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
    K_inv = np.linalg.inv(K(dim, k_max, r_u, r_u) + lam*np.eye(r_u.shape[0]))
    C = np.einsum("ijk,jl->ilk", np.einsum("ij,jkl->ikl", K_inv.T, get_matrix_for_arrival_power_estimation(k_max, vartheta_vec, r_u, dim)), K_inv) # matrix for power estimation
    D = np.einsum("ijk,jl->ilk", np.einsum("ij,jkl->ikl", K_inv.T, get_matrix_for_weighted_arrival_power_estimation(k_max, vartheta_vec, r_u, dim)), K_inv) # matrix for weighted power estimation
    
    # input signals
    p = get_primary_field(k_max, phi_vec, r_u)
    p = add_noise(p, SNR=30)
    
    # signal processing
    P = np.matmul(np.matmul(C.T, p), p)
    Pw = np.matmul(np.matmul(D.T, p), p)
    
    # preparing for drawing
    P -= np.min(P)
    P /= np.max(P)
    Pw -= np.min(Pw)
    Pw /= np.max(Pw)
    
    # drawing
    fig1, ax1 = plt.subplots(figsize=(8,6))
    ax1.vlines(phi_deg, 0, 1.05, color="dimgrey", linestyles='-', label="True DOAs")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks([0, 40, 90, 110, 150, 180])
    ax1.tick_params(labelsize=14)
    ax1.set_xlabel("Angle [deg]", fontsize=18)
    ax1.set_ylabel("Normalized power", fontsize=18)
    ax1.grid(ls="--")
    ax1.plot(vartheta_deg, P, lw=2, markevery=5, label="Normal", color="dimgray", ls="-.")
    ax1.plot(vartheta_deg, Pw, lw=2, markevery=5, label="Weighted", color="black", ls="-")
    ax1.legend(loc='upper right', fontsize=15)
    
    plt.show()