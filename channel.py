#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:34:09 2020

@author: tyuan
"""
import math
#from sympy import *
#import cvxpy as cp
import numpy as np
class Channel():
    def __init__(self):
        #carrier frequency
        self.f_c = 2e9#2Ghz
        self.f_c_mmwave = 38e9#2Ghz
        self.d_0 =5 #m
        self.light_speed = 3e8
        self.a = 10
        self.b=0.6
        
        self.zeta_LoS = 2
        self.zeta_NLoS = 2.4
        self.X_sigma_LoS = 5.3
        self.X_sigma_NLoS = 5.27
        self.X_0 = 20*math.log(self.d_0*self.f_c*4*math.pi/self.light_speed, 10)
        self.sigma_2 = math.pow(10,-95/10)/1000 # in W
        #3.1622776602e-13
        self.kappa = 0.2
        self.alpha = 2.3
        
        self.g_0 = 0.5e-10 #W
        self.p_los_U2V = -135 #dB -138
        self.p_los_U2R = -250
#        -250 #dB
        
        self.dis_max = 700 #m
        
    def R_U2R(self, height):
        di = height
        for distance in np.arange(self.dis_max, height, np.round((height-self.dis_max)/600, decimals=1)):
            if self.g_U2R(height, distance)[1] >= self.p_los_U2R:
                di = distance.copy()
                break
        return math.sqrt(pow(di,2)-pow(height,2)) 
        
    def R_U2V(self, height):
        di = height
        for distance in np.arange(self.dis_max,int(height),np.round((height-self.dis_max)/600, decimals=1)):
            if self.g_U2V(height, distance)[1] >= self.p_los_U2V:
                di = distance.copy()
                break
        return math.sqrt(pow(di,2)-pow(height,2)) 
#    
#    def R_R2R(self, height):
#        di = height
#        for distance in np.arange(self.dis_max,int(height),np.round((height-self.dis_max)/600, decimals=1)):
#            if self.g_R2R(distance)[1] >= self.p_los_RR2V:
#                di = distance.copy()
#                break
#        return math.sqrt(pow(di,2)-pow(height,2))  
#    
#    def g_R2R(self, distance): 
#
#        varphi_NLoS = -(10* self.zeta_NLoS *math.log(distance,10) + self.X_0 + self.X_sigma_NLoS)
#        g = math.pow(distance,-2)* varphi_NLoS
#        
#        return g
    
    def g_U2R(self, height, distance):  
        varphi_0 = math.pow(4*math.pi*self.f_c*self.d_0/self.light_speed,-2) #
        P_Los = 1/(1+self.a*math.exp(-self.b*(180/math.pi*math.asin(height/distance)-self.a)))
        P_hat = P_Los+ self.kappa*(1-P_Los)
        g = varphi_0*P_hat/math.pow(distance,self.alpha)#W
        g_db = 10*math.log(g)
        return g, g_db
    
    def g_U2V(self, height, distance): 
        X_0 = 20*math.log(self.d_0*self.f_c_mmwave*4*math.pi/self.light_speed, 10)
        P_LOS = 1/(1+self.a*math.exp(-self.b*(180/math.pi*math.asin(height/distance)-self.a)))
        P_NLOS = 1- P_LOS
        varphi_LoS = -(10* self.zeta_LoS *math.log(distance,10) + X_0 + self.X_sigma_LoS) #dB
        varphi_NLoS = -(10* self.zeta_NLoS *math.log(distance,10) + X_0 + self.X_sigma_NLoS)
        g_ = P_LOS*varphi_LoS +P_NLOS*varphi_NLoS #dB
        g = math.pow(10,g_/10)# W
        return g, g_
    
    
#    def sending_rate_U2R(self, height, distance):  
#        power_u2r = 1 #w
#        Bandwidth = 1e3 #10Mhz =10e3khz
#        varphi_0 = math.pow(4*math.pi*self.f_c*self.d_0/self.light_speed,-2) #
#        r_0_ = varphi_0*power_u2r/self.sigma_2 #in W/W
#        P_Los = 1/(1+self.a*math.exp(-self.b*(180/math.pi*math.asin(height/distance)-self.a)))
#        P_hat = P_Los+ self.kappa*(1-P_Los)
#        X_u2r = Bandwidth* math.log(1+r_0_*P_hat/(math.pow(distance,(self.alpha/2))),2)/1e3 #in Mbps
#        return X_u2r
#    
#    def sending_rate_U2S(self, height, distance):  
#        power_u2s = 1 #w
#        Bandwidth = 1e3 #10Mhz =10e3khz
#        varphi_0 = math.pow(4*math.pi*self.f_c*self.d_0/self.light_speed,-2) #
#        r_0_ = varphi_0*power_u2s/self.sigma_2 #in W/W
#        P_Los = 1/(1+self.a*math.exp(-self.b*(180/math.pi*math.asin(height/distance)-self.a)))
#        P_hat = P_Los+ self.kappa*(1-P_Los)
#        X_u2s = Bandwidth* math.log(1+r_0_*P_hat/(math.pow(distance,(self.alpha/2))),2)/1e3 #in Mbps
#        return X_u2s
#    
#    def sending_rate_U2U(self, distance):  
#        power_u2u = 1 #w
#        Bandwidth = 1e3 #10Mhz =10e3khz
#        varphi_0 = math.pow(4*math.pi*self.f_c*self.d_0/self.light_speed,-2) #    
#        g = math.pow(distance,-2)* varphi_0
#        X_u2u = Bandwidth* math.log(1+g*power_u2u/self.sigma_2)/1e3 #in Mbps
#        return X_u2u
#    
#    def sending_rate_G2G(self, distance, C_type): 
##        if C_type == 'R2V' or  C_type == 'S2v':
##            power = 0.1#w
##            Bandwidth = 10e3 #10Mhz =10e3khz
#        if C_type == 'R2R':
#            power = 0.5 #W
#            Bandwidth = 1e3 #10Mhz =10e3khz
#        elif C_type == 'R2S':
#            power = 1#w
#            Bandwidth = 1e3 #20Mhz =20e3khz
#        else:
#            print(C_type)
#            
#        # include S2X, R2X (X include R, S, V) with different power
#        varphi_NLoS = -(10* self.zeta_NLoS *math.log(distance,10) + self.X_0 + self.X_sigma_NLoS)
#        
#        g = math.pow(distance,-2)* varphi_NLoS
#        eta = math.pow(10, g/10)*power/self.sigma_2 # W/W
#        X_G2G = Bandwidth* math.log(1+eta)/1e3 #in Mbps
#        return X_G2G
    
    def sending_rate_B2V(self, Num, distance): 
        power = 1#w
        Bandwidth = 60e3 #60Mhz =60e3khz
        g = -(10* self.zeta_NLoS *math.log(distance,10) + self.X_0 )
        eta = math.pow(10, g/10)*power/self.sigma_2 # W/W
        X_G2G = Bandwidth/Num* math.log(1+eta)/1e3 #in Mbps
        return X_G2G
    
    def sending_rate_G2V(self, Num, distance): 
        power = 1#w
        Bandwidth = 1e3 #1Mhz =1e3khz
            
        # include S2X, R2X (X include R, S, V) with different power
#        varphi_NLoS = -(10* self.zeta_NLoS *math.log(distance,10) + self.X_0 + self.X_sigma_NLoS)
#        g = math.pow(distance,-2)* varphi_NLoS
        g = -(10* self.zeta_NLoS *math.log(distance,10) + self.X_0 )
        eta = math.pow(10, g/10)*power/self.sigma_2 # W/W
        X_G2G = Bandwidth/Num* math.log(1+eta)/1e3 #in Mbps
        return X_G2G
    
    def sending_rate_U2V(self, Num, height, distance):
        power_u2v = 1#w
        Bandwidth = 1e3 #2Mhz =2e3khz
        
        P_LOS = 1/(1+self.a*math.exp(-self.b*(180/math.pi*math.asin(height/distance)-self.a)))
        P_NLOS = 1- P_LOS
        
        varphi_LoS = -(10* self.zeta_LoS *math.log(distance,10) + self.X_0 + self.X_sigma_LoS) #dB
        varphi_NLoS = -(10* self.zeta_NLoS *math.log(distance,10) + self.X_0 + self.X_sigma_NLoS)
        
        g_ = P_LOS*varphi_LoS +P_NLOS*varphi_NLoS
        eta = math.pow(10,g_/10)*power_u2v/self.sigma_2 # W/W
        X_u2v = Bandwidth/Num* math.log(1+eta,2)/1e3 #in Mbps
    
        return X_u2v
    

if __name__ == "__main__":

    chanel=Channel()
    g, g_db = chanel.g_U2R(2, 4)
    g2,g2_db = chanel.g_U2V(4,4.1)
  
    import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # from matplotlib import cm
    h_low, h_high = 50, 100
#    r_low, r_high = 1, 50
#    fig = plt.figure()
#    ax = Axes3D(fig)
##    ax = fig.add_subplot(projection='3d')
#    # X, Y value
#    n = int((h_high - h_low)/100)
#    h = np.arange(h_low, h_high, 1)
#    r = np.arange(r_low, r_high, 1)
#    R = []
#    for h_ in h:
#        for r_ in r:
#            d_ =math.sqrt(pow(h_,2)+pow(r_,2))
#            if d_ > h_:
#                R.append(chanel.g_U2R(h_, d_)[1])
#            else:
#                R.append(0)
#    Z= np.array(R)
#    Z = Z.reshape((len(h),len(r))).T
#
#    R2 = []
#    for h_ in h:
#        for r_ in r:
#            d_ =math.sqrt(pow(h_,2)+pow(r_,2))
#            if d_ > h_:
#                R2.append(chanel.g_U2V(h_, d_)[1])
#            else:
#                R2.append(0)
#    Z2= np.array(R2)
#    Z2 = Z2.reshape((len(h),len(r))).T
#
#    h, r = np.meshgrid(h, r)    # x-y 平面的网格
#
##    ax.set_zlabel('label text flipped') 
#    ax.set_xlabel('UAV altitude (m)')
#    ax.set_ylabel('Horizontal distance (m)')
##    ax.set_zlabel('Channel Coefficient (1e-10)')#
#    ax.set_zlabel('Average path loss (dB)')#
##    ax.plot_surface(h, r, -Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'), alpha=0.5, label='U2R')
#    ax.plot_surface(h, r, -Z2, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'),label='U2V')
##    ax.legend()
##    ax.text2D(0.25, 0.5, "U2R", transform=ax.transAxes)
##    ax.text(1, 2.7, 4.3, "U2R")
##    ax.text(1, 2.7, -0.2, "U2V")
##    ax.text2D(0.05, 0.95, "2D Text", transform=ax.transAxes)
#    ax.azim = 225
#    plt.savefig('runs/P_los.eps', dpi=400)
#    plt.show()
#    
    fig2 = plt.figure()
    h2 = np.arange(h_low, h_high, 1)
    R_ = []
#    R2_ = []
    for h_ in h2:
        R_.append(chanel.R_U2V(h_))
#        R2_.append(chanel.R_U2R(h_))
        
    plt.plot(h2, R_, label ='U2V')
#    plt.plot(h2, R2_,'red', label ='U2R' )
    plt.xlabel("UAV altitude (m)")
    plt.ylabel("Coverage R (m)")
    plt.legend()
    plt.show()
    plt.savefig('runs/coverage.eps', dpi=400)
    
    Y =[]
    distance = range(20,100)
    fig = plt.figure()
    for dis in distance:
        X_U2V = chanel.sending_rate_G2V(20, dis) #in kbps
        Y.append(X_U2V)
    plt.plot(distance, Y, label ='U2V')
    
    X_U2R = chanel.sending_rate_U2R(40,50)
