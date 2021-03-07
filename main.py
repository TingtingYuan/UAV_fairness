#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:58:20 2019

@author: tyuan
"""

from Env_grid import Env, file_to_csv
import tensorflow as tf
import sys
import json
#import seaborn as sns
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from helper import setup_exp, setup_run
import numpy as np
import argparse

def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.set_defaults(render=False)
    return parser.parse_args(args)

def remove_zero_rows(X):
     # X is a scipy sparse matrix. We want to remove all zero rows from it
     nonzero_row_indice, _ = X.nonzero()
     unique_nonzero_indice = np.unique(nonzero_row_indice)
     return X[unique_nonzero_indice]

def playGame(DDPG_config, args=None, train_indicator=1, Fix_traffic=None, OPT = 0):   
    # train_indicator: 1 means Train, 0 means simply Run
    folder = setup_run(DDPG_config, 'U')
    Read_others_m = DDPG_config['Read_other_methods']
    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K 
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    K.set_session(sess)
       
    env = Env(DDPG_config, folder)
    env2 = Env(DDPG_config, folder)
    env3 = Env(DDPG_config, folder)
    env_tr = Env(DDPG_config, folder)
    env_tr_UAV = Env(DDPG_config, folder)
    env_fair = Env(DDPG_config, folder)
    env_tr_energy = Env(DDPG_config, folder)
    env_alpha2, env_inf=  Env(DDPG_config, folder), Env(DDPG_config, folder)
    
    env_enu = Env(DDPG_config, folder)

    action_dim, state_dim, state_dim2 = env.a_dim, env.s_dim, env.s_dim_2
    
    Branch = DDPG_config['Branch_NN']
    algo = DDPG(action_dim, state_dim, DDPG_config, state_dim2, Branch)
    # for SCV
    # v_p = algo.vehicle_data_dis(DDPG_config)
    # Train
    algo.train(env, DDPG_config, env_enu, env2, env3, env_tr, env_tr_UAV, env_fair, env_tr_energy, env_alpha2, env_inf, folder, Read_others_m)
    
    # export_path = env.folder
    algo.save_weights(env.folder)
           
    print("All is over")
    return env  
    
def Convert(string): 
    li = list(string.split(",")) 
    return li     
 
    
if __name__ == "__main__":
    
    # VANILLA
    if len(sys.argv) == 1:
        with open('DDPG.json') as jconfig:
            DDPG_config = json.load(jconfig)
        from DDPG.ddpg import DDPG, plot_com
        
        # DDPG_config['EXPERIMENT'] = setup_exp()
        MAX_Group = DDPG_config['GROUP']
        MAX_step = DDPG_config['MAX_STEPS']
        folder_data = DDPG_config['Folder_data']
        
        env = playGame(DDPG_config, None, 1, None, 0) # training

        Reward = file_to_csv(env.folder + 'TotalRewardLog.csv')
        reward_plot = Convert(Reward[1:-1])
        reward_plot = list(map(float, reward_plot))
        plot_com(env.folder, [reward_plot], MAX_Group, 1, "Reward",  "Reward","Episode",["DDPG"]) 

        print("Finish.")
        
        