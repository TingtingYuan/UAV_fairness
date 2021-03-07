#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:07:44 2020
# mean traffic, no interval difference
@author: eva
"""
import itertools
import numpy as np
from helper import pretty
import math
from channel import Channel
import pickle
        
OMTRAFFIC = 'Traffic.txt'
ACTIONLOG = 'Action.csv'
ACTIONLOG_ONE_TIME = 'Action_one_times.txt'
STATIONLOG = 'State.csv'
STATIONLOG_fix = 'State_fix.csv'

TRAFFICLOG = 'TrafficLog.csv'
BALANCINGLOG = 'BalancingLog.csv'
UTILOG = 'UtilizationLog.csv'
WHOLELOG = 'Log.csv'
OMLOG = 'omnetLog.csv'
TOPO = 'Networks/abilene.txt'
C_loation = 'Location_list.txt'
#candidate location of UAVs
   
# FROM MATRIX
def matrix_to_rl(matrix):
    return matrix[(matrix!=-1)]

matrix_to_log_v = matrix_to_rl

def matrix_to_omnet_v(matrix):
    return matrix.flatten()

def vector_to_file(vector, file_name, action):
    string = ','.join(pretty(_) for _ in vector)
    with open(file_name, action) as file:
        return file.write(string + '\n')
    
def dict_to_file(dictionary, file_name):
    file = open(file_name+'.pickle', 'wb')
    pickle.dump(dictionary, file)
    file.close() 
    
def file_to_dic(file_name):
    list_file = open(file_name+'.pickle','rb')
    list2 = pickle.load(list_file)
    return list2
    
def file_to_csv(file_name):
    # reads file, outputs csv
    with open(file_name, 'r') as file:
        return file.readline().strip().strip(',')
    
def csv_to_matrix(string, nodes_num):
    # reads text, outputs matrix
    v = np.asarray(tuple(float(x) for x in string.split(',')[:nodes_num**2]))
    M = np.split(v, nodes_num)
    return np.vstack(M)   

# TO RL
def rl_state(env):
    Work = []
    for u in env.U:
        if u:
            Work.append(1)
        else:
            Work.append(0)
    return np.concatenate((matrix_to_rl(env.state_grid/env.MAX_v_grid_num), matrix_to_rl(env.UAV_location/env.Grid_num_state),
                           matrix_to_rl(env.UAV_battery),  matrix_to_rl(np.array(Work))))


def Norm(rsu_lo):
    rsu_norm = (rsu_lo-rsu_lo.min(axis=0))/(rsu_lo.max(axis=0)-rsu_lo.min(axis=0))+0.01
    return rsu_norm


class Env():
    def __init__(self, DDPG_config, folder, UAV_num = -1):
        self.Battery = DDPG_config['Battery_update']
        self.Energe_efficient = DDPG_config['Energe_efficient']
        self.alpha = DDPG_config['alpha']
        if UAV_num ==-1:
            self.UAV_NUMBER_Initial =  DDPG_config['UAV_NUMBER'] 
        else:
            self.UAV_NUMBER_Initial = UAV_num
        self.Grid_num = 10
        self.Grid_num_state =  DDPG_config['Grid_num'] #10
        self.a = DDPG_config['Grid_num']
        self.Cell_length =  DDPG_config['Area_size']/self.Grid_num_state # length of each cell
        self.MAX_v_grid_num =  DDPG_config['MAX_vehicle_num_grid']

        self.ACTION_TYPE = DDPG_config['ACTION_TYPE']
        self.Action_precision = DDPG_config['Action_precision']
        
        self.Height_L = np.around(DDPG_config['HEIGHT_l']/self.Cell_length, self.Action_precision)
        self.Height_U = np.around(DDPG_config['HEIGHT_u']/self.Cell_length, self.Action_precision)

        self.N_grid = self.Grid_num_state/self.Grid_num
        self.charge_station = np.array([5.,5., 0])*self.N_grid
        self.RSU_location = np.array([[2,8,0], [1,1,0], [7,2,0]])*self.N_grid
        if self.N_grid == 1:
            self.BS_location =  np.array([[5, 5, 0]])*self.N_grid
        else:
            self.BS_location =  np.array([[2, 2,0], [8, 2,0], [5, 5, 0], [2, 8,0], [8, 8,0],  [6, 8,0]])*self.N_grid

        self.BS_Number = len(self.BS_location)
        self.RSU_Number = len(self.RSU_location)
        self.ACTIVE_NODES = self.BS_Number+self.RSU_Number
        
        self.R_RSU = DDPG_config['R_RSU']/self.Cell_length # 200m
        self.MAX_STEPS = DDPG_config['MAX_STEPS']
        self.Flying_speed = DDPG_config['Flying_speed']
        self.deta_t = DDPG_config['deta_t']
        
        self.UAV_B =  DDPG_config['UAV_B'] #wh
        self.folder = folder
        self.delta_engery = 1
        self.uav_dim = 3
        self.s_dim = self.Grid_num_state**2 + self.UAV_NUMBER_Initial*self.uav_dim + self.UAV_NUMBER_Initial*2
        self.s_dim_1 = self.Grid_num_state**2 
        self.s_dim_2 = self.UAV_NUMBER_Initial*self.uav_dim + self.UAV_NUMBER_Initial*2
        
        if self.ACTION_TYPE == 'CP':
            self.a_dim = self.UAV_NUMBER_Initial*self.uav_dim
        elif self.ACTION_TYPE == 'CP_2D':
             self.a_dim = self.UAV_NUMBER_Initial*(self.uav_dim-1)
        elif self.ACTION_TYPE == 'CP_charge':
            self.a_dim = self.UAV_NUMBER_Initial*self.uav_dim + self.UAV_NUMBER_Initial*2

        self.Grid_I_withoutU = self.Grid_access_point_noUAV()

    
    def reset(self, day, F_G):
        self.end_point = 0 #begin game
        self.U = np.full(self.UAV_NUMBER_Initial, True)# True is work       
        self.UAV_location = np.array( [self.charge_station.tolist()]*self.UAV_NUMBER_Initial)
        self.UAV_battery = np.ones(self.UAV_NUMBER_Initial) 
        # initial the batery of UAV is (100%, 100%, 100%)
        self.F_G = F_G[day]        
        state = self.get_data(0)
        self.UAV_NUMBER = self.UAV_NUMBER_Initial
        self.T_k= np.zeros([self.Grid_num_state, self.Grid_num_state])
        return state 
        
                    
    def step(self, ac, state, step_num, a_pre, Test = False):     
        V_g = state[:self.Grid_num_state**2]*self.MAX_v_grid_num
        self.state_grid  = V_g.reshape(self.Grid_num_state, self.Grid_num_state)
        self.vehicle_number = int(np.sum(self.state_grid))
        # use self.Grid_I_withoutU to get self.re
        self.get_grid_access_rate_grid(self.state_grid)
        r_pre = np.copy(self.re)
        self.capacity_access_grid_pre = np.copy(self.capacity_access_grid)
        action = ac.copy()
        #get real action， based on action self.UAV_NUMBER
        action_real = self.realaction(action) #self.U, self.UAV_Num
        self.upd_UAV_location(action_real)
        # calculate uav radius self.UAV_R_U2V
        self.UAV_cover(action_real)
        deta_f_time, ft_id, flying_time = self.flying_time(action_real, a_pre)   
        action_ =  np.array([self.charge_station.tolist()]*self.UAV_NUMBER_Initial)
        reward = 0
        if len(ft_id) == 0:
            reward_ = r_pre
        else:
            reward_ =  r_pre*flying_time[ft_id[0]]/self.deta_t
        for u_id, ft in enumerate(ft_id): # ft is the sequence of arrived work UAVs
            action_[ft][:] = np.copy(action_real[ft][:])
            self.Grid_access_point(action_) #self.Grid_I->based on UAV location
            self.get_grid_access_rate_grid(self.state_grid, action_, deta_f_time[u_id]/self.deta_t) #self.G_num_a, self.capacity_access_grid
            reward += (self.re - r_pre)/abs(r_pre) * deta_f_time[u_id]/self.deta_t
            reward_ += self.re * deta_f_time[u_id]/self.deta_t

        delta_engery = self.upd_env_Battery(action_real, a_pre, flying_time) # self.U update
        # if  self.Energe_efficient:
        #     reward /= delta_engery/self.UAV_B
        #     reward_ /= delta_engery/self.UAV_B]
        # self.re_energy = self.re/delta_engery
        new_state = self.get_data(step_num+1)

        if Test:
            return new_state, reward_, action_real, delta_engery, r_pre
        else:
            return new_state, reward, self.end_point, action_real,  self.UAV_battery
    
    def get_data(self, step):    
        if step >= self.MAX_STEPS:
            step_t = step-1
        else:
            step_t = step            
        self.state_grid = self.F_G[step_t]
        self.vehicle_number = int(np.sum(self.state_grid))
        self.step_num = step
        state = rl_state(self)
        return state   
                
    
    def flying_time(self, action_real, a_pre):
        flying_time=np.zeros(self.UAV_NUMBER_Initial)
        flying_time_seq=np.zeros(self.UAV_NUMBER_Initial)
        for uid, u in enumerate(action_real):
            if self.U[uid]:
                distance = np.linalg.norm(u - a_pre[uid])*self.Cell_length
                flying_time[uid] = distance/self.Flying_speed
                flying_time_seq[uid] = distance/self.Flying_speed
            else:
                distance = np.linalg.norm(self.charge_station - a_pre[uid])*self.Cell_length
                flying_time[uid] = distance/self.Flying_speed
                flying_time_seq[uid] = 10000000
        time_sort = np.sort(flying_time_seq)
        deta_f_time =[]
        for u_id in range(self.UAV_NUMBER-1):
            deta_f_time.append(time_sort[u_id+1]- time_sort[u_id])
        deta_f_time.append(self.deta_t-time_sort[self.UAV_NUMBER-1])
        ft_id = np.argsort(flying_time_seq)
        return deta_f_time, ft_id[:self.UAV_NUMBER], flying_time
    
    
    
    def step_with_traffic_a(self, a_pre, step_num, ac, No_engery= False, pre_return=False):
#        No_engery= true for testing for no_energy_effic
        action = np.copy(ac)
        self.get_data(step_num)
        self.get_grid_access_rate_grid(self.state_grid)
        r_pre = float(np.copy(self.re))
        self.capacity_access_grid_pre = np.copy(self.capacity_access_grid)
        self.UAV_NUMBER = self.UAV_NUMBER_Initial
        for u in range(self.UAV_NUMBER_Initial):
            if not self.U[u]:
                action[u][:] = list(self.charge_station)
                self.UAV_NUMBER -= 1

        action_real = np.array(action)
        self.upd_UAV_location(action_real)
        # calculate uav radius self.UAV_R
        self.UAV_cover(action_real)
        deta_f_time, ft_id, flying_time = self.flying_time(action_real, a_pre)
        action_ = np.array([self.charge_station.tolist()]*self.UAV_NUMBER_Initial)
        reward = 0
        # , reward_1, reward_2 = 0, 0, 0
        
        if len(ft_id) == 0:
            reward_1 = r_pre
        else:
            reward_1 = r_pre*flying_time[ft_id[0]]/self.deta_t
        for u_id, ft in enumerate(ft_id):
            action_[ft][:] = action_real[ft][:].copy()
            self.Grid_access_point(action_) #self.Grid_I->基于UAV location
            self.get_grid_access_rate_grid(self.state_grid, action_, deta_f_time[u_id]/self.deta_t) #self.G_num_a
            reward += (self.re - r_pre)/abs(r_pre) * deta_f_time[u_id]/self.deta_t
            reward_1 += self.re* deta_f_time[u_id]/self.deta_t
            # reward_2 += self.re* deta_f_time[u_id]/self.deta_t
        # reward_3 += reward_2
        delta_engery = self.upd_env_Battery(action_real, a_pre, flying_time) #self.U
        # if self.Energe_efficient and not No_engery:    
        #     reward /= delta_engery/self.UAV_B
            # reward_1 /= delta_engery/self.UAV_B

        new_state = self.get_data(step_num+1)
        # if pre_return:
        #     return reward, reward_1, reward_2, reward_3, action_real, new_state, delta_engery, self.re_thr, r_pre
        # else:
        #     return reward, reward_1, reward_2, reward_3, action_real, new_state, delta_engery, self.re_thr
        return reward, reward_1, action_real, delta_engery
        
    def UAV_cover(self, action):
        channel = Channel()
        R2 = []
        for u_id, u in enumerate(action):
            # charging and not server
            if (u == self.charge_station).all():
                r2 = 0
            else:                
                r2 = channel.R_U2V(u[-1]*self.Cell_length)/self.Cell_length
            R2.append(r2)
        self.UAV_R_U2V = np.array(R2)
        
    def get_fairness_Factor(self, work_time=1):
            G = np.copy(self.Grid_I)
            G[G<self.ACTIVE_NODES]=0
            G[G>=self.ACTIVE_NODES]=1
            T_k_pre = np.copy(self.T_k)
            T_k = np.copy(self.T_k)
            T_k += G*work_time
            C_fairness=T_k/self.MAX_STEPS
            f_fair = pow(np.sum(C_fairness),2)/(self.Grid_num_state*np.sum(pow(C_fairness,2)))
            self.re_fairness =f_fair*(np.sum(T_k -T_k_pre))
            self.T_k = np.copy(T_k)

    def get_grid_access_rate_grid(self, State_Grid, action=None, work_time=0): 
        if action is not None:
            self.Grid_access_down(self.Grid_I, State_Grid)   
            self.get_fairness_Factor(work_time)
        else:
            self.Grid_access_down(self.Grid_I_withoutU, State_Grid)
            
    def grid_num_flows(self, Grid, State_Grid):
        num = self.ACTIVE_NODES + self.UAV_NUMBER_Initial
        G_num_a = np.zeros(num)
        for i in range(num):
            G_num_a[i] = int(np.sum(State_Grid[Grid == i]))
        self.G_num_a = G_num_a   
        if np.sum(G_num_a) != np.sum(State_Grid):
            print("grid_num_flows")

    def Grid_access_down(self, Grid, State_Grid): 
        diff = 1/2
        self.grid_num_flows(Grid, State_Grid) #self.G_num_a
        channel = Channel()
        capacity_access = np.zeros([self.Grid_num_state, self.Grid_num_state])
        n = 0
        for g1 in range(self.Grid_num_state):
            for g2 in range(self.Grid_num_state):
                grid_lo = [g1+diff, g2+diff, 0]
                access_node = int(Grid[g1, g2])
                
                if State_Grid[g1,g2]== 0:
                    capacity_access[g1,g2] =0
                else:
                    if access_node >= self.ACTIVE_NODES: #UAV
                        u_lo = self.UAV_location[access_node-self.ACTIVE_NODES]
                        height = u_lo[-1]*self.Cell_length
                        distance = max(0.1*self.Cell_length, np.linalg.norm(u_lo-grid_lo)*self.Cell_length)
                        capacity_access[g1,g2] = channel.sending_rate_U2V(self.G_num_a[access_node], height, distance)
                        n+=1
                    elif access_node >= self.BS_Number: #RSU as AP
                        r_lo = self.RSU_location[access_node-self.BS_Number]
                        distance = max(0.1*self.Cell_length, np.linalg.norm(r_lo-grid_lo)*self.Cell_length)                       
                        capacity_access[g1,g2] = channel.sending_rate_G2V(self.G_num_a[access_node], distance)
                    else:
                        B_lo = self.BS_location[access_node]
                        distance = max(0.1*self.Cell_length, np.linalg.norm(B_lo-grid_lo)*self.Cell_length)
                        capacity_access[g1,g2] = channel.sending_rate_B2V(self.G_num_a[access_node], distance)
                            
        #capacity_access = np.clip( capacity_access, 0, 5)
        self.capacity_access_grid = np.copy(capacity_access)
        
        F1 = capacity_access*State_Grid
        
        if self.alpha==0:
            F = capacity_access*State_Grid 
        elif self.alpha==1:
            F = np.log(capacity_access+1)*State_Grid
        else:
            F_1 = np.where(capacity_access > 0, pow(capacity_access, 1-self.alpha)/(1-self.alpha), 0)
            F = F_1*State_Grid 
            
        self.re = np.sum(F)/self.vehicle_number   
        self.re_thr = np.sum(F1)/self.vehicle_number
        self.re_cov = np.sum(State_Grid[Grid >= self.ACTIVE_NODES]) 
        

        alpha_list = [2, 5]
        for a_id, alpah in enumerate(alpha_list):
            F_ = np.where(capacity_access > 0, pow(capacity_access, 1-alpah)/(1-alpah), 0)
            F_ *= State_Grid 
            if alpah == 2:
                self.re_2 =np.sum(F_)/self.vehicle_number
            else:
                self.re_inf =np.sum(F_)/self.vehicle_number    
                

    def action_emu_others(self, T_k):
        reward_cov, reward_fair =[], []
        positon_all = {}
        k = 0
        for i in range(int(self.Grid_num_state/self.N_grid)):
            for j in range(int(self.Grid_num_state/self.N_grid)):
                h = self.Height_U
                if self.UAV_NUMBER_Initial==5:
                    location = np.array([[i, j, h],list(self.charge_station), list(self.charge_station), list(self.charge_station), list(self.charge_station)])
                elif self.UAV_NUMBER_Initial==4:
                    location = np.array([[i, j, h],list(self.charge_station), list(self.charge_station), list(self.charge_station)])
                elif self.UAV_NUMBER_Initial==3:
                    location = np.array([[i, j, h],list(self.charge_station), list(self.charge_station)])
                elif self.UAV_NUMBER_Initial==2:
                    location = np.array([[i, j, h],list(self.charge_station)])
                else:
                    location = np.array([[i, j, h]])
                positon_all[k] = [i, j, h]
                k+=1
                self.upd_UAV_location(location)
                # calculate uav radius self.UAV_R
                self.UAV_cover(location)
                self.Grid_access_point(location)
                self.T_k = np.copy(T_k)
                self.get_fairness_Factor(1)
                self.re_cov = np.sum(self.state_grid[self.Grid_I >= self.ACTIVE_NODES]) 
                reward_cov.append(self.re_cov)
                reward_fair.append(self.re_fairness) 
                
        m = np.array(reward_cov).argsort()[::-1][0:self.UAV_NUMBER_Initial]
        action_cov = []
        for m_ in m :
            action_cov.append(positon_all[m_])
        
        m = np.array(reward_fair).argsort()[::-1][0:self.UAV_NUMBER_Initial]
        action_fair = []
        for m_ in m :
            action_fair.append(positon_all[m_])   
        return np.array(action_cov), np.array(action_fair)


    def action_emu(self, step_num, a_pre_engery_re, a_pre_fair, a_pre_thr, a_pre_heur, a_pre_alpha2, a_pre_inf,\
                    U_re_energy, U_fair, U_tr, U_cov, U_alpha2, U_inf, T_k, Test=False):
        
        action_cov_, action_fair_ = self.action_emu_others(T_k)
        
        
        self.UAV_cover(np.array([[0, 0, self.Height_U]]))
        R_max_UAV =  self.UAV_R_U2V
        self.get_data(step_num)
        state_grid =  self.state_grid
        self.get_grid_access_rate_grid(state_grid)
        positon_all = {}
        reward_re, reward_thr, reward_re_energy, reward_cov, reward_fair = [], [], [], [], []
        reward_alpha2, reward_alpha_inf = [], []
        
        clist =[]
        grid_lim = max(1,int(np.percentile(state_grid,80)))
        dis_lim_BS = 3 * self.N_grid
        cov_num_lim = np.mean(state_grid)*5
        loction_list = []

        First = True
        k_ = 0
        N_low = 8
        N_high = 25
        while (len(clist) <= N_low or len(clist) > N_high) and k_ < 5 and grid_lim>=1:
            k_+=1
            clist =[]
            position_can = []
            k = 0
            for i in range(int(self.Grid_num_state/self.N_grid)):
                for j in range(int(self.Grid_num_state/self.N_grid)):
                        h = self.Height_U
                        No = False
                        if Test:
                            for b in self.BS_location:
                                distance = np.linalg.norm(np.array([i,j,0]) - np.array(b))
                                if distance <= dis_lim_BS:
                                    No = True
                                    break
                            if not No:
                                for r in self.RSU_location:
                                    distance = np.linalg.norm(np.array([i,j,0]) - np.array(r))
                                    if distance < 2*self.R_RSU:
                                        No = True
                                        break
                        if not No and state_grid[i,j] > grid_lim:
                           if np.sum(state_grid[np.clip(i - int(R_max_UAV),0, self.Grid_num_state-1): np.clip(i + int(R_max_UAV),0, self.Grid_num_state-1),\
                                   np.clip(j - int(R_max_UAV),0, self.Grid_num_state-1): np.clip(j + int(R_max_UAV),0, self.Grid_num_state-1)]) >= cov_num_lim:
                                position_can.append([i, j, h])
                                clist.append(k)
                        if First:
                            positon_all[k] = [i, j, self.Height_U]
                        k+=1
            First = False  
            if  len(clist) <= N_low:
                grid_lim -= 1
                if Test:
                    dis_lim_BS -= 0.2
                cov_num_lim -= 2
            if len(clist) > N_high:
                grid_lim += 1
                if Test:
                    dis_lim_BS += 0.2
                cov_num_lim += 2
        if len(clist) > N_high:
            clist = clist[:N_high]
        
        if len(clist)<self.UAV_NUMBER_Initial+1:
            print("max cov")
            clist = state_grid.flatten().argsort()[-self.UAV_NUMBER_Initial-4:][::-1].tolist()
        print("positon_can: " + str(len(clist)))
        
        Reward = np.zeros([k]*self.UAV_NUMBER_Initial)
        
        if self.UAV_NUMBER_Initial>1:
            List = list(itertools.combinations(clist, self.UAV_NUMBER_Initial))
            dis_lim2 = 3+1
            N_p = 0
            N_p_low = min(50, 10*self.UAV_NUMBER_Initial)
            N_p_high = min(150, 20*self.UAV_NUMBER_Initial)
            k_ = 0
            while (N_p < N_p_low or N_p > N_p_high) and 1 < dis_lim2<= 5 and k_<4: 
                k_ += 1
                if N_p < N_p_low:
                    dis_lim2 -= 1
                elif N_p > N_p_high:
                    dis_lim2 += 1
                N_p = 0
                location = []
                loction_list = []
                for L_id, L in enumerate(List):
                    P = []
                    for l_id, l in enumerate(L):
                        P.append(positon_all[l])
                    No = False
                    for i in range(self.UAV_NUMBER_Initial-1):
                        distance = np.linalg.norm(np.array(P[i]) - np.array(P[i+1]))
                        if distance < dis_lim2:
                            No = True
                            break
                    if not No:
                        N_p += 1  
                        location = np.array(P)
                        loction_list.append(location)
            print("possible action List length: " +str(N_p))

        if self.UAV_NUMBER_Initial<=1 or len(loction_list) <2 or N_p<1:
            if self.UAV_NUMBER_Initial>1:
                print("list is none")
            List = list(itertools.combinations(clist, self.UAV_NUMBER_Initial))
            location = []
            N_p = 0
            for L_id, L in enumerate(List):
                P = []
                for l_id, l in enumerate(L):
                    P.append(positon_all[l])
                N_p += 1  
                location = np.array(P)
                loction_list.append(location)
            print("New possible action List length: " +str(N_p))
                        
        loction_list.append(action_cov_)
        loction_list.append(action_fair_)
        for location in loction_list:
            self.upd_UAV_location(location)
            # calculate uav radius self.UAV_R
            self.UAV_cover(location)
            self.Grid_access_point(location)
            self.T_k = np.copy(T_k)
            self.get_grid_access_rate_grid(self.state_grid, location, 1)
 
            reward_re.append(self.re) 
            reward_cov.append(self.re_cov)
            reward_fair.append(self.re_fairness)               
            reward_thr.append(self.reward_with_fly_time(location, U_tr, a_pre_thr, 1))
            reward_re_energy.append(self.reward_with_fly_time(location, U_re_energy, a_pre_engery_re, 3))
            reward_alpha2.append(self.reward_with_fly_time(location, U_alpha2, a_pre_alpha2, 5))
            reward_alpha_inf.append(self.reward_with_fly_time(location, U_inf, a_pre_inf, 6))

            Reward[L] = self.re
                
        m = np.argmax(reward_re)  
        action_re = loction_list[m]
        
        m = np.argmax(reward_thr)  
        action_thr = loction_list[m] 
        
        m = np.argmax(reward_cov)  
        action_cov = loction_list[m]  
        
        m = np.argmax(reward_fair)  
        action_fair = loction_list[m]  
        
        m = np.argmax(reward_re_energy)  
        action_re_energy = loction_list[m]

        m = np.argmax(reward_alpha2)  
        action_alpha2 = loction_list[m]
        
        m = np.argmax(reward_alpha_inf)  
        action_alpha_inf = loction_list[m]
              
        if Test:            
            return action_re, action_thr, action_alpha2, action_alpha_inf, action_cov, action_fair, action_re_energy
        else:
            return action_re, action_thr, action_cov, action_fair, action_re_energy, action_alpha2, action_alpha_inf, Reward, positon_all
   

    def reward_with_fly_time(self, lo, U_work, a_pre, type_reward = 0, u_id_try = -1):
        action_pre = np.copy(a_pre)
        if u_id_try !=-1:
            for u_id in range(self.UAV_NUMBER_Initial):
                if u_id != u_id_try:
                    action_pre[u_id][:] = list(self.charge_station)
                    
        location = np.copy(lo)
        if (U_work==False).all():
            return 0
        for uid, U_value in enumerate(U_work):
            if not U_value:
                location[uid][:] = list(self.charge_station)
        
        deta_f_time, ft_id, flying_time = self.flying_time(location, action_pre)
        delta_engery = self.upd_env_Battery(location, action_pre, flying_time, Try=True) # self.U
        
        action_ = np.array( [self.charge_station.tolist()]*self.UAV_NUMBER_Initial)       
        self.get_grid_access_rate_grid(self.state_grid)

        reward = 0
        for u_id, ft in enumerate(ft_id):
            if not (location[ft][:] == self.charge_station).all():
                action_[ft][:] = location[ft][:]
                self.Grid_access_point(action_) #self.Grid_I->based on UAV location
                self.get_grid_access_rate_grid(self.state_grid, action_,deta_f_time[u_id]/self.deta_t) #self.G_num_a
                if type_reward == 1:
                    r = self.re_thr
                elif type_reward == 2:  
                    r = self.re_fairness
                elif type_reward == 3:   
                    r = self.re
                elif type_reward == 4: 
                    r = self.re_cov
                elif type_reward == 5: 
                    r = self.re_2
                elif type_reward == 6: 
                    r = self.re_inf            
                reward += r * deta_f_time[u_id]/self.deta_t
        if self.Energe_efficient:
            reward = reward/delta_engery

        return reward
        

    def realaction(self, action):
        self.UAV_NUMBER = self.UAV_NUMBER_Initial 
          
        if self.ACTION_TYPE == 'CP':
            action_real = action[:self.UAV_NUMBER_Initial*self.uav_dim].reshape(self.UAV_NUMBER_Initial, self.uav_dim)
            for i in range(self.UAV_NUMBER_Initial):
                if not self.U[i]:
                    action_real[i][:] = list(self.charge_station)
                    self.UAV_NUMBER -= 1
                else:
                    action_real[i][:] *= [self.Grid_num_state, self.Grid_num_state, (self.Height_U-self.Height_L)]
                    action_real[i][-1] += self.Height_L
            # action_real = np.around(action_real, self.Action_precision) 
            # action_real[:,:-1] = np.around(action_real[:,:-1], self.Action_precision)
        
        elif self.ACTION_TYPE == 'CP_2D':      
            action_real = action[:self.UAV_NUMBER_Initial*(self.uav_dim-1)].reshape(self.UAV_NUMBER_Initial, self.uav_dim-1)
            action_real = np.append(action_real, [[self.Height_U]]*self.UAV_NUMBER_Initial, axis = 1)
            for i in range(self.UAV_NUMBER_Initial):
                if self.Battery and (action_real[i][0]<=0.01 and action_real[i][1]<=0.01):
                    action_real[i][:] = list(self.charge_station)
                else:                        
                    action_real[i][:-1] *= [self.Grid_num, self.Grid_num]
            # action_real = np.around(action_real, self.Action_precision)
            
        elif self.ACTION_TYPE == 'CP_charge':
            action_real = action[:self.UAV_NUMBER_Initial*self.uav_dim].reshape(self.UAV_NUMBER_Initial, self.uav_dim)
            #last several bit is charge or not
            for i in range(self.UAV_NUMBER_Initial):
                a_1 = self.UAV_NUMBER_Initial*3+2*i
                a_2 = self.UAV_NUMBER_Initial*3+2*i+1
                if action[a_1]> action[a_2]:
                    action_real[i][:] *= [self.Grid_num, self.Grid_num,  (self.Height_U-self.Height_L)]
                    action_real[i][-1] += self.Height_L
                else:
                    #charging later is bigger
                    action_real[i][:] = self.charge_station
            # action_real = np.around(action_real, self.Action_precision)

        return action_real    

    # update the location of UAVs
    def upd_UAV_location(self, action):
        self.UAV_location = action
        
        # update the battery of UAVs
    def upd_env_Battery(self, action, a_pre, flying_time, Try = False):
        Delta_engery = 0
        for uid, u in enumerate(action):
            if self.U[uid]:
                cost_p = self.Battery_cost(uid, flying_time[uid])
            else: #cost for fly back
                cost_p = self.Battery_cost(uid, flying_time[uid], False)
            Delta_engery += cost_p
            if not Try and self.Battery:
                if not self.U[uid]:
                    self.UAV_battery[uid] = 1
                    self.U[uid] = True
                else:
                    self.UAV_battery[uid] = (self.UAV_battery[uid]*self.UAV_B -cost_p)/self.UAV_B
                if self.UAV_battery[uid] <= 0.1: # go to charge at next step
                    self.U[uid] = False   

        Delta_engery = max(Delta_engery, 1)
        return Delta_engery
    
    def Battery_cost(self, uid, flying_time, Commu = True):
        D_t =self.deta_t #s
        d_t = flying_time #s
        v = self.Flying_speed#m/s
        P0 = 79.85628 #J/s
        P1 = 88.6279 #J/s
        P2 = 0.01848# 
        PCV = 1 #w=J/s
        v0 = 120 #m/s
        v1 = 4.03 
        N_c = 0
        # power of flying with speed v
        Pv = P0*(1+3*pow(v,2)/pow(v0,2))+P1*math.sqrt(math.sqrt(1+pow(v,4)/(4*pow(v1,4)))-pow(v,2)/(2*pow(v1,2)))+1/2*P2*pow(v,3)
        if Commu:
            le = np.sum(self.state_grid[self.Grid_I == uid+self.ACTIVE_NODES])
            N_c += le
            cost_p = d_t*Pv+(P0+P2)*(D_t-d_t) + PCV*N_c
        else:
            cost_p = d_t*Pv+(P0+P2)*(D_t-d_t)
        return cost_p
        
        
        
    def Grid_access_point_noUAV(self):
        #默认是BS
        diff = 1/2
        G_without_UAV = np.zeros([self.Grid_num_state, self.Grid_num_state])
        for g1 in range(self.Grid_num_state):
            for g2 in range(self.Grid_num_state):
                l1 = g1+diff
                l2 = g2+diff
                d = []
                for r_id, r in enumerate(self.RSU_location):
                    distance = np.linalg.norm(r-[l1, l2, 0])
                    d.append(distance)
                minpos = d.index(min(d))
                if min(d) <= self.R_RSU:
                    G_without_UAV[g1, g2] = minpos+ self.BS_Number
                else:
                    d = []
                    for b_id, b in enumerate(self.BS_location):
                        distance = np.linalg.norm(b-[l1, l2, 0])
                        d.append(distance)
                    minpos = d.index(min(d))
                    G_without_UAV[g1, g2] = minpos
        self.G_without_UAV = G_without_UAV
        return G_without_UAV

            
        
    def Grid_access_point(self, UAV_location):
        dis_menory = {}
        diff = 1/2
        G = np.copy(self.G_without_UAV)
        for u_id, u in enumerate(UAV_location):
            if not (u == self.charge_station).all():
                x_low = max(0, int(u[0]-self.UAV_R_U2V[u_id])-1)
                x_high = min(self.Grid_num_state-1,1+math.ceil(u[0]-self.UAV_R_U2V[u_id]))
                y_low = max(0, int(u[1]-self.UAV_R_U2V[u_id])-1)
                y_high =  min(self.Grid_num_state-1,1+math.ceil(u[1]-self.UAV_R_U2V[u_id]))
                for x in range(x_low, x_high+1):
                    for y in range(y_low, y_high+1):
                        distance = np.linalg.norm(u-[x+diff, y+diff, 0]) 
                        if distance<=self.UAV_R_U2V[u_id]:
                            if (x,y) not in dis_menory.keys():
                                dis_menory[(x,y)] = []
                            dis_menory[(x,y)].append(distance)
                            if (distance<= dis_menory[(x,y)]).all():
                                G[x, y] = u_id + self.ACTIVE_NODES
        self.Grid_I = G
