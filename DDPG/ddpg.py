# mean traffic, no interval difference
#import sys
import numpy as np
# import numpy.random as nr
import matplotlib.pyplot as plt
import seaborn as sns
#from .ou import OUNoise
#from tqdm import tqdm
from .actor import Actor
from .critic import Critic
#from utils.stats import gather_stats
# from utils.networks import tfSummary, OrnsteinUhlenbeckProcess
from utils.memory_buffer import MemoryBuffer #priority
from Env_grid import vector_to_file, dict_to_file, file_to_dic, file_to_csv
#from itertools import combinations, permutations
#import pickle
import json

import os


def get_key (dict, value):
    return [k for k, v in dict.items() if v == value]
           
def Convert(string): 
    li = list(string.split(",")) 
    li_d = [float(i) for i in li]
    return li_d 

def plot_com(plots_dir, data_set, num1, num2, name, y_label, x_label ="Episode", legends = ["DDPG", "Fix", "OPT", "MCN", "THR"], range_y=None):
#    get_color = lambda : "#" + "".join(np.random.choice(list("02468acef"), size=6))
#    colors = sns.color_palette('colorblind')
    colors = ['r', 'royalblue', 'c', 'orange', 'lightgreen', 'lightcoral' ]
#    markers = ['s', 'o', '^', 'v', '<', '*','.']
    a4_dims = (9, 5)
    plt.figure(figsize=a4_dims)
    Episode = 0
    line=[]
    for i, data in enumerate(data_set):
        data_new = []
        if num1>1:
            for j in range(0,np.array(data).size,num1):
                data_new.append(np.mean(data[j:j+num1]))
        else:
            data_new = data
        if i == 0:
            Episode = int(np.array(data_new).size/num2)
        data_new = np.reshape(data_new[:num2*Episode],(Episode, num2)).T
        l = sns.tsplot(data_new, color=colors[i], legend = True)
#        , marker = markers[i])
#     get_color()  
        line.append(l)
#    plt.show()
    plt.legend(l, labels=legends)
    plt.rc('xtick', labelsize=18) 
    plt.rc('ytick', labelsize=18) 
    plt.rc('axes', labelsize=18) 
    plt.rc('axes', titlesize=18) 
    plt.rc('legend', fontsize=18)
    
    if Episode < 500:
        scale = 50
    elif Episode < 1000:
        scale = 100
    elif Episode < 2000:
        scale = 200
    elif Episode < 4000:
        scale = 400
    else:
        scale = 1000
        
    scale_ls = np.arange(0, Episode, scale)
#                         int(Episode/5))
    index_ls = scale_ls*num2*2
    plt.xticks(scale_ls, index_ls, fontsize=16)
    if range_y!=None:
        plt.ylim((0, range_y))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(plots_dir+ name+ 'line_figure.pdf', dpi=400)
    
def fx_normal(x):
    a = 20
    mu = 15
    return 1/(np.sqrt(2*np.pi)*a)*np.exp(-(x-mu)*(x-mu)/(2*a*a))

class DDPG:
    """ Deep Deterministic Policy Gradient (DDPG) Helper Class
    """

    def __init__(self, act_dim, env_dim, DDPG_config, state_dim2=0, Branch=False, act_range=1):
#        buffer_size = 20000, gamma = 0.99, lr = 0.001, tau = 0.001
        """ Initialization
        """
        
        lr_a = DDPG_config['LRA']
        lr_c = DDPG_config['LRC']
        tau  =  DDPG_config['TAU']
        gamma = DDPG_config['GAMMA']
        LRDecay = DDPG_config['LRDecay']
        buffer_size = DDPG_config['BUFFER_SIZE']
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.act_range = act_range
        self.env_dim = env_dim
        self.state_dim2 = state_dim2
#        (k,) + env_dim
        self.gamma = gamma
        self.lr = lr_a
        self.Branch = Branch
        # Create actor and critic networks
        self.actor = Actor(self.env_dim, act_dim, act_range, lr_a, tau, state_dim2, LRDecay, Branch)
        self.critic = Critic(self.env_dim, act_dim, lr_c, tau,state_dim2, LRDecay, Branch)
        self.buffer = MemoryBuffer(buffer_size)
        self.Energe_efficinet =  DDPG_config['Energe_efficient']
        N = DDPG_config['UAV_NUMBER']
        if self.Energe_efficinet:
            self.times_plot = 0.1
        else:
            # if DDPG_config['alpha'] ==1:
            #     self.times_plot = 1
            # else:
            if N == 1:   
                self.times_plot = 1
            elif N == 2:
                self.times_plot = 3
            elif N == 3:
                self.times_plot = 2
            else:
                self.times_plot = 1
            
        
    def policy_action(self, s):
        """ Use the actor to predict value
        """
        if self.Branch:
            return self.actor.predict_2b(s)[0]
        else:          
            return self.actor.predict(s)[0]

    def bellman(self, rewards, q_values, dones):
        """ Use the Bellman Equation to compute the critic target
        """
        critic_target = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                critic_target[i] = rewards[i]
            else:
                critic_target[i] = rewards[i] + self.gamma * q_values[i]
        return critic_target

    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """
        self.buffer.memorize(state, action, reward, done, new_state)

    def sample_batch(self, batch_size):
        return self.buffer.sample_batch(batch_size)

    def update_models(self, states, actions, critic_target, Grad_clip=0):
        """ Update actor and critic networks from sampled experience
        """
        # Train critic
        if not self.Branch:
            loss = self.critic.train_on_batch(states, actions, critic_target)
            # Q-Value Gradients under Current Policy
            actions = self.actor.model.predict(states)
            grads = self.critic.gradients(states, actions)
        else:
            state_size2 = self.state_dim2
            state_size1 = self.env_dim-state_size2
            state1 = states[:,:state_size1]
            state2 = states[:,state_size1:]
            new_states =[state1, state2]
            
            loss = self.critic.train_on_batch_2b(states, actions, critic_target)

            actions = self.actor.model.predict(new_states)
#            actions = self.actor.predict_2b(states)
            grads = self.critic.gradients_2b(states, actions)
#        acs = actions 
        
        if Grad_clip > 0:
           grads = np.clip(grads, -Grad_clip, Grad_clip)
        # Train actor
        self.actor.train(states, actions, np.array(grads).reshape((-1, self.act_dim)))
        # Transfer weights to target networks at rate Tau
        self.actor.transfer_weights()
        self.critic.transfer_weights()
        break_mark = False
        if loss < 0.00001:
            break_mark = True
        if loss > 200:
            break_mark = True
        print("Loss: "+ str(loss))
        print("Grad: " + str(np.mean(grads)))
        return break_mark
        
    def Grid_all_File(self, Grid_num, MAX_GROUP, MAX_STEPS):
#        dic2 = './DATA/Flow_File_Df_10km_2/'
        dic2 = './DATA/Flow_File_Df_10km_Rome/'
        V_dic = {}
        F_G = {}
        V_num = []
        for day in range(MAX_GROUP):
            with open(dic2+str(day)+'_Vehicle.json','r') as f:
                diction = json.load(fp=f)
            V_dic[day] = diction
            if len(diction.keys())!= MAX_STEPS:
                print("error")
        for day in range(MAX_GROUP):
            f_g ={}
            V_n = []
            for step in range(MAX_STEPS):
                vehicle_location =  np.array(V_dic[day][str(step)])[:,:2]/100*Grid_num
                vehicle_number = len(vehicle_location) 
                V_n.append(vehicle_number)
                v_loca = vehicle_location
                v_lo =v_loca.astype(np.int32)
                flow_n_grid = np.zeros([Grid_num, Grid_num])
                for f in range(vehicle_number):
                    source = v_lo[f]
                    source_x = np.clip(source[0],0, Grid_num-1)
                    source_y = np.clip(source[1] ,0, Grid_num-1)
#                    if source_x < Grid_num and source_y < Grid_num:
                    flow_n_grid[source_x, source_y] += 1                                       
                f_g[step] = flow_n_grid  
            F_G[day] = f_g
            V_num.append(V_n)
        return F_G, V_num
    


    def Grid_all_generate(self, Grid_num, MAX_GROUP, MAX_STEPS, Max_num):
        F_G = {}
        a=[[2]*3, [1]*8,[2]*6,[3]*4,[4]*4,[5]*5,[6]*4,[7]*3,[6]*5,[5]*6,[4.5]*10,[5]*6,[6]*5,[7]*3,[6]*5,[5]*3,[4]*4,[3]*4,[2]*4,[1]*6]
        loc_1 = [y for x in a for y in x]
        b=[[2]*3, [1]*9,[2]*8,[3]*5,[4]*6,[5]*5,[6]*4,[5]*8,[4]*10,[5]*8,[6]*5,[5]*6,[4]*6,[3]*6,[2]*6,[1]*6]
        loc_2 = np.array([y for x in b for y in x])/1.2
        for day in range(MAX_GROUP):
            f_g ={}
            G = []
            for step in range(MAX_STEPS):
#                fx_normal(step)*300
                if day <5:
                    loc_ = loc_1
                else:
                    loc_ = loc_2
                loc =loc_[step]
                F_G_day = np.random.normal(loc, scale= 5, size=(Grid_num, Grid_num))
                F_G_day = np.clip(F_G_day.astype(np.int32), 0, Max_num)
                I_1 = np.random.randint(0, Grid_num, (70,1))
                I_2= np.random.randint(0, Grid_num, (70,1))
                F_G_day[I_1, I_2] = 0
#                F_G_day[I_1[20:30], I_2[20:30]] = 3
                F_G_day[I_1[10:20], I_2[10:20]] = 2
                F_G_day[I_1[:10], I_2[:10]] = 1
                f_g[step] = F_G_day
                G.append(np.sum(F_G_day))
            F_G[day] = f_g
#            plt.plot(G)
#            scale_ls = np.arange(0,97,12)
#            index_ls = ['0:00','3:00','6:00','9:00','12:00','15:00','18:00', '21:00', '24:00']
#            plt.xticks(scale_ls,index_ls) 
#            plt.show() 
        return F_G
    
    def Pre_Momory(self, env, action, o_state, n_state, reward):
        act_buffer = np.copy(action)
        for i in range(env.UAV_NUMBER_Initial):
            if (act_buffer[i] == env.charge_station).all():
                continue
            else:
                act_buffer[i][-1] -= env.Height_L  
                act_buffer[i] /= [env.Grid_num_state, env.Grid_num_state, (env.Height_U-env.Height_L)]
        a = act_buffer.reshape(env.UAV_NUMBER_Initial*3)
#        self.memorize(o_state, a, 20*reward, 0, n_state)
        return [o_state, a, reward, 0, n_state]
    
    
    def Other_methods(self, F_G, MAX_STEPS, UAV_num, env_emu, env_cov, env3,\
                      env_tr,env_tr_UAV,env_fair, env_re_energy, env_alpha2, env_inf, MAX_Group, folder):                         
    
        Reward_heru, Reward_opt, Reward_tr, Reward_re_energy, Reward_fair = [], [], [], [], []
        Reward_alpha2, Reward_inf = [], []
        Capacity_grid_heru, Capacity_grid_opt, Capacity_grid_tr, \
         Capacity_grid_re_energy, Capacity_grid_fair = {},{},{},{},{}
        Capacity_grid_alpha2, Capacity_grid_inf = {},{}
        E_tr, E_heru, E_fair, E_re_energy =[],[],[],[]
        E_alpha2, E_inf = [], []       
        Flow_grid ={}       
        R_t_opt_all, R_t_tr_all,R_t_re_energy_all, R_t_heru_all, R_t_fair_all = [], [], [], [], []
        R_t_alpha2_all, R_t_inf_all = [], []
        for day in range(MAX_Group):  
            total_reward_heru = 0
            total_reward_opt = 0
            total_reward_tr = 0
            total_reward_alpha2, total_reward_inf = 0,0
            total_reward_fair, total_reward_re_energy = 0, 0
            env_cov.reset(day, F_G)
            env3.reset(day, F_G)
            env_tr.reset(day, F_G)
            env_tr_UAV.reset(day, F_G)
            env_fair.reset(day, F_G)
            env_re_energy.reset(day, F_G)
            env_alpha2.reset(day, F_G)
            env_inf.reset(day, F_G)
            env_emu.reset(day, F_G)
            a_t_real_tr, a_t_real_re_energy, a_t_real_fair=\
                np.zeros(([UAV_num,3])), np.zeros(([UAV_num,3])), np.zeros(([UAV_num,3]))
            a_t_real_cov, a_t_real_alpha2, a_t_real_inf = np.zeros(([UAV_num,3])), np.zeros(([UAV_num,3])), np.zeros(([UAV_num,3]))
            
            for s in range(MAX_STEPS): 
                action_re, action_thr, action_cov, action_fair,\
                action_re_energy, action_alpha2, action_inf, re_fix, loction_list= env_emu.action_emu(
                       s, a_t_real_re_energy, a_t_real_fair, a_t_real_tr, a_t_real_cov, a_t_real_alpha2, a_t_real_inf,\
                       env_re_energy.U, env_fair.U, env_tr.U,  env_cov.U, env_alpha2.U, env_inf.U, env_fair.T_k)

                r_t_tr,r_t_tr2, a_t_real_tr, delta_engery_tr = env_tr.step_with_traffic_a(a_t_real_tr, s, action_thr)
                
                r_t_re_energy,r_t_re_energy2, a_t_real_re_energy, delta_engery_re_energy = env_re_energy.step_with_traffic_a(
                                                a_t_real_re_energy, s, action_re_energy)
                r_t_fair, r_t_fair2, a_t_real_fair, delta_engery_fair = env_fair.step_with_traffic_a(
                                                a_t_real_fair, s, action_fair)            
                r_t_alpha2, r_t_alpha22, a_t_real_alpha2, delta_engery_alpha2 = env_alpha2.step_with_traffic_a(
                                                a_t_real_alpha2, s, action_alpha2)             
                r_t_inf, r_t_inf2, a_t_real_inf, delta_engery_inf = env_inf.step_with_traffic_a(
                                                a_t_real_inf, s, action_inf)                
                r_t_cov, r_t_cov2, a_t_real_cov, delta_engery_cov = env_cov.step_with_traffic_a(a_t_real_cov, s, action_cov)
                
                
                R_t_tr_all.append(r_t_tr2)
                R_t_heru_all.append(r_t_cov2)
                # R_t_opt_all.append(r_t_opt2)
#                R_t_tr_UAV_all.append(r_t_tr_UAV2)
                R_t_fair_all.append(r_t_fair2)
                R_t_re_energy_all.append(r_t_re_energy2)
                R_t_alpha2_all.append(r_t_alpha22)
                R_t_inf_all.append(r_t_inf2)
                
                E_tr.append(delta_engery_tr)
                E_heru.append(delta_engery_cov)
                E_fair.append(delta_engery_fair)
                E_re_energy.append(delta_engery_re_energy)
                E_alpha2.append(delta_engery_alpha2)
                E_inf.append(delta_engery_inf)
                
                total_reward_tr += r_t_tr
                total_reward_heru += r_t_cov
                total_reward_fair += r_t_fair
                total_reward_re_energy += r_t_re_energy
                total_reward_alpha2 += r_t_alpha2
                total_reward_inf += r_t_inf
            

                R_ = r_t_re_energy
                     
                print("Step "+str(s)+ ": action:" 
                      +str(np.around(a_t_real_cov.reshape(1, UAV_num*3)[0], decimals=2))
                      +str(a_t_real_re_energy.reshape(1, UAV_num*3)[0])
                      +str(a_t_real_fair.reshape(1, UAV_num*3)[0])
                      +" MC: " + str(np.around(self.times_plot*r_t_cov, decimals=3))
                      +" Greedy: "+ str(np.around(self.times_plot*R_, decimals=3))
                      +" FC: "+ str(np.around(self.times_plot*r_t_fair, decimals=3)))
                
                      
                Capacity_grid_heru[day, s] = env_cov.capacity_access_grid
                Capacity_grid_tr[day, s] =  env_tr.capacity_access_grid
                Capacity_grid_re_energy[day, s] =  env_re_energy.capacity_access_grid
                Capacity_grid_fair[day, s] =  env_fair.capacity_access_grid
                Capacity_grid_alpha2[day, s] = env_alpha2.capacity_access_grid
                Capacity_grid_inf[day, s] = env_inf.capacity_access_grid
                Flow_grid[day,s]=env3.state_grid
            
            Reward_heru.append(total_reward_heru)
            Reward_opt.append(total_reward_opt)
            Reward_tr.append(total_reward_tr)
            Reward_re_energy.append(total_reward_re_energy)
            Reward_fair.append(total_reward_fair)
            Reward_alpha2.append(total_reward_alpha2)
            Reward_inf.append(total_reward_inf)
            R = total_reward_re_energy
            print("Total reward: "+ str(total_reward_fair)+" " +str(R)+" " +str())
            
            dict_to_file(Capacity_grid_heru,folder +'C_heru_Log')
            dict_to_file(Capacity_grid_opt,folder +'C_opt_Log')
            dict_to_file(Capacity_grid_re_energy,folder +'C_re_energy_Log')
            dict_to_file(Capacity_grid_fair,folder +'C_fair_Log')
            dict_to_file(Capacity_grid_alpha2,folder +'C_alpha2_Log')
            dict_to_file(Capacity_grid_inf,folder +'C_inf_Log')
            
            vector_to_file(R_t_tr_all, folder + 'Reward_tr_all.csv', 'w')
            vector_to_file(R_t_re_energy_all, folder + 'Reward_re_energy_all.csv', 'w') 
            vector_to_file(R_t_heru_all, folder + 'Reward_heru_all.csv', 'w') 
            # vector_to_file(R_t_opt_all, folder + 'Reward_opt_all.csv', 'w') 
            vector_to_file(R_t_fair_all, folder + 'Reward_fair_all.csv', 'w') 
            vector_to_file(R_t_alpha2_all, folder + 'Reward_alpha2_all.csv', 'w') 
            vector_to_file(R_t_inf_all, folder + 'Reward_inf_all.csv', 'w') 
            dict_to_file(Flow_grid, folder +'Flow_grid')
            
        return Reward_heru, Reward_opt, Reward_tr, Reward_fair, Reward_re_energy, Reward_alpha2, Reward_inf
    
    def test(self, env, DDPG_config, F_G, folder_data):
        MAX_Group = DDPG_config['GROUP']
        MAX_STEPS = DDPG_config['MAX_STEPS']
        Reward_log, Energy_log, Pre_log =[],[], []
        UAV_num = env.UAV_NUMBER_Initial
        self.load_weights(folder_data, Test=True)
        Capacity_grid_drl ={}
        
#        print("Do other methods first: ")
#        Reward_heru, Reward_opt, Reward_tr, Reward_fair, Reward_re_energy, \
#        Reward_alpha2, Reward_inf = self.Other_methods(F_G, MAX_STEPS, UAV_num, env_enu, env_cov, env3,
#                env_tr, env_tr_UAV, env_fair, env_re_energy, env_alpha2, env_inf, MAX_Group, folder_others)
#        vector_to_file([Reward_heru], folder_others + 'TotalRewardheru.csv', 'w')
#        vector_to_file([Reward_opt], folder_others + 'TotalRewardopt.csv', 'w')
#        vector_to_file([Reward_tr], folder_others + 'TotalRewardtr.csv', 'w')
#        vector_to_file([Reward_fair], folder_others + 'TotalRewardfair.csv', 'w')
#        vector_to_file([Reward_re_energy], folder_others + 'TotalReward_re_energy.csv', 'w')
#        vector_to_file([Reward_alpha2], folder_others + 'TotalRewardalpha2.csv', 'w')
#        vector_to_file([Reward_inf], folder_others + 'TotalRewardinf.csv', 'w')
        
        for day in range(MAX_Group): 
            state = env.reset(day, F_G)
            a_t_real = np.zeros(([UAV_num,3]))
            R =[]
            E = 0
            for j in range(MAX_STEPS):
                a_pre = np.copy(a_t_real)
                a = self.policy_action(state)
                new_state, r, a_t_real, delta_engery, r_pre = env.step(a, state, j, a_pre, True) #True for test. reward without re_pre
                state = new_state
                R.append(r)
                Pre_log.append(r_pre)
                E += delta_engery            
                Capacity_grid_drl[day, j] = env.capacity_access_grid                
            Reward_log.append(R)
            Energy_log.append(E)
            # Pre_log.append(Pre)
        return Reward_log, Energy_log, Capacity_grid_drl, Pre_log
     
    def vehicle_data_dis(self, DDPG_config):
        # SCV and DCV of vehicles' distribution
        from numpy import mean, std
        MAX_Group = DDPG_config['GROUP']
        MAX_STEPS = DDPG_config['MAX_STEPS']
        Grid_num_state =  DDPG_config['Grid_num'] #10
        F_G = file_to_dic('./DATA/SDNdata_15min_3km/' + 'F_G_Rio_3km')
        # v_p = np.zeros([Grid_num_state, Grid_num_state])
        v =  np.zeros([MAX_Group, MAX_STEPS, Grid_num_state, Grid_num_state])
        v_p =  np.zeros([MAX_Group, MAX_STEPS, Grid_num_state, Grid_num_state])
        CV = 0
        for d in range(MAX_Group):
            for s in range(MAX_STEPS):
                v_n = np.sum(F_G[d][s])
                v_p[d, s] = F_G[d][s]/v_n
                v[d, s] = F_G[d][s]
                CV += std(v[d, s])/mean(v[d, s])
        CV_mean = CV/MAX_Group/MAX_STEPS
        Std_grid_all = np.zeros([MAX_Group, Grid_num_state, Grid_num_state])
        Std_grid = np.zeros([Grid_num_state, Grid_num_state]) #for each two steps
        Std_grid_v = np.zeros([Grid_num_state, Grid_num_state]) #for each two steps for number
        for i in range(Grid_num_state):
            for j in range(Grid_num_state): 
                s_temp_mean = 0
                s_temp_v_mean = 0
                for d in range(MAX_Group):
                    s_temp = 0
                    s_temp_v = 0
                    for s in range(MAX_STEPS-1):
                        data_p = v_p[d,s:s+2,i,j]#ratio
                        data = v[d,s:s+2,i,j]#number
                        s_temp_v += std(data)#number
                        s_temp += std(data_p)#ratio
                    V_p =  v_p[d,:,i,j]#ratio
                    V = v[d,:,i,j]#number
                    V_mean = mean(V_p)#ratio
                    if V_mean>0:
                        Std_grid_all[d,i,j]= std(V_p)/V_mean#ratio
                        s_temp_mean += s_temp/V_mean#ratio
                        s_temp_v_mean += s_temp_v/mean(V)#number
                Std_grid[i,j] = s_temp_mean/((MAX_STEPS-1)*MAX_Group)
                Std_grid_v[i,j] = s_temp_v_mean/((MAX_STEPS-1)*MAX_Group)
                            # std(data)/V_mean #Coefficient of Variation 极差
        print("CV:" + str(CV_mean))
        print("SCV:"+str(mean(Std_grid)))
        print("TSV:"+str(mean(mean(Std_grid_all, axis=0))))
        return mean(Std_grid), mean(mean(Std_grid_all, axis=0))
        
    def train(self, env, DDPG_config, env_enu, env_cov, env3, env_tr, env_tr_UAV, env_fair, env_re_energy, env_alpha2, env_inf, folder, Read):
        UAV_num = env.UAV_NUMBER_Initial
#        CITY =  DDPG_config["CITY"]
#        TRAFFIC_File = DDPG_config["TRAFFIC_File"]
        alpha = DDPG_config["alpha"]
        folder_data = DDPG_config['Folder_data']
        folder_others = folder_data+'alpha_'+str(alpha)+'_UAV_'+str(UAV_num)+'/'
        if not os.path.exists(folder_others):
            os.makedirs(folder_others)
        MAX_Group = DDPG_config['GROUP']
        MAX_STEPS = DDPG_config['MAX_STEPS']
        EPISODE = DDPG_config['EPISODE_COUNT']
        buffer_size = DDPG_config['BUFFER_SIZE']
        Batch_size = DDPG_config['BATCH_SIZE']
        Grid_num_state =  DDPG_config['Grid_num'] #10
#        MAX_v_grid_num =  DDPG_config['MAX_vehicle_num_grid']
        pretrain = DDPG_config['Pretrain']
        Read_traffic = DDPG_config['Read_traffic']
        Do_Others_methods = DDPG_config['Do_others']
        Read = DDPG_config['Read_other_methods']
        if Read_traffic:
#            if CITY == "Rio":
                # for Rio 3km*3km
            F_G = file_to_dic('DATA/U96_Rio/DATA/' + 'F_G_Rio_3km')
        else:

            F_G, V_num = self.Grid_all_File(Grid_num_state, MAX_Group, MAX_STEPS)
            dict_to_file(F_G, folder_data + 'F_G')
            vector_to_file([V_num], folder_data + 'V_num.csv', 'w') 

        if pretrain:
            self.load_weights(folder_data)
        if Do_Others_methods:
            if Read:
                Reward_heru = Convert(file_to_csv(folder_others + 'TotalRewardheru.csv')[1:-1])[:MAX_Group]
                Reward_opt = Convert(file_to_csv(folder_others + 'TotalRewardopt.csv')[1:-1])[:MAX_Group]
                Reward_fair= Convert(file_to_csv(folder_others + 'TotalRewardfair.csv')[1:-1])[:MAX_Group]
                Reward_tr = Convert(file_to_csv(folder_others + 'TotalRewardtr.csv')[1:-1])[:MAX_Group]
                Reward_re_energy= Convert(file_to_csv(folder_others + 'TotalReward_re_energy.csv')[1:-1])[:MAX_Group]
                Reward_alpha2 = Convert(file_to_csv(folder_others + 'TotalRewardalpha2.csv')[1:-1])[:MAX_Group]
                Reward_inf = Convert(file_to_csv(folder_others + 'TotalRewardinf.csv')[1:-1])[:MAX_Group]
            else:
                print("Do other methods first: ")
                Reward_heru, Reward_opt, Reward_tr, Reward_fair, Reward_re_energy, \
                Reward_alpha2, Reward_inf = self.Other_methods(F_G, MAX_STEPS, UAV_num, env_enu, env_cov, env3,
                        env_tr, env_tr_UAV, env_fair, env_re_energy, env_alpha2, env_inf, MAX_Group, folder_others)
                vector_to_file([Reward_heru], folder_others + 'TotalRewardheru.csv', 'w')
                vector_to_file([Reward_opt], folder_others + 'TotalRewardopt.csv', 'w')
                vector_to_file([Reward_tr], folder_others + 'TotalRewardtr.csv', 'w')
                vector_to_file([Reward_fair], folder_others + 'TotalRewardfair.csv', 'w')
                vector_to_file([Reward_re_energy], folder_others + 'TotalReward_re_energy.csv', 'w')
                vector_to_file([Reward_alpha2], folder_others + 'TotalRewardalpha2.csv', 'w')
                vector_to_file([Reward_inf], folder_others + 'TotalRewardinf.csv', 'w')
                # dict_to_file(Buffer, folder_others +'Buffer')

        Reward_log = []
        
        step_num=0
        break_mark = False
        Capacity_grid = {}
        Step_num_each_episode = []
        time = 0
        if pretrain:
            noise_level = 0.05
        else:
            noise_level = 0.6
        Total_reward_opt = []
        Total_reward_thr = []
        for e in range(EPISODE):
            print(str(e) + "-th Episode ")
            noise_level /= 1.0002
            noise_level = max(noise_level,0.05)
            if e % 500 == 0 and e>0:
                self.save_weights(folder, e)
                vector_to_file([Reward_log], folder + 'TotalRewardLog.csv', 'w')
                # vector_to_file([Engery_log], folder + 'TotalEngeryLog.csv', 'w')
                dict_to_file(Capacity_grid,folder +'C_ddpg')
                if Do_Others_methods:
                    N = int(len(Reward_log)/MAX_Group)
                    R_fair = Reward_fair*N
                    R_Greedy = Reward_re_energy*N
                    Data_plot = [Reward_log, R_fair, R_Greedy]
                    plot_com(folder, Data_plot, MAX_Group, 20, "Reward1", "Reward", "Episode", ["FP","FC", "GP"])
                else:
                    plot_com(folder, [Reward_log], MAX_Group, 20, "Reward",  "Accumulated Reward", "Episode",["DDPG"]) 

            total_reward = 0
            time += 1
            for day in range(MAX_Group): 
                # Reset episode
                cumul_reward, done = 0, False                                   
                state = env.reset(day, F_G)
                actions, states, rewards = [], [], []              
                a_t_real = np.zeros(([env.UAV_NUMBER_Initial,3]))
                Mem = True
                for j in range(MAX_STEPS):                    
                    a_pre = np.copy(a_t_real)
                    if e%2==0 and step_num>buffer_size:
                        a = self.policy_action(state)
                        Mem = False
                    elif not pretrain:
                        if step_num <= buffer_size:
                            a = np.random.rand(1, self.act_dim)[0]
                        else:
                            a = self.policy_action(state)
                            noi = noise_level*(np.random.rand(self.act_dim)-0.5)*2
                            a = np.clip(a+noi, 0, 1)
                    else:
                        a = self.policy_action(state)
                        noi = noise_level*(np.random.rand(self.act_dim)-0.5)*2
                        a = np.clip(a+noi, 0, 1)
                    a = np.around(a, 2)
                    a[2::3]=np.around(a[2::3], 1)
                    # Retrieve new state, reward, and whether the state is terminal
                    new_state, r, done, a_t_real, State_battery = env.step(a, state, j, a_pre)
                    Capacity_grid[day,j]= env.capacity_access_grid
                    # Add outputs to memory buffer
                    if Mem:
                        self.memorize(state, a, self.times_plot*r, done, new_state)
                    # Update current state
                    state = new_state
                    cumul_reward += r
                    total_reward += r                   
                    step_num +=1
                    print("Step "+str(j)+ ": action:" +str(np.around(a_t_real.reshape(1, UAV_num*3)[0], decimals=2))+ 
                          ". Rew:" + str(np.around(self.times_plot*r, decimals=2)))
                    if done == 1 :
                        print("Game end with steps "+ str(j))
                        break
             
                if e >1 and e%2==0 and step_num>Batch_size:
                    Step_num_each_episode.append(j)
                    Reward_log.append(cumul_reward)
                    # Engery_log.append(delta_engery)
                    if Do_Others_methods:
                        total_reward_heru = Reward_heru[day]
                        total_reward_fair = Reward_fair[day]
                        total_reward_re_energy = Reward_re_energy[day]
                    
                if Do_Others_methods:    
                    if e==0 and env.ACTION_TYPE == 'CP':
                        Total_reward_opt.append(cumul_reward)
                    elif e==1 and env.ACTION_TYPE == 'CP':
                        Total_reward_thr.append(cumul_reward)
                
                # NN update
                if step_num>Batch_size:
                    batch_s = Batch_size
                    # Sample experience from buffer
                    states, actions, rewards, dones, new_states, _ = self.sample_batch(batch_s)
                    # Predict target q-values using target networks
                    if self.Branch:
                         q_values = self.critic.target_predict_2b([new_states, self.actor.target_predict_2b(new_states)])
                    else:
                         q_values = self.critic.target_predict([new_states, self.actor.target_predict(new_states)])
                    # Compute critic target
                    critic_target = self.bellman(rewards, q_values, dones)
                    # Train both networks on sampled batch, update target networks
                    break_mark = self.update_models(states, actions, critic_target, Grad_clip=1)

                    
                if Do_Others_methods:  

                    if e >1 and e%2==0 and step_num>Batch_size:
                        R = total_reward_re_energy
                        print("DAY REWARD @ " + str(day) + "-th day: "
                          + str(np.around(cumul_reward, decimals=3))+" @ " 
                          +" Greedy:"+str(np.around(R, decimals=3)) +" @ "
                          +" MC: "+str(np.around(total_reward_heru, decimals=3)) +" @ "
                          +" MF:"+ str(np.around(total_reward_fair, decimals=3)))
                        print("")
                else:
                    if e >1 and e%2==0 and step_num>Batch_size:
                        print("DAY REWARD @ " + str(day) + "-th day: "+ str(np.around(cumul_reward, decimals=3)))
                        print("")
                        
            if e >1 and e%2==0 and step_num>Batch_size and Do_Others_methods:
                R = Reward_re_energy 
                print("TOTAL REWARD @ " + str(e) + "-th Episode: "
                            + str(np.around(total_reward, decimals=3))
                            + " Greedy: " + " @ " + str(np.around(np.sum(R), decimals=3))
                            + " MC: " + " @ " + str(np.around(np.sum(Reward_heru), decimals=3)) 
                            + " MF: " + " @ " + str(np.around(np.sum(Reward_fair), decimals=3))) 
                print("")
            if break_mark:
                break

            


    def save_weights(self, path, e=0):
        if e!=0:
            path += 'E_{}'.format(e)
        self.actor.save(path)
        self.critic.save(path)
        

    def load_weights(self, path, Test=False):
        if not Test:
            self.critic.load_weights(path)
        self.actor.load_weights(path,Test)

