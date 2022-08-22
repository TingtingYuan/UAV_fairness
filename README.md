


# Deep Deterministic Policy Gradient (DDPG) for UAV placement to improve fainess of communication
Paper link: https://ieeexplore.ieee.org/document/9585312


This is the code for implementing the DDPG algorithm for dynamic UAV placement for ITS.

## Installation

- Known dependencies: Python (3.6.2), tensorflow (1.13.1), numpy (1.14.5), seaborn(0.9.0), keras(2.1.6)


## Command-line options

- To train, run `main.py`:

``python3 main.py``


### Environment options

- `--Env_grid`: defines which environment of communication in ITS.

- `--EPISODE_COUNT` total number of training episodes (default: `20000`)

- `--GROUP` number of days (default: `"7'`)

- `--MAX_STEPS`: number of time step in one day (default: `"96"`)

- `--UAV_NUMBER` number of UAVs (default: `"2'`)

- `--alpha` factor for utility function (default: `"0'`)

- `--R_RSU` coverage radius of RSUs 

- `--R_BS` coverage radius of BSs 

- `--HEIGHT_l` lower-bound height of UAVs (default: `"50'`)
 
- `--HEIGHT_u` upper-bound height of UAVs (default: `"100'`)

- `--UAV_B` battery of UAVs (default: `"700e3'`)

- `--Flying_speed` maximum flying speed of UAVs (default: `"20'`)

### Core training parameters

- `--LRA`: learning rate of actor (default: `1e-4`)

- `--LRC`: learning rate of actor (default: `1e-3`)

- `--gamma`: discount factor (default: `0.95`)

- `--BATCH_SIZE`: batch size (default: `800`)


## Code structure

- `./main.py`: contains code for training DDPG in the ITS

- `./Env_grid.py`: contains code for environment model of the UAV-assisted ITS

- `./channel.py`: the model of communication channels of the ITS

- `./DDPG.json`: main parameters in setting

- `./DDPG/ddpg.py`: core code for the DDPG algorithm

- `./DDPG/ReplayBuffer.py`: replay buffer code for DDPG

- `./DDPG/acotr.py`: the actor of DDPG

- `./DDPG/critic.py`: the critic of DDPG

- `./DATA/U96_Rio/DATA`: contains data of vehicles loaction


