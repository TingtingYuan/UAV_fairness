import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, concatenate, LSTM, Reshape, BatchNormalization, Lambda, Flatten

class Critic:
    """ Critic for the DDPG Algorithm, Q-Value function approximator
    """

    def __init__(self, inp_dim, out_dim, lr, tau,state_size2, LRDecay, Branch):
        # Dimensions and Hyperparams
        self.LRDecay = LRDecay
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.tau, self.lr = tau, lr
        self.state_size2 = state_size2
        # Build models and target models
        if Branch:
            self.model = self.network_2branch()
            self.target_model = self.network_2branch()
            # Function to compute Q-value gradients (Actor Optimization)
            self.action_grads = K.function([self.model.input[0], self.model.input[1], self.model.input[2]], K.gradients(self.model.output, [self.model.input[2]]))
        else:   
            self.model = self.network()
            self.target_model = self.network()
            # Function to compute Q-value gradients (Actor Optimization)
            self.action_grads = K.function([self.model.input[0], self.model.input[1]], K.gradients(self.model.output, [self.model.input[1]]))
        if LRDecay:
            self.model.compile(Adam(self.lr, decay=1e-6), 'mse')
#            self.target_model.compile(Adam(self.lr , decay=0.001), 'mse')
        else:
            self.model.compile(Adam(self.lr), 'mse')
#            self.target_model.compile(Adam(self.lr), 'mse')



    def network(self):
        """ Assemble Critic network to predict q-values
        """
        state = Input(shape=(self.env_dim,))
        action = Input(shape=(self.act_dim,))
        x = Dense(256, activation='relu')(state)
#        a_layer = Dense(24, activation='linear')(action)
        x = concatenate([x, action])
#        x = Dense(64, activation='linear', kernel_initializer=RandomUniform())(x)
        x = Dense(128, activation='selu')(x)
#        x = Dense(32, activation='linear')(x)
        out = Dense(1, activation='linear', kernel_initializer=RandomUniform())(x)
        # self.act_dim
        return Model([state, action], out)

    def network_2branch(self):
        """ Assemble Critic network to predict q-values
        """
#        state = Input(shape=(self.env_dim,))
        action = Input(shape=(self.act_dim,))
        state_size2 = self.state_size2
        state_size1 = self.env_dim-state_size2
        S1 = Input(shape=[state_size1], name='c_S1')
        S2 = Input(shape=[state_size2], name='c_S2')
        x = Dense(64, activation='selu')(S1)
        x = Dense(16, activation='selu')(x)
        
        x = concatenate([x, S2])
        x = concatenate([x, action])
#        x = Dense(256, activation='relu')(state)
#        a_layer = Dense(24, activation='linear')(action)
#        x = concatenate([x, action])
#        x = Dense(64, activation='linear', kernel_initializer=RandomUniform())(x)
        x = Dense(32, activation='selu')(x)
        x = Dense(16, activation='linear')(x)
#        x = Dense(32, activation='linear')(x)
        out = Dense(1, activation='linear', kernel_initializer=RandomUniform())(x)
        # self.act_dim
        return Model([S1, S2, action], out)


    def gradients(self, states, actions):
        """ Compute Q-value gradients w.r.t. states and policy-actions
        """
#        with tf.Session() as sess:
#            print('Learning rate: %f' % (sess.run(self.lr)))
#        print("LR Critic:" + str(self.lr))
        return self.action_grads([states, actions])
    
    def gradients_2b(self, states, actions):
        """ Compute Q-value gradients w.r.t. states and policy-actions
        """
        state_size2 = self.state_size2
        state_size1 = self.env_dim-state_size2
        state1 = states[:,:state_size1]
        state2 = states[:,state_size1:]
        return self.action_grads([state1, state2 , actions])

    def target_predict(self, inp):
        """ Predict Q-Values using the target network
        """
        return self.target_model.predict(inp)
    
    def target_predict_2b(self, inp):
        """ Predict Q-Values using the target network
        """
        state_size2 = self.state_size2
        state_size1 = self.env_dim-state_size2
        state1 = inp[0][:,:state_size1]
        state2 = inp[0][:,state_size1:]
        action = inp[1]
        return self.target_model.predict([state1,state2,action])

    def train_on_batch(self, states, actions, critic_target):
        """ Train the critic network on batch of sampled experience
        """
        return self.model.train_on_batch([states, actions], critic_target)
    
    def train_on_batch_2b(self, states, actions, critic_target):
        """ Train the critic network on batch of sampled experience
        """
        state_size2 = self.state_size2
        state_size1 = self.env_dim-state_size2
        state1 = states[:,:state_size1]
        state2 = states[:,state_size1:]
        return self.model.train_on_batch([state1, state2 , actions], critic_target)
    
    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.target_model.set_weights(target_W)
    
    def transfer_weights_pre(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            noise = np.random.rand(1)[0]/10
            target_W[i] = W[i]+noise
        self.target_model.set_weights(target_W)

    def save(self, path):
        self.model.save_weights(path + 'critic.h5')
        self.target_model.save_weights(path + 'critic_target.h5')

    def load_weights(self, path):
        self.model.load_weights(path+ 'critic.h5')
        self.target_model.load_weights(path+ 'critic_target.h5')