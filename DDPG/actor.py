import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Input, Dense, Reshape, LSTM, Lambda, BatchNormalization, GaussianNoise, Flatten
from keras.layers import concatenate
from keras.optimizers import Adam
import keras

class Actor:
    """ Actor Network for the DDPG Algorithm
    """

    def __init__(self, inp_dim, out_dim, act_range, lr, tau, state_size2, LRDecay, branch):
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.state_size2 = state_size2
        self.act_range = act_range
        self.tau = tau
        self.lr = lr
        self.branch = branch
        if branch:
            self.model = self.network_2branch()
            self.target_model =  self.network_2branch()
        else:
            self.model = self.network()
            self.target_model = self.network()
        self.adam_optimizer = self.optimizer(LRDecay)

    def network(self):
        """ Actor Network for Policy function Approximation, using a tanh
        activation for continuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
        """
        inp = Input(shape=(self.env_dim,))
#        Conv2D(64, kernel_size=3, activation=’relu’, input_shape=(28,28,1))
        #
        x = Dense(256, activation='relu')(inp)
        x = GaussianNoise(1.0)(x)
#        #
#        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = GaussianNoise(1.0)(x)
        
        x = Dense(32, activation='relu')(x)
        x = GaussianNoise(1.0)(x)
        #
        out = Dense(self.act_dim, activation='sigmoid', kernel_initializer=RandomUniform())(x)
#        out = Lambda(lambda i: i * self.act_range)(out)
        #
        return Model(inp, out)
    
    def network_2branch(self):
        state_size2 = self.state_size2
        state_size1 = self.env_dim-state_size2
        action_dim = self.act_dim
        
        S1 = Input(shape=[state_size1], name='a_S1')
        h0 = Dense(64, activation= 'selu', kernel_initializer="glorot_normal", name='a_h0')(S1)
        h2 = Dense(16, activation= 'selu', kernel_initializer= "glorot_normal", name='a_h2')(h0)

        S2 = Input(shape=[state_size2], name='a_S2')        
        h1_ = Dense(16, activation= 'selu', kernel_initializer="glorot_normal", name='a_h0_')(S2)
        h2_ = concatenate([h2, h1_], name='a_h2_')
        h3 = Dense(8, activation= 'selu', kernel_initializer= "glorot_normal", name='a_h3')(h2_)
        
        V = Dense(action_dim, activation='sigmoid', kernel_initializer="glorot_normal", name='a_V')(h3)
        return  Model(inputs=[S1, S2], outputs=V)

    def predict(self, state):
        """ Action prediction
        """
        return self.model.predict(np.expand_dims(state, axis=0))
    
    def predict_2b(self, state):
        """ Action prediction
        """
        state_size2 = self.state_size2
        state_size1 = self.env_dim-state_size2
        state1 = state[:state_size1]
        state2 = state[state_size1:]
        return self.model.predict([state1.reshape(-1, state_size1), state2.reshape(-1,state_size2) ])
#                np.expand_dims(state, axis=0))

    def target_predict(self, inp):
        """ Action prediction (target network)
        """
        return self.target_model.predict(inp)
    
    def target_predict_2b(self, inp):
        """ Action prediction (target network)
        """
        state_size2 = self.state_size2
        state_size1 = self.env_dim-state_size2
        state1 = inp[:,:state_size1]
        state2 = inp[:,state_size1:]
#        return self.target_model.predict([state1.reshape(-1, self.env_dim-self.state_size2), state2.reshape(-1,self.state_size2) ])
        return self.target_model.predict([state1,state2])
    
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

    def train(self, states, actions, grads):
        """ Actor Training
        """
        if self.branch:
            state_size2 = self.state_size2
            state_size1 = self.env_dim-state_size2
            state1 = states[:,:state_size1]
            state2 = states[:,state_size1:]
            self.adam_optimizer([state1, state2, grads])
        else:
            self.adam_optimizer([states, grads])
        

    def optimizer(self, LRDecay= False):
        """ Actor Optimizer
        """
        action_gdts = K.placeholder(shape=(None, self.act_dim))
        params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.model.trainable_weights)
        
        if LRDecay:
            decay = 0.9
            step_rate = 1000
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.lr, global_step, step_rate, decay, staircase=True)
            self.lr = learning_rate
        if self.branch:
            return K.function([self.model.input[0], self.model.input[1], action_gdts],[tf.train.AdamOptimizer(self.lr).apply_gradients(grads)])
        else:
            return K.function([self.model.input, action_gdts], [tf.train.AdamOptimizer(self.lr).apply_gradients(grads)])
        
#                          [tf.train.AdamOptimizer(self.lr).apply_gradients(grads)])

    def save(self, path):
        self.model.save_weights(path + 'actor.h5')
        self.target_model.save_weights(path + 'actor_target.h5')

    def load_weights(self, path, test= False):
        self.model.load_weights(path+ 'actor.h5')
        if not test:
            self.target_model.load_weights(path+ 'actor_target.h5')
