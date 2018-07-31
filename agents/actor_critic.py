import numpy as np
import tensorflow as tf

from keras import layers, models, optimizers
from keras import backend as K
from keras.initializers import TruncatedNormal, RandomUniform
from keras.regularizers import l2, l1


class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""

        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Batch Normalization
        normalized_states = layers.BatchNormalization()(states)

        # Weight initialization
        init = RandomUniform(minval=-0.003, maxval=0.003, seed=None)

        # Add hidden layers
        net = layers.Dense(units=400, activation=None, kernel_initializer=init)(normalized_states)
        #net = layers.Dropout(0.25)(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)

        net = layers.Dense(units=300, activation=None, kernel_initializer=init)(net)
        #net = layers.Dropout(0.25)(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)

        # Add hidden layers

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid', kernel_initializer=init, name='raw_actions')(net)

        # Scale output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=0.0001)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)

        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[loss],
            updates=updates_op)

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Batch Normalization on input states - as suggested by the Lilicrap, ext paper
        normalized_states = layers.BatchNormalization()(states)

        # L2 regularization
        l2_lambda = 0.001

        # Weight initialization
        #init = TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
        init = RandomUniform(minval=-0.003, maxval=0.003, seed=None)

        """
        Add hidden layer(s) for state pathway
        """
        net_states = layers.Dense(units=400, activation=None, kernel_initializer=init, kernel_regularizer=l2(l2_lambda))(normalized_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)
        #net_states = layers.Dropout(0.25)(net_states)

        net_states = layers.Dense(units=300, activation=None, kernel_initializer=init, kernel_regularizer=l2(l2_lambda))(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)
        #net_states = layers.Dropout(0.25)(net_states)

        """
        Add hidden layer(s) for action pathway
        """

        # Actions not included until 2nd layer (as per parer)
        net_actions = layers.Dense(units=300, activation=None, kernel_initializer=init, kernel_regularizer=l2(l2_lambda))(actions)
        net_actions = layers.Activation('relu')(net_actions)
        #net_actions = layers.Dropout(0.25)(net_actions)

        # Add hidden layer(s) for state pathway

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to produce action values (Q values)
        # L2 regularization (weight decay) added to Q network - as suggested by the paper
        Q_values = layers.Dense(units=1, activation='linear', kernel_initializer=init, kernel_regularizer=l2(l2_lambda), name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)

