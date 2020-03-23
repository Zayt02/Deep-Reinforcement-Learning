import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from Memory import Memory, PER
import os


class D2DQN:
    def __init__(self, state_shape, action_size, use_per=True, use_target_net=True, use_duel=True, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.98, mem_size=20001, batch_size=32, tau=0.05):
        self.state_shape = state_shape
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = epsilon_decay
        # self.epsilon_speed = episodes * 2
        self.discount_factor = discount_factor
        self.tau = tau
        self.use_per = use_per
        self.memory = PER(mem_size, beta=0.4) if use_per else Memory(mem_size)
        self.batch_size = batch_size
        self.optimizer = Adam()
        self.use_duel = use_duel
        self.model = self._build_model()
        self.use_target_net = use_target_net  # double q networks
        if use_target_net:
            self.target_model = self._build_model()

    def _build_model(self):
        inp = Input(self.state_shape)
        out = Dense(100, activation='relu')(inp)
        # out = BatchNormalization()(out)
        out = Dense(32, activation='relu')(out)
        if self.use_duel:
            value_out = Dense(1, activation='linear')(out)
            advantage_out = Dense(self.action_size, activation='linear')(out)
            out = value_out + advantage_out - tf.reduce_mean(advantage_out, 1, keepdims=True)
        else:
            out = Dense(self.action_size, activation='linear')(out)
        model = Model(inp, out)
        model.compile(loss="mse", optimizer=self.optimizer)

        return model

    def choose_action(self, state):
        prob = np.random.uniform(0, 1)
        if prob >= self.epsilon:
            return np.argmax(self.model.predict(np.array([state]))[0])
        else:
            return np.random.randint(self.action_size)

    def get_td_error(self, state, action, reward, next_state):
        out = self.model.predict(np.array([state]))
        next_state_ = np.array([next_state])
        if self.use_target_net:
            max_Q = self.target_model.predict(next_state_)[0, np.argmax(
                self.model.predict(next_state_)[0])]
        else:
            max_Q = np.amax(self.model.predict(next_state_)[0])
        return np.abs(out[0][action] - reward - self.discount_factor * max_Q)

    def update_epsilon(self, loop_counter):
        if self.epsilon > self.epsilon_min:
            # self.epsilon *= 1/(1+loop_counter/self.epsilon_speed)
            self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        if self.use_per:
            td_error = self.get_td_error(state, action, reward, next_state)
            self.memory.add([state, action, reward, next_state, done], td_error)
        else:
            self.memory.add([state, action, reward, next_state, done])

    def train(self):
        """
        samples is a set of (S, A, R, S', done)
        """
        if self.use_per:
            samples, is_weights = self.memory.sample(self.batch_size)
        else:
            samples = self.memory.sample(self.batch_size)
        states = tf.convert_to_tensor(np.array([sample[0] for sample in samples]), dtype='float32')
        actions = np.array([sample[1] for sample in samples])
        rewards = np.array([sample[2] for sample in samples])
        next_states = np.array([sample[3] for sample in samples])
        done = np.array([sample[4] for sample in samples])

        if self.use_target_net:
            max_next_Qvals_index = np.argmax(self.model.predict_on_batch(next_states), axis=1)
            next_Qvals = self.target_model.predict_on_batch(next_states)
            next_Qvals = np.array([next_Qvals[i, max_next_Qvals_index[i]] for i in range(len(samples))])
        else:
            next_Qvals = np.amax(self.model.predict_on_batch(next_states), axis=1)

        yhat = rewards + self.discount_factor * done * next_Qvals
        out = self.model(states).numpy()
        for i in range(len(samples)):
            out[i][actions[i]] = yhat[i]
        if self.use_per:
            self.model.train_on_batch(states, out, is_weights)
        else:
            self.model.train_on_batch(states, out)

    def update_target(self):
        if self.use_target_net:
            weights = self.model.trainable_variables
            target_weights = self.target_model.trainable_variables
            for i in range(len(weights)):
                target_weights[i].assign(self.tau * weights[i] + (1 - self.tau) * target_weights[i])
        else:
            pass

    def save(self, path):
        self.model.save(path + "/model.h5")
        if self.use_target_net:
            self.target_model.save(path + "/target_model.h5")

    def load(self, path):
        self.model = load_model(path + "/model.h5")
        if self.use_target_net:
            if 'target_model.h5' in os.listdir(path):
                self.target_model = load_model(path + "/target_model.h5")
            else:
                raise Exception("Can't not find target model")


