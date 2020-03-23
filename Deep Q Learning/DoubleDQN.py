import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from Memory import Memory


class DoubleDQN:
    def __init__(self, state_shape, action_size, discount_rate=0.97, epsilon=1.0,
                 epsilon_decay=0.98, mem_size=20001, batch_size=32, tau=0.05):
        self.state_shape = state_shape
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = epsilon_decay
        # self.epsilon_speed = episodes * 2
        self.discount_rate = discount_rate
        self.tau = tau
        self.memory = Memory(mem_size)
        self.batch_size = batch_size
        self.optimizer = Adam()
        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        inp = Input(self.state_shape)
        out = Dense(48, activation='relu')(inp)
        out = Dense(32, activation='relu')(out)
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

    def update_epsilon(self, loop_counter):
        if self.epsilon > self.epsilon_min:
            # self.epsilon *= 1/(1+loop_counter/self.epsilon_speed)
            self.epsilon *= self.epsilon_decay

    def train(self):
        """
        samples is a set of (S, A, R, S', done)
        """
        samples = self.memory.sample(self.batch_size)
        states = tf.convert_to_tensor(np.array([sample[0] for sample in samples]), dtype='float32')
        actions = np.array([sample[1] for sample in samples])
        rewards = np.array([sample[2] for sample in samples])
        next_states = np.array([sample[3] for sample in samples])
        done = np.array([sample[4] for sample in samples])
        maxQ_next_actions_index = np.argmax(self.model.predict_on_batch(next_states), axis=1)
        target_next_Qvals = np.array([self.target_model.predict_on_batch(next_states)[i, maxQ_next_actions_index[i]]
                                      for i in range(len(samples))])
        target_Qvals = rewards + self.discount_rate * done * target_next_Qvals

        out = self.model(states).numpy()
        for i in range(len(samples)):
            out[i][actions[i]] = target_Qvals[i]
        self.model.train_on_batch(states, out)

        # def train(self):
        #     """
        #     samples is a set of (S, A, R, S', done)
        #     """
        # samples = self.memory.sample(self.batch_size)
        #
        # for i in range(len(samples)):
        #     state = tf.convert_to_tensor(samples[i][0], dtype='float32')
        #     action = samples[i][1]
        #     reward = samples[i][2]
        #     next_state = samples[i][3]
        #     done = samples[i][4]
        #     target_next_Qval = self.target_model.predict(next_state)[0, np.argmax(self.model.predict(next_state)[0])]
        #     target_Qval = reward + self.discount_rate * done * target_next_Qval
        #     with tf.GradientTape() as tape:
        #         out = self.model(state)
        #         y_train = out.numpy()
        #         y_train[0][action] = target_Qval
        #         loss = (out - y_train) ** 2
        #     grad = tape.gradient(loss, self.model.trainable_variables)
        #     self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

    def remember(self, state, action, reward, next_state, done):
        self.memory.add([state, action, reward, next_state, done])

    def update_target(self):
        weights = self.model.trainable_variables
        target_weights = self.target_model.trainable_variables
        for i in range(len(weights)):
            target_weights[i].assign(self.tau * weights[i] + (1 - self.tau) * target_weights[i])

    def save(self, path):
        self.model.save(path + "/model.h5")
        self.target_model.save(path + "/target_model.h5")

    def load(self, path):
        self.model = load_model(path + "/model.h5")
        self.target_model = load_model(path + "/target_model.h5")


