import numpy as np
import random
from collections import deque
import tensorflow as tf
from Model import Actor, Critic
from tensorflow.keras.models import load_model


class DDPG:
    def __init__(self, actor: Actor, critic: Critic, mem_size=50001,
                 noise_generator=None, random_seed=2):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.actor = actor
        self.critic = critic
        self.memory = Memory(mem_size)
        self.noise_generator = noise_generator if noise_generator is not None \
            else OrnsteinUhlenbeckActionNoise(np.zeros((1,)))
        self.noise_scale = 0.5

    def choose_action(self, states):
        return self.actor.model.predict(states)

    # def make_noise(self, times, shape):
    #     noise = self.noise_generator.get_noise(times)
    #     noise = np.random.normal(noise, noise * self.noise_scale, shape)
    #     return noise

    # def get_action_with_noise(self, states, times):
    #     actions = self.choose_action(states)
    #     noise = self.make_noise(times, actions.shape)
    #     # print("pre-actions", actions)
    #     actions = np.random.normal(actions, np.abs(np.multiply(actions, noise)))
    #     # print("noise: ", noise, "actions: ", actions)
    #     return actions

    def get_action_with_noise(self, states):
        return self.choose_action(states)[0] + self.noise_generator()

    def train(self, samples):
        """
        samples is a set of (S, A, R, S', done)
        """
        states = tf.convert_to_tensor(np.array([sample[0] for sample in samples]), dtype='float32')
        actions = tf.convert_to_tensor(np.array([sample[1] for sample in samples]), dtype='float32')
        rewards = np.array([[sample[2]] for sample in samples])
        next_states = np.array([sample[3] for sample in samples])
        done = np.array([[sample[4]] for sample in samples])

        next_actions = self.actor.predict_target(next_states)
        next_Qvals = self.critic.predict_target([next_states, next_actions])
        critic_y_train = rewards + self.critic.discount_rate * done * next_Qvals

        self.critic.train([states, actions], critic_y_train)
        self.actor.train(states, self.critic.model)

    def _update_target(self, weights, target_weights, tau):
        for i in range(len(weights)):
            target_weights[i].assign(tau * weights[i] + (1 - tau) * target_weights[i])

    def update_all_target(self):
        self._update_target(self.actor.model.trainable_variables,
                            self.actor.target_model.trainable_variables, self.actor.tau)
        self._update_target(self.critic.model.trainable_variables,
                            self.critic.target_model.trainable_variables, self.critic.tau)

    def save_weights(self, path):
        self.actor.model.save_weights(path + "/actor.h5")
        self.actor.target_model.save_weights(path + "/actor_target.h5")
        self.critic.model.save_weights(path + "/critic.h5")
        self.critic.target_model.save_weights(path + "/critic_target.h5")

    def load_weights(self, path):
        self.actor.model.load_weights(path + "/actor.h5")
        self.actor.target_model.load_weights(path + "/actor_target.h5")
        self.critic.model.load_weights(path + "/critic.h5")
        self.critic.target_model.load_weights(path + "/critic_target.h5")

    def save(self, path):
        self.actor.model.save(path + "/actor.h5")
        self.actor.target_model.save(path + "/actor_target.h5")
        self.critic.model.save(path + "/critic.h5")
        self.critic.target_model.save(path + "/critic_target.h5")

    def load(self, path):
        self.actor.model = load_model(path + "/actor.h5")
        self.actor.target_model = load_model(path + "/actor_target.h5")
        self.critic.model = load_model(path + "/critic.h5")
        self.critic.target_model = load_model(path + "/critic_target.h5")


class Memory:
    def __init__(self, mem_size):
        self.memory = deque(maxlen=mem_size)

    def sample(self, batch_size):
        memory = random.choices(self.memory, k=batch_size)
        return memory

    def add(self, memory):
        # state, action, reward, next_state
        self.memory.append(memory)


class ExpActionNoise:
    """
    noise = exp(-alpha * times)
    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.noise = np.e

    def reset(self):
        self.noise = np.e

    def get_noise(self, times):
        self.noise = np.exp(-self.alpha * times)
        return self.noise


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.x_prev = 0.0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
