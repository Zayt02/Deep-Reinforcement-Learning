import gym
import numpy as np
import tensorflow as tf
from DDPG import DDPG
from Model import Actor, Critic

EPISODES = 1000
BATCH_SIZE = 32


def train(env: gym.Env, path):
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    ubound = env.action_space.high
    lbound = env.action_space.low
    # print(state_shape, action_shape)

    actor = Actor(state_shape, action_shape, ubound)
    critic = Critic(state_shape, action_shape)
    agent = DDPG(actor, critic)

    print(actor.model.summary())
    print(critic.model.summary())
    print(actor.target_model.summary())
    print(critic.target_model.summary())

    for time in range(EPISODES):
        state = env.reset()
        score = 0
        agent.noise_generator.reset()
        for _ in range(1000):
            if time > EPISODES / 5:  # for learning optimal solution
                action = agent.choose_action(np.array([state]))[0]
            else:
                action = np.clip(agent.get_action_with_noise(np.array([state])),
                                 lbound, ubound)
            next_state, reward, done, info = env.step(action)
            score += reward
            if not done:
                agent.memory.add([state, action, reward, next_state, 1.])
                state = next_state
                if len(agent.memory.memory) > BATCH_SIZE:
                    samples = agent.memory.sample(BATCH_SIZE)
                    agent.train(samples)
            else:
                agent.memory.add([state, action, reward, next_state, 0.])
                break
        print(time, score)
        agent.update_all_target()
        if time % 500 == 0:
            agent.save(path+"/whole_model/save" + str(time))


def play(env: gym.Env, path):
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    ubound = env.action_space.high
    lbound = env.action_space.low

    actor = Actor(state_shape, action_shape, ubound)
    critic = Critic(state_shape, action_shape)
    agent = DDPG(actor, critic)
    agent.load(path+"/whole_model/save400")

    print(actor.model.summary())
    print(critic.model.summary())
    print(actor.target_model.summary())
    print(critic.target_model.summary())

    for time in range(EPISODES):
        state = env.reset()
        score = 0
        for _ in range(1000):
            env.render()
            action = agent.choose_action(np.array([state]))[0]
            state, reward, done, info = env.step(action)
            score += reward
        print(score)
    env.close()


if __name__ == "__main__":
    with tf.device('/GPU:0'):
        env = gym.make("Pendulum-v0")
        play(env, "save_Pendulumv0")
