import numpy as np
import gym
from gym import wrappers
from DoubleDQN import DoubleDQN
from D2DQN_PER_tensorflow import D2DQN

EPISODES = 300
BATCH_SIZE = 32


def train(env, agent, save_path):
    # with tf.device('/cpu:0'):
    # make environment
    scores = []

    for episode in range(EPISODES):
        state = env.reset()
        score = 0
        s = 0

        for i in range(500):
            # env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            s += reward
            score += reward
            if done:
                agent.remember(state, action, reward, next_state, 0.)
                print("episode: {}/{}, score: {}, s = {}, e: {:.2})"
                      .format(episode, EPISODES, score, s, agent.epsilon))
                scores.append([i+1, score])
                break
            agent.remember(state, action, reward, next_state, 1.)
            state = next_state
            if agent.memory.get_current_capacity() > BATCH_SIZE:
                agent.train()
        agent.update_target()
        agent.update_epsilon(episode+1)

        if (episode + 1) % 30 == 0:
            agent.save(save_path)


def play(env, agent, run_eps):
    for episode in range(run_eps):
        state = env.reset()
        score = 0
        for i in range(500):
            env.render()
            action = np.argmax(agent.model.predict(np.array([state]))[0])
            next_state, reward, done, _ = env.step(action)
            score += reward
            if done:
                print("episode: {}/{}, score: {}"
                      .format(episode, EPISODES, score))
                break
            state = next_state


if __name__ == '__main__':
    env_to_wrap = gym.make("Acrobot-v1")
    state_shape = env_to_wrap.observation_space.shape
    action_size = env_to_wrap.action_space.n
    agent = D2DQN(state_shape, action_size, use_per=True, use_duel=True,
                  use_target_net=True, epsilon_decay=0.97)
    # agent = DoubleDQN(state_shape, action_size)
    agent.load("save/Acrobot-v1/90")
    # env = wrappers.Monitor(env_to_wrap, 'result/Acrobot-v1/video.mp4', force=True)

    # train(env_to_wrap, agent, "save/Acrobot-v1")
    play(env_to_wrap, agent, 5)
    # env.cloes()
    env_to_wrap.close()

