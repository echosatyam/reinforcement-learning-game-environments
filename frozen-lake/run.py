# %%
import os
import random
import time

import gym
import numpy as np

env = gym.make('FrozenLake-v0')
print(env.observation_space)
action_space_size = env.action_space.n
state_space_size = env.observation_space.n
q_table = np.zeros((state_space_size, action_space_size))
print(q_table)

num_episodes = 10000
max_steps_per_episode = 100
learning_rate = .1
discount_rate = .99
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = .01
exploration_decay_rate = .001

rewards_all_episodes = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    rewards_current_episode = 0
    for step in range(max_steps_per_episode):
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        q_table[state, action] = q_table[state, action]*(1-learning_rate) + \
            learning_rate*(reward+discount_rate *
                           (np.max(q_table[new_state, :])))
        state = new_state
        rewards_current_episode += reward
        if done:
            break
    if(episode % 10000 == 0):
        print(f"episode: {episode} completed")
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate-min_exploration_rate) * \
        np.exp(-exploration_decay_rate*episode)
    rewards_all_episodes.append(rewards_current_episode)

rewards_per_thousand_episodes = np.split(
    np.array(rewards_all_episodes), num_episodes/1000)
count = 1000
print(" Average reward per thousand episodes\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000
print('\n\n**** Q-table\n\n')
print(q_table)

# %%
count = 0
for episode in range(10):
    state = env.reset()
    done = False
    print(f"Episode : {episode}")
    time.sleep(1)
    for step in range(max_steps_per_episode):
        os.system('clear')
        env.render()
        time.sleep(.1)
        action = np.argmax(q_table[state, :])
        new_state, reward, done, info = env.step(action)
        state = new_state
        if done:
            os.system('clear')
            env.render()
            if reward == 1:
                print('You Won')
                count += 1
            else:
                print("Lost the game")
            time.sleep(2)
            os.system('clear')
            break
print(f"{count} count of wins")
time.sleep(3)
env.close()
os.system("clear")
