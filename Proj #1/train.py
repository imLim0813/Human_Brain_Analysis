import numpy as np
import matplotlib.pyplot as plt

from Env import env, func
from DRL_model import TD3
from sys_util import makedir
from itertools import count
import torch

func.os_driver(False)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

env = env.Load()
agent = TD3.make_model()

trial = input("Please Enter the model's name : ")
directory = makedir(trial)

max_episode = 100000
start_train = 100
batch_size = 128
tau = 0.01
frame_size = 84

mode = input('Please select the mode : ')

if mode == 'train':
    reward_list = []
    for episode in range(max_episode):
        total_reward = 0
        step = 0
        env.reset()
        state = []

        for i in range(4):
            state.append(env.to_frame(frame_size, frame_size).squeeze().copy() / 255)
        state = np.array(state)

        for t in count():
            reward = 0

            if episode < start_train:
                action_r = (np.random.normal(0, 0.2, size=1)).clip(0, 1)
                action_theta = np.random.normal(0, 0.4, size=1).clip(-1, 1)
            else:
                action_r, action_theta = agent.select_action(state, noise=0.1)
                action_r = np.array([action_r])
                action_theta = np.array([action_theta])

            next_state = []

            for _ in range(4):
                next_tmp, reward_tmp, done, _ = env.step(action_r, action_theta)
                next_tmp = env.to_frame(frame_size, frame_size).squeeze().copy() / 255
                next_state.append(next_tmp)
                reward += reward_tmp

            next_state = np.array(next_state)

            action = np.array([action_r.item(), action_theta.item()], dtype=float)
            agent.replay_buffer.push((state, next_state, action, reward, float(done)))
            state = next_state.copy()

            total_reward += reward

            if done:
                break

        print('\rEpisode : {}, Total Step : {}, Total Reward : {:.2f}'.format(episode, env.count, total_reward), end='')
        if episode == start_train:
            print('')
            print('=' * 50)
            print('***** Now Train begins.. *****')
            print('=' * 50)

        reward_list.append(total_reward)

        if episode > start_train and total_reward == max(reward_list):
            agent.save(directory=directory, epoch=episode)

        if episode > start_train:
            agent.update(batch_size, episode)

        if total_reward > 29000:
            np.save(directory + '/{}_lr.npy'.format(trial), reward_list)
            break

elif mode == 'test':
    epoch = int(input('Please Enter epoch number : '))
    agent.load(directory, epoch, device=device)

    test_reward = 0
    RL_action = []
    RL_pos = []
    for episode in range(1):
        env.reset()
        state = []

        for i in range(4):
            state.append(env.to_frame(frame_size, frame_size, trial).squeeze().copy() / 255)
        state = np.array(state)

        for step_ in count():
            reward = 0

            action_r, action_theta = agent.select_action(state, noise=0)

            action_r = np.array([action_r])
            action_theta = np.array([action_theta])

            next_state = []

            for _ in range(4):
                RL_pos.append(env.state)
                RL_action.append([action_r, action_theta])
                next_tmp, reward_tmp, done, _ = env.step(action_r, action_theta)
                next_tmp = env.to_frame(frame_size, frame_size, trial).squeeze().copy() / 255
                next_state.append(next_tmp)
                reward += reward_tmp

            next_state = np.array(next_state)

            test_reward += reward

            if done or step_ > env.parameter.duration:
                print('Episode : {}, Reward : {:.2f}, Step : {}'.format(int(episode), test_reward, int(env.count)))
                test_reward = 0
                break

            state = next_state.copy()

    RL_action = np.array(RL_action).squeeze()
    np.save(directory + '/{}.npy'.format('DRL_action'), np.array(RL_action))
    np.save(directory + '/{}.npy'.format('DRL_position'), np.array(RL_pos))