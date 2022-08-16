import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pickle

from Env import env, func
from DRL_model import TD3_extract
from sys_util import makedir
from itertools import count
from glob import glob


def extract_data(subj, epoch, directory, mode, device):
    agent.load(directory, epoch, device)

    data_list = glob('./Data/Human/behavior/{}/*.pkl'.format(subj))
    data_list.sort()

    if not os.path.exists('./Data/Human/HRL/{}'.format(subj)):
        os.makedirs('./Data/Human/HRL/{}'.format(subj))

    if not os.path.exists('./Data/Human/HRL/{}/{}'.format(subj, mode.capitalize())):
        os.makedirs('./Data/Human/HRL/{}/{}'.format(subj, mode.capitalize()))

    for idx in range(6):
        data = []
        with open(data_list[idx], 'rb') as f:
            tmp = pickle.load(f)
        data.append(np.concatenate([np.array(tmp['cursor']) - 35, np.array(tmp['target'])], axis=1))
        data = np.array(data).squeeze().reshape(-1, 4)

        print('=' * 50)
        print('Run number : {}'.format(idx + 1))
        print('=' * 50)
        print('Behavior Data shape :', data.shape)

        actor_out = {}
        critic_out = {}

        actor_list = ['pixel', 'conv1', 'conv2', 'conv3', 'conv4', 'linear', 'actor1', 'actor2', 'action']
        for name in actor_list:
            actor_out[name] = []

        critic_list = ['pixel', 'conv1', 'conv2', 'conv3', 'conv4', 'linear', 'critic1', 'critic2', 'Q_val']
        for name in critic_list:
            critic_out[name] = []

        total_pos = data
        target_pos = data[:, 2:]

        for episode in range(1):
            env.reset()
            state = []

            for i in range(4):
                state.append(env.to_frame(84, 84, trial).squeeze().copy() / 255)
            state = np.array(state)

            for step_ in range(int(target_pos.shape[0] / 4)):

                action, conv_data, actor = agent.select_action(state, noise=0)
                conv_data_2, critic, Q = agent.critic.Q1(torch.FloatTensor(state).to(device),
                                                         torch.FloatTensor(action).reshape(-1, 2).to(device))

                actor_out['pixel'].append(conv_data[0].detach().cpu().numpy().reshape(1, -1).tolist())
                actor_out['conv1'].append(conv_data[1].detach().cpu().numpy().reshape(1, -1).tolist())
                actor_out['conv2'].append(conv_data[2].detach().cpu().numpy().reshape(1, -1).tolist())
                actor_out['conv3'].append(conv_data[3].detach().cpu().numpy().reshape(1, -1).tolist())
                actor_out['conv4'].append(conv_data[4].detach().cpu().numpy().reshape(1, -1).tolist())
                actor_out['linear'].append(conv_data[5].detach().cpu().numpy().reshape(1, -1).tolist())
                actor_out['actor1'].append(actor[0].detach().cpu().numpy().reshape(1, -1).tolist())
                actor_out['actor2'].append(actor[1].detach().cpu().numpy().reshape(1, -1).tolist())
                actor_out['action'].append(action.reshape(1, -1))

                critic_out['pixel'].append(conv_data_2[0].detach().cpu().numpy().reshape(1, -1).tolist())
                critic_out['conv1'].append(conv_data_2[1].detach().cpu().numpy().reshape(1, -1).tolist())
                critic_out['conv2'].append(conv_data_2[2].detach().cpu().numpy().reshape(1, -1).tolist())
                critic_out['conv3'].append(conv_data_2[3].detach().cpu().numpy().reshape(1, -1).tolist())
                critic_out['conv4'].append(conv_data_2[4].detach().cpu().numpy().reshape(1, -1).tolist())
                critic_out['linear'].append(conv_data_2[5].detach().cpu().numpy().reshape(1, -1).tolist())
                critic_out['critic1'].append(critic[0].detach().cpu().numpy().reshape(1, -1).tolist())
                critic_out['critic2'].append(critic[1].detach().cpu().numpy().reshape(1, -1).tolist())
                critic_out['Q_val'].append(Q.reshape(-1))

                next_state = []

                for _ in range(4):
                    env.cursor.cur_x = total_pos[env.count][0]
                    env.cursor.cur_y = total_pos[env.count][1]
                    next_tmp, reward_tmp, done, _ = env.step(np.array([0]), np.array([0]), path=target_pos)
                    next_tmp = env.to_frame(84, 84, trial).squeeze().copy() / 255
                    next_state.append(next_tmp)

                next_state = np.array(next_state)

                state = next_state.copy()

        with open('./Data/Human/HRL/{}/{}/Human_RL_actor_{}.pkl'.format(subj, mode.capitalize(), idx+1), 'wb') as f:
            pickle.dump(actor_out, f)

        with open('./Data/Human/HRL/{}/{}/Human_RL_critic_{}.pkl'.format(subj, mode.capitalize(), idx+1), 'wb') as f:
            pickle.dump(critic_out, f)

        RL_action = np.array(actor_out['action']).squeeze()
        np.save('./Data/Human/HRL/{}/{}/Human_RL_action_{}.npy'.format(subj, mode.capitalize(), idx + 1), np.array(RL_action))
        print('Run {} has been saved..'.format(idx + 1))
        print('')


if __name__ == '__main__':
    func.os_driver(False)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    env = env.Load()
    agent = TD3_extract.make_model()

    print('=' * 50)
    trial = input("Please Enter the model's name : ")
    directory = makedir(trial)
    epoch = input("Please Enter the model's epoch :")
    print('=' * 50)
    subj = input("Please Enter the subject name :")

