import sys, os
import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from glob import glob
from Anlz.Human.data_anlz import import_behav, sampling, cal_theta

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"


def plot_human_dist(subj):
    plt.rc('font', size=20)

    behav_data = import_behav(subj, run='All')
    sampled_data = sampling(behav_data)

    directory = './Result/Human_dist/{}'.format(subj)

    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.rc('font', size=15)

    plt.hist(sampled_data['radian'][:, 0])
    plt.title('Human_r')
    plt.savefig(directory + '/Human_r_hist.JPEG', dpi=300)

    plt.hist(sampled_data['radian'][:, 1])
    plt.title('Human_theta')
    plt.savefig(directory + '/Human_theta_hist.JPEG', dpi=300)

    r = sampled_data['radian'][:, 0]
    theta = sampled_data['radian'][:, 1]
    rad = np.deg2rad(theta)

    human_data = np.concatenate([r.reshape(-1, 1), rad.reshape(-1, 1)], axis=1)
    rbins = np.linspace(0, 6, 13)
    abins = np.linspace(0, 2 * np.pi, 37)

    hist, _, _ = np.histogram2d(rad, r, bins=(abins, rbins))
    A, R = np.meshgrid(abins, rbins)
    human_dist = hist.T.reshape(-1, 1)

    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8, 8))
    normalize = matplotlib.colors.LogNorm()
    pc = ax.pcolormesh(A, R, hist.T, cmap='Oranges', norm=normalize)
    fig.colorbar(pc)
    ax.grid(True)
    plt.savefig(directory + '/Human_Distribution.JPEG', dpi=300)

    return r, theta, human_dist


def plot_target_dist():
    plt.rc('font', size=20)
    tmp = np.load('../../Data/total_path.npy')
    behav_data = {'target': []}

    for i in range(20):
        for idx in range(1500 * i, 1500 * i + 1465):
            behav_data['target'].append(tmp[idx])
    behav_data['target'] = np.array(behav_data['target'])

    target_data = []
    for idx in range(0, behav_data['target'].shape[0] - 1, 1):
        tmp = behav_data['target'][idx + 1] - behav_data['target'][idx]
        target_data.append(tmp)
    target_data.append([0, 0])
    target_data = np.array(target_data)

    data = {'joystick': target_data}

    tmp = {'joystick': data['joystick'], 'radian': []}
    for idx in range(tmp['joystick'].shape[0]):
        if (np.sqrt(data['joystick'][idx][0] ** 2 + data['joystick'][idx][1] ** 2)) < 100:
            tmp['radian'].append([np.sqrt(data['joystick'][idx][0] ** 2 + data['joystick'][idx][1] ** 2),
                                  cal_theta(data['joystick'], idx)])
        else:
            tmp['radian'].append([0, 0])
    tmp['radian'] *= 6
    tmp['radian'] = np.array(tmp['radian'])

    r = tmp['radian'][:, 0]
    theta = tmp['radian'][:, 1]
    rad = np.deg2rad(theta)

    target_data = np.concatenate([r.reshape(-1, 1), rad.reshape(-1, 1)], axis=1)

    directory = './Analysis/Result/Target_dist'

    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.rc('font', size=15)

    plt.hist(r)
    plt.title('Target_r')
    plt.savefig(directory + '/Target_r_hist.JPEG', dpi=300)

    plt.hist(theta)
    plt.title('Target_theta')
    plt.savefig(directory + '/Target_theta_hist.JPEG', dpi=300)

    rbins = np.linspace(0, 6, 13)
    abins = np.linspace(0, 2 * np.pi, 37)

    hist, _, _ = np.histogram2d(rad, r, bins=(abins, rbins))
    A, R = np.meshgrid(abins, rbins)
    target_dist = hist.T.reshape(-1, 1)

    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8, 8))
    normalize = matplotlib.colors.LogNorm()
    pc = ax.pcolormesh(A, R, hist.T, cmap='Oranges', norm=normalize)
    fig.colorbar(pc)
    ax.grid(True)
    plt.savefig(directory + '/TARGET_Distribution.JPEG', dpi=300)

    return r, theta, target_dist


def plot_RL_dist(model_name, data):
    plt.rc('font', size=20)
    RL_action = data

    RL_action[:, 0] *= 6
    RL_action[:, 1] *= 180

    tmp = []
    for idx in range(RL_action[:, 1].shape[0]):
        if RL_action[:, 1][idx] < 0:
            tmp.append(RL_action[:, 1][idx] + 360)
        else:
            tmp.append(RL_action[:, 1][idx])
    RL_action[:, 1] = np.array(tmp)

    directory = './Result/{}'.format(model_name)

    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.hist(RL_action[:, 0])
    plt.title('{}_r'.format(model_name))
    plt.savefig(directory + '/{}_r_hist.JPEG'.format(model_name), dpi=300)

    plt.hist(RL_action[:, 1])
    plt.title('{}_theta'.format(model_name))
    plt.savefig(directory + '/{}_theta_hist.JPEG'.format(model_name), dpi=300)

    r = RL_action[:, 0]
    theta = RL_action[:, 1]
    rad = np.deg2rad(theta)

    rbins = np.linspace(0, 6, 13)
    abins = np.linspace(0, 2 * np.pi, 37)

    hist, _, _ = np.histogram2d(rad, r, bins=(abins, rbins))
    A, R = np.meshgrid(abins, rbins)
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8, 8))
    normalize = matplotlib.colors.LogNorm()
    pc = ax.pcolormesh(A, R, hist.T, cmap='Oranges', norm=normalize)
    cb = fig.colorbar(pc)
    ax.grid(True)
    plt.savefig(directory + '/{}_Distribution.JPEG'.format(model_name), dpi=300)

    RL_dist = hist.T.reshape(-1, 1)

    return r, theta, RL_dist


def plot_correlation(corr_dict):
    plt.rc('font', size=20)
    corr_list = []
    for name in corr_dict.keys():
        for name_ in corr_dict.keys():
            corr_list.append(np.corrcoef(corr_dict[name].reshape(-1), corr_dict[name_].reshape(-1))[0][1])
    corr_list = np.array(corr_list)
    corr_list = corr_list.reshape(-1, len(list(corr_dict.keys())))

    plt.rc('font', size=15)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr_list, cmap='Oranges')
    fig.colorbar(cax)

    for (i, j), z in np.ndenumerate(corr_list):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')

    ax.set_xticklabels([''] + list(['Human', 'Target', 'RGB', '15frame', '4frame']))
    ax.set_yticklabels([''] + list(['Human', 'Target', 'RGB', '15frame', '4frame']))
    plt.xticks(rotation=30)
    plt.savefig('./Distribution_corr.jpg', dpi=300)

    return corr_list


def plot_all_dist():
    plt.rc('font', size=30)
    dir_path = "./Result/"

    name_list = []
    dist_list = []

    for (root, directories, files) in os.walk(dir_path):
        for file in files:
            if 'Distribution' in file:
                file_path = os.path.join(root, file)
                name_list.append('_'.join(file_path.split('/')[-1].split('.')[0].split('_')[:-1]))
                dist_list.append(cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB))
    name_list = np.array(name_list)
    dist_list = np.array(dist_list)

    idx = np.array(name_list).argsort()

    name_list = name_list[idx]
    dist_list = dist_list[idx]

    rows = name_list.shape[0] // 3 + 1
    cols = 3

    axes = []
    fig = plt.figure(figsize=(20, 15))

    for a in range(len(name_list)):
        print(idx)
        axes.append(fig.add_subplot(rows, cols, a + 1 if a < 2 else a + 2))
        subplot_title = ()
        axes[-1].set_title(name_list[a])
        plt.axis('off')
        plt.imshow(dist_list[a])
    fig.tight_layout()
    plt.savefig('./All_Distribution.JPEG')


if __name__ == '__main__':
    subj = input('Please enter the subject number : ')
    human_r, human_theta, human_dist = plot_human_dist(subj)
    target_r, target_theta, target_dist = plot_target_dist()

    corr_dict = {'Human': human_dist, 'Target': target_dist}

    data_list = glob('../../Data/RL_model/*/')
    rl_dict = {}

    for idx, path in enumerate(data_list):
        name = path.split('/')[-2]
        pos = np.load(path+'DRL_action.npy')

        rl_dict[name] = pos

    for name, data in rl_dict.items():
        _, _, tmp = plot_RL_dist(name, data)
        corr_dict[name] = tmp

    corr_list = []
    for name in corr_dict.keys():
        for name_ in corr_dict.keys():
            corr_list.append(np.corrcoef(corr_dict[name].reshape(-1), corr_dict[name_].reshape(-1))[0][1])
    corr_list = np.array(corr_list)
    corr_list = corr_list.reshape(-1, len(list(corr_dict.keys())))

    plt.rc('font', size=20)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr_list, cmap='Oranges')
    fig.colorbar(cax)

    for (i, j), z in np.ndenumerate(corr_list):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')

    ax.set_xticklabels([''] + list(corr_dict.keys()))
    ax.set_yticklabels([''] + list(corr_dict.keys()))
    plt.xticks(rotation=30)

