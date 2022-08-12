import pickle
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import rsatoolbox.data as rsd
import rsatoolbox.rdm as rsr
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches

from glob import glob
from numpy import mean

sys.path.append((os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))))
from Anlz.Human.data_anlz import import_behav, cal_theta


def RDM_extract(subj, net_name, train):
    """이 함수는 사람의 이미지를 학습된 강화학습 모델의 입력으로 사용하여 얻은 결과(행동)의 RDM을 추출하기 위한 함수입니다.

    Args:
        subj : 이 매개변수는 피험자 번호를 입력으로 받습니다.
        net_name  : 이 매개변수는 강화학습 모델의 네트워크를 선택합니다. [1] Actor [2] Critic
        train : 이 매개변수는 학습 된 모델인지 아닌지 선택합니다.

    Returns:
        생성된 RDM을 pkl 파일로 저장합니다.
    """

    model_name = net_name

    if train == True:
        train = 'Trained'
    else:
        train = 'Untrained'

    if not os.path.exists(
            './Result/RSA_HDF/{}/{}/{}'.format(subj, train, model_name.capitalize())):
        os.makedirs('./Result/RSA_HDF/{}/{}/{}'.format(subj, train, model_name.capitalize()))

    data_list = glob(
        '../../Data/Human/image/{}/{}/{}/*.pkl'.format(subj, train, model_name.capitalize()))
    data_list.sort()

    dnn_data = {}
    for idx, data in enumerate(data_list):
        with open(data, 'rb') as f:
            tmp = pickle.load(f)

        if idx == 0:
            for name in tmp.keys():
                dnn_data[name] = [np.array(tmp[name]).squeeze()]

        else:
            for name in tmp.keys():
                dnn_data[name].append(np.array(tmp[name]).squeeze())

    for name in dnn_data.keys():
        dnn_data[name] = np.array(dnn_data[name])

    if model_name == 'Actor':
        del dnn_data['action']
    elif model_name == 'Critic':
        del dnn_data['Q_val']

    for name in dnn_data.keys():
        dnn_data[name] = dnn_data[name].reshape(-1, dnn_data[name].shape[2])

    time_data = import_behav(subj)

    RDM_dict = {}
    for name in ['pixel', 'conv1', 'conv2', 'conv3', 'conv4', 'linear', '{}1'.format(model_name.lower()),
                 '{}2'.format(model_name.lower())]:
        tmp = np.array(dnn_data[name]).squeeze()

        pixel_data = []
        for idx in range(tmp.shape[0]):
            for _ in range(4):
                pixel_data.append(tmp[idx])
        pixel_data = np.array(pixel_data)

        RL_data = pd.DataFrame(pixel_data)
        RL_data['time'] = time_data['time']
        time_step = []
        for i in range(0, 120):
            for j in range(50):
                time_step.append(RL_data['time'][1465 * i] + (j + 1) * 0.5)

        tmp = []
        for i in time_step:
            if np.isnan(np.array(
                    (RL_data[(RL_data['time'] < i) & (RL_data['time'] >= (i - 0.5))].mean().values[:-1]))).any():
                pass
            else:
                tmp.append(RL_data[(RL_data['time'] < i) & (RL_data['time'] >= (i - 0.5))].mean().values[:-1])

        tmp = np.array(tmp)

        RDM_dict[name] = []

        for idx in range(120):
            measurements = tmp[idx * 50: (idx + 1) * 50]
            nCond = measurements.shape[0]
            nVox = measurements.shape[1]
            des = {'session': 1, 'subj': 1}
            obs_des = {'conds': np.array(['cond_%02d' % x for x in np.arange(nCond)])}
            chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
            data = rsd.Dataset(measurements=measurements,
                               descriptors=des,
                               obs_descriptors=obs_des,
                               channel_descriptors=chn_des)

            RDM_corr = rsr.calc_rdm(data, method='correlation', descriptor='conds')
            RDM_dict[name].append(RDM_corr)

        with open('./Result/RSA_HDF/{}/{}/{}/RDM.pkl'.format(subj, train, model_name.capitalize()),
                  'wb') as f:
            pickle.dump(RDM_dict, f)


def RDM_spearman(subj, net_name):
    """이 함수는 RSA 분석을 위한 함수입니다.

    Args:
        subj : 이 매개변수는 피험자 번호를 입력으로 받습니다.
        net_name  : 이 매개변수는 강화학습 모델의 네트워크를 선택합니다. [1] Actor [2] Critic

    Returns:
        생성된 RDM을 pkl 파일로 저장합니다.
    """

    model_name = net_name
    time_data = import_behav(subj)
    if not os.path.exists(
            './Result/RSA_HDF/{}/{}/{}'.format(subj, 'Image', model_name.capitalize())):
        os.makedirs('./Result/RSA_HDF/{}/{}/{}'.format(subj, 'Image', model_name.capitalize()))

    # Pixel
    with open('./Result/RSA_HDF/{}/{}/{}/RDM.pkl'.format(subj, 'Trained', model_name.capitalize()), 'rb') as f:
        RDM_dict = pickle.load(f)

    RDM_corr = {}
    for name in list(RDM_dict.keys()):
        RDM_corr[name] = []

        for idx in range(120):
            spearman_corr = scipy.stats.spearmanr(RDM_dict['pixel'][idx].get_vectors().squeeze(),
                                                  RDM_dict[name][idx].get_vectors().squeeze()).correlation
            RDM_corr[name].append(spearman_corr)

    with open('./Result/RSA_HDF/{}/{}/{}/RDM.pkl'.format(subj, 'Untrained', model_name.capitalize()), 'rb') as f:
        RDM_dict = pickle.load(f)

    RDM_corr_u = {}
    for name in list(RDM_dict.keys()):
        RDM_corr_u[name] = []

        for idx in range(120):
            spearman_corr = scipy.stats.spearmanr(RDM_dict['pixel'][idx].get_vectors().squeeze(),
                                                  RDM_dict[name][idx].get_vectors().squeeze()).correlation
            RDM_corr_u[name].append(spearman_corr)

    corr_df = pd.DataFrame(np.array(list(RDM_corr.values())).reshape(-1, 1), columns=['corr'])
    corr_df['layer_name'] = np.array([[name] * 120 for name in RDM_corr.keys()]).reshape(-1, 1)
    corr_df['corr_2'] = np.array(list(RDM_corr_u.values())).reshape(-1, 1)

    sns.set(font_scale=2.5)
    fig = plt.figure(figsize=(25, 6))
    sns.pointplot(x="layer_name", y="corr_2", data=corr_df, estimator=mean, ci='sem', capsize=.05, errwidth=1)
    sns.pointplot(x="layer_name", y="corr", data=corr_df, estimator=mean, ci='sem', capsize=.05, errwidth=1,
                  color='#ff9999')
    red_patch = mpatches.Patch(color='#ff9999', label='Trained')
    black_patch = mpatches.Patch(color='blue', label='Untrained')
    plt.legend(handles=[red_patch, black_patch], bbox_to_anchor=(0.95, 0.8, 0.3, 0.3), loc='upper right')
    plt.xticks(np.arange(8))
    plt.ylim(0, 1.1)

    tmp_list = []
    for idx, name_2 in enumerate(RDM_corr.keys()):
        tmp = scipy.stats.ttest_1samp(RDM_corr[name_2], 0).pvalue
        t_val = scipy.stats.ttest_1samp(RDM_corr[name_2], 0).statistic
        if tmp < 0.001:
            plt.text(idx - 0.16, 0.1, '***')
        elif tmp < 0.01:
            plt.text(idx - 0.16, 0.1, '**')
        elif tmp < 0.05:
            plt.text(idx - 0.16, 0.1, '*')
        tmp_list.append(str('{:.2f}'.format(t_val)) + '(DOF=119)')
    pixel_df = pd.DataFrame(np.array(tmp_list), columns=['1sample'])

    tmp_list = []
    for idx, name_2 in enumerate(RDM_corr.keys()):
        tmp = scipy.stats.ttest_rel(RDM_corr[name_2], RDM_corr_u[name_2]).pvalue
        t_val = scipy.stats.ttest_rel(RDM_corr[name_2], RDM_corr_u[name_2]).statistic
        print('layer name :', name_2, 'p-value', tmp)
        if tmp < 0.001:
            plt.text(idx - 0.16, 0.2, '***')
        elif tmp < 0.01:
            plt.text(idx - 0.16, 0.2, '**')
        elif tmp < 0.05:
            plt.text(idx - 0.16, 0.2, '*')
        tmp_list.append(str('{:.2f}'.format(t_val)) + '(DOF=119)')
    pixel_df['pairwise'] = np.array(tmp_list)

    plt.gcf().subplots_adjust(bottom=0.2)
    plt.ylabel('Spearman correlation')
    plt.xlabel('Layer name')
    plt.title('Pixel')
    plt.savefig('./Result/RSA_HDF/{}/{}/{}/{}.jpg'.format(subj, 'Image', model_name.capitalize(), 'Pixel'), dpi=300)

    # HDF 1. 상대 거리
    data = import_behav(subj)
    Human_pos = np.concatenate([data['cursor'], data['target']], axis=1)
    pos_df = pd.DataFrame(Human_pos, columns=['cur_x', 'cur_y', 'tar_x', 'tar_y'])
    rt_x = (pos_df['tar_x'] - pos_df['cur_x']).values
    rt_y = (pos_df['tar_y'] - pos_df['cur_y']).values
    rt_pos = np.concatenate([rt_x.reshape(-1, 1), rt_y.reshape(-1, 1)], axis=1)
    relative_df = pd.DataFrame(rt_pos, columns=['err_x', 'err_y'])
    relative_df['time'] = time_data['time']

    time_step = []
    for i in range(0, 120):
        for j in range(50):
            time_step.append(relative_df['time'][1465 * i] + (j + 1) * 0.5)

    tmp = []
    for i in time_step:
        if np.isnan(np.array((relative_df[(relative_df['time'] < i) & (relative_df['time'] >= (i - 0.5))].mean().values[
                              :-1]))).any():
            pass
        else:
            data = relative_df[(relative_df['time'] < i) & (relative_df['time'] >= (i - 0.5))]
            pos = data.mean().values[:-1]
            tmp.append(pos)
    tmp = np.array(tmp)

    tmp_x = []
    for idx in range(tmp.shape[0]):
        tmp_x.append(np.sqrt(tmp[idx][0] ** 2 + tmp[idx][1] ** 2))

    tmp_y = []
    for idx in range(tmp.shape[0]):
        tmp_y.append(cal_theta(idx, tmp))

    tmp = np.concatenate([np.array(tmp_x).reshape(-1, 1), np.array(tmp_y).reshape(-1, 1)], axis=1)
    relative_pos = tmp.copy()
    z_tmp = scipy.stats.zscore(tmp)

    with open('./Result/RSA_HDF/{}/{}/{}/RDM.pkl'.format(subj, 'Trained', model_name.capitalize()), 'rb') as f:
        RDM_dict = pickle.load(f)

    del RDM_dict['pixel']
    RDM_dict['HDF_rp'] = []

    for idx in range(120):
        measurements = z_tmp[idx * 50: (idx + 1) * 50]
        nCond = measurements.shape[0]
        nVox = measurements.shape[1]
        # now create a  dataset object
        des = {'session': 1, 'subj': 1}
        obs_des = {'conds': np.array(['cond_%02d' % x for x in np.arange(nCond)])}
        chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des)

        RDM_corr = rsr.calc_rdm(data, method='euclidean', descriptor='conds')
        RDM_dict['HDF_rp'].append(RDM_corr)

    col_order = ['HDF_rp', 'conv1', 'conv2', 'conv3', 'conv4', 'linear', '{}1'.format(model_name.lower()),
                 '{}2'.format(model_name.lower())]

    RDM_corr_t = {}
    for name in col_order:
        RDM_corr_t[name] = []

        for idx in range(120):
            spearman_corr = scipy.stats.spearmanr(RDM_dict['HDF_rp'][idx].get_vectors().squeeze(),
                                                  RDM_dict[name][idx].get_vectors().squeeze()).correlation
            RDM_corr_t[name].append(spearman_corr)

    with open('./Result/RSA_HDF/{}/{}/{}/RDM.pkl'.format(subj, 'Untrained', model_name.capitalize()), 'rb') as f:
        RDM_dict = pickle.load(f)

    del RDM_dict['pixel']

    RDM_dict['HDF_rp'] = []
    for idx in range(120):
        measurements = z_tmp[idx * 50: (idx + 1) * 50]
        nCond = measurements.shape[0]
        nVox = measurements.shape[1]
        # now create a  dataset object
        des = {'session': 1, 'subj': 1}
        obs_des = {'conds': np.array(['cond_%02d' % x for x in np.arange(nCond)])}
        chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des)

        RDM_corr = rsr.calc_rdm(data, method='euclidean', descriptor='conds')
        RDM_dict['HDF_rp'].append(RDM_corr)

    RDM_corr_u = {}
    for name in col_order:
        RDM_corr_u[name] = []

        for idx in range(120):
            spearman_corr = scipy.stats.spearmanr(RDM_dict['HDF_rp'][idx].get_vectors().squeeze(),
                                                  RDM_dict[name][idx].get_vectors().squeeze()).correlation
            RDM_corr_u[name].append(spearman_corr)

    del RDM_corr_u['HDF_rp']
    del RDM_corr_t['HDF_rp']

    corr_df = pd.DataFrame(np.array(list(RDM_corr_t.values())).reshape(-1, 1), columns=['corr'])
    corr_df['layer_name'] = np.array([[name] * 120 for name in RDM_corr_t.keys()]).reshape(-1, 1)
    corr_df['corr_2'] = np.array(list(RDM_corr_u.values())).reshape(-1, 1)

    sns.set(font_scale=1)
    sns.pointplot(x="layer_name", y="corr_2", data=corr_df, estimator=mean, ci='sd', capsize=.05, errwidth=1)
    sns.pointplot(x="layer_name", y="corr", data=corr_df, estimator=mean, ci='sd', capsize=.05, errwidth=1,
                  color='#ff9999')
    red_patch = mpatches.Patch(color='#ff9999', label='Trained')
    black_patch = mpatches.Patch(color='blue', label='Untrained')
    plt.legend(handles=[red_patch, black_patch])
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.ylabel('Spearman correlation')
    plt.xlabel('Layer name')
    plt.title('HDF : Relative Position')
    plt.savefig(
        './Result/RSA_HDF/{}/{}/{}/{}.jpg'.format(subj, 'Image', model_name.capitalize(), 'Relative Position'),
        dpi=300)

    # HDF 2. 상대 속도
    x_dif = []
    for idx in range(pos_df.shape[0] - 1):
        if (idx + 1) % 1465 != 0:
            x_dif.append(pos_df.iloc[idx + 1]['cur_x'] - pos_df.iloc[idx]['cur_x'])
        else:
            x_dif.append(0)
    x_dif.append(0)

    y_dif = []
    for idx in range(pos_df.shape[0] - 1):
        if (idx + 1) % 1465 != 0:
            y_dif.append(pos_df.iloc[idx + 1]['cur_y'] - pos_df.iloc[idx]['cur_y'])
        else:
            y_dif.append(0)
    y_dif.append(0)

    cur_dif = np.concatenate([np.array(x_dif).reshape(-1, 1), np.array(y_dif).reshape(-1, 1)], axis=1)

    x_dif = []
    for idx in range(pos_df.shape[0] - 1):
        if (idx + 1) % 1465 != 0:
            x_dif.append(pos_df.iloc[idx + 1]['tar_x'] - pos_df.iloc[idx]['tar_x'])
        else:
            x_dif.append(0)
    x_dif.append(0)

    y_dif = []
    for idx in range(pos_df.shape[0] - 1):
        if (idx + 1) % 1465 != 0:
            y_dif.append(pos_df.iloc[idx + 1]['tar_y'] - pos_df.iloc[idx]['tar_y'])
        else:
            y_dif.append(0)
    y_dif.append(0)

    tar_dif = np.concatenate([np.array(x_dif).reshape(-1, 1), np.array(y_dif).reshape(-1, 1)], axis=1)

    relative_df = pd.DataFrame(tar_dif - cur_dif, columns=['x_dif', 'y_dif'])
    relative_df['time'] = time_data['time']

    time_step = []
    for i in range(0, 120):
        for j in range(50):
            time_step.append(relative_df['time'][1465 * i] + (j + 1) * 0.5)

    tmp = []
    for i in time_step:
        if np.isnan(np.array((relative_df[(relative_df['time'] < i) & (relative_df['time'] >= (i - 0.5))].mean().values[
                              :-1]))).any():
            pass
        else:
            data = relative_df[(relative_df['time'] < i) & (relative_df['time'] >= (i - 0.5))]
            pos = data.mean().values[:-1]
            tmp.append(pos)
    tmp = np.array(tmp)

    tmp_x = []
    for idx in range(tmp.shape[0]):
        tmp_x.append(np.sqrt(tmp[idx][0] ** 2 + tmp[idx][1] ** 2))

    tmp_y = []
    for idx in range(tmp.shape[0]):
        tmp_y.append(cal_theta(idx, tmp))

    tmp = np.concatenate([np.array(tmp_x).reshape(-1, 1), np.array(tmp_y).reshape(-1, 1)], axis=1)

    relative_vel = tmp.copy()
    z_tmp = scipy.stats.zscore(tmp)

    with open('./Result/RSA_HDF/{}/{}/{}/RDM.pkl'.format(subj, 'Trained', model_name.capitalize()), 'rb') as f:
        RDM_dict = pickle.load(f)

    del RDM_dict['pixel']

    RDM_dict['HDF_rv'] = []

    for idx in range(120):
        measurements = z_tmp[idx * 50: (idx + 1) * 50]
        nCond = measurements.shape[0]
        nVox = measurements.shape[1]
        # now create a  dataset object
        des = {'session': 1, 'subj': 1}
        obs_des = {'conds': np.array(['cond_%02d' % x for x in np.arange(nCond)])}
        chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des)

        RDM_corr = rsr.calc_rdm(data, method='euclidean', descriptor='conds')
        RDM_dict['HDF_rv'].append(RDM_corr)

    col_order = ['HDF_rv', 'conv1', 'conv2', 'conv3', 'conv4', 'linear', '{}1'.format(model_name.lower()),
                 '{}2'.format(model_name.lower())]

    RDM_corr_t = {}
    for name in col_order:
        RDM_corr_t[name] = []

        for idx in range(120):
            spearman_corr = scipy.stats.spearmanr(RDM_dict['HDF_rv'][idx].get_vectors().squeeze(),
                                                  RDM_dict[name][idx].get_vectors().squeeze()).correlation
            RDM_corr_t[name].append(spearman_corr)

    with open('./Result/RSA_HDF/{}/{}/{}/RDM.pkl'.format(subj, 'Untrained', model_name.capitalize()), 'rb') as f:
        RDM_dict = pickle.load(f)

    del RDM_dict['pixel']

    RDM_dict['HDF_rv'] = []
    for idx in range(120):
        measurements = z_tmp[idx * 50: (idx + 1) * 50]
        nCond = measurements.shape[0]
        nVox = measurements.shape[1]
        # now create a  dataset object
        des = {'session': 1, 'subj': 1}
        obs_des = {'conds': np.array(['cond_%02d' % x for x in np.arange(nCond)])}
        chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des)

        RDM_corr = rsr.calc_rdm(data, method='euclidean', descriptor='conds')
        RDM_dict['HDF_rv'].append(RDM_corr)

    RDM_corr_u = {}
    for name in col_order:
        RDM_corr_u[name] = []

        for idx in range(120):
            spearman_corr = scipy.stats.spearmanr(RDM_dict['HDF_rv'][idx].get_vectors().squeeze(),
                                                  RDM_dict[name][idx].get_vectors().squeeze()).correlation
            RDM_corr_u[name].append(spearman_corr)

    del RDM_corr_u['HDF_rv']
    del RDM_corr_t['HDF_rv']

    corr_df = pd.DataFrame(np.array(list(RDM_corr_t.values())).reshape(-1, 1), columns=['corr'])
    corr_df['layer_name'] = np.array([[name] * 120 for name in RDM_corr_t.keys()]).reshape(-1, 1)
    corr_df['corr_2'] = np.array(list(RDM_corr_u.values())).reshape(-1, 1)

    sns.set(font_scale=1)
    sns.pointplot(x="layer_name", y="corr_2", data=corr_df, estimator=mean, ci='sd', capsize=.05, errwidth=1)
    sns.pointplot(x="layer_name", y="corr", data=corr_df, estimator=mean, ci='sd', capsize=.05, errwidth=1,
                  color='#ff9999')
    red_patch = mpatches.Patch(color='#ff9999', label='Trained')
    black_patch = mpatches.Patch(color='blue', label='Untrained')
    plt.legend(handles=[red_patch, black_patch])
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.ylabel('Spearman correlation')
    plt.xlabel('Layer name')
    plt.title('HDF : Relative Velocity')
    plt.savefig(
        './Result/RSA_HDF/{}/{}/{}/{}.jpg'.format(subj, 'Image', model_name.capitalize(), 'Relative Velocity'),
        dpi=300)

    # HDF 3. 상대 속도 + 상대 거리
    relative = np.concatenate([relative_pos, relative_vel], axis=1)
    z_tmp = scipy.stats.zscore(relative)

    with open('./Result/RSA_HDF/{}/{}/{}/RDM.pkl'.format(subj, 'Trained', model_name.capitalize()), 'rb') as f:
        RDM_dict = pickle.load(f)

    del RDM_dict['pixel']

    RDM_dict['HDF_rel'] = []

    for idx in range(120):
        measurements = z_tmp[idx * 50: (idx + 1) * 50]
        nCond = measurements.shape[0]
        nVox = measurements.shape[1]
        # now create a  dataset object
        des = {'session': 1, 'subj': 1}
        obs_des = {'conds': np.array(['cond_%02d' % x for x in np.arange(nCond)])}
        chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des)

        RDM_corr = rsr.calc_rdm(data, method='euclidean', descriptor='conds')
        RDM_dict['HDF_rel'].append(RDM_corr)

    col_order = ['HDF_rel', 'conv1', 'conv2', 'conv3', 'conv4', 'linear', '{}1'.format(model_name.lower()),
                 '{}2'.format(model_name.lower())]

    RDM_corr_t = {}
    for name in col_order:
        RDM_corr_t[name] = []

        for idx in range(120):
            spearman_corr = scipy.stats.spearmanr(RDM_dict['HDF_rel'][idx].get_vectors().squeeze(),
                                                  RDM_dict[name][idx].get_vectors().squeeze()).correlation
            RDM_corr_t[name].append(spearman_corr)

    with open('./Result/RSA_HDF/{}/{}/{}/RDM.pkl'.format(subj, 'Untrained', model_name.capitalize()), 'rb') as f:
        RDM_dict = pickle.load(f)

    del RDM_dict['pixel']

    RDM_dict['HDF_rel'] = []
    for idx in range(120):
        measurements = z_tmp[idx * 50: (idx + 1) * 50]
        nCond = measurements.shape[0]
        nVox = measurements.shape[1]
        # now create a  dataset object
        des = {'session': 1, 'subj': 1}
        obs_des = {'conds': np.array(['cond_%02d' % x for x in np.arange(nCond)])}
        chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des)

        RDM_corr = rsr.calc_rdm(data, method='euclidean', descriptor='conds')
        RDM_dict['HDF_rel'].append(RDM_corr)

    RDM_corr_u = {}
    for name in col_order:
        RDM_corr_u[name] = []

        for idx in range(120):
            spearman_corr = scipy.stats.spearmanr(RDM_dict['HDF_rel'][idx].get_vectors().squeeze(),
                                                  RDM_dict[name][idx].get_vectors().squeeze()).correlation
            RDM_corr_u[name].append(spearman_corr)

    del RDM_corr_u['HDF_rel']
    del RDM_corr_t['HDF_rel']

    corr_df = pd.DataFrame(np.array(list(RDM_corr_t.values())).reshape(-1, 1), columns=['corr'])
    corr_df['layer_name'] = np.array([[name] * 120 for name in RDM_corr_t.keys()]).reshape(-1, 1)
    corr_df['corr_2'] = np.array(list(RDM_corr_u.values())).reshape(-1, 1)
    sns.set(font_scale=1)
    sns.pointplot(x="layer_name", y="corr_2", data=corr_df, estimator=mean, ci='sd', capsize=.05, errwidth=1)
    sns.pointplot(x="layer_name", y="corr", data=corr_df, estimator=mean, ci='sd', capsize=.05, errwidth=1,
                  color='#ff9999')
    red_patch = mpatches.Patch(color='#ff9999', label='Trained')
    black_patch = mpatches.Patch(color='blue', label='Untrained')
    plt.legend(handles=[red_patch, black_patch])
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.ylabel('Spearman correlation')
    plt.xlabel('Layer name')
    plt.title('HDF : Relative velocity + distance')
    plt.savefig('./Result/RSA_HDF/{}/{}/{}/{}.jpg'.format(subj, 'Image', model_name.capitalize(), 'Relative'),
                dpi=300)

    # HDF 4. 커서 속도
    x_dif = []
    for idx in range(pos_df.shape[0] - 1):
        if (idx + 1) % 1465 != 0:
            x_dif.append(pos_df.iloc[idx + 1]['cur_x'] - pos_df.iloc[idx]['cur_x'])
        else:
            x_dif.append(0)
    x_dif.append(0)

    y_dif = []
    for idx in range(pos_df.shape[0] - 1):
        if (idx + 1) % 1465 != 0:
            y_dif.append(pos_df.iloc[idx + 1]['cur_y'] - pos_df.iloc[idx]['cur_y'])
        else:
            y_dif.append(0)
    y_dif.append(0)

    cur_dif = np.concatenate([np.array(x_dif).reshape(-1, 1), np.array(y_dif).reshape(-1, 1)], axis=1)
    relative_df = pd.DataFrame(cur_dif, columns=['x_dif', 'y_dif'])
    relative_df['time'] = time_data['time']

    time_step = []
    for i in range(0, 120):
        for j in range(50):
            time_step.append(relative_df['time'][1465 * i] + (j + 1) * 0.5)

    tmp = []
    for i in time_step:
        if np.isnan(np.array((relative_df[(relative_df['time'] < i) & (relative_df['time'] >= (i - 0.5))].mean().values[
                              :-1]))).any():
            pass
        else:
            data = relative_df[(relative_df['time'] < i) & (relative_df['time'] >= (i - 0.5))]
            pos = data.mean().values[:-1]
            tmp.append(pos)
    tmp = np.array(tmp)

    tmp_x = []
    for idx in range(tmp.shape[0]):
        tmp_x.append(np.sqrt(tmp[idx][0] ** 2 + tmp[idx][1] ** 2))

    tmp_y = []
    for idx in range(tmp.shape[0]):
        tmp_y.append(cal_theta(idx, tmp))

    tmp = np.concatenate([np.array(tmp_x).reshape(-1, 1), np.array(tmp_y).reshape(-1, 1)], axis=1)

    abs_cursor = tmp
    z_tmp = scipy.stats.zscore(tmp)

    with open('./Result/RSA_HDF/{}/{}/{}/RDM.pkl'.format(subj, 'Trained', model_name.capitalize()), 'rb') as f:
        RDM_dict = pickle.load(f)

    del RDM_dict['pixel']

    RDM_dict['HDF_cv'] = []

    for idx in range(120):
        measurements = z_tmp[idx * 50: (idx + 1) * 50]
        nCond = measurements.shape[0]
        nVox = measurements.shape[1]
        # now create a  dataset object
        des = {'session': 1, 'subj': 1}
        obs_des = {'conds': np.array(['cond_%02d' % x for x in np.arange(nCond)])}
        chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des)

        RDM_corr = rsr.calc_rdm(data, method='euclidean', descriptor='conds')
        RDM_dict['HDF_cv'].append(RDM_corr)

    col_order = ['HDF_cv', 'conv1', 'conv2', 'conv3', 'conv4', 'linear', '{}1'.format(model_name.lower()),
                 '{}2'.format(model_name.lower())]
    RDM_corr_t = {}
    for name in col_order:
        RDM_corr_t[name] = []

        for idx in range(120):
            spearman_corr = scipy.stats.spearmanr(RDM_dict['HDF_cv'][idx].get_vectors().squeeze(),
                                                  RDM_dict[name][idx].get_vectors().squeeze()).correlation
            RDM_corr_t[name].append(spearman_corr)

    with open('./Result/RSA_HDF/{}/{}/{}/RDM.pkl'.format(subj, 'Untrained', model_name.capitalize()), 'rb') as f:
        RDM_dict = pickle.load(f)

    del RDM_dict['pixel']

    RDM_dict['HDF_cv'] = []
    for idx in range(120):
        measurements = z_tmp[idx * 50: (idx + 1) * 50]
        nCond = measurements.shape[0]
        nVox = measurements.shape[1]
        # now create a  dataset object
        des = {'session': 1, 'subj': 1}
        obs_des = {'conds': np.array(['cond_%02d' % x for x in np.arange(nCond)])}
        chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des)

        RDM_corr = rsr.calc_rdm(data, method='euclidean', descriptor='conds')
        RDM_dict['HDF_cv'].append(RDM_corr)

    RDM_corr_u = {}
    for name in col_order:
        RDM_corr_u[name] = []

        for idx in range(120):
            spearman_corr = scipy.stats.spearmanr(RDM_dict['HDF_cv'][idx].get_vectors().squeeze(),
                                                  RDM_dict[name][idx].get_vectors().squeeze()).correlation
            RDM_corr_u[name].append(spearman_corr)

    del RDM_corr_u['HDF_cv']
    del RDM_corr_t['HDF_cv']

    corr_df = pd.DataFrame(np.array(list(RDM_corr_t.values())).reshape(-1, 1), columns=['corr'])
    corr_df['layer_name'] = np.array([[name] * 120 for name in RDM_corr_t.keys()]).reshape(-1, 1)
    corr_df['corr_2'] = np.array(list(RDM_corr_u.values())).reshape(-1, 1)

    sns.set(font_scale=1)
    fig = plt.figure(figsize=(10, 3))
    sns.pointplot(x="layer_name", y="corr_2", data=corr_df, estimator=mean, ci='sd', capsize=.05, errwidth=1)
    sns.pointplot(x="layer_name", y="corr", data=corr_df, estimator=mean, ci='sd', capsize=.05, errwidth=1,
                  color='#ff9999')
    red_patch = mpatches.Patch(color='#ff9999', label='Trained')
    black_patch = mpatches.Patch(color='blue', label='Untrained')
    plt.legend(handles=[red_patch, black_patch])
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.ylabel('Spearman correlation')
    plt.xlabel('Layer name')
    plt.title('HDF : Cursor velocity')
    plt.savefig('./Result/RSA_HDF/{}/{}/{}/{}.jpg'.format(subj, 'Image', model_name.capitalize(), 'Cursor Velocity'),
                dpi=300)

    # HDF 5. 타겟 속도
    x_dif = []
    for idx in range(pos_df.shape[0] - 1):
        if (idx + 1) % 1465 != 0:
            x_dif.append(pos_df.iloc[idx + 1]['tar_x'] - pos_df.iloc[idx]['tar_x'])
        else:
            x_dif.append(0)
    x_dif.append(0)

    y_dif = []
    for idx in range(pos_df.shape[0] - 1):
        if (idx + 1) % 1465 != 0:
            y_dif.append(pos_df.iloc[idx + 1]['tar_y'] - pos_df.iloc[idx]['tar_y'])
        else:
            y_dif.append(0)
    y_dif.append(0)

    tar_dif = np.concatenate([np.array(x_dif).reshape(-1, 1), np.array(y_dif).reshape(-1, 1)], axis=1)

    relative_df = pd.DataFrame(cur_dif, columns=['x_dif', 'y_dif'])
    relative_df['time'] = time_data['time']

    time_step = []
    for i in range(0, 120):
        for j in range(50):
            time_step.append(relative_df['time'][1465 * i] + (j + 1) * 0.5)

    tmp = []
    for i in time_step:
        if np.isnan(np.array((relative_df[(relative_df['time'] < i) & (relative_df['time'] >= (i - 0.5))].mean().values[
                              :-1]))).any():
            pass
        else:
            data = relative_df[(relative_df['time'] < i) & (relative_df['time'] >= (i - 0.5))]
            pos = data.mean().values[:-1]
            tmp.append(pos)
    tmp = np.array(tmp)

    tmp_x = []
    for idx in range(tmp.shape[0]):
        tmp_x.append(np.sqrt(tmp[idx][0] ** 2 + tmp[idx][1] ** 2))

    tmp_y = []
    for idx in range(tmp.shape[0]):
        tmp_y.append(cal_theta(idx, tmp))

    tmp = np.concatenate([np.array(tmp_x).reshape(-1, 1), np.array(tmp_y).reshape(-1, 1)], axis=1)

    abs_target = tmp.copy()

    z_tmp = scipy.stats.zscore(tmp)

    with open('./Result/RSA_HDF/{}/{}/{}/RDM.pkl'.format(subj, 'Trained', model_name.capitalize()), 'rb') as f:
        RDM_dict = pickle.load(f)

    del RDM_dict['pixel']
    RDM_dict['HDF_tv'] = []

    for idx in range(120):
        measurements = z_tmp[idx * 50: (idx + 1) * 50]
        nCond = measurements.shape[0]
        nVox = measurements.shape[1]
        # now create a  dataset object
        des = {'session': 1, 'subj': 1}
        obs_des = {'conds': np.array(['cond_%02d' % x for x in np.arange(nCond)])}
        chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des)

        RDM_corr = rsr.calc_rdm(data, method='euclidean', descriptor='conds')
        RDM_dict['HDF_tv'].append(RDM_corr)

    col_order = ['HDF_tv', 'conv1', 'conv2', 'conv3', 'conv4', 'linear', '{}1'.format(model_name.lower()),
                 '{}2'.format(model_name.lower())]
    RDM_corr_t = {}
    for name in col_order:
        RDM_corr_t[name] = []

        for idx in range(120):
            spearman_corr = scipy.stats.spearmanr(RDM_dict['HDF_tv'][idx].get_vectors().squeeze(),
                                                  RDM_dict[name][idx].get_vectors().squeeze()).correlation
            RDM_corr_t[name].append(spearman_corr)

    with open('./Result/RSA_HDF/{}/{}/{}/RDM.pkl'.format(subj, 'Untrained', model_name.capitalize()), 'rb') as f:
        RDM_dict = pickle.load(f)

    del RDM_dict['pixel']
    RDM_dict['HDF_tv'] = []
    for idx in range(120):
        measurements = z_tmp[idx * 50: (idx + 1) * 50]
        nCond = measurements.shape[0]
        nVox = measurements.shape[1]
        # now create a  dataset object
        des = {'session': 1, 'subj': 1}
        obs_des = {'conds': np.array(['cond_%02d' % x for x in np.arange(nCond)])}
        chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des)

        RDM_corr = rsr.calc_rdm(data, method='euclidean', descriptor='conds')
        RDM_dict['HDF_tv'].append(RDM_corr)

    RDM_corr_u = {}
    for name in col_order:
        RDM_corr_u[name] = []

        for idx in range(120):
            spearman_corr = scipy.stats.spearmanr(RDM_dict['HDF_tv'][idx].get_vectors().squeeze(),
                                                  RDM_dict[name][idx].get_vectors().squeeze()).correlation
            RDM_corr_u[name].append(spearman_corr)

    del RDM_corr_u['HDF_tv']
    del RDM_corr_t['HDF_tv']

    corr_df = pd.DataFrame(np.array(list(RDM_corr_t.values())).reshape(-1, 1), columns=['corr'])
    corr_df['layer_name'] = np.array([[name] * 120 for name in RDM_corr_t.keys()]).reshape(-1, 1)
    corr_df['corr_2'] = np.array(list(RDM_corr_u.values())).reshape(-1, 1)

    sns.set(font_scale=1)
    fig = plt.figure(figsize=(10, 3))
    sns.pointplot(x="layer_name", y="corr_2", data=corr_df, estimator=mean, ci='sd', capsize=.05, errwidth=1)
    sns.pointplot(x="layer_name", y="corr", data=corr_df, estimator=mean, ci='sd', capsize=.05, errwidth=1,
                  color='#ff9999')
    red_patch = mpatches.Patch(color='#ff9999', label='Trained')
    black_patch = mpatches.Patch(color='blue', label='Untrained')
    plt.legend(handles=[red_patch, black_patch])
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.ylabel('Spearman correlation')
    plt.xlabel('Layer name')
    plt.title('HDF : Target velocity')
    plt.savefig('./Result/RSA_HDF/{}/{}/{}/{}.jpg'.format(subj, 'Image', model_name.capitalize(), 'Target Velocity'),
                dpi=300)

    # HDF 6. 절대 속도
    absolute = np.concatenate([abs_cursor, abs_target], axis=1)
    z_tmp = scipy.stats.zscore(absolute)

    with open('./Result/RSA_HDF/{}/{}/{}/RDM.pkl'.format(subj, 'Trained', model_name.capitalize()), 'rb') as f:
        RDM_dict = pickle.load(f)

    del RDM_dict['pixel']
    RDM_dict['HDF_abs'] = []

    for idx in range(120):
        measurements = z_tmp[idx * 50: (idx + 1) * 50]
        nCond = measurements.shape[0]
        nVox = measurements.shape[1]
        # now create a  dataset object
        des = {'session': 1, 'subj': 1}
        obs_des = {'conds': np.array(['cond_%02d' % x for x in np.arange(nCond)])}
        chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des)

        RDM_corr = rsr.calc_rdm(data, method='euclidean', descriptor='conds')
        RDM_dict['HDF_abs'].append(RDM_corr)

    col_order = ['HDF_abs', 'conv1', 'conv2', 'conv3', 'conv4', 'linear', '{}1'.format(model_name.lower()),
                 '{}2'.format(model_name.lower())]

    RDM_corr_t = {}
    for name in col_order:
        RDM_corr_t[name] = []

        for idx in range(120):
            spearman_corr = scipy.stats.spearmanr(RDM_dict['HDF_abs'][idx].get_vectors().squeeze(),
                                                  RDM_dict[name][idx].get_vectors().squeeze()).correlation
            RDM_corr_t[name].append(spearman_corr)

    with open('./Result/RSA_HDF/{}/{}/{}/RDM.pkl'.format(subj, 'Untrained', model_name.capitalize()), 'rb') as f:
        RDM_dict = pickle.load(f)

    del RDM_dict['pixel']

    RDM_dict['HDF_abs'] = []
    for idx in range(120):
        measurements = z_tmp[idx * 50: (idx + 1) * 50]
        nCond = measurements.shape[0]
        nVox = measurements.shape[1]
        # now create a  dataset object
        des = {'session': 1, 'subj': 1}
        obs_des = {'conds': np.array(['cond_%02d' % x for x in np.arange(nCond)])}
        chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des)

        RDM_corr = rsr.calc_rdm(data, method='euclidean', descriptor='conds')
        RDM_dict['HDF_abs'].append(RDM_corr)

    RDM_corr_u = {}
    for name in col_order:
        RDM_corr_u[name] = []

        for idx in range(120):
            spearman_corr = scipy.stats.spearmanr(RDM_dict['HDF_abs'][idx].get_vectors().squeeze(),
                                                  RDM_dict[name][idx].get_vectors().squeeze()).correlation
            RDM_corr_u[name].append(spearman_corr)

    del RDM_corr_u['HDF_abs']
    del RDM_corr_t['HDF_abs']

    corr_df = pd.DataFrame(np.array(list(RDM_corr_t.values())).reshape(-1, 1), columns=['corr'])
    corr_df['layer_name'] = np.array([[name] * 120 for name in RDM_corr_t.keys()]).reshape(-1, 1)
    corr_df['corr_2'] = np.array(list(RDM_corr_u.values())).reshape(-1, 1)

    sns.set(font_scale=1)
    fig = plt.figure(figsize=(10, 3))
    sns.pointplot(x="layer_name", y="corr_2", data=corr_df, estimator=mean, ci='sd', capsize=.05, errwidth=1)
    sns.pointplot(x="layer_name", y="corr", data=corr_df, estimator=mean, ci='sd', capsize=.05, errwidth=1,
                  color='#ff9999')
    red_patch = mpatches.Patch(color='#ff9999', label='Trained')
    black_patch = mpatches.Patch(color='blue', label='Untrained')
    plt.legend(handles=[red_patch, black_patch])
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.ylabel('Spearman correlation')
    plt.xlabel('Layer name')
    plt.title('HDF : Absolute velocity')
    plt.savefig(
        './Result/RSA_HDF/{}/{}/{}/{}.jpg'.format(subj, 'Image', model_name.capitalize(), 'Absolute Velocity'),
        dpi=300)

    # HDF 7. Total
    total = np.concatenate([relative, absolute], axis=1)
    z_tmp = scipy.stats.zscore(total)

    with open('./Result/RSA_HDF/{}/{}/{}/RDM.pkl'.format(subj, 'Trained', model_name.capitalize()), 'rb') as f:
        RDM_dict = pickle.load(f)

    del RDM_dict['pixel']

    RDM_dict['HDF_total'] = []

    for idx in range(120):
        measurements = z_tmp[idx * 50: (idx + 1) * 50]
        nCond = measurements.shape[0]
        nVox = measurements.shape[1]
        # now create a  dataset object
        des = {'session': 1, 'subj': 1}
        obs_des = {'conds': np.array(['cond_%02d' % x for x in np.arange(nCond)])}
        chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des)

        RDM_corr = rsr.calc_rdm(data, method='euclidean', descriptor='conds')
        RDM_dict['HDF_total'].append(RDM_corr)

    col_order = ['HDF_total', 'conv1', 'conv2', 'conv3', 'conv4', 'linear', '{}1'.format(model_name.lower()),
                 '{}2'.format(model_name.lower())]

    RDM_corr_t = {}
    for name in col_order:
        RDM_corr_t[name] = []

        for idx in range(120):
            spearman_corr = scipy.stats.spearmanr(RDM_dict['HDF_total'][idx].get_vectors().squeeze(),
                                                  RDM_dict[name][idx].get_vectors().squeeze()).correlation
            RDM_corr_t[name].append(spearman_corr)

    with open('./Result/RSA_HDF/{}/{}/{}/RDM.pkl'.format(subj, 'Untrained', model_name.capitalize()), 'rb') as f:
        RDM_dict = pickle.load(f)

    del RDM_dict['pixel']

    RDM_dict['HDF_total'] = []
    for idx in range(120):
        measurements = z_tmp[idx * 50: (idx + 1) * 50]
        nCond = measurements.shape[0]
        nVox = measurements.shape[1]
        # now create a  dataset object
        des = {'session': 1, 'subj': 1}
        obs_des = {'conds': np.array(['cond_%02d' % x for x in np.arange(nCond)])}
        chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des)

        RDM_corr = rsr.calc_rdm(data, method='euclidean', descriptor='conds')
        RDM_dict['HDF_total'].append(RDM_corr)

    RDM_corr_u = {}
    for name in col_order:
        RDM_corr_u[name] = []

        for idx in range(120):
            spearman_corr = scipy.stats.spearmanr(RDM_dict['HDF_total'][idx].get_vectors().squeeze(),
                                                  RDM_dict[name][idx].get_vectors().squeeze()).correlation
            RDM_corr_u[name].append(spearman_corr)

    del RDM_corr_u['HDF_total']
    del RDM_corr_t['HDF_total']

    corr_df = pd.DataFrame(np.array(list(RDM_corr_t.values())).reshape(-1, 1), columns=['corr'])
    corr_df['layer_name'] = np.array([[name] * 120 for name in RDM_corr_t.keys()]).reshape(-1, 1)
    corr_df['corr_2'] = np.array(list(RDM_corr_u.values())).reshape(-1, 1)

    sns.set(font_scale=2.5)
    fig = plt.figure(figsize=(25, 6))
    sns.pointplot(x="layer_name", y="corr_2", data=corr_df, estimator=mean, ci='sem', capsize=.05, errwidth=1)
    sns.pointplot(x="layer_name", y="corr", data=corr_df, estimator=mean, ci='sem', capsize=.05, errwidth=1,
                  color='#ff9999')
    red_patch = mpatches.Patch(color='#ff9999', label='Trained')
    black_patch = mpatches.Patch(color='blue', label='Untrained')
    plt.legend(handles=[red_patch, black_patch], bbox_to_anchor=(0.95, 0.8, 0.3, 0.3), loc='upper right')
    plt.xticks(np.arange(8))
    plt.ylim(-0.2, 0.6)

    print('=' * 50)
    print('1samp ttest')
    print('=' * 50)
    tmp_list = []
    for idx, name_2 in enumerate(RDM_corr_t.keys()):
        tmp = scipy.stats.ttest_1samp(RDM_corr_t[name_2], 0).pvalue
        t_val = scipy.stats.ttest_1samp(RDM_corr_t[name_2], 0).statistic
        print('layer_name :', name_2, 'p-value :', tmp)
        if tmp < 0.001:
            plt.text(idx - 0.16, 0.4, '***')
        elif tmp < 0.01:
            plt.text(idx - 0.16, 0.4, '**')
        elif tmp < 0.05:
            plt.text(idx - 0.16, 0.4, '*')
        tmp_list.append(str('{:.2f}'.format(t_val)) + '(DOF=119)')
    hdf_df = pd.DataFrame(tmp_list, columns=['1sample'])

    tmp_list = []
    print('=' * 50)
    print('pairwise ttest')
    print('=' * 50)
    for idx, name_2 in enumerate(RDM_corr_t.keys()):
        tmp = scipy.stats.ttest_rel(RDM_corr_t[name_2], RDM_corr_u[name_2]).pvalue
        t_val = scipy.stats.ttest_rel(RDM_corr_t[name_2], RDM_corr_u[name_2]).statistic
        print('layer_name :', name_2, 'p-value :', tmp)
        if tmp < 0.001:
            plt.text(idx - 0.16, 0.5, '***')
        elif tmp < 0.01:
            plt.text(idx - 0.16, 0.5, '**')
        elif tmp < 0.05:
            plt.text(idx - 0.16, 0.5, '*')
        tmp_list.append(str('{:.2f}'.format(t_val)) + '(DOF=119)')
    hdf_df['pairwise'] = np.array(tmp_list)

    plt.gcf().subplots_adjust(bottom=0.2)
    plt.ylabel('Spearman correlation')
    plt.xlabel('Layer name')
    plt.title('HDF')
    plt.savefig('./Result/RSA_HDF/{}/{}/{}/{}.jpg'.format(subj, 'Image', model_name.capitalize(), 'Total'), dpi=300)

    with open('./Result/RSA_HDF/{}/HDF_RDM.pkl'.format(subj), 'wb') as f:
        pickle.dump(RDM_dict['HDF_total'], f)

    return pixel_df, hdf_df


if __name__ == '__main__':

    # 1. Extract Representational Dissimilarity Matrix
    subj = input('Subj number : ')
    net_name = input('Network name : ')

    RDM_extract(subj, net_name, True)
    RDM_extract(subj, net_name, False)

    # 2. RSA Analysis (Pixel-RL, HDF-RL)
    RDM_spearman(subj, net_name)