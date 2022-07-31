import pickle
import cv2
import os

import nibabel
import nibabel as nib
import pygame as G
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pygame.rect import *
from PIL import Image
from nilearn import image

from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
from nilearn.glm.first_level import compute_regressor
from nilearn.plotting import plot_design_matrix
from Env.func import euclidean_distance, os_driver, distance

os_driver(True)


def import_behav(participant, run='All'):
    """이 함수는 실험을 통해 얻은 행동데이터를 호출하기 위해 사용되는 함수입니다.

    Args:
        participant : 이 매개변수는 피험자 번호를 입력으로 받습니다.
        run : 이 매개변수는 Run 번호를 지정할 때 사용합니다. default : 'All'

        ex) participant : AG01, run : 1

    Returns:
        행동 데이터를 딕셔너리로 반환합니다.
        딕셔너리의 key는 ['cursor', 'target', 'joystick', 'hit', 'time']으로 구성되어 있습니다.

    """

    if run != 'All':
        with open('../../Data/Human/behavior/{}/behavior_data_{}.pkl'.format(participant, run), 'rb') as f:
            data = pickle.load(f)

        for name in data.keys():
            data[name] = np.array(data[name])

        data['cursor'] = np.array(data['cursor']) - 35

    if run == 'All':
        data = {}

        cursor = []
        target = []
        joystick = []
        hit = []
        time = []

        run_list = [i for i in range(1, 7)]
        for run in run_list:
            tmp = import_behav(participant, run)

            cursor.append(tmp['cursor'])
            target.append(tmp['target'])
            joystick.append(tmp['joystick'])
            hit.append(tmp['hit'])
            time.append(np.array(tmp['time']) + 700 * (run - 1))

        data['cursor'] = np.array(cursor).reshape(-1, np.array(tmp['cursor']).shape[1])
        data['target'] = np.array(target).reshape(-1, np.array(tmp['target']).shape[1])
        data['joystick'] = np.array(joystick).reshape(-1, np.array(tmp['joystick']).shape[1])
        data['hit'] = np.array(hit).reshape(-1, np.array(tmp['hit']).shape[1])
        data['time'] = np.array(time).reshape(-1, np.array(tmp['time']).shape[1])

    return data


def to_video(data, run, trial, participant):
    """이 함수는 실험을 동영상화 시키기 위한 함수입니다.
       단, 이 함수는 VIDEO DRIVER가 off된 상황에서는 오류를 일으킵니다.

    Args:
        data : 이 매개변수는 행동 데이터를 입력으로 받습니다.
        run  : 이 매개변수는 Run 번호를 지정할 때 사용합니다.
        trial : 이 매개변수는 trial 번호를 지정할 때 사용합니다.
        participant : 이 매개변수는 저장될 폴더의 이름을 위해 사용합니다.

    Returns:
        실험의 지정된 run과 지정된 trial을 name의 이름으로 동영상을 저장합니다.

    """

    tmp = data
    G.init()
    screen = G.display.set_mode([1920, 1080])
    frame = []

    for i in range(20 * 1465 * (run - 1) + 1465 * (trial - 1), 20 * 1465 * (run - 1) + 1465 * trial):
        screen.fill(G.Color(0, 0, 0))
        t_rect = Rect(int(tmp['target'][i][0]), int(tmp['target'][i][1]), 70, 70)
        G.draw.ellipse(screen, G.Color(0, 0, 255), t_rect, 0)

        if tmp['hit'][:-1][i][0] == 1:
            G.draw.ellipse(screen, G.Color(255, 0, 0), t_rect, 0)

        G.draw.line(screen, G.Color(255, 255, 255), (int(tmp['cursor'][:-1][i][0]) - 10 + 35, int(tmp['cursor'][:-1][i][1]) + 35),
                    (int(tmp['cursor'][:-1][i][0]) + 10 + 35, int(tmp['cursor'][:-1][i][1]) + 35), 3)
        G.draw.line(screen, G.Color(255, 255, 255), (int(tmp['cursor'][:-1][i][0]) + 35, int(tmp['cursor'][:-1][i][1] - 10 + 35)),
                    (int(tmp['cursor'][:-1][i][0]) + 35, int(tmp['cursor'][:-1][i][1] + 10 + 35)), 3)
        G.display.flip()
        G.image.save(screen, 'abc.BMP')
        png = Image.open('./abc.BMP')
        png.load()

        background = Image.new("RGB", png.size, (255, 255, 255))
        background.paste(png, mask=list(png.split())[3])
        frame.append(background)

    frame_array = []
    for i in range(len(frame)):
        frame_array.append(cv2.cvtColor(np.array(frame[i]), cv2.COLOR_RGB2BGR))

    height, width, layers = frame_array[0].shape
    size = (width, height)

    if not os.path.exists('./Video'):
        os.makedirs('./Video')

    if not os.path.exists('./Video/{}'.format(participant)):
        os.makedirs('./Video/{}'.format(participant))

    if not os.path.exists('./Video/{}/Run{}'.format(participant, run)):
        os.makedirs('./Video/{}/Run{}'.format(participant, run))

    out = cv2.VideoWriter('./Video/{}/Run{}/Trial {:02d}.mp4'.format(participant, run, trial), fourcc=0x7634706d,
                          fps=60, frameSize=size)
    for i in range(1465):
        out.write(frame_array[i])
    out.release()


def to_image(data, run, trial, participant):
    """이 함수는 실험을 이미지화 시키기 위한 함수입니다.

    Args:
        data : 이 매개변수는 행동 데이터를 입력으로 받습니다.
        run  : 이 매개변수는 Run 번호를 지정할 때 사용합니다.
        trial : 이 매개변수는 trial 번호를 지정할 때 사용합니다.
        participant : 이 매개변수는 이미지가 저장될 폴더를 위해 사용합니다.

    Returns:
        실험의 지정된 run과 지정된 trial을 participant 폴더 내 이미지를 저장합니다.

    """

    tmp = data
    G.init()
    screen = G.display.set_mode([1920, 1080])
    frame = []
    screen.fill(G.Color(255, 255, 255))
    color_list = [(0, 0, 0), (78, 112, 189)]

    for i in range(20 * 1465 * (run - 1) + 1465 * (trial - 1), 20 * 1465 * (run - 1) + 1465 * (trial)):
        G.event.pump()

        G.draw.circle(screen, G.Color(color_list[0]), (int(tmp['target'][i][0]), int(tmp['target'][i][1])), 3, 0)
        G.draw.circle(screen, G.Color(color_list[1]), (int(tmp['cursor'][i][0]), int(tmp['cursor'][i][1])), 3, 0)

        G.display.flip()

    string_image = G.image.tostring(screen, 'RGB')
    temp_surf = G.image.fromstring(string_image, (1920, 1080), 'RGB')
    tmp = G.surfarray.array3d(temp_surf)
    tmp = tmp.transpose((1, 0, 2))

    plt.imshow(tmp)
    plt.axis('off')
    plt.text(20, 0, '- : target')
    plt.text(20, 70, '- : cursor', c='b')

    if not os.path.exists('./Path'):
        os.makedirs('./Path')

    if not os.path.exists('./Path/{}'.format(participant)):
        os.makedirs('./Path/{}'.format(participant))

    if not os.path.exists('./Path/{}/Run{}'.format(participant, run)):
        os.makedirs('./Path/{}/Run{}'.format(participant, run))

    plt.savefig('./Path/{}/Run{}/Trial {:02d}.png'.format(participant, run, trial), dpi=300)


def plot_learning_curve(behav_data, mode, subj):
    """이 함수는 피험자 hit_rate를 이미지화 시키기 위한 함수입니다.

    Args:
        behav_data : 이 매개변수는 행동 데이터를 입력으로 받습니다.
        mode : 이 매개변수는 실험에 사용된 mode를 지정할 때 사용합니다.
        subj : 이 매개변수는 이미지가 저장될 폴더를 위해 사용합니다.

    Returns:
        실험의 지정된 run과 지정된 trial을 participant 폴더 내 이미지를 저장합니다.

    """
    if not os.path.exists('./Learning_curve'):
        os.makedirs('./Learning_curve')

    if not os.path.exists('./Learning_curve/{}'.format(subj)):
        os.makedirs('./Learning_curve/{}'.format(subj))

    if mode == 'Base':

        hit_rate = []
        for trial in range(20 * 6):
            hit_sum = np.array(behav_data['hit'][1465 * trial: 1465 * (trial + 1)]).sum()
            hit_rate.append(float('{:.2f}'.format(hit_sum / 1465)))

        plt.rcParams["font.family"] = 'AppleGothic'
        plt.rcParams["font.size"] = 12
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(1, 121, 1), hit_rate, color='gray')

        for idx in range(0, 120, 20):
            plt.axvline([idx], color='#ff9999', linestyle=':', linewidth=1)

        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.title('Learning rate ( {} )'.format(subj))
        plt.ylabel('Hit rate')
        plt.xlim(0, 120)
        plt.xlabel('Trial number')

        for idx, i in enumerate(range(7, 120, 20)):
            plt.text(i, 0.1, 'Run {}'.format(idx + 1))

        plt.savefig('./Learning_curve/{}/Learning rate({}).jpg'.format(subj, mode))

    if mode == 'Adap':
        hit_rate = []
        for trial in range(20 * 6):
            hit_sum = np.array(behav_data['hit'][1465 * trial: 1465 * (trial + 1)]).sum()
            hit_rate.append(float('{:.2f}'.format(hit_sum / 1465)))
        hit_rate.append(hit_rate[-1])

        color_list = ['gray', '#ff9999', '#ff9999', 'gray', 'gray', '#ff9999', '#ff9999', 'gray', 'gray', '#ff9999',
                      '#ff9999', 'gray']

        plt.rcParams["font.family"] = 'AppleGothic'
        plt.rcParams["font.size"] = 12
        plt.figure(figsize=(10, 4))

        label_list = ['Adap', 'Base']

        for i in range(12):
            if i < 10:
                plt.plot(np.arange(i * 10, 10 * (i + 1) + 1, 1), hit_rate[10 * i: 10 * (i + 1) + 1],
                         color=color_list[i])
            else:
                plt.plot(np.arange(i * 10, 10 * (i + 1) + 1, 1), hit_rate[10 * i: 10 * (i + 1) + 1],
                         color=color_list[i], label=label_list[i % 2])

        for idx in range(0, 120, 10):
            plt.axvline([idx], color='#ff9999', linestyle=':', linewidth=1)

        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.title('Learning rate ( {} )'.format(subj))
        plt.ylabel('Hit rate')
        plt.xlim(0, 120)
        plt.xlabel('Trial number')

        for idx, i in enumerate(range(7, 120, 20)):
            plt.text(i, 0.4, 'Run {}'.format(idx + 1))

        plt.legend(loc='lower right')

        plt.savefig('./Learning_curve/{}/Learning rate({}).jpg'.format(subj, mode))


def total_fMRI(participant, save=True):
    """이 함수는 피험자의 각 run 별 fMRI데이터를 하나로 합치기 위한 함수입니다.
    """
    data_path = '../../Data/Human/fMRI/{}/'.format(participant)

    if os.path.isfile(os.path.join(data_path, 'pb04.total.nii')):
        print('=' * 50)
        print('Total fMRI file already exists..')
        print('=' * 50)

        concat_img = image.load_img(os.path.join(data_path, 'pb04.total.nii'))

        return concat_img

    r01_nii = os.path.join(data_path, 'pb04.r01.nii')
    r02_nii = os.path.join(data_path, 'pb04.r02.nii')
    r03_nii = os.path.join(data_path, 'pb04.r03.nii')
    r04_nii = os.path.join(data_path, 'pb04.r04.nii')
    r05_nii = os.path.join(data_path, 'pb04.r05.nii')
    r06_nii = os.path.join(data_path, 'pb04.r06.nii')

    n1_img = nib.load(r01_nii)
    n2_img = nib.load(r02_nii)
    n3_img = nib.load(r03_nii)
    n4_img = nib.load(r04_nii)
    n5_img = nib.load(r05_nii)
    n6_img = nib.load(r06_nii)

    n1_header = n1_img.header

    concat_img = image.concat_imgs([n1_img, n2_img, n3_img, n4_img, n5_img, n6_img])
    concat_header = concat_img.header
    concat_header['cal_max'] = n1_header['cal_max']

    if save:
        print('=' * 50)
        print('Total fMRI file has been saved..')
        print('=' * 50)
        nibabel.save(concat_img, os.path.join(data_path, 'pb04.total.nii'))

    return concat_img


def total_data(behav_data):
    """이 함수는 rest data를 기존 실험 데이터에 추가 시키기 위한 함수입니다.

    Args:
        behav_data : 이 매개변수는 행동 데이터를 입력으로 받습니다.

    """
    for name in behav_data.keys():
        tmp = []

        if name == 'cursor' or name == 'target':
            rest_data = [960, 540]

        if name == 'joystick':
            rest_data = [0, 0]

        if name == 'hit':
            rest_data = [0]

        for i in range(1, behav_data[name].shape[0] + 1, 1):
            tmp.append(behav_data[name][i - 1])

            if i % 1465 == 0 and name != 'time':
                for _ in range(586):
                    tmp.append(rest_data)

            if i % 1465 == 0 and name == 'time':
                for idx in range(1, 587, 1):
                    if i != 175800:
                        time_diff = behav_data[name][i] - behav_data[name][i - 1]
                    if i == 175800:
                        time_diff = 4200 - behav_data[name][i - 1]
                    tmp.append(behav_data[name][i - 1] + time_diff * idx / 586)

        behav_data[name] = np.array(tmp)

    return behav_data


def sampling(data):
    """이 함수는 행동데이터로부터 적절한 regressor를 추출하기 위한 함수입니다.

    Args:
        data : 이 매개변수는 행동 데이터를 입력으로 받습니다.

    """
    tmp = {}
    state = np.concatenate([data['cursor'], data['target']], axis=1)
    tmp['distance'] = data['target'] - data['cursor']
    tmp['euclidean'] = []
    for idx in range(state.shape[0]):
        tmp['euclidean'].append(euclidean_distance(state[idx]))
    tmp['euclidean'] = np.array(tmp['euclidean']).reshape(-1, 1)
    tmp['joystick'] = data['joystick']
    tmp['radian'] = []
    for idx in range(tmp['joystick'].shape[0]):
        tmp['radian'].append(
            [np.sqrt(data['joystick'][idx][0] ** 2 + data['joystick'][idx][1] ** 2), cal_theta(data['joystick'], idx)])
    tmp['radian'] = np.array(tmp['radian'])
    tmp['hit'] = data['hit']
    tmp['time'] = data['time']

    return tmp


def cal_theta(data, idx):
    """이 함수는 Cartesian coordinate을 Polar coordinate으로 변환시 사용됩니다.

    Args:
        data : 이 매개변수는 조이스틱 데이터를 입력으로 받습니다.
        idx : 각 row를 지정하기 위해 사용됩니다.

    """
    if data[idx][1] == 0 or data[idx][0] == 0:
        theta = 0
        return theta

    if np.rad2deg(np.arctan(data[idx][1] / data[idx][0])) > 0:
        if data[idx][0] > 0:
            theta = np.rad2deg(np.arctan(data[idx][1] / data[idx][0]))
        elif data[idx][0] < 0:
            theta = np.rad2deg(np.arctan(data[idx][1] / data[idx][0])) + 180

    elif np.rad2deg(np.arctan(data[idx][1] / data[idx][0])) < 0:
        if data[idx][0] > 0:
            theta = 360 + np.rad2deg(np.arctan(data[idx][1] / data[idx][0]))
        elif data[idx][0] < 0:
            theta = np.rad2deg(np.arctan(data[idx][1] / data[idx][0])) + 180

    return theta


def make_design_matrix(data, name, subj):
    """이 함수는 행동 데이터로부터 디자인 매트릭스를 만들기 위한 함수입니다.

    Args:
        name : 이 매개변수는 key값을 지정하기 위한 매개변수다.
        subj : 이 매개변수는 피험자 번호를 지정하기 위한 매개변수다.
    """
    TR_df = pd.DataFrame(data[name], columns=['column_{}'.format(i) for i in range(data[name].shape[1])])
    TR_df['time'] = np.array(data['time'])
    tmp = []
    for i in np.arange(0.5, 4200.1, 0.5):
        tmp.append(TR_df[(TR_df['time'] < i) & (TR_df['time'] >= (i - 0.5))].mean().values[:-1])
    resample_df = pd.DataFrame(tmp, columns=TR_df.keys()[:-1].tolist())
    resample_df['time'] = [0.5 * i for i in range(0, 8400, 1)]

    tr = 0.5
    n_scans = 1400 * 6
    frame_times = np.arange(n_scans) * 0.5
    conditions = TR_df.keys()[:-1].tolist()

    print('=' * 50)
    print('Motion data directory')
    print('=' * 50)
    print('../../Data/Human/fMRI/{}/motion_demean.1D'.format(subj))
    print('')

    add_reg_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']

    motion_r01 = pd.read_csv('../../Data/Human/fMRI/{}/motion_demean.{}.1D'.format(subj, 'r01'),
                             delimiter=' ', names=add_reg_names)
    motion_r02 = pd.read_csv('../../Data/Human/fMRI/{}/motion_demean.{}.1D'.format(subj, 'r02'),
                             delimiter=' ', names=add_reg_names)
    motion_r03 = pd.read_csv('../../Data/Human/fMRI/{}/motion_demean.{}.1D'.format(subj, 'r03'),
                             delimiter=' ', names=add_reg_names)
    motion_r04 = pd.read_csv('../../Data/Human/fMRI/{}/motion_demean.{}.1D'.format(subj, 'r04'),
                             delimiter=' ', names=add_reg_names)
    motion_r05 = pd.read_csv('../../Data/Human/fMRI/{}/motion_demean.{}.1D'.format(subj, 'r05'),
                             delimiter=' ', names=add_reg_names)
    motion_r06 = pd.read_csv('../../Data/Human/fMRI/{}/motion_demean.{}.1D'.format(subj, 'r06'),
                             delimiter=' ', names=add_reg_names)
    motion = pd.concat([motion_r01, motion_r02, motion_r03, motion_r04, motion_r05, motion_r06], axis=0)

    onset = np.arange(0, 4200, 0.5)
    duration = np.array([0.5] * 8400)

    trial_type = ['stim' for i in range(50)]

    for i in range(20):
        trial_type.append('rest')
    trial_type = np.array(trial_type * 20 * 6)

    event_df = pd.DataFrame()

    event_df['trial_type'] = trial_type
    event_df['onset'] = onset
    event_df['duration'] = duration

    dsg_mat = make_first_level_design_matrix(frame_times=frame_times, events=event_df, drift_model='polynomial',
                                             drift_order=3, add_regs=motion.values, add_reg_names=add_reg_names,
                                             hrf_model='spm')

    hrf_reg = []
    for name in conditions:
        exp_condition = np.array((onset, duration, resample_df[name])).reshape(3, -1)
        signal, _name = compute_regressor(exp_condition, hrf_model='spm', frame_times=frame_times)
        hrf_reg.append(signal)
    hrf_reg = np.array(hrf_reg)
    hrf_reg = hrf_reg.reshape(-1, 8400).transpose((1, 0))
    hrf_reg = pd.DataFrame(hrf_reg, columns=conditions)

    dsg_mat.drop('rest', axis=1, inplace=True)
    dsg_mat.drop('stim', axis=1, inplace=True)

    hrf_reg.set_index(onset, inplace=True)

    reg_df = pd.concat([hrf_reg, dsg_mat], axis=1)

    plot_design_matrix(reg_df)

    print('=' * 50)
    print('Desing Matrix has been maded!')
    print('=' * 50)
    print('')

    return reg_df


def Generalized_Linear_Model(subj, fMRI_data, design_matrix, name):
    mask_img = image.load_img('../../Data/Human/fMRI/{}/full_mask.nii'.format(subj))
    anat_img = image.load_img('../../Data/Human/fMRI/{}/anat_final.nii.gz'.format(subj))

    func_img = fMRI_data

    fmri_glm = FirstLevelModel(0.5, mask_img=mask_img, hrf_model='spm', drift_model='polynomial', drift_order=3,
                               minimize_memory=False)

    glm_fit = fmri_glm.fit(func_img, design_matrices=design_matrix)

    contrast = np.zeros(design_matrix.shape[1])
    for i in range(0, design_matrix.keys().tolist().index('tx'), 1):
        contrast[i] = 1

    if not os.path.exists('./fMRI'):
        os.makedirs('./fMRI')

    if not os.path.exists('./fMRI/{}'.format(subj)):
        os.makedirs('./fMRI/{}'.format(subj))

    z_map = glm_fit.compute_contrast(contrast, output_type='z_score')
    z_map.to_filename('./fMRI/{}/{}.{}.nii.gz'.format(subj, name, 'z_score'))

    print('=' * 50)
    print('NII file has been maded!')
    print('=' * 50)
    print('')


if __name__ == '__main__':
    print('=' * 50)
    print('Please select the number of analyses')
    num = int(input('[1] Video [2] Path [3] Learning curve [4] GLM // '))
    print('=' * 50)

    subj = input('Subj number : ')

    if num == 1:
        run = int(input('Run number : '))
        trial = int(input('Trial number : '))
        data = import_behav(subj, run='All')
        to_video(data, run, trial, subj)

    if num == 2:
        run = int(input('Run number : '))
        trial = int(input('Trial number : '))
        data = import_behav(subj, run='All')
        to_image(data, run, trial, subj)

    if num == 3:
        data = import_behav(subj, run='All')
        mode = int(input('Mode select : [1] Base [2] Adaptation // '))
        if mode == 1:
            mode = 'Base'
        elif mode == 2:
            mode = 'Adap'
        plot_learning_curve(data, mode, subj)

    if num == 4:
        data = import_behav(subj, run='All')
        total_behav = total_data(data)
        sample_behav = sampling(total_behav)
        fMRI_data = total_fMRI(subj, True)
        for idx, tmp in enumerate(sample_behav.keys()):
            print('[{}] {} '.format(idx+1, tmp), end='')
        print('')
        key = input('Please choose the feature')
        dsg_mat = make_design_matrix(sample_behav, key, subj)
        Generalized_Linear_Model(subj, fMRI_data, dsg_mat, key)
