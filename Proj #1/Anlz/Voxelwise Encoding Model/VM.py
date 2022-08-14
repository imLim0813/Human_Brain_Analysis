import os, sys
import pickle
import numpy as np
import nibabel as nib
import nilearn
import nilearn.plotting
import copy
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

from glob import glob
from nilearn import masking, plotting
from nilearn import image
from scipy import stats
from voxelwise_tutorials.utils import explainable_variance
from nilearn.glm import threshold_stats_img

sys.path.append((os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))))
from Anlz.Human.data_anlz import import_behav
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import check_cv
from voxelwise_tutorials.utils import generate_leave_one_run_out
from voxelwise_tutorials.delayer import Delayer
from himalaya.kernel_ridge import KernelRidgeCV
from sklearn.pipeline import make_pipeline
from himalaya.backend import set_backend
from sklearn import set_config

subj = 'AG06'
net_name = 'Actor'
trained_ = 'Trained'


def ev_fmri(subj):
    """이 함수는 fMRI 데이터를 호출하기 위한 함수입니다.

    Args:
        subj : 이 매개변수는 피험자 번호를 지정하기 위한 매개변수다.

    Returns:
        fMRI data : rest time을 제외한 피험자 fMRI 데이터

    """
    data_path = '../../Data/Human/fMRI/{}/'.format(subj)
    data = []
    for idx in range(1, 7, 1):

        tmp = os.path.join(data_path, 'pb04.r0{}.nii'.format(idx))
        f_img = image.get_data(tmp)
        img = []
        for idx_2 in range(20):
            for j in range(50):
                img.append(f_img[:, :, :, 70 * idx_2 + j])
        f_img = np.array(img)
        mask = nib.load('../../Data/Human/fMRI/{}/full_mask.nii'.format(subj))
        f_img = f_img.transpose((1, 2, 3, 0))

        f_img = nib.Nifti1Image(f_img, mask.affine, mask.header)
        f_img = masking.apply_mask(f_img, mask)

        f_img = f_img.reshape(f_img.shape[0], -1)
        f_img = stats.zscore(f_img, 0)
        data.append(f_img)
        print('=' * 50)
        print('pb04.r0{} has been stored!'.format(idx))
        print('=' * 50)

    data = np.array(data)

    print('Data type : {}, Data shape : {}'.format(type(data), data.shape))

    return data


# fMRI 데이터 분석
# ev = explainable_variance(ev_data)
#
# voxel_1 = np.nanargmax(ev)
# time = np.arange(ev_data.shape[1]) * 0.5
# plt.figure(figsize=(10, 3))
# plt.plot(time, ev_data[:, :, voxel_1].T, color='C0', alpha=0.5)
# plt.plot(time, ev_data[:, :, voxel_1].mean(0), color='C1', label='average')
# plt.xlabel('Time (sec)')
# plt.title('Voxel with large explainable variance (%.2f)' % ev[voxel_1])
# plt.legend()
# plt.tight_layout()
# plt.show()
#
# voxel_1 = np.nanargmin(ev)
# time = np.arange(ev_data.shape[1]) * 0.5
# plt.figure(figsize=(10, 3))
# plt.plot(time, ev_data[:, :, voxel_1].T, color='C0', alpha=0.5)
# plt.plot(time, ev_data[:, :, voxel_1].mean(0), color='C1', label='average')
# plt.xlabel('Time (sec)')
# plt.title('Voxel with large explainable variance (%.2f)' % ev[voxel_1])
# plt.legend()
# plt.tight_layout()
# plt.show()


def vem(subj, ev_data):
    mask_file = nib.load('../../Data/Human/fMRI/{}/full_mask.nii'.format(subj))
    anat_file = nilearn.image.load_img('../../Data/Human/fMRI/{}/anat_final.nii.gz'.format(subj))

    ev_data = ev_data.reshape(-1, ev_data.shape[2])

    data_list = glob(
        '/mnt/sdb2/imlim/proj_AGI/RL_data/{}/{}/Human_RL_{}_*.pkl'.format(subj, trained_, net_name.lower()))

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

    RL_data = dnn_data
    del RL_data['pixel']

    if net_name == 'Critic':
        del RL_data['Q_val']
    elif net_name == 'Actor':
        del RL_data['action']

    for name in RL_data.keys():
        RL_data[name] = RL_data[name].reshape(-1, RL_data[name].shape[2])

    for key in RL_data.keys():
        data = np.array(RL_data[key]).squeeze()
        tmp = []
        for idx in range(data.shape[0]):
            for _ in range(4):
                tmp.append(data[idx])
        tmp = np.array(tmp)
        RL_data[key] = tmp
        print('Key : {}'.format(key))
        print('=' * 50)
        print('Upsampling has been completed..')
        print('=' * 50)

    for key in dnn_data.keys():
        print('{} shape : {}'.format(key, dnn_data[key].shape))

    time_data = import_behav(subj)
    ds_data = {}

    for key in RL_data.keys():
        print('=' * 50)
        print('Key name : {}'.format(key))
        print('=' * 50)
        RL_df = pd.DataFrame(RL_data[key])
        RL_df['time'] = time_data['time']
        time_step = []

    # Each run has 20 trials. So, total number of trials is 120.
    for i in range(0, 120):

        # Each trial runs 25 seconds. TR is 0.5 seconds. So, 25 / 0.5 = 50.
        for j in range(50):
            time_step.append(RL_df['time'][1465 * i] + (j + 1) * 0.5)

    tmp = []

    # Downsampling
    for i in time_step:
        print('\rTime step {:.2f} has been processed..'.format(i), end='')
        if np.isnan(np.array((RL_df[(RL_df['time'] < i) & (RL_df['time'] >= (i - 0.5))].mean().values[:-1]))).any():
            pass
        else:
            tmp.append(RL_df[(RL_df['time'] < i) & (RL_df['time'] >= (i - 0.5))].mean().values[:-1])

    tmp = np.array(tmp)
    ds_data[key] = tmp
    print('')

    for key in ds_data.keys():
        print('{} shape : {}'.format(key, ds_data[key].shape))

    y_train, y_test = ev_data[:5000], ev_data[5000:]
    for name in ds_data.keys():

        train, test = ds_data[name][:5000], ds_data[name][5000:]

        std = StandardScaler()
        pca = PCA(n_components=100)

        pca_data = pca.fit_transform(std.fit_transform(train))
        print('Key : {}'.format(name))
        print('=' * 50)
        print('PCA operation has been completed..')
        print('Explainable Variance : {}'.format(pca.explained_variance_ratio_.cumsum()[-1]))
        print('=' * 50)

        if name == 'conv1':
            X_train = pca_data
            X_test = pca.transform(std.transform(test))
        else:
            X_train = np.concatenate([X_train, pca_data], axis=1)
            X_test = np.concatenate([X_test, pca.transform(std.transform(test))], axis=1)

    run_onsets = np.arange(0, 5000, 1000)
    n_samples_train = X_train.shape[0]
    cv = generate_leave_one_run_out(n_samples_train, run_onsets)
    cv = check_cv(cv)  # copy the cross-validation splitter into a reusable list

    # CCN에 따르면 prediction accuracy 높게 tuning 필요하다고 함.

    delayer = Delayer(delays=[4, 6, 8, 10, 12])
    backend = set_backend("torch_cuda", on_error="warn")

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    alphas = np.logspace(1, 3, 100)

    kernel_ridge_cv = KernelRidgeCV(
        alphas=alphas, cv=cv,
        solver_params=dict(n_targets_batch=5000, n_alphas_batch=5,
                           n_targets_batch_refit=100))

    scaler = StandardScaler(with_mean=True, with_std=True)

    pipeline = make_pipeline(
        delayer,
        kernel_ridge_cv,
    )

    set_config(display='diagram')  # requires scikit-learn 0.23

    _ = pipeline.fit(X_train, y_train)

    scores = pipeline.score(X_test, y_test)
    print("(n_voxels,) =", scores.shape)

    scores = backend.to_numpy(scores)
    for idx in range(scores.shape[0]):
        if scores[idx] < 0:
            scores[idx] = 0
        else:
            scores[idx] = scores[idx]

    Y_pred = pipeline.predict(X_test)

    r_score = []
    for idx in range(Y_pred.shape[1]):
        r_score.append(scipy.stats.pearsonr(Y_pred[:, idx], y_test[:, idx]))

    r_score = np.array(r_score)

    def z_transform(r, n):
        z = np.log((1 + r) / (1 - r)) * (np.sqrt(n - 3) / 2)
        return z

    z_score = z_transform(r_score[:, 0], n=1000)

    thresholded_map2, threshold2 = threshold_stats_img(
        masking.unmask(z_score, mask_file), alpha=0.0001, height_control='fdr')

    if not os.path.exists('./VM_Result/{}/{}/{}'.format(subj, trained_, net_name)):
        os.makedirs('./VM_Result/{}/{}/{}'.format(subj, trained_, net_name))

    # R2_score
    nib.save(masking.unmask(scores, mask_file), './VM_Result/{}/{}/{}/r2_score.nii'.format(subj, trained_, net_name))

    # pearson correlation
    nib.save(masking.unmask(r_score[:, 0], mask_file),
             './VM_Result/{}/{}/{}/r_score.nii'.format(subj, trained_, net_name))

    # z score
    nib.save(masking.unmask(r_score[:, 0], mask_file),
             './VM_Result/{}/{}/{}/z_score.nii'.format(subj, trained_, net_name))

    # FDR 0.0001
    nib.save(thresholded_map2, './VM_Result/{}/{}/{}/FDR(0.0001).nii'.format(subj, trained_, net_name))


if __name__ == '__main__':
    subj = input('Subj number : ')
    vem(subj, ev_fmri(subj))
