import os, sys, pickle
import nibabel as nib
import numpy as np
import rsatoolbox
import rsatoolbox.data as rsd
import rsatoolbox.rdm as rsr
import scipy.stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nilearn import image, plotting, masking
from numpy import mean
from glob import glob

subj = 'AG06'

full_mask = nib.load('/mnt/sdb2/imlim/proj_AGI/fMRI_data/{}/full_mask.nii'.format(subj))
anat_file = nib.load('/mnt/sdb2/imlim/proj_AGI/fMRI_data/{}/anat_final.nii.gz'.format(subj))

path_list = glob('/mnt/sdb2/imlim/proj_AGI/atlases/data/roi/*.nii')

early_roi_list = [path_list[-4], path_list[-3], path_list[-2], path_list[-1]]
ppc_roi_list = [path_list[2], path_list[-5], path_list[-6], path_list[4]]
motor_roi_list = [path_list[5], path_list[7], path_list[6]]

for path in early_roi_list:
    roi = image.load_img(path)
    plotting.plot_glass_brain(roi, title=path.split('/')[-1].split('.')[0])
    plt.show()

fMRI_data_paths = list(glob('/mnt/sdb2/imlim/proj_AGI/fMRI_data/{}/pb02.r0*.nii'.format(subj)))
fMRI_data_paths.sort()
print(fMRI_data_paths)

fMRI_data = []
for path in fMRI_data_paths:
    fMRI_data.append(nib.load(path))

roi_name = 'Motor/Frontal'
roi_RDM = {}
for idx, path in enumerate(motor_roi_list):
    tmp = image.load_img(path)
    if idx == 0:
        img1 = tmp
    else:
        img1 = image.math_img('img1+img2', img1=img1, img2=tmp)
roi = img1
plotting.plot_glass_brain(roi)
fmri_array = masking.apply_mask(fMRI_data, roi)
print('Masked image shape : {}'.format(fmri_array.shape))

print(fmri_array.shape)
# img = []
# for idx in range(120):
#     for j in range(50):
#         img.append(fmri_array[70 * idx + j ])
# fmri_array = np.array(img)
print('Resampled image shape : {}'.format(fmri_array.shape))
roi_RDM[roi_name] = []

for idx in range(120):
    print('\rRDM process // Index {} has been processed..'.format(idx + 1), (idx * 70 + 12) / 2, ((idx) * 70 + 62) / 2,
          end='')
    measurements = fmri_array[idx * 70 + 12: (idx) * 70 + 62]
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

    RDM_corr = rsr.calc_rdm(data, method='correlation', descriptor='conds')
    #     fig, ax, ret_val = rsatoolbox.vis.show_rdm(RDM_corr, figsize=(6,6), show_colorbar='panel')
    roi_RDM[roi_name].append(RDM_corr)

with open('/mnt/sdb2/imlim/proj_AGI/Result/Human_RDM/AG06/Trained/Actor/RDM.pkl', 'rb') as f:
    train_RDM = pickle.load(f)

if not os.path.exists('./RDM/untrained/Actor'):
    os.makedirs('./RDM/untrained/Actor')

with open('/mnt/sdb2/imlim/proj_AGI/Result/Human_RDM/AG06/HDF_RDM.pkl', 'rb') as f:
    HDF_RDM = pickle.load(f)

train_RDM['HDF'] = HDF_RDM

RDM_corr = {}
for layer in list(roi_RDM.keys()):
    print('=' * 50)
    print('Layer name : {}'.format(layer))
    print('=' * 50)

    RDM_corr[layer] = {}
    for name in list(train_RDM.keys()):

        RDM_corr[layer][name] = []

        for idx in range(120):
            spearman_corr = scipy.stats.spearmanr(roi_RDM[layer][idx].get_vectors().squeeze(),
                                                  train_RDM[name][idx].get_vectors().squeeze()).correlation
            RDM_corr[layer][name].append(spearman_corr)
    print('RDM process has been processed..')

with open('/mnt/sdb2/imlim/proj_AGI/Result/Human_RDM/AG06/Untrained/Actor/RDM.pkl', 'rb') as f:
    train_RDM = pickle.load(f)

train_RDM['HDF'] = HDF_RDM

RDM_corr_u = {}
for layer in list(roi_RDM.keys()):
    print('=' * 50)
    print('Layer name : {}'.format(layer))
    print('=' * 50)

    RDM_corr_u[layer] = {}
    for name in list(train_RDM.keys()):

        RDM_corr_u[layer][name] = []

        for idx in range(120):
            spearman_corr = scipy.stats.spearmanr(roi_RDM[layer][idx].get_vectors().squeeze(),
                                                  train_RDM[name][idx].get_vectors().squeeze()).correlation
            RDM_corr_u[layer][name].append(spearman_corr)
    print('RDM process has been processed..')

from matplotlib.offsetbox import OffsetImage,AnnotationBbox
from matplotlib.cbook import get_sample_data
import scipy.stats
plotting.plot_glass_brain(roi, cmap='Reds', vmin=0, vmax = 3, output_file = '{}_roi.png'.format(list(roi_RDM.keys())[0].split('/')[0]))
im = plt.imread(('{}_roi.png'.format(list(roi_RDM.keys())[0].split('/')[0])))

for name in RDM_corr.keys():

    corr_df = pd.DataFrame(RDM_corr[name])
    sns.set(font_scale=2)
    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(111)
    sns.pointplot(data=corr_df, estimator=mean, ci='sem', capsize=.05, errwidth=1, color='#ff9999')
    corr_df = pd.DataFrame(RDM_corr_u[name])
    sns.pointplot(data=corr_df, estimator=mean, ci='sem', capsize=.05, errwidth=1)
    import matplotlib.patches as mpatches

    red_patch = mpatches.Patch(color='#ff9999', label='Trained')
    black_patch = mpatches.Patch(color='blue', label='Untrained')
    plt.legend(handles=[red_patch, black_patch], loc='upper left')
    plt.ylim(0, 0.22)
    plt.ylabel('Spearman correlation')
    plt.xlabel('ROI name')
    plt.xticks(np.arange(0, 9, 1), list(RDM_corr[roi_name].keys()), rotation=45)
    plt.yticks(np.arange(0, 0.21, 0.05))
    plt.title('Name : {}'.format(name))
    print('pairwise')
    tmp_list = []
    for idx, name_2 in enumerate(RDM_corr[name].keys()):
        tmp = scipy.stats.ttest_rel(RDM_corr[name][name_2], RDM_corr_u[name][name_2]).pvalue
        t_val = scipy.stats.ttest_rel(RDM_corr[name][name_2], RDM_corr_u[name][name_2]).statistic
        print(tmp)
        if tmp < 0.001:
            plt.text(idx - 0.16, 0.02, '***')
        elif tmp < 0.01:
            plt.text(idx - 0.16, 0.02, '**')
        elif tmp < 0.05:
            plt.text(idx - 0.16, 0.02, '*')
        tmp_list.append(str('{:.2f}'.format(t_val)))
    ROI_df = pd.DataFrame(np.array(tmp_list), columns=['pairwise'])

    tmp_list = []
    print('1sample')
    for idx, name_2 in enumerate(RDM_corr[name].keys()):
        tmp = scipy.stats.ttest_1samp(RDM_corr[name][name_2], 0).pvalue
        t_val = scipy.stats.ttest_1samp(RDM_corr[name][name_2], 0).statistic
        print(tmp)
        if tmp < 0.001:
            plt.text(idx - 0.16, 0.01, '***')
        elif tmp < 0.01:
            plt.text(idx - 0.16, 0.01, '**')
        elif tmp < 0.05:
            plt.text(idx - 0.16, 0.01, '*')
        tmp_list.append(str('{:.2f}'.format(t_val)))

    ROI_df['1sample'] = np.array(tmp_list)

    newax = fig.add_axes([0.65, 0.75, 0.2, 0.2], anchor='NE', zorder=-1)
    newax.imshow(im)
    newax.axis('off')
    plt.gcf().subplots_adjust(bottom=0.15, top=0.75)
    plt.savefig('/mnt/sdb2/imlim/proj_AGI/Result/Human_RDM/AG05/Trained/Critic/{}.jpg'.format(name.split('/')[0]),
                dpi=300)