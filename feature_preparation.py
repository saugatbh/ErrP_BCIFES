
"""
This script is used to extract the features necessary that are to be applied as inputs
to the classifier. The following steps are followed:
1. Apply a temporal filter for 6 Hz.
2. Resample the epochs by 16.
3. Apply Laplacian filter on FCx, Cz and CPz.
4. Use baseline correction

Usage: python feature_preparation.py

Author: Saugat Bhattacharyya,
        Ulster University
"""

import scipy as sp
import numpy as np
import scipy.io as sio
from scipy.signal import remez


def detrend(trials):
    eeg = trials
    ones = np.matrix(np.ones((eeg.shape[0], 1)))
    r = np.matrix(np.arange(0, 1 + 0.000001, 1. / (eeg.shape[1] - 1)))
    p = ones * r
    e = np.diag(np.mean(eeg[:, -4:], axis=-1) - np.mean(eeg[:, :5], axis=-1))
    detrended = eeg - (e * p + np.reshape(np.mean(eeg[:, :5], axis=-1), (eeg.shape[0], 1)))
    return detrended


def preprocess(epoch, srate, h, dsample, reshape=True):
    pre = int(0.2 * srate + .5)
    f = []
    for c in epoch:  # we use 64 EEG channels
        f.append(sp.convolve(c - c[:pre].mean(), h, mode='same')[int(dsample / 2)::dsample])
    if reshape:
        return sp.concatenate(f)
    else:
        return sp.array(f)


# Constants
channels = ['FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CPz', 'CP4', 'P3', 'Pz', 'P4']
type_of_analysis = ['fes', 'vis']
sample_rate = 256
no_electrodes = len(channels)
epoch_length = int(sample_rate * 6.0)
pre_cue = int(sample_rate * 1.5)
save_preprocessed_epoch = True

# This initializes the filter
filt_len, low_cut, high_cut, dsample = (.4, 4, 6, 16)
h = remez(int(sample_rate * filt_len), [0, low_cut, high_cut, sample_rate / 2.], [1, 0], Hz=sample_rate)

# Down-sampling constants
dsampled_srate = int(sample_rate/dsample)
dsampled_length = int(6*dsampled_srate)
print(dsampled_srate, dsampled_length)

# Neighbors for Laplacian
selected_channels = ['FCz', 'Cz', 'CPz']
neighbors_FCz = ['FC3', 'FC4', 'Cz']
neighbors_Cz = ['C3', 'FCz', 'C4', 'CPz']
neighbors_CPz = ['CP3', 'Cz', 'CP4', 'Pz']

# paths
data_path = 'C:/Users/sauga/Google Drive/Datasets/bcifes/'  # Change the path according to the location of the data
save_results = 'results/'

# Load the epochs and labels from the mat files
print('Loading epochs and labels')
correctness = {}
epochs = {}
people = {}
resampled_epochs = {}
spat_filt_epochs = {}
baseline_corrected_epochs = {}
for analysis in type_of_analysis:
    print(analysis)
    resampled_epochs[analysis] = {}
    spat_filt_epochs[analysis] = {}
    baseline_corrected_epochs[analysis] = {}
    if analysis == 'fes':
        subjects = ['1', '2', '3', '13', '15', '16', '17', '18']
        people[analysis] = len(subjects)
        sub_index = [0, 1, 2, 3, 4, 5, 6, 7]
        corr = sio.loadmat(data_path + 'correctness_fes.mat')['correctness'][0]
        epochs[analysis] = {}
        correctness[analysis] = {}
        for sIdx, subject in enumerate(subjects):
            epochs[analysis][sIdx] = sio.loadmat(data_path + 's' + subject + '_valid.mat')['EEGEpoch']
            correctness[analysis][sIdx] = corr[sub_index[sIdx]][0][0]
            print(epochs[analysis][sIdx].shape, correctness[analysis][sIdx].shape)
    elif analysis == 'vis':
        subjects = ['4', '5', '6', '7', '8', '9', '10', '12']
        people[analysis] = len(subjects)
        sub_index = [0, 1, 2, 3, 4, 5, 6, 7]
        corr = sio.loadmat(data_path + 'correctness_vis.mat')['correctness_lr']
        epochs[analysis] = {}
        correctness[analysis] = {}
        for sIdx, subject in enumerate(subjects):
            epochs[analysis][sIdx] = sio.loadmat(data_path + 's' + subject + '_valid.mat')['EEGEpoch']
            correctness[analysis][sIdx] = corr[0][sub_index[sIdx]][0][0]
            print(epochs[analysis][sIdx].shape, correctness[analysis][sIdx].shape)
    for sIdx in range(people[analysis]):
        # Detrend and resample epochs
        print('Detrending and resampling Subject', sIdx, 'Group:', analysis)
        ep0 = epochs[analysis][sIdx].T
        resampled_epochs[analysis][sIdx] = np.empty((np.asarray(ep0).shape[0], no_electrodes, dsampled_length))
        for te in range(np.asarray(ep0).shape[0]):
            resampled_epochs[analysis][sIdx][te, :, :] = detrend(preprocess(ep0[te, :, :epoch_length], sample_rate,
                                                                            h, dsample, reshape=False))
        # print(resampled_epochs[analysis][sIdx].shape)

        # Laplacian Spatial Filter
        print('Applying Laplacian Spatial Filter Subject', sIdx, 'Group:', analysis)
        ep1 = resampled_epochs[analysis][sIdx]
        spat_filt_epochs[analysis][sIdx] = np.empty((np.asarray(ep0).shape[0], len(selected_channels),
                                                     int(dsampled_srate * 6)))
        for c, channel in enumerate(selected_channels):
            for te in range(np.asarray(ep1).shape[0]):
                if channel == 'FCz':
                    sig = ep1[te, channels.index(channel), :]
                    n_sig = []
                    for idx in neighbors_FCz:
                        n_sig.append(ep1[te, channels.index(idx), :])
                    mean_sig = np.mean(n_sig)
                    spat_filt_epochs[analysis][sIdx][te, c, :] = (sig - mean_sig) * 1.0 / len(neighbors_FCz)
                elif channel == 'Cz':
                    sig = ep1[te, channels.index(channel), :]
                    n_sig = []
                    for idx in neighbors_Cz:
                        n_sig.append(resampled_epochs[analysis][sIdx][te, channels.index(idx), :])
                    mean_sig = np.mean(n_sig)
                    spat_filt_epochs[analysis][sIdx][te, c, :] = (sig - mean_sig) * 1.0 / len(neighbors_Cz)
                elif channel == 'CPz':
                    sig = ep1[te, channels.index(channel), :]
                    n_sig = []
                    for idx in neighbors_CPz:
                        n_sig.append(resampled_epochs[analysis][sIdx][te, channels.index(idx), :])
                    mean_sig = np.mean(n_sig)
                    spat_filt_epochs[analysis][sIdx][te, c, :] = (sig - mean_sig) * 1.0 / len(neighbors_CPz)

        # Baseline Corrected
        print('Baseline Correction Subject', sIdx, 'Group:', analysis)
        ep2 = spat_filt_epochs[analysis][sIdx]
        baseline_corrected_epochs[analysis][sIdx] = np.empty((np.asarray(ep2).shape[0], len(selected_channels),
                                                              int(dsampled_srate * 1.5)))
        for trial in range(np.asarray(ep2).shape[0]):
            for c in range(len(selected_channels)):
                baseline_region = ep2[trial, c, int(0. * dsampled_srate):int(.3 * dsampled_srate)]
                roi_epoch = ep2[trial, c, int(2.5 * dsampled_srate):int(4 * dsampled_srate)]
                count = 0
                corrected = []
                while count < int(dsampled_srate * 1.5):
                    corrected.extend(roi_epoch[count:count + len(baseline_region)] - baseline_region)
                    count += len(baseline_region)
                baseline_corrected_epochs[analysis][sIdx][trial, c, :] = corrected
        print(baseline_corrected_epochs[analysis][sIdx].shape)
if save_preprocessed_epoch:
    preprocessed_data = {'epochs': baseline_corrected_epochs, 'classes': correctness}
    np.save('pickles/pre-processed_epochs.npy', preprocessed_data, allow_pickle=True)

