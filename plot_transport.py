
"""
Plot the trasnportation during Optimal Transport

Usage: python plot_transport.py

Author: Saugat Bhattacharyya,
        Ulster University
"""

import numpy as np
from sklearn.model_selection import KFold
import ot
from ot.plot import plot2D_samples_mat
from sklearn.preprocessing import power_transform
import matplotlib.pylab as pl

data = np.load('pickles/pre-processed_epochs.npy', allow_pickle=True)

type_of_analysis = ['fes', 'vis']
sampling_rate = 256.0
people = 8
kf = KFold(n_splits=people, random_state=people, shuffle=True)
eeg_epochs = {}
labels = {}
kf = KFold(n_splits=people, random_state=people, shuffle=True)
for analysis in type_of_analysis:
    eeg_epochs[analysis] = data[()]['epochs'][analysis]
    labels[analysis] = data[()]['classes'][analysis]
    for fold, (train, test) in enumerate(kf.split(np.zeros((people, 1)))):
        train_epochs = []
        test_epochs = []
        train_labels = []
        test_labels = []
        for sIdx in train:
            train_epochs.extend(eeg_epochs[analysis][sIdx])
            train_labels.extend(labels[analysis][sIdx])
        for sIdx in test:
            test_epochs.extend(eeg_epochs[analysis][sIdx])
            test_labels.extend(labels[analysis][sIdx])
        train_epochs = np.array(train_epochs).reshape(np.array(train_epochs).shape[0],
                                                      np.array(train_epochs).shape[1]*np.array(train_epochs).shape[2])
        train_epochs = power_transform(train_epochs)
        train_labels = np.array(train_labels)
#         print(train_labels)
        train_labels_bin = np.zeros_like(train_labels)
        test_labels_bin = np.zeros_like(test_labels)
        for idx, x in enumerate(train_labels):
            if x == -1:
                train_labels_bin[idx] = 0
            else:
                train_labels_bin[idx] = 1
        test_epochs = np.array(test_epochs).reshape(np.array(test_epochs).shape[0],
                                                    np.array(test_epochs).shape[1]*np.array(test_epochs).shape[2])
        test_labels = np.array(test_labels)
        for idx, x in enumerate(test_labels):
            if x == -1:
                test_labels_bin[idx] = 0
            else:
                test_labels_bin[idx] = 1
        test_epochs = power_transform(test_epochs)
        ot_emd_laplace = ot.da.SinkhornTransport(reg_e=10)
        ot_emd_laplace.fit(Xs=train_epochs, Xt=test_epochs, ys=train_labels_bin, yt=test_labels)
        transp_Xs = ot_emd_laplace.transform(Xs=train_epochs)
        transp_Xt = ot_emd_laplace.inverse_transform(Xs=test_epochs)
        pl.figure(figsize=(10, 10))
        pl.subplot(2, 2, 1)
        pl.scatter(train_epochs[:, 0], train_epochs[:, -1], c=train_labels, marker='+', cmap=pl.cm.jet,
                   label='Source samples')
        pl.xticks([])
        pl.yticks([])
        pl.legend(loc=0)
        pl.title('Source  samples')

        pl.subplot(2, 2, 2)
        pl.scatter(test_epochs[:, 0], test_epochs[:, -1], c=test_labels, marker='o', cmap=pl.cm.jet,
                   label='Target samples')
        pl.xticks([])
        pl.yticks([])
        pl.legend(loc=0)
        pl.title('Target samples')

        pl.subplot(2, 2, 3)
        plot2D_samples_mat(train_epochs, test_epochs, ot_emd_laplace.coupling_)
        pl.scatter(train_epochs[:, 0], train_epochs[:, -1], c=train_labels, marker='+',
                   cmap=pl.cm.jet, label='Source samples')
        pl.scatter(test_epochs[:, 0], test_epochs[:, -1], c=test_labels, marker='o',
                   cmap=pl.cm.jet, label='Target samples')
        pl.xticks([])
        pl.yticks([])
        pl.legend(loc=0)
        pl.title('Main coupling coefficients')

        pl.subplot(2, 2, 4)
        pl.scatter(test_epochs[:, 0], test_epochs[:, -1], c=test_labels, marker='o',
                   label='Target samples', cmap=pl.cm.jet, s=30)
        pl.scatter(transp_Xs[:, 0], transp_Xs[:, -1], c=train_labels,
                   marker='+', label='Transported samples', cmap=pl.cm.jet, s=30)
        pl.title('Transported samples')
        pl.xticks([])
        pl.yticks([])
        pl.legend(loc=0)

        pl.tight_layout()
        pl.savefig('results/transported_source_'+analysis+'_Subject'+str(test[0])+'.pdf')
        pl.clf()
