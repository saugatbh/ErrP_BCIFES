
"""
This script classifies the correct and incorrect feedbacks using Random Forest Classifier.
The features were prepared using feature_preparation.py

Saves the result as csv files.

Usage: python classification.py

Author: Saugat Bhattacharyya,
        Ulster University
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import power_transform
from classifiers import RFClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score

data = np.load('pickles/pre-processed_epochs.npy', allow_pickle=True)

type_of_analysis = ['fes', 'vis']
sampling_rate = 256.0
people = 8
kf = KFold(n_splits=people, random_state=people, shuffle=True)
eeg_epochs = {}
labels = {}
performance = {}
predicted = {}
for analysis in type_of_analysis:
    predicted[analysis] = {}
    performance[analysis] = np.zeros((people, 5))
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
                                                      np.array(train_epochs).shape[1] * np.array(train_epochs).shape[2])
        train_epochs = power_transform(train_epochs)
        train_labels = np.array(train_labels)
        test_epochs = np.array(test_epochs).reshape(np.array(test_epochs).shape[0],
                                                    np.array(test_epochs).shape[1] * np.array(test_epochs).shape[2])
        test_labels = np.array(test_labels)
        test_epochs = power_transform(test_epochs)
        # print(train_epochs.shape, test_epochs.shape)
        # print(train_labels.shape, test_labels.shape)
        classif = RFClassifier(estimators=100)
        classif.train(train_epochs, train_labels)
        predicted[analysis][test[0]] = classif.test_prediction(test_epochs)
        performance[analysis][test[0], 0] = accuracy_score(test_labels, predicted[analysis][test[0]]) * 100
        performance[analysis][test[0], 1] = precision_score(test_labels, predicted[analysis][test[0]], pos_label=1,
                                                            average='weighted') * 100
        performance[analysis][test[0], 2] = recall_score(test_labels, predicted[analysis][test[0]], pos_label=1,
                                                         average='weighted') * 100
        performance[analysis][test[0], 3] = f1_score(test_labels, predicted[analysis][test[0]], pos_label=1,
                                                     average='weighted') * 100
        performance[analysis][test[0], 4] = roc_auc_score(test_labels, predicted[analysis][test[0]],
                                                          average='weighted') * 100

        # Performance
        rows = []
        for sIdx in range(people):
            rows.append(analysis.upper() + '0' + str(sIdx + 1))
        mean_performance = np.mean(performance[analysis], axis=0)
        std_performance = np.std(performance[analysis], axis=0)
        rows.append('Mean')
        rows.append('SD')
        columns = ['Balanced Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC']
        df_soleRF = pd.DataFrame(data=np.vstack((performance[analysis], mean_performance, std_performance)),
                                 index=rows, columns=columns)
        df_filename = 'results/Classification_performance_' + analysis + '.csv'
        df_soleRF.to_csv(df_filename)