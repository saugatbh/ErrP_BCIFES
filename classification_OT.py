
"""
This script classifies the correct and incorrect feedbacks using commonly
used ML algorithms and Optimal Algorithm.
The features were prepared using feature_preparation.py

Saves the result as csv files.

Usage: python classification_OT.py

Author: Saugat Bhattacharyya,
        Ulster University
"""

import numpy as np
import pandas as pd
import ot
from sklearn.model_selection import KFold
from sklearn.preprocessing import power_transform
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from classifiers import RFClassifier, LDAClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score

data = np.load('pickles/pre-processed_epochs.npy', allow_pickle=True)

type_of_analysis = ['fes', 'vis']
classifiers = ['LDA', 'Logistic Regression', 'SVM', 'Bagging', 'Adaboost', 'RF']
sampling_rate = 256.0
people = 8
kf = KFold(n_splits=people, random_state=people, shuffle=True)
eeg_epochs = {}
labels = {}
performance = {}
predicted = {}
for analysis in type_of_analysis:
    performance[analysis] = {}
    predicted[analysis] = {}
    eeg_epochs[analysis] = data[()]['epochs'][analysis]
    labels[analysis] = data[()]['classes'][analysis]
    print(len(eeg_epochs[analysis]), len(labels[analysis]))
    for cl in classifiers:
        predicted[analysis][cl] = {}
        performance[analysis][cl] = np.zeros((people, 5))
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
                                                          np.array(train_epochs).shape[1] *
                                                          np.array(train_epochs).shape[2])
            train_epochs = power_transform(train_epochs)
            train_labels = np.array(train_labels)
            test_epochs = np.array(test_epochs).reshape(np.array(test_epochs).shape[0],
                                                        np.array(test_epochs).shape[1] * np.array(test_epochs).shape[2])
            test_labels = np.array(test_labels)
            test_epochs = power_transform(test_epochs)
            train_labels_bin = np.zeros_like(train_labels)
            test_labels_bin = np.zeros_like(test_labels)
            for idx, x in enumerate(train_labels):
                if x == -1:
                    train_labels_bin[idx] = 0
                else:
                    train_labels_bin[idx] = 1
            for idx, x in enumerate(test_labels):
                if x == -1:
                    test_labels_bin[idx] = 0
                else:
                    test_labels_bin[idx] = 1
            ot_emd_laplace = ot.da.SinkhornTransport(reg_e=10)
            ot_emd_laplace.fit(Xs=train_epochs, Xt=test_epochs, ys=train_labels_bin, yt=test_labels_bin)  #
            transp_Xs = ot_emd_laplace.transform(Xs=train_epochs)
            if cl == 'LDA':
                classif = LDAClassifier()
                classif.train(transp_Xs, train_labels)
                predicted[analysis][cl][test[0]] = classif.test_prediction(test_epochs)
                performance[analysis][cl][test[0], 0] = accuracy_score(test_labels,
                                                                       predicted[analysis][cl][test[0]]) * 100
                performance[analysis][cl][test[0], 1] = precision_score(test_labels, predicted[analysis][cl][test[0]],
                                                                        pos_label=1,
                                                                        average='weighted') * 100
                performance[analysis][cl][test[0], 2] = recall_score(test_labels, predicted[analysis][cl][test[0]],
                                                                     pos_label=1,
                                                                     average='weighted') * 100
                performance[analysis][cl][test[0], 3] = f1_score(test_labels, predicted[analysis][cl][test[0]],
                                                                 pos_label=1,
                                                                 average='weighted') * 100
                performance[analysis][cl][test[0], 4] = roc_auc_score(test_labels, predicted[analysis][cl][test[0]],
                                                                      average='weighted') * 100
            if cl == 'Bagging':
                classif = BaggingClassifier(base_estimator=lda(solver="lsqr", shrinkage='auto'), n_estimators=100,
                                            random_state=people)
                classif.fit(transp_Xs, train_labels)
                predicted[analysis][cl][test[0]] = classif.predict(test_epochs)
                performance[analysis][cl][test[0], 0] = accuracy_score(test_labels,
                                                                       predicted[analysis][cl][test[0]]) * 100
                performance[analysis][cl][test[0], 1] = precision_score(test_labels, predicted[analysis][cl][test[0]],
                                                                        pos_label=1,
                                                                        average='weighted') * 100
                performance[analysis][cl][test[0], 2] = recall_score(test_labels, predicted[analysis][cl][test[0]],
                                                                     pos_label=1,
                                                                     average='weighted') * 100
                performance[analysis][cl][test[0], 3] = f1_score(test_labels, predicted[analysis][cl][test[0]],
                                                                 pos_label=1,
                                                                 average='weighted') * 100
                performance[analysis][cl][test[0], 4] = roc_auc_score(test_labels, predicted[analysis][cl][test[0]],
                                                                      average='weighted') * 100
            if cl == 'Adaboost':
                classif = AdaBoostClassifier(n_estimators=100, random_state=people)
                classif.fit(transp_Xs, train_labels)
                predicted[analysis][cl][test[0]] = classif.predict(test_epochs)
                performance[analysis][cl][test[0], 0] = balanced_accuracy_score(test_labels,
                                                                                predicted[analysis][cl][test[0]]) * 100
                performance[analysis][cl][test[0], 1] = precision_score(test_labels, predicted[analysis][cl][test[0]],
                                                                        pos_label=1,
                                                                        average='weighted') * 100
                performance[analysis][cl][test[0], 2] = recall_score(test_labels, predicted[analysis][cl][test[0]],
                                                                     pos_label=1,
                                                                     average='weighted') * 100
                performance[analysis][cl][test[0], 3] = f1_score(test_labels, predicted[analysis][cl][test[0]],
                                                                 pos_label=1,
                                                                 average='weighted') * 100
                performance[analysis][cl][test[0], 4] = roc_auc_score(test_labels, predicted[analysis][cl][test[0]],
                                                                      average='weighted') * 100
            if cl == 'Logistic Regression':
                classif = LogisticRegression(C=1000, random_state=people)
                classif.fit(transp_Xs, train_labels)
                predicted[analysis][cl][test[0]] = classif.predict(test_epochs)
                performance[analysis][cl][test[0], 0] = balanced_accuracy_score(test_labels,
                                                                                predicted[analysis][cl][test[0]]) * 100
                performance[analysis][cl][test[0], 1] = precision_score(test_labels, predicted[analysis][cl][test[0]],
                                                                        pos_label=1,
                                                                        average='weighted') * 100
                performance[analysis][cl][test[0], 2] = recall_score(test_labels, predicted[analysis][cl][test[0]],
                                                                     pos_label=1,
                                                                     average='weighted') * 100
                performance[analysis][cl][test[0], 3] = f1_score(test_labels, predicted[analysis][cl][test[0]],
                                                                 pos_label=1,
                                                                 average='weighted') * 100
                performance[analysis][cl][test[0], 4] = roc_auc_score(test_labels, predicted[analysis][cl][test[0]],
                                                                      average='weighted') * 100
            if cl == 'SVM':
                classif = LinearSVC(random_state=people)
                classif.fit(transp_Xs, train_labels)
                predicted[analysis][cl][test[0]] = classif.predict(test_epochs)
                performance[analysis][cl][test[0], 0] = balanced_accuracy_score(test_labels,
                                                                                predicted[analysis][cl][test[0]]) * 100
                performance[analysis][cl][test[0], 1] = precision_score(test_labels, predicted[analysis][cl][test[0]],
                                                                        pos_label=1,
                                                                        average='weighted') * 100
                performance[analysis][cl][test[0], 2] = recall_score(test_labels, predicted[analysis][cl][test[0]],
                                                                     pos_label=1,
                                                                     average='weighted') * 100
                performance[analysis][cl][test[0], 3] = f1_score(test_labels, predicted[analysis][cl][test[0]],
                                                                 pos_label=1,
                                                                 average='weighted') * 100
                performance[analysis][cl][test[0], 4] = roc_auc_score(test_labels, predicted[analysis][cl][test[0]],
                                                                      average='weighted') * 100

            if cl == 'RF':
                classif = RFClassifier(estimators=100)
                classif.train(transp_Xs, train_labels)
                predicted[analysis][cl][test[0]] = classif.test_prediction(test_epochs)
                performance[analysis][cl][test[0], 0] = balanced_accuracy_score(test_labels,
                                                                                predicted[analysis][cl][test[0]]) * 100
                performance[analysis][cl][test[0], 1] = precision_score(test_labels, predicted[analysis][cl][test[0]],
                                                                        pos_label=1,
                                                                        average='weighted') * 100
                performance[analysis][cl][test[0], 2] = recall_score(test_labels, predicted[analysis][cl][test[0]],
                                                                     pos_label=1,
                                                                     average='weighted') * 100
                performance[analysis][cl][test[0], 3] = f1_score(test_labels, predicted[analysis][cl][test[0]],
                                                                 pos_label=1,
                                                                 average='weighted') * 100
                performance[analysis][cl][test[0], 4] = roc_auc_score(test_labels, predicted[analysis][cl][test[0]],
                                                                      average='weighted') * 100

            # Performances
            rows = []
            for sIdx in range(people):
                rows.append(analysis.upper() + '0' + str(sIdx + 1))
            mean_performance = np.mean(performance[analysis][cl], axis=0)
            std_performance = np.std(performance[analysis][cl], axis=0)
            rows.append('Mean')
            rows.append('SD')
            columns = ['Balanced Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC']
            df_OTRF = pd.DataFrame(data=np.vstack((performance[analysis][cl], mean_performance, std_performance)),
                                   index=rows, columns=columns)
            df_filename = 'results/Classification_performance_' + analysis + '_' + cl + '.csv'
            df_OTRF.to_csv(df_filename)
