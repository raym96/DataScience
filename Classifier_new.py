import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.multiclass import OneVsRestClassifier

import inspect

class Classifier:

    def svm(self, X_train, y_train, X_test=None, kernel='rbf', C=1.0, degree=3, class_weight = None):
        '''
        This function fits an SVM on (X_train, y_train) and returns predictions for X_test.
        Possible kernels: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’
        will be used.
        The fit time scales at least quadratically with the number of samples and may be impractical beyond tens of
        thousands of samples.
        Parameters:
        - C: inverse of the regularization strength, smaller values specify stronger regularization
        - class_weight: {dict, ‘balanced’}, optional
            The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies
        '''

        classifier = svm.SVC(kernel=kernel, C=C, degree=degree, class_weight = class_weight)

        # If the user hasn't provided any test data set, we simply predict on the train dataset
        if X_test is None:
            y_pred = classifier.fit(X_train, y_train).predict(X_train)
            # distances from the separating hyperplane
            distances = classifier.decision_function(X_train)
        else:
            y_pred = classifier.fit(X_train, y_train).predict(X_test)
            distances = classifier.decision_function(X_test)

        return classifier, y_pred, distances

    def plot_confusion_matrix(self, y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
        """
        This function plots and returns the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        # in case we have a list instead of a numpy array convert it
        classes = np.array(classes)
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.show()

        return cm

    def roc(self, y_true, distances):
        '''
        This function plots the Receiver Operating Characteristic Curve and returns the False Positive Rate,
        True Positive Rate and Thresholds used.
        :param y_true: true labels
        :param distances: could be probability estimates, confidences values or non-thresholded measure
        of decision
        :return: fpr, tpr and thresholds
        '''
        fpr, tpr, thresholds = roc_curve(y_true, distances, pos_label=1)
        plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        return fpr, tpr, thresholds

    # AUROC
    def auroc(self, y_true, distances):
        '''
        This function calculates the Area Under the Receiver Operating Characteristic Curve
        :param y_true: trues labels
        :param distances: target scores, could be probability estimates, confidences values or non-thresholded measure
        of decision
        :return: area under the Receiver Operating Characteristic Curve
        '''
        return roc_auc_score(y_true, distances)

    # Accuracy, F1 score, etc... to do
    def metrics(self, y_true, y_pred, average = None):
        '''
        This function calculates the metrics: accuracy, precision, recall, F1-score
        :param average: the average method used for multi-class labels
        return a dictionary of metrics
        '''
        res = {}
        for metric in [accuracy_score, precision_score, recall_score, f1_score]:
            if 'average' in inspect.getfullargspec(metric).args:
                res[metric.__name__] = metric(y_true = y_true, y_pred = y_pred, average = average)
            else:
                res[metric.__name__] = metric(y_true = y_true, y_pred = y_pred)
        return res
    
    # multi-class example
    def svm_multi(self, X_train, y_train, X_test=None, kernel='rbf', C=1.0, degree=3, class_weight = None):
        '''
        This function fits an SVM on (X_train, y_train) and returns predictions for X_test.
        Possible kernels: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’
        will be used.
        The fit time scales at least quadratically with the number of samples and may be impractical beyond tens of
        thousands of samples.
        Parameters:
        - C: inverse of the regularization strength, smaller values specify stronger regularization
        - class_weight: {dict, ‘balanced’}, optional
            The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies
        '''

        classifier = OneVsRestClassifier(svm.SVC(kernel=kernel, C=C, degree=degree, class_weight = class_weight))  # wrap into one-vs-rest

        # If the user hasn't provided any test data set, we simply predict on the train dataset
        if X_test is None:
            y_pred = classifier.fit(X_train, y_train).predict(X_train)
            # distances from the separating hyperplane
            distances = classifier.decision_function(X_train)
        else:
            y_pred = classifier.fit(X_train, y_train).predict(X_test)
            distances = classifier.decision_function(X_test)
        return classifier, y_pred, distances
