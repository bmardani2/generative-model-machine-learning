import os
from builtins import range, input
import numpy as np
import pandas as pd
import seaborn as sns
# from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, auc, roc_curve
from itertools import cycle
from sklearn.preprocessing import label_binarize
# from  sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
# from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from scipy import interp


class Generative(object):
    """
    generative approach without naive bayes assumption
    """
    def __init__(self, n):
        self.n = n
        


    def prob_dens_func(self, i, std, mean):
        '''
        prob_dens_func function
        '''
        pdf = (1/(((2*np.pi)**-(self.n/2))* (np.linalg.det(std))**-0.5)) * np.exp((i - mean).T * np.linalg.pinv(std) * (i - mean))
        return pdf


    def cal_mean_cov(self, X, Y):
        '''
        calculate mean and covariance of any class
        '''
        k = len(np.unique(Y))
        mean = dict()
        cov = dict()
        for s in range(1, k+1):
            l = []
            for i in range(len(Y)):
                if Y[i] == s :
                    l.append(X[i])
            l = np.array(l)
            myangin = l.mean(axis=0)
            covar = np.cov(l.T)#covariance of feature
            mean[s] = myangin
            cov[s] = covar
        return mean, cov

    def evaluation(self, Y_eval, result):
        '''
        for calculating precision, recall and f1-score
        '''
        print(classification_report(Y_eval, label))
        return result

    
    def eval_roc_auc(self, y_test, y_score):
        '''
        calculate ROC-AUC estimation
        '''
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = y_test.shape[1]
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure()
        lw = 2
        plt.plot(fpr[2], tpr[2], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()

    def prior_of_any_class(self, Y):
        '''
        calculate prior probablity of any class
        '''
        k = len(np.unique(Y))
        class_prior = dict()
        for s in range(1, k+1):
            counter = 0
            for i in range(len(Y)):
                if Y[i] == s :
                    counter += 1
            res = counter/len(Y)
            class_prior[s] = res
        return class_prior
        
    def predict_class(self, X, class_prior, cov, mean, Y):
        '''
        predict target class
        '''
        k = len(np.unique(Y))
        prediction = []
        out_come = []
        for i in range(len(X)):
            label = []
            for j in range(1, k):
                res = class_prior[j] * self.prob_dens_func(i, cov[j], mean[j])
                label.append(res)
            ind = np.argmax(label,1)
            out_come.append(label)
            prediction.append(ind)
        return prediction, out_come
    

    def softmax(self, x):
        """
        Compute softmax values for each sets of scores in x.
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def pred_eval(self, X, Y):
        class_num = np.unique(Y)
        y_bin = label_binarize(Y, classes=class_num)

        mean, cov = self.cal_mean_cov(X, Y)
        self._pred_eval(X, Y, mean, cov)

        
    def _pred_eval(self, X, Y, mean, cov):
        print("mean of any class :" + "\n")
        print(mean, "\n", "\n")
        print("---"*20)
        print("covariance of any class :" + "\n")
        print(cov, "\n", "\n")
        print("---"*20)
        class_prior = self.prior_of_any_class(Y)
        print("prior of any class :" + "\n")
        print(class_prior, "\n", "\n")
        expectancy, out_come = self.predict_class(X, class_prior, cov, mean, Y)
        #print(expectancy)
        self.evaluation(Y_eval, expectancy)
        out_come = np.array(out_come)
        out_come = out_come[0]
        self.eval_roc_auc(y_bin, out_come) 


class NiveBayesGenerative(Generative):
    '''
    Implementation of generative approache without naive bayes assumption
    '''

    def __init__(self, n):
        return super().__init__(n)

    def NB_covar_cal(self, X):
        '''
        '''
        return np.cov(X, rowvar=False)

    def predict_class(self, X, class_prior, cov, mean, Y):
        '''
        predict target class
        '''
        k = len(np.unique(Y))
        prediction = []
        out_come = []
        for i in range(len(X)):
            label = []
            for j in range(1, k):
                res = class_prior[j] * self.prob_dens_func(i, cov, mean[j])
                label.append(res)
            ind = np.argmax(label,1)
            out_come.append(label)
            prediction.append(ind)
        return prediction, out_come

    def cal_mean(self, X, Y):
        '''
        calculate mean and covariance of any class
        '''
        k = len(np.unique(Y))
        mean = dict()
        cov = dict()
        for s in range(1, k+1):
            l = []
            for i in range(len(Y)):
                if Y[i] == s :
                    l.append(X[i])
            l = np.array(l)
            myangin = l.mean(axis=0)
            mean[s] = myangin
        return mean


    def pred_eval(self, X, Y):
        class_num = np.unique(Y)
        y_bin = label_binarize(Y, classes=class_num)

        cov = self.NB_covar_cal(X)
        mean = self.cal_mean(X, Y)
        self._pred_eval(X, Y, mean, cov)
 





