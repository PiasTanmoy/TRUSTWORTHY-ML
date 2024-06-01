from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from sklearn import metrics
import pandas as pd


# for decompensation, in-hospital mortality

def print_metrics_binary(y_true, predictions, verbose=1):
    predictions = np.array(predictions)
    if len(predictions.shape) == 1:
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    print(predictions.shape)
    print(predictions[1])

    # implemented by Tanmoy
    ths = np.arange(0.18, 0.5, 0.005)
    result_metrics = []

    #for th in ths:
    th = 0.22

    res = []
    for i in range(predictions.shape[0]):
      if predictions[i][1] > th:
        res.append(1)
      else:
        res.append(0)

    # Changing Threshold
    y_pred = res
    cf = metrics.confusion_matrix(y_true, y_pred)

    #cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1))


    if verbose:
        print("confusion matrix:")
        print(cf)
    cf = cf.astype(np.float32)

    TN = cf[0][0]
    FN = cf[1][0]
    TP = cf[1][1]
    FP = cf[0][1]

    TPR = (TP) / (TP + FN)
    TNR = (TN) / (TN + FP)

    balanced_accuracy = (TPR + TNR) / 2

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])

    f1_c1 = (2*prec1*rec1)/(prec1+rec1)
    f0_c0 = (2*prec0*rec0)/(prec0+rec0)

    y_true_inverse = []
    for i in y_true:
        y_true_inverse.append(1 - i)
    y_true_inverse = np.array(y_true_inverse)

    '''
    AUROC and AUPRC are threshold agnostic. 
    In other words, threshold is not required to calculate these metrics. 
    So, this can be used to select the model.
    '''
    auroc = metrics.roc_auc_score(y_true, predictions[:, 1])
    #auroc_inv = metrics.roc_auc_score(y_true_inverse, predictions[:, 0])

    (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, predictions[:, 1])
    (precisions_inv, recalls_inv, thresholds_inv) = metrics.precision_recall_curve(y_true_inverse, predictions[:, 0])

    auprc = metrics.auc(recalls, precisions)
    auprc_inv = metrics.auc(recalls_inv, precisions_inv)

    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])


    # tanmoy
    #y_pred = predictions.argmax(axis=1)
    f1_score_macro = metrics.f1_score(y_true, y_pred, average='macro')
    f1_score_micro = metrics.f1_score(y_true, y_pred, average='micro')
    f1_score_weighted = metrics.f1_score(y_true, y_pred, average='weighted')
    MCC = metrics.matthews_corrcoef(y_true, y_pred)
    bal_acc= metrics.balanced_accuracy_score(y_true, y_pred)







    if verbose:
        print("recall class 1 = {}".format(rec1))
        print("precision class 1 = {}".format(prec1))
        print("f1_c1 = {}".format(f1_c1))

        print("recall class 0 = {}".format(rec0))
        print("precision class 0 = {}".format(prec0))
        print("f0_c0 = {}".format(f0_c0))

        print("accuracy = {}".format(acc))
        print("Balance Acc = {}".format(bal_acc))
        print("AUC of ROC = {}".format(auroc))
        print("MCC = {}".format(MCC))
        print("AUC of PRC = {}".format(auprc))

        print("f1_score_macro = {}".format(f1_score_macro))
        print("f1_score_micro = {}".format(f1_score_micro))
        print("f1_score_weighted = {}".format(f1_score_weighted))
        print("min(+P, Se) = {}".format(minpse))


        print("threshold: ", th)
        print(round(rec1, 2))
        print(round(prec1, 2))
        print(round(auprc, 2))
        print(round(f1_c1, 2))

        print(round(rec0, 2))
        print(round(prec0, 2))
        print(round(auprc_inv, 2))
        print(round(f0_c0, 2))

        print(round(acc, 2))
        print(round(bal_acc, 2))
        print(round(auroc, 2))
        #print(round(auroc_inv, 2)) # ROC curve is same for class 0 and 1
        print(round(MCC, 2))

        #result_metrics.append([th, rec1, prec1, f1_c1, rec0, prec0, f0_c0, acc, bal_acc, auroc, MCC])

    # mat_file = pd.DataFrame(result_metrics,
    #                         columns = ['th', 'rec1', 'prec1', 'f1_c1', 'rec0', 'prec0', 'f0_c0', 'acc', 'bal_acc', 'auroc', 'MCC'],
    #                         dtype=float)
    # mat_file.to_csv("/home/tanmoy/Downloads/Trustworthiness/threshold_variation.tsv", sep='\t')
    # print("File saved!!")

    return {"acc": acc,
            "prec0": prec0,
            "prec1": prec1,
            "rec0": rec0,
            "rec1": rec1,
            "auroc": auroc,
            "auprc": auprc,
            "f1_macro":f1_score_macro,
            "f1_micro":f1_score_micro,
            "f1_weighted":f1_score_weighted,
            "minpse": minpse,
            "MCC": MCC,
            "bal_acc": bal_acc,
            "f1_c1": f1_c1,
            "f0_c0": f0_c0,
            }


# for phenotyping

def print_metrics_multilabel(y_true, predictions, verbose=1):
    y_true = np.array(y_true)
    predictions = np.array(predictions)

    auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
    ave_auc_micro = metrics.roc_auc_score(y_true, predictions,
                                          average="micro")
    ave_auc_macro = metrics.roc_auc_score(y_true, predictions,
                                          average="macro")
    ave_auc_weighted = metrics.roc_auc_score(y_true, predictions,
                                             average="weighted")

    if verbose:
        print("ROC AUC scores for labels:", auc_scores)
        print("ave_auc_micro = {}".format(ave_auc_micro))
        print("ave_auc_macro = {}".format(ave_auc_macro))
        print("ave_auc_weighted = {}".format(ave_auc_weighted))

    return {"auc_scores": auc_scores,
            "ave_auc_micro": ave_auc_micro,
            "ave_auc_macro": ave_auc_macro,
            "ave_auc_weighted": ave_auc_weighted}


# for length of stay

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.1))) * 100


def print_metrics_regression(y_true, predictions, verbose=1):
    predictions = np.array(predictions)
    predictions = np.maximum(predictions, 0).flatten()
    y_true = np.array(y_true)

    y_true_bins = [get_bin_custom(x, CustomBins.nbins) for x in y_true]
    prediction_bins = [get_bin_custom(x, CustomBins.nbins) for x in predictions]
    cf = metrics.confusion_matrix(y_true_bins, prediction_bins)
    if verbose:
        print("Custom bins confusion matrix:")
        print(cf)

    kappa = metrics.cohen_kappa_score(y_true_bins, prediction_bins,
                                      weights='linear')
    mad = metrics.mean_absolute_error(y_true, predictions)
    mse = metrics.mean_squared_error(y_true, predictions)
    mape = mean_absolute_percentage_error(y_true, predictions)

    if verbose:
        print("Mean absolute deviation (MAD) = {}".format(mad))
        print("Mean squared error (MSE) = {}".format(mse))
        print("Mean absolute percentage error (MAPE) = {}".format(mape))
        print("Cohen kappa score = {}".format(kappa))

    return {"mad": mad,
            "mse": mse,
            "mape": mape,
            "kappa": kappa}


class LogBins:
    nbins = 10
    means = [0.611848, 2.587614, 6.977417, 16.465430, 37.053745,
             81.816438, 182.303159, 393.334856, 810.964040, 1715.702848]


def get_bin_log(x, nbins, one_hot=False):
    binid = int(np.log(x + 1) / 8.0 * nbins)
    if binid < 0:
        binid = 0
    if binid >= nbins:
        binid = nbins - 1

    if one_hot:
        ret = np.zeros((LogBins.nbins,))
        ret[binid] = 1
        return ret
    return binid


def get_estimate_log(prediction, nbins):
    bin_id = np.argmax(prediction)
    return LogBins.means[bin_id]


def print_metrics_log_bins(y_true, predictions, verbose=1):
    y_true_bins = [get_bin_log(x, LogBins.nbins) for x in y_true]
    prediction_bins = [get_bin_log(x, LogBins.nbins) for x in predictions]
    cf = metrics.confusion_matrix(y_true_bins, prediction_bins)
    if verbose:
        print("LogBins confusion matrix:")
        print(cf)
    return print_metrics_regression(y_true, predictions, verbose)


class CustomBins:
    inf = 1e18
    bins = [(-inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, +inf)]
    nbins = len(bins)
    means = [11.450379, 35.070846, 59.206531, 83.382723, 107.487817,
             131.579534, 155.643957, 179.660558, 254.306624, 585.325890]


def get_bin_custom(x, nbins, one_hot=False):
    for i in range(nbins):
        a = CustomBins.bins[i][0] * 24.0
        b = CustomBins.bins[i][1] * 24.0
        if a <= x < b:
            if one_hot:
                ret = np.zeros((CustomBins.nbins,))
                ret[i] = 1
                return ret
            return i
    return None


def get_estimate_custom(prediction, nbins):
    bin_id = np.argmax(prediction)
    assert 0 <= bin_id < nbins
    return CustomBins.means[bin_id]


def print_metrics_custom_bins(y_true, predictions, verbose=1):
    return print_metrics_regression(y_true, predictions, verbose)
