import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, roc_auc_score, roc_curve, auc, \
    mean_absolute_error, precision_score, recall_score, classification_report, confusion_matrix, balanced_accuracy_score, precision_recall_curve, matthews_corrcoef

def measure_performance(eval_y, scores_y, threshold  = 0.5):
    m = {}

    predict_y = []
    for p in scores_y:
        if p < threshold: predict_y.append(0)
        else: predict_y.append(1)

    m['AUC'] = roc_auc_score(eval_y, scores_y)
    m['F1 C1'] = f1_score(eval_y, predict_y)
    m['Accuracy'] = accuracy_score(eval_y, predict_y)
    m['Balanced_accuracy'] = balanced_accuracy_score(eval_y, predict_y)
    m['Precision  C1'] = precision_score(eval_y, predict_y)
    m['Recall C1'] = recall_score(eval_y, predict_y)

    tn, fp, fn, tp = confusion_matrix(eval_y, predict_y).ravel()
    recall0 = tn / (tn + fp)
    precision0 = tn / (tn + fn)

    m['F1 C0'] = 2 * precision0 * recall0 / (precision0 + recall0)
    m['Precision C0'] = precision0
    m['Recall C0'] = recall0
    m['FPR'] = fp / (fp + tn)
    m['FNR'] = fn / (fn + tp)
    m['TPR'] = tp / (fn + tp)
    m['TNR'] = tn / (tn + fp)

    try:
        curve_precision, curve_recall, _ = precision_recall_curve(eval_y, scores_y)
        m['AUC_PR_C1'] = auc(curve_recall, curve_precision)
    except:
        m['AUC_PR_C1'] = 0

    #true_y_filp = ((eval_y) == 0).astype(np.int)
    true_y_filp = []
    for y in eval_y:
        if y == 0: true_y_filp.append(1)
        else: true_y_filp.append(0)

    # score_y_filp = (1-scores_y)
    score_y_filp = []
    for y in scores_y:
        score_y_filp.append(1-y)
    

    try:
        curve_precision0, curve_recall0, _ = precision_recall_curve(true_y_filp, score_y_filp)
        m['AUC_PR_C0'] = auc(curve_recall0, curve_precision0)
    except:
        m['AUC_PR_C0'] = 0

    m['tp'] = tp
    m['tn'] = tn
    m['fp'] = fp
    m['fn'] = fn

    MCC = matthews_corrcoef(eval_y, predict_y)
    m['MCC'] = MCC

    minpse = np.max([min(x, y) for (x, y) in zip(curve_precision0, curve_recall0)])
    m['minpse'] = minpse

    m_df = pd.DataFrame(m.items(), columns=['Metrics', 'Value'])

    return m_df