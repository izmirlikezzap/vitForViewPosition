from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn import metrics

def evaluate(predictions, labels):
    '''
    returns tn, fp, fn, tp, accuracy, precision, sensitivity, specifity, kappa, f1, auc, final_score
    '''
    classes = ["0", "1"]
    th = 0.5
    # gt = gt_data.flatten()
    # pr = pr_data.flatten()
    result_list = []
    # labels = ["accuracy", "precision", "sensitivity", "specifity", "kappa", "f1", "auc", "final score"]
    # result_list.append(labels)
    gt = labels.tolist()
    pr = predictions.tolist()
        
    classified_pr = [1 if i >= th else 0 for i in pr]
    tn, fp, fn, tp = confusion_matrix(gt, classified_pr).ravel()
    if tn + fp + fn + tp == 0:
        accuracy = 0
    else:
        accuracy = (tn + tp) / (tn + fp + fn + tp)
    if fp + tp == 0:
        precision = 0
    else:
        precision = (tp) / (fp + tp)
    if tp + fn == 0:
        sensitivity = 0
    else:
        sensitivity = (tp) / (tp + fn)
    if tn + fp == 0:
        specifity = 0
    else:
        specifity = (tn) / (tn + fp)
    kappa = metrics.cohen_kappa_score(gt, pr)
    f1 = metrics.f1_score(gt, pr, average='micro')
    auc = metrics.roc_auc_score(gt, pr)
    final_score = (kappa + f1 + auc) / 3.0

    result_labels = ["True Negative", "False Positive", "False Negative", "True Positive", "Accuracy", "Precision", "Sensitivity", "Specifity", "Kappa", "F1", "AUC", "Final Score"]
    result = [tn, fp, fn, tp,accuracy, precision, sensitivity, specifity, kappa, f1, auc, final_score]
    return result_labels, result


# calculate kappa, F-1 socre and AUC value
def ODIR_Metrics_for_classes(gt_data, pr_data, name):
    classes = ["0", "1"]
    th = 0.5
    # gt = gt_data.flatten()
    # pr = pr_data.flatten()
    result_list = []
    # labels = ["accuracy", "precision", "sensitivity", "specifity", "kappa", "f1", "auc", "final score"]
    # result_list.append(labels)
    for i, classification in enumerate(classes):
        gt = gt_data[:, i]
        pr = pr_data[:, i]
        
        classified_pr = [1 if i >= th else 0 for i in pr]
        """
        cm = confusion_matrix(gt, classified_pr)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        
        plt.figure()
        disp.plot()
        title_name = name.replace('_', ' ')
        plt.title(f'{title_name} {classes[i]}')
        plt.savefig(f'{name}/{name}_{classes[i]}.png')
        plt.close('all') """
        tn, fp, fn, tp = confusion_matrix(gt, classified_pr).ravel()
        if tn + fp + fn + tp == 0:
            accuracy = 0
        else:
            accuracy = (tn + tp) / (tn + fp + fn + tp)
        if fp + tp == 0:
            precision = 0
        else:
            precision = (tp) / (fp + tp)
        if tp + fn == 0:
            sensitivity = 0
        else:
            sensitivity = (tp) / (tp + fn)
        if tn + fp == 0:
            specifity = 0
        else:
            specifity = (tn) / (tn + fp)
        kappa = metrics.cohen_kappa_score(gt, pr > th)
        f1 = metrics.f1_score(gt, pr > th, average='micro')
        auc = metrics.roc_auc_score(gt, pr)
        final_score = (kappa + f1 + auc) / 3.0

        result = [tn, fp, fn, tp,accuracy, precision, sensitivity, specifity, kappa, f1, auc, final_score]
        result_list.append(result)

    return result_list

