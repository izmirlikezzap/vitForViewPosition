from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
import numpy as np


def evaluate(predictions, labels, class_names):

    # Sınıfların sayısını belirle
    num_classes = len(class_names)

    # Confusion Matrix (her sınıf için)
    cm = confusion_matrix(labels, predictions)
    print("Confusion Matrix:")
    print(cm)

    # Class-specific metrics (Precision, Recall, F1 per class)
    class_report = classification_report(labels, predictions, target_names=class_names, zero_division=0)
    print("\nClassification Report:")
    print(class_report)

    # Genel Doğruluk (Accuracy)
    accuracy = metrics.accuracy_score(labels, predictions)

    # Precision, Recall, F1 Score (macro)
    precision_macro = metrics.precision_score(labels, predictions, average='macro', zero_division=0)
    recall_macro = metrics.recall_score(labels, predictions, average='macro', zero_division=0)
    f1_macro = metrics.f1_score(labels, predictions, average='macro', zero_division=0)

    # Cohen's Kappa
    kappa = metrics.cohen_kappa_score(labels, predictions)

    # ROC AUC (çok sınıflı destek)
    try:
        auc = metrics.roc_auc_score(
            metrics.label_binarize(labels, classes=range(num_classes)),
            metrics.label_binarize(predictions, classes=range(num_classes)),
            average='macro',
            multi_class="ovr"
        )
    except ValueError:
        auc = None  # Eğer ROC AUC hesaplanamazsa

    # Sonuçları döndür
    result_labels = [
        "Accuracy",
        "Precision (macro)",
        "Recall (macro)",
        "F1 Score (macro)",
        "Cohen's Kappa",
        "ROC AUC",
    ]
    result = [accuracy, precision_macro, recall_macro, f1_macro, kappa, auc]

    return cm, class_report, result_labels, result


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

