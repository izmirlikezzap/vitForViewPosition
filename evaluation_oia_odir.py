# -*- coding: utf-8 -*-
""" This file is only an example approach for evaluating the classification 
    performance in Ocular Disease Intelligent Recognition (ODIR-2019). 
    
    To run this file, sklearn and numpy packages are required 
    in a Python 3.0+ environment.
    
    Author: Shanggong Medical Technology, China.
    Date: July, 2019.
"""
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import sys
import xlrd
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils import read_csv_file
import os
from glob import glob
import re


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


# read the ground truth from xlsx file and output case id and eight labels
def importGT(filepath):
    data = xlrd.open_workbook(filepath)
    table = data.sheets()[0]
    data = [[int(table.row_values(i, 0, 1)[0])] + table.row_values(i, -8) for i in range(1, table.nrows)]
    return np.array(data)


# read the submitted predictions in csv format and output case id and eight labels 
def importPR(gt_data, filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        pr_data = [[int(row[0])] + list(map(float, row[1:])) for row in reader]
    pr_data = np.array(pr_data)

    # Sort columns if they are not in predefined order
    order = ['ID', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    order_index = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    order_dict = {item: ind for ind, item in enumerate(order)}
    sort_index = [order_dict[item] for ind, item in enumerate(header) if item in order_dict]
    wrong_col_order = 0
    if (sort_index != order_index):
        wrong_col_order = 1
        pr_data[:, order_index] = pr_data[:, sort_index]

        # Sort rows if they are not in predefined order
    wrong_row_order = 0
    order_dict = {item: ind for ind, item in enumerate(gt_data[:, 0])}
    order_index = [v for v in order_dict.values()]
    sort_index = [order_dict[item] for ind, item in enumerate(pr_data[:, 0]) if item in order_dict]
    if (sort_index != order_index):
        wrong_row_order = 1
        pr_data[order_index, :] = pr_data[sort_index, :]

    # If have missing results
    missing_results = 0
    if (gt_data.shape != pr_data.shape):
        missing_results = 1
    return pr_data, wrong_col_order, wrong_row_order, missing_results


# calculate kappa, F-1 socre and AUC value
def ODIR_Metrics_for_classes(gt_data, pr_data, name):
    classes = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
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
        cm = confusion_matrix(gt, classified_pr)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        plt.figure()
        disp.plot()
        title_name = name.replace('_', ' ')
        plt.title(f'{title_name} {classes[i]}')
        plt.savefig(f'{name}/{name}_{classes[i]}.png')
        plt.close('all')
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

        result = [accuracy, precision, sensitivity, specifity, kappa, f1, auc, final_score]
        result_list.append(result)

    return result_list
    # with open(f'{name}/{name}_class_results.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(result_list)
    # return kappa, f1, auc, final_score


# calculate kappa, F-1 score and AUC value
def ODIR_Metrics(gt_data, pr_data, name):
    th = 0.5
    gt = gt_data.flatten()
    pr = pr_data.flatten()
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
    kappa = metrics.cohen_kappa_score(gt, pr > th)
    f1 = metrics.f1_score(gt, pr > th, average='micro')
    auc = metrics.roc_auc_score(gt, pr)
    final_score = (kappa + f1 + auc) / 3.0

    # result_list = []
    # labels = ["accuracy", "precision", "sensitivity", "specifity", "kappa", "f1", "auc", "final score"]
    # result_list.append(labels)
    result = [accuracy, precision, sensitivity, specifity, kappa, f1, auc, final_score]
    # result_list.append(result)
    # with open(f'{name}/{name}_results.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(result_list)
    return result


# calculate kappa, F-1 socre and AUC value
def confusion_matrix_for_all(gt_data, pr_data, name, matrix_name, title_name):
    gt_data = np.argmax(gt_data, axis=-1)
    pr_data = np.argmax(pr_data, axis=-1)
    cm = confusion_matrix(gt_data, pr_data)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'])
    plt.figure()
    disp.plot()
    title_name = title_name.replace('_', ' ')
    plt.title(f'{title_name}')
    plt.savefig(f'{name}/{matrix_name}.png')
    plt.close('all')
    # np.savetxt('inception_overall_processed_test_confusion_matrix.csv', cm, delimiter=',', fmt='%i')

    # return kappa, f1, auc, final_score


def calculate_score(GT_filepath, PR_filepath, name):
    try:
        os.mkdir(f"{name}")
        print(f"Directory {name} Created ")
    except FileExistsError:
        print(f"Directory {name} already exists")
    val_annotations = read_csv_file(GT_filepath)
    val_annotations = val_annotations[1:]
    gt_data = []
    for ann in val_annotations:
        gt_data.append([int(ann[0]), float(ann[-8]), float(ann[-7]), float(ann[-6]), float(ann[-5]), float(ann[-4]),
                        float(ann[-3]), float(ann[-2]), float(ann[-1])])
    gt_data = np.array(gt_data)
    pr_data, wrong_col_order, wrong_row_order, missing_results = importPR(gt_data, PR_filepath)

    ODIR_Metrics(gt_data[:, 1:], pr_data[:, 1:], name=name)
    ODIR_Metrics_for_classes(gt_data[:, 1:], pr_data[:, 1:], name=name)
    confusion_matrix_for_all(gt_data=gt_data[:, 1:], pr_data=pr_data[:, 1:], name=name, matrix_name=name,
                             title_name=name)


def plot_graphic(data_array, name):
    plt.figure()
    plt.plot(data_array)
    title_name = name.replace('_', ' ')
    plt.title(f'{title_name}')
    plt.savefig(f'{name}.png')
    plt.close('all')


def plot_graphic_all(train_array, test_array, val_array, name):
    plt.figure()
    plt.plot(train_array, '-r', label="Train")
    plt.plot(test_array, '-g', label="Test")
    plt.plot(val_array, '-b', label="Validation")
    title_name = name.replace('_', ' ')
    plt.title(f'{title_name}')
    plt.legend()
    plt.savefig(f'{name}.png')
    plt.close('all')


def calculate_score_for_all_epochs(GT_filepath, PR_folder, model_name):
    val_annotations = read_csv_file(GT_filepath)
    val_annotations = val_annotations[1:]
    gt_data = []
    for ann in val_annotations:
        gt_data.append([int(ann[0]), float(ann[-8]), float(ann[-7]), float(ann[-6]), float(ann[-5]), float(ann[-4]),
                        float(ann[-3]), float(ann[-2]), float(ann[-1])])
    gt_data = np.array(gt_data)
    pr_files = glob(f'{PR_folder}/{model_name}*.csv')
    pr_files.sort(key=natural_keys)
    labels = ["accuracy", "precision", "sensitivity", "specifity", "kappa", "f1", "auc", "final score"]
    classfications = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

    class_based_results = []
    model_results = []
    for pr_file in pr_files:
        pr_data, wrong_col_order, wrong_row_order, missing_results = importPR(gt_data, pr_file)
        pr_file = pr_file.split('/')[1]
        try:
            os.mkdir(f"{pr_file[:-4]}")
            print(f"Directory {pr_file[:-4]} Created ")
        except FileExistsError:
            print(f"Directory {pr_file[:-4]} already exists")
        model_results.append(ODIR_Metrics(gt_data[:, 1:], pr_data[:, 1:], name=pr_file[:-4]))
        class_based_results.append(ODIR_Metrics_for_classes(gt_data[:, 1:], pr_data[:, 1:], name=pr_file[:-4]))
        confusion_matrix_for_all(gt_data=gt_data[:, 1:], pr_data=pr_data[:, 1:], name=pr_file[:-4],
                                 matrix_name=pr_file[:-4],
                                 title_name=pr_file[:-4])

    model_results = np.array(model_results)
    class_based_results = np.array(class_based_results)
    for i, label in enumerate(labels):
        data_array = model_results[:, i]
        plot_graphic(data_array=data_array, name=f"{model_name}_{label}")
        for j, class_label in enumerate(classfications):
            data_array = class_based_results[:, j, i]
            plot_graphic(data_array=data_array, name=f"{model_name}_{label}_{class_label}")
    return model_results, class_based_results


# calculate_score('Annotations/test.csv', 'inception_test_result.csv', name='inception_test')
'''
models = ["inception", "resnet", "vgg"]
for model in models:
    train_model_results, train_class_based_results = calculate_score_for_all_epochs(
        'Annotations/processed_train_annotations.csv', "oia-odir-train-csv-results", model)
    test_model_results, test_class_based_results = calculate_score_for_all_epochs(
        'Annotations/processed_test_annotations.csv', "oia-odir-test-csv-results", model)
    val_model_results, val_class_based_results = calculate_score_for_all_epochs(
        'Annotations/processed_val_annotations.csv', "oia-odir-val-csv-results", model)

    labels = ["accuracy", "precision", "sensitivity", "specifity", "kappa", "f1", "auc", "final score"]
    classfications = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    for i, label in enumerate(labels):
        train_array = train_model_results[:, i]
        test_array = test_model_results[:, i]
        val_array = val_model_results[:, i]
        plot_graphic_all(train_array=train_array, test_array=test_array, val_array=val_array,
                         name=f"{model}_{label}_all")
        for j, class_label in enumerate(classfications):
            train_array = train_class_based_results[:, j, i]
            test_array = test_class_based_results[:, j, i]
            val_array = val_class_based_results[:, j, i]
            plot_graphic_all(train_array=train_array, test_array=test_array, val_array=val_array,
                             name=f"{model}_{label}_{class_label}_all")
                        
                        
'''
loss_files = glob('oia-odir-csv-results/*')
for file_path in loss_files:
    name = file_path.split('/')[1][:-4] + "_loss"
    file = read_csv_file(file_path)
    labels = file[0]
    file = file[1:]
    numeric_list = []
    for row in file:
        row = row[0].split(',')
        row = list(map(float, row))
        numeric_list.append(row)
    npfile = np.array(list(zip(*numeric_list)))
    train_loss = npfile[1]
    test_loss = npfile[3]
    val_loss = npfile[5]
    plot_graphic_all(train_array=train_loss, test_array=test_loss, val_array=val_loss, name=name)

# Run in terminal python evaluation.py on_site_test_annotation.xlsx *_result.csv
