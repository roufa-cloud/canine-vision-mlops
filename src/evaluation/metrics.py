""" This script defines functions to evaluate classification models"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, dataset, verbose=1):
    """ Evaluate a compiled model
    Args:
        model(tf.keras.Model): compiled Keras model
        dataset(tf.data.Dataset): dataset to evaluate on
        verbose(int): 0=silent, 1=progress bar, 2=one line per epoch
    Returns:
        a dict of evaluation metrics
    """
    results = model.evaluate(dataset, verbose=verbose, return_dict=True)
    return results


def get_predictions(model, dataset):
    """ Extract predictions and true labels from dataset
    
    Args:
        model(tf.keras.Model): trained Keras model
        dataset(tf.data.Dataset): dataset to predict on
    Returns:
        y_pred(np.array): predicted class indices
        y_true(np.array): true class indices
    """
    
    y_probs = model.predict(dataset, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = []
    for _, labels in dataset:
        y_true.extend(labels.numpy())
    y_true = np.array(y_true)
    return y_pred, y_true


def compute_confusion_matrix(y_true, y_pred, num_classes):
    """ compute confusion matrix
    Args:
        y_true(np.array): true class indices
        y_pred(np.array): predicted class indices
        num_classes(int): number of classes
    Returns:
        confusion matrix (np.ndarray)
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    return cm


def get_classification_report(y_true, y_pred, class_names, output_dict=True):
    """ Generate classification report with per-class metrics
    
    Args:
        y_true(np.array): true class indices
        y_pred(np.array): predicted class indices
        class_names(list): list of class names corresponding to indices
        output_dict(bool): if True, return report as dict; else return string
    Returns:
        classification report (dict or string)
    """

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=output_dict,
        zero_division=0,
    )
    return report


def get_per_class_metrics(y_true, y_pred, class_names):
    """ extract per-class precision, recall, f1-score, support
    Args:
        y_true(np.array): true class indices
        y_pred(np.array): predicted class indices
        class_names(list): list of class names
    Returns:
        dict of per-class metrics sorted by f1-score
    """

    report = get_classification_report(y_true, y_pred, class_names, output_dict=True)
    per_class = {}
    for name in class_names:
        if name in report:
            per_class[name] = {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
    sorted_classes = sorted(per_class.items(), key=lambda x: x[1]["f1"])
    return dict(sorted_classes)


def top_confused_pairs(cm, class_names, top_k=10):
    """ extract top K most confused class pairs from confusion matrix 
    Args:
        cm(np.ndarray): confusion matrix
        class_names(list): list of class names
        top_k(int): number of top confused pairs to return
    Returns:
        A list of (true_class, predicted_class, count) tuples,
        sorted by count descending
    """

    num_classes = cm.shape[0]
    confused = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confused.append((class_names[i], class_names[j], cm[i, j]))
    confused.sort(key=lambda x: x[2], reverse=True)
    return confused[:top_k]


def get_accuracy_by_class(y_true, y_pred, num_classes):
    """
    Compute per-class accuracy (recall)
    Args:
        y_true(np.array): true class indices
        y_pred(np.array): predicted class indices
        num_classes(int): number of classes
    Returns:
        np.array of per-class accuracy
    """

    cm = confusion_matrix(y_true, y_pred, num_classes)
    acc = np.zeros(num_classes)
    for i in range(num_classes):
        total = cm[i, :].sum()
        if total > 0:
            acc[i] = cm[i, i]/total
        else:
            acc[i] = 0.0 
    return acc

