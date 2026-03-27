import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any


def plot_training_history(history, title="Training History", figsize=(14, 5), save_path=None):
    """ 
    Plot training and validation metrics over epochs
    Args:
        history(dict): dictionary from model.fit().history containing keys
        title(str): title for the plots
        figsize(tuple): figure size for the plots
        save_path(str): optional path to save figure,
                        if None, it will just display plot without saving
    """

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # loss
    axes[0].plot(history["loss"], label="Train Loss", marker="o")
    axes[0].plot(history["val_loss"], label="Val Loss", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} — Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # accuracy 
    axes[1].plot(history["accuracy"], label="Train Accuracy", marker="o")
    axes[1].plot(history["val_accuracy"], label="Val Accuracy", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title} — Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_multiple_histories(histories, metric="val_accuracy", title="Model Comparison",
                            figsize=(10, 6), save_path=None,):
    """ 
    Plot a single metric from multiple training runs for comparison 
    Args:
        histories(dict): dictionary of {model_name: history_dict} where history_dict is from model.fit().history
        metric(str): metric to plot ("val_accuracy" or "val_loss" etc)
        title(str): title for the plot
        figsize(tuple): figure size for the plot
        save_path(str): optional path to save figure,
                        if None, it will just display plot without saving
    """

    plt.figure(figsize=figsize)
    for name, history in histories.items():
        if metric in history:
            plt.plot(history[metric], label=name, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_confusion_matrix(cm, class_names, normalize=False, title="Confusion Matrix", 
                          figsize=(16, 14), save_path=None):
    """
    plot confusion matrix as a heatmap  
    Args:
        cm(np.ndarray): confusion matrix of shape (num_classes, num_classes)
        class_names(list): list of class names for axis labels
        normalize(bool): if True, normalize each row to [0, 1] to show percentage
        title(str): title for plot
        figsize(tuple): figure size for plot
        save_path(str): optional path to save figure,
                        if None, it will just display plot without saving
    """

    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums!=0)
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=False,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Percentage" if normalize else "Count"},
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_top_k_confusion_pairs(confused_pairs, top_k=15, title="Top Confused Class Pairs",
                               figsize=(10, 8),save_path=None):
    """Plot the top-K most confused class pairs as a horizontal bar chart
    Args:
        confused_pairs(list): output from top_confused_pairs() of shape (true_class, pred_class, count)
        top_k(int): number of top pairs to plot
        title(str): title for plot
        figsize(tuple): figure size for plot
        save_path(str): optional path to save figure,
                        if None, it will just display plot without saving
    """

    pairs = confused_pairs[:top_k]
    labels = [f"{true_cls} → {pred_cls}" for true_cls, pred_cls, _ in pairs]
    counts = [count for _, _, count in pairs]
    plt.figure(figsize=figsize)
    plt.barh(labels, counts, color="coral", edgecolor="black")
    plt.xlabel("Number of Misclassifications")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_per_class_f1(per_class_metrics, top_n=20, worst=True, title="Per-Class F1 Scores", 
                      figsize=(10, 8),save_path=None):
    """ Plot per-class F1 scores as a horizontal bar chart
    Args:
        per_class_metrics(dict): output from per_class_metrics(), it is a dict mapping 
                                 class_name to a dict of metrics
        top_n(int): number of classes to plot
        worst(bool): if True, plot worst N classes, if False, plot best N classes
        title(str): title for plot
        figsize(tuple): figure size for plot
        save_path(str): optional path to save figure,
                        if None, it will just display plot without saving              
"""

    items = list(per_class_metrics.items())
    if worst:
        items = items[:top_n] 
        color = "coral"
        title_suffix = f" (Worst {top_n})"
    else:
        items = items[-top_n:] 
        color = "mediumseagreen"
        title_suffix = f" (Best {top_n})"
    class_names_subset = [name for name, _ in items]
    f1_scores = [metrics["f1"] for _, metrics in items]
    plt.figure(figsize=figsize)
    plt.barh(class_names_subset, f1_scores, color=color, edgecolor="black")
    plt.xlabel("F1 Score")
    plt.title(title+title_suffix)
    plt.xlim(0, 1)
    if not worst:
        plt.gca().invert_yaxis()  
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_sample_predictions(images, y_true, y_pred, class_names, n_samples=16, n_cols=4, 
                            figsize=(12, 12), save_path=None):
    """ Plot sample images with true and predicted labels
    
    Args:
        images(np.ndarray): array of images to plot with shape (N, H, W, 3)
        y_true(np.ndarray): true labels
        y_pred(np.ndarray): predicted labels
        class_names(list): lsit of class names
        n_samples(int): number of images to plot
        n_cols(int): number of columns in the plot grid
        figsize(tuple): figure size
        save_path(str): optional path to save figure,
                        if None, it will just display plot without saving
    """

    n_rows = (n_samples+n_cols-1)//n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    for i in range(n_samples):
        if i >= len(images):
            break
        img = images[i]
        true_label = class_names[y_true[i]]
        pred_label = class_names[y_pred[i]]
        axes[i].imshow(img.astype(np.uint8))
        axes[i].axis("off")
        color = "green" if y_true[i] == y_pred[i] else "red"
        title = f"True: {true_label}\nPred: {pred_label}"
        axes[i].set_title(title, fontsize=8, color=color)
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()


def models_comparison(results):
    """
    Print a comparison table of multiple models 
    Args:
        results(dict): dict mapping model_name to a dict of metrics
                       e.g. {"Model A": {"accuracy": 0.9, "f1": 0.8}, ...}
    """

    all_metrics = set()
    for metrics in results.values():
        all_metrics.update(metrics.keys())
    all_metrics = sorted(all_metrics)
    header = f"{'Model':<28s} "+" ".join([f"{m:>16s}" for m in all_metrics])
    print(header)
    print("=" * len(header))
    for model, metrics in results.items():
        row = f"{model:<28s} "
        for metric in all_metrics:
            value = metrics.get(metric, float("nan"))
            row += f"{value:>16.4f} "
        print(row)

