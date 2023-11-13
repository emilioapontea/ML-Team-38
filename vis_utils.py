from typing import Sequence, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Subset

def plot_loss_history(train_loss_history: float, val_loss_history: float) -> None:
    plt.figure()
    epoch_idxs = range(len(train_loss_history))

    plt.plot(epoch_idxs, train_loss_history, "-b", label="training")
    plt.plot(epoch_idxs, val_loss_history, "-r", label="validation")
    plt.title("Loss history")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.show()

def plot_acc_history(train_acc_history: float, val_acc_history: float) -> None:
    plt.figure()
    epoch_idxs = range(len(train_acc_history))

    plt.plot(epoch_idxs, train_acc_history, "-b", label="training")
    plt.plot(epoch_idxs, val_acc_history, "-r", label="validation")
    plt.title("Accuracy history")
    plt.legend()
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Epochs")
    plt.show()

def generate_confusion_data(
    model: nn.Module,
    dataset: DataLoader,
    class_labels: Sequence[str] | None = None,
) -> Tuple[Sequence[int], Sequence[int], Sequence[str]]:

    preds = np.zeros(len(dataset)).astype(np.int32)
    targets = np.zeros(len(dataset)).astype(np.int32)

    if class_labels is None:
        labels = np.arange(len(dataset)).astype(str).tolist()
    else:
        labels = class_labels

    model.eval()

    model_output = []

    for i, (x, y) in enumerate(dataset):
            targets[i:i+len(y)] = y
            model_output = model(x)
            preds[i:i+len(y)] = model_output.argmax(-1)

    class_labels = label_to_idx

    preds = torch.tensor(preds)
    targets = torch.tensor(targets)

    model.train()

    return targets.cpu().numpy(), preds.cpu().numpy(), labels


def generate_confusion_matrix(
    targets: np.ndarray, preds: np.ndarray, num_classes: int, normalize=True
) -> np.ndarray:

    confusion_matrix = np.zeros((num_classes, num_classes))

    for target, prediction in zip(targets, preds):
        confusion_matrix[target, prediction] += 1
    if normalize:
        confusion_matrix /= np.sum(confusion_matrix, axis=-1).reshape(num_classes,1)

    return confusion_matrix


def plot_confusion_matrix(
    confusion_matrix: np.ndarray, class_labels: Sequence[str]
) -> None:

    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)

    num_classes = len(class_labels)

    ax.imshow(confusion_matrix, cmap="Blues")

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Ground-Truth label")
    ax.set_title("Confusion Matrix")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(num_classes):
        for j in range(num_classes):
            _ = ax.text(
                j,
                i,
                f"{confusion_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
            )

    plt.show()


def generate_and_plot_confusion_matrix(
    model: nn.Module, dataset: DataLoader, use_cuda: bool = False
) -> None:

    targets, predictions, class_labels = generate_confusion_data(
        model, dataset
    )

    confusion_matrix = generate_confusion_matrix(
        np.array(targets, dtype=np.int32),
        np.array(predictions, np.int32),
        len(class_labels),
    )

    plot_confusion_matrix(confusion_matrix, class_labels)