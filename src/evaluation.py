import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def evaluate_classifier(y_true, y_pred, label_names=None, plot_confusion=True):
    if label_names is None:
        label_names = {0: 'fatigued', 1: 'impulsive', 2: 'careful', 3: 'focused'}

    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    target_names = [label_names[label] for label in labels]

    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0)
    print('Classification Report:\n')
    print(report)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f'Accuracy: {acc:.4f}')
    print(f'Macro F1: {macro_f1:.4f}')
    print(f'Weighted F1: {weighted_f1:.4f}')

    if plot_confusion:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
               xticklabels=target_names, yticklabels=target_names,
               ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')

        fig.tight_layout()
        plt.show()

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'report': report,
    }
