import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_processed_data(data_dir='Data/processed'):
    data_dir = Path(data_dir)
    X = np.load(data_dir / 'X_features.npy')
    y = np.load(data_dir / 'y_labels.npy')
    return X, y


def get_train_test_split(X, y, test_size=0.2, random_state=42, stratify=True):
    if stratify:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
