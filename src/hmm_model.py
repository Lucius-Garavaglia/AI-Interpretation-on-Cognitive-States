import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split

from data_loader import load_processed_data, get_train_test_split
from evaluation import evaluate_classifier


LABEL_NAMES = {0: 'fatigued', 1: 'impulsive', 2: 'careful', 3: 'focused'}


def train_state_hmms(X, y, n_components=2, random_state=42):
    models = {}
    for state in sorted(np.unique(y)):
        X_state = X[y == state]
        if len(X_state) < n_components:
            raise ValueError(f'Not enough samples to train HMM for state {state}')

        model = GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=200, random_state=random_state)
        model.fit(X_state)
        models[state] = model
    return models


def predict_hmm(models, X):
    predictions = []
    for sample in X:
        sample = sample.reshape(1, -1)
        scores = {state: model.score(sample) for state, model in models.items()}
        predictions.append(max(scores, key=scores.get))
    return np.array(predictions, dtype=int)


def run_hmm_experiment(test_size=0.2, random_state=42, n_components=2):
    X, y = load_processed_data()
    X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=test_size, random_state=random_state)

    models = train_state_hmms(X_train, y_train, n_components=n_components, random_state=random_state)
    y_pred = predict_hmm(models, X_test)
    results = evaluate_classifier(y_test, y_pred, label_names=LABEL_NAMES, plot_confusion=True)
    return models, results


if __name__ == '__main__':
    run_hmm_experiment()
