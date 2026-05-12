import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.class_weight import compute_class_weight

from data_loader import load_processed_data, get_train_test_split
from evaluation import evaluate_classifier


LABEL_NAMES = {0: 'fatigued', 1: 'impulsive', 2: 'careful', 3: 'focused'}


def build_decision_tree(max_depth=None, min_samples_leaf=20, random_state=42):
    class_weight = 'balanced'
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        class_weight=class_weight,
    )


def run_decision_tree_experiment(test_size=0.2, random_state=42, max_depth=None, min_samples_leaf=20):
    X, y = load_processed_data()
    X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=test_size, random_state=random_state)

    clf = build_decision_tree(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print('Decision Tree trained with max_depth=', max_depth, 'min_samples_leaf=', min_samples_leaf)
    results = evaluate_classifier(y_test, y_pred, label_names=LABEL_NAMES, plot_confusion=True)

    feature_importances = clf.feature_importances_
    important_idx = np.argsort(feature_importances)[::-1][:15]
    print('\nTop 15 feature importances:')
    for idx in important_idx:
        print(f'  Feature {idx}: importance={feature_importances[idx]:.4f}')

    return clf, results


if __name__ == '__main__':
    run_decision_tree_experiment()
