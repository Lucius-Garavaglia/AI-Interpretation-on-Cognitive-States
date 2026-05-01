import argparse

from decision_tree import run_decision_tree_experiment
from cnn_model import run_cnn_experiment
from hmm_model import run_hmm_experiment


def main():
    parser = argparse.ArgumentParser(description='Run cognitive state models on processed EEG features.')
    parser.add_argument('model', choices=['decision_tree', 'cnn', 'hmm'], help='Which model to run')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test split fraction')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    parser.add_argument('--max_depth', type=int, default=None, help='Decision tree max depth')
    parser.add_argument('--epochs', type=int, default=20, help='CNN training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='CNN batch size')
    args = parser.parse_args()

    if args.model == 'decision_tree':
        run_decision_tree_experiment(test_size=args.test_size, random_state=args.random_state, max_depth=args.max_depth)
    elif args.model == 'cnn':
        run_cnn_experiment(test_size=args.test_size, random_state=args.random_state, batch_size=args.batch_size, epochs=args.epochs)
    elif args.model == 'hmm':
        run_hmm_experiment(test_size=args.test_size, random_state=args.random_state)


if __name__ == '__main__':
    main()
