import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from data_loader import load_processed_data
from evaluation import evaluate_classifier

LABEL_NAMES = {0: 'fatigued', 1: 'impulsive', 2: 'careful', 3: 'focused'}
NUM_STATES = 4

def build_supervised_hmm(X, y):
    """
    Manually calculates the parameters for a Supervised HMM 
    using the chronological labels.
    """
    n_features = X.shape[1]
    
    # 1. Start Probabilities (overall frequency of each state)
    startprob = np.array([0.25, 0.25, 0.25, 0.25])

    # 2. Transition Matrix (count actual state-to-state transitions over time)
    transmat = np.zeros((NUM_STATES, NUM_STATES))
    
    # Add a small pseudo-count (Laplace smoothing) so rare transitions don't have 0% probability
    pseudo_count = 1e-3 
    transmat += pseudo_count
    
    for i in range(len(y) - 1):
        transmat[y[i], y[i+1]] += 1
        
    # Normalize rows so they sum to 1
    for i in range(NUM_STATES):
        transmat[i] /= np.sum(transmat[i])

    # 3. Emission Probabilities (Calculate Gaussian Mean & Variance for each state)
    means = np.zeros((NUM_STATES, n_features))
    covars = np.zeros((NUM_STATES, n_features))

    for i in range(NUM_STATES):
        X_state = X[y == i]
        if len(X_state) > 0:
            means[i] = np.mean(X_state, axis=0)
            # Add tiny value to prevent zero variance crashes on perfectly static features
            covars[i] = np.var(X_state, axis=0) + 1e-6 
        else:
            covars[i] = np.ones(n_features)

    # 4. Construct the HMM
    # init_params="" prevents the model from overwriting our injected parameters
    model = GaussianHMM(n_components=NUM_STATES, covariance_type="diag", init_params="")
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = covars
    
    return model

def run_hmm_experiment(test_size=0.2):
    X, y = load_processed_data()
    physio_sum = np.sum(np.abs(X[:, 53:]))
    print(f"Sum of all physiological data: {physio_sum}")

    # CRITICAL FIX: DO NOT SHUFFLE!
    # We must split the array chronologically to preserve the temporal sequence.
    # The first 80% of trials are training, the final 20% are testing.
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Scale the features so the Variance/Covariance calculations aren't distorted
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build the model using our chronologically ordered data
    model = build_supervised_hmm(X_train, y_train)
    
    # .predict() runs the Viterbi Algorithm, which uses the Transition Matrix
    # to find the most mathematically logical sequence of states over time
    y_pred = model.predict(X_test)
    
    print("True Supervised HMM trained using temporal sequences.")
    results = evaluate_classifier(y_test, y_pred, label_names=LABEL_NAMES, plot_confusion=True)
    return model, results

if __name__ == '__main__':
    run_hmm_experiment()