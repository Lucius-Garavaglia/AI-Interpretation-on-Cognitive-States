"""
Preprocessing module for EEG cognitive state classification.

Pipeline:
1. Load behavioral data → filter RT=0 rows → compute cognitive state labels (4 states)
2. Load EEG → window around each trial → extract features (53 EEG + optional physio)
3. Optional: load physiological signals for supplementary features

Cognitive States (derived from accuracy + reaction time):
  3: Focused   - correct + fast (optimal)
  2: Careful   - correct + slow (deliberate)
  1: Impulsive - incorrect + fast (rash, but they tried)
  0: Fatigued  - incorrect + slow (disengaged, but they tried)

Thresholds are computed from the BASELINE (session 101, NoMusic) CORRECT responses only.
RT=0 rows are filtered out (no response given, time passing between trials).

Key data findings:
- Session 100 = practice (skipped)
- Sessions 101-104 = NoMusic, RelaxMusic, StressMusic, NewMusic
- RT=0 means no response, not a fast response
- Incorrect answers with RT>0 are rare but real (impulsive errors)
- Singular correct response with rt = 0 is likely a data quirk, not a real trial (filtered out)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')


class EEGPreprocessor:
    """
    Takes raw CSV files from the brain-wearable-monitoring dataset
    and produces labeled EEG feature matrices ready for ML models.
    """
    
    def __init__(self, data_path="Data/raw"):
        self.data_path = Path(data_path)
        self.eeg_sampling_rate = 256  # Muse headband: 256 Hz
        
        # EEG channels from Muse headband
        self.eeg_channels = ['TP9', 'AF7', 'AF8', 'TP10']
        
    # ==========================================
    # PART 1: BEHAVIORAL DATA & COGNITIVE LABELS
    # ==========================================
    
    def load_behavioral_data(self, subject_id, experiment="Experiment_1"):
        """Load n_back_responses.csv for a subject."""
        filepath = self.data_path / experiment / subject_id / "n_back_responses.csv"
        
        if not filepath.exists():
            print(f"  Warning: {filepath} not found")
            return None
        
        return pd.read_csv(filepath)
    
    def extract_trial_data(self, behavior_df):
        """
        Extract individual trial data from the wide-format behavioral DataFrame.
        
        The CSV has columns like Fixation101.ACC, Fixation101.RT,
        Stimulus101.OnsetTime for each session.
        
        Experiment 1: sessions 101-104 (NoMusic, RelaxMusic, StressMusic, NewMusic)
        Experiment 2: sessions 101-103 (NoMusic, Perfume, Coffee)
        
        Filters out:
        - Session 100 (practice)
        - RT=0 rows (no response given, time passing between trials)
        """
        # Auto-detect session numbers from column names
        session_nums = set()
        for col in behavior_df.columns:
            if 'Fixation' in col and '.ACC' in col:
                try:
                    num = int(col.replace('Fixation', '').split('.')[0])
                    if num >= 101:  # Skip practice session 100
                        session_nums.add(num)
                except ValueError:
                    pass
        
        if len(session_nums) == 0:
            print("  No session data found in behavioral file")
            return None
        
        print(f"  Found sessions: {sorted(session_nums)}")
        
        trials = []
        total_with_rt = 0
        total_all = 0
        
        for session_num in sorted(session_nums):
            acc_col = f"Fixation{session_num}.ACC"
            rt_col = f"Fixation{session_num}.RT"
            onset_col = f"Stimulus{session_num}.OnsetTime"
            
            # Only process sessions with all required columns
            if not all(c in behavior_df.columns for c in [acc_col, rt_col, onset_col]):
                continue
            
            # Get non-null trial data
            session_trials = behavior_df[[acc_col, rt_col, onset_col]].dropna()
            total_all += len(session_trials)
            
            # FILTER: Keep only rows where RT > 0 (actual responses)
            actual_responses = session_trials[session_trials[rt_col] > 0]
            total_with_rt += len(actual_responses)
            
            # Get condition name from Running[Block]
            condition = f"session_{session_num}"
            if 'Running[Block]' in behavior_df.columns:
                unique_conditions = behavior_df['Running[Block]'].dropna().unique()
                if len(unique_conditions) > 0:
                    condition = str(unique_conditions[0])
            
            for idx, row in actual_responses.iterrows():
                trials.append({
                    'session': session_num,
                    'condition': condition,
                    'accuracy': int(row[acc_col]),
                    'reaction_time': float(row[rt_col]),
                    'onset_time_ms': float(row[onset_col]),
                })
        
        print(f"  Total non-null trials: {total_all}")
        print(f"  Trials with RT > 0: {total_with_rt} (kept)")
        print(f"  Trials with RT = 0: {total_all - total_with_rt} (filtered - no response)")
        
        return pd.DataFrame(trials)
    
    def compute_baseline_thresholds(self, trial_df):
        """
        Compute RT thresholds from the BASELINE (session 101, NoMusic).
        Uses only CORRECT responses to establish the "fast" cutoff.
        
        This way, we measure stimulus effects relative to unstimulated
        correct performance, not contaminated by non-responses.
        """
        # Baseline = session 101, correct answers only
        baseline_trials = trial_df[(trial_df['session'] == 101) & 
                                   (trial_df['accuracy'] == 1)]
        
        if len(baseline_trials) == 0:
            # Fallback: use first session's correct answers
            first_session = trial_df['session'].min()
            baseline_trials = trial_df[(trial_df['session'] == first_session) & 
                                       (trial_df['accuracy'] == 1)]
            print(f"  Session 101 correct not found, using session {first_session}")
        
        rt_median = baseline_trials['reaction_time'].median()
        print(f"  Baseline correct trials: {len(baseline_trials)}")
        print(f"  Baseline median RT (correct only): {rt_median:.0f} ms")
        
        return rt_median
    
    def compute_cognitive_state(self, accuracy, reaction_time, rt_threshold):
        """
        Classify a single trial into one of 4 cognitive states.
        
        States:
          3: Focused   - correct + fast (peak performance)
          2: Careful   - correct + slow (speed-accuracy tradeoff)
          1: Impulsive - incorrect + fast (responding without thinking)
          0: Fatigued  - incorrect + slow (disengaged but still trying)
        """
        correct = (accuracy >= 0.5)
        fast = (reaction_time <= rt_threshold)
        
        if correct and fast:
            return 3   # Focused
        elif correct and not fast:
            return 2   # Careful
        elif not correct and fast:
            return 1   # Impulsive
        else:
            return 0   # Fatigued
    
    def create_labels(self, trial_df):
        """
        Add cognitive state labels using baseline-referenced thresholds.
        """
        rt_threshold = self.compute_baseline_thresholds(trial_df)
        
        trial_df['cognitive_state'] = trial_df.apply(
            lambda row: self.compute_cognitive_state(
                row['accuracy'], 
                row['reaction_time'],
                rt_threshold
            ),
            axis=1
        )
        
        state_names = {
            0: 'fatigued',
            1: 'impulsive', 
            2: 'careful',
            3: 'focused'
        }
        trial_df['state_label'] = trial_df['cognitive_state'].map(state_names)
        
        return trial_df, rt_threshold
    
    # ==========================================
    # PART 2: EEG WINDOWING & FEATURE EXTRACTION
    # ==========================================
    
    def load_eeg(self, subject_id, experiment="Experiment_1"):
        """Load EEG recording. Handles partial downloads gracefully."""
        filepath = self.data_path / experiment / subject_id / "EEG_recording.csv"
        
        if not filepath.exists():
            print(f"  Warning: EEG file not found at {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath)
            print(f"  Loaded EEG: {len(df):,} rows")
            return df
        except Exception as e:
            print(f"  Error loading EEG: {e}")
            return None
    
    def get_eeg_window(self, eeg_df, onset_time_ms, 
                       window_before_ms=500, window_after_ms=2000):
        """
        Extract EEG segment around a trial event.
        
        onset_time_ms: Stimulus onset time in ms from experiment start
        window_before_ms: EEG to include BEFORE stimulus (500ms)
        window_after_ms: EEG to include AFTER stimulus (2000ms = covers response)
        """
        eeg_start_time = eeg_df.iloc[0, 0]
        onset_time_sec = onset_time_ms / 1000.0 + eeg_start_time
        
        window_start = onset_time_sec - (window_before_ms / 1000.0)
        window_end = onset_time_sec + (window_after_ms / 1000.0)
        
        timestamps = eeg_df.iloc[:, 0].values
        mask = (timestamps >= window_start) & (timestamps <= window_end)
        
        return eeg_df[mask]
    
    def extract_eeg_features(self, eeg_window, channels=None):
        """
        Extract 13 features per EEG channel from a time window.
        
        Features per channel:
        - 5 band powers: delta, theta, alpha, beta, gamma
        - 4 statistical: mean, std, skewness, kurtosis
        - 3 Hjorth: activity, mobility, complexity
        - 1 ratio: theta/beta ratio
        
        Total: 4 channels × 13 = 52 features
        Plus: 1 frontal alpha asymmetry = 53 features total
        """
        if channels is None:
            channels = self.eeg_channels
        
        available_channels = [ch for ch in channels if ch in eeg_window.columns]
        
        if len(available_channels) == 0:
            return None
        
        all_channel_features = {}
        feature_vector = []
        
        # NumPy 2.0+ compatibility for trapz
        try:
            trapz_func = np.trapezoid
        except AttributeError:
            trapz_func = np.trapz
        
        for channel in channels:
            if channel not in eeg_window.columns:
                feature_vector.extend([0.0] * 13)
                all_channel_features[channel] = [0.0] * 13
                continue
            
            data = eeg_window[channel].dropna().values
            
            if len(data) < 32:
                feature_vector.extend([0.0] * 13)
                all_channel_features[channel] = [0.0] * 13
                continue
            
            # ---- FREQUENCY DOMAIN: Band Powers ----
            freqs, psd = signal.welch(
                data, 
                fs=self.eeg_sampling_rate,
                nperseg=min(128, len(data))
            )
            
            delta = trapz_func(psd[(freqs >= 0.5) & (freqs <= 4)])
            theta = trapz_func(psd[(freqs >= 4) & (freqs <= 8)])
            alpha = trapz_func(psd[(freqs >= 8) & (freqs <= 13)])
            beta  = trapz_func(psd[(freqs >= 13) & (freqs <= 30)])
            gamma = trapz_func(psd[(freqs >= 30) & (freqs <= 45)])
            
            # ---- TIME DOMAIN: Statistical Features ----
            mean_val = float(np.mean(data))
            std_val  = float(np.std(data))
            skew_val = float(skew(data)) if len(data) > 2 else 0.0
            kurt_val = float(kurtosis(data)) if len(data) > 3 else 0.0
            
            # ---- HJORTH PARAMETERS ----
            activity = float(np.var(data))
            mobility = 0.0
            complexity = 0.0
            
            if activity > 0:
                diff1 = np.diff(data)
                mobility = float(np.sqrt(np.var(diff1) / activity))
                
                if mobility > 0:
                    diff2 = np.diff(diff1)
                    complexity = float(np.sqrt(np.var(diff2) / np.var(diff1)) / mobility)
            
            # ---- THETA/BETA RATIO (attention biomarker) ----
            tbr = theta / beta if beta > 0 else 0.0
            
            channel_features = [
                delta, theta, alpha, beta, gamma,
                mean_val, std_val, skew_val, kurt_val,
                activity, mobility, complexity,
                tbr
            ]
            
            all_channel_features[channel] = channel_features
            feature_vector.extend(channel_features)
        
        # ---- FRONTAL ALPHA ASYMMETRY (emotional/motivational state) ----
        if 'AF7' in all_channel_features and 'AF8' in all_channel_features:
            af7_alpha = all_channel_features['AF7'][2]
            af8_alpha = all_channel_features['AF8'][2]
            
            if af7_alpha + af8_alpha > 0:
                alpha_asymmetry = (af7_alpha - af8_alpha) / (af7_alpha + af8_alpha)
            else:
                alpha_asymmetry = 0.0
            
            feature_vector.append(alpha_asymmetry)
        else:
            feature_vector.append(0.0)
        
        return np.array(feature_vector, dtype=np.float32)
    
    # ==========================================
    # PART 3: PHYSIOLOGICAL SIGNALS (OPTIONAL)
    # ==========================================
    
    def load_physiological(self, subject_id, experiment="Experiment_1",
                           sensors=['EDA', 'HR']):
        """Load Empatica wristband data (heart rate, skin conductance, etc.)."""
        physio_data = {}
        
        for side in ['Left', 'Right']:
            for sensor in sensors:
                filepath = self.data_path / experiment / subject_id / f"{side}_{sensor}.csv"
                
                if filepath.exists():
                    try:
                        df = pd.read_csv(filepath)
                        if len(df) > 0:
                            physio_data[f"{side}_{sensor}"] = df
                    except Exception:
                        pass
        
        if physio_data:
            print(f"  Loaded {len(physio_data)} physiological signal files")
        
        return physio_data
    
    def extract_physio_features(self, physio_data, eeg_window_start, eeg_window_end):
        """Extract mean values of physiological signals during an EEG window."""
        features = []
        
        for name, df in physio_data.items():
            try:
                timestamps = df.iloc[:, 0].values
                values = df.iloc[:, 1].values if df.shape[1] > 1 else df.iloc[:, 0].values
                
                mask = (timestamps >= eeg_window_start) & (timestamps <= eeg_window_end)
                window_values = values[mask]
                
                features.append(float(np.mean(window_values)) if len(window_values) > 0 else 0.0)
            except Exception:
                features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    # ==========================================
    # PART 4: BUILD COMPLETE DATASET
    # ==========================================
    
    def process_subject(self, subject_id, experiment="Experiment_1",
                        use_physio=False):
        """
        Process one subject end-to-end:
        1. Create cognitive state labels from behavioral data
        2. Window EEG around each trial
        3. Extract features (EEG + optional physiological)
        4. Return X (features) and y (labels)
        """
        print(f"\n{'='*50}")
        print(f"Processing: {experiment}/{subject_id}")
        print(f"{'='*50}")
        
        # --- Step 1: Behavioral data → labels ---
        behavior_df = self.load_behavioral_data(subject_id, experiment)
        if behavior_df is None:
            return None, None, None
        
        trial_df = self.extract_trial_data(behavior_df)
        if trial_df is None or len(trial_df) == 0:
            print("  No trials extracted")
            return None, None, None
        
        trial_df, rt_threshold = self.create_labels(trial_df)
        
        print(f"  Total trials with responses: {len(trial_df)}")
        print(f"  State distribution:")
        for state_id, state_name in [(0, 'fatigued'), (1, 'impulsive'), 
                                       (2, 'careful'), (3, 'focused')]:
            count = (trial_df['cognitive_state'] == state_id).sum()
            pct = 100 * count / len(trial_df) if len(trial_df) > 0 else 0
            print(f"    {state_id} ({state_name:10s}): {count:4d} trials ({pct:.1f}%)")
        
        # --- Step 2: Load EEG ---
        eeg_df = self.load_eeg(subject_id, experiment)
        if eeg_df is None:
            return None, None, trial_df
        
        # --- Step 3: Load physiological data (optional) ---
        physio_data = {}
        if use_physio:
            physio_data = self.load_physiological(subject_id, experiment)
        
        # --- Step 4: Extract features for each trial ---
        X_features = []
        y_labels = []
        trial_indices = []
        
        for idx, trial in trial_df.iterrows():
            eeg_window = self.get_eeg_window(
                eeg_df,
                trial['onset_time_ms'],
                window_before_ms=500,
                window_after_ms=2000
            )
            
            if eeg_window is None or len(eeg_window) < 16:
                continue
            
            eeg_feats = self.extract_eeg_features(eeg_window)
            
            if eeg_feats is None:
                continue
            
            # Optional: add physiological features
            if use_physio and len(physio_data) > 0:
                window_start = eeg_window.iloc[0, 0]
                window_end = eeg_window.iloc[-1, 0]
                physio_feats = self.extract_physio_features(physio_data, window_start, window_end)
                combined = np.concatenate([eeg_feats, physio_feats])
            else:
                combined = eeg_feats
            
            X_features.append(combined)
            y_labels.append(trial['cognitive_state'])
            trial_indices.append(idx)
        
        X = np.array(X_features, dtype=np.float32)
        y = np.array(y_labels, dtype=np.int32)
        
        print(f"\n  Final dataset: {X.shape[0]} samples × {X.shape[1]} features")
        print(f"  Class balance: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # Check for NaN/Inf
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"  ⚠️  NaN values: {nan_count}, Inf values: {inf_count}")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X, y, trial_df


# ==========================================
# PROCESS ALL SUBJECTS
# ==========================================
if __name__ == "__main__":
    preprocessor = EEGPreprocessor("Data/raw")
    
    # Find all subjects in Experiment_1
    exp_path = Path("Data/raw/Experiment_1")
    subjects = sorted([d.name for d in exp_path.iterdir() 
                       if d.is_dir() and not d.name.startswith('Excluded')])
    
    print(f"\n{'='*60}")
    print(f"PROCESSING ALL SUBJECTS IN EXPERIMENT 1")
    print(f"{'='*60}")
    print(f"Found {len(subjects)} subjects: {subjects}")
    
    all_X = []
    all_y = []
    all_subjects = []
    
    for subject in subjects:
        X, y, trial_df = preprocessor.process_subject(subject, "Experiment_1", use_physio=False)
        
        if X is not None and len(X) > 0:
            all_X.append(X)
            all_y.append(y)
            all_subjects.extend([subject] * len(y))
    
    if len(all_X) > 0:
        X_combined = np.vstack(all_X)
        y_combined = np.concatenate(all_y)
        
        print(f"\n{'='*60}")
        print(f"COMBINED DATASET")
        print(f"{'='*60}")
        print(f"Total subjects processed: {len(set(all_subjects))}")
        print(f"Total samples: {X_combined.shape[0]}")
        print(f"Total features: {X_combined.shape[1]}")
        print(f"Class distribution:")
        for state_id, state_name in [(0, 'fatigued'), (1, 'impulsive'), 
                                       (2, 'careful'), (3, 'focused')]:
            count = (y_combined == state_id).sum()
            print(f"  {state_id} ({state_name:10s}): {count:5d} ({100*count/len(y_combined):.1f}%)")
        
        # Save the combined dataset
        np.save("Data/processed/X_features.npy", X_combined)
        np.save("Data/processed/y_labels.npy", y_combined)
        print(f"\nSaved to Data/processed/")
    else:
        print("\nNo subjects processed successfully.")