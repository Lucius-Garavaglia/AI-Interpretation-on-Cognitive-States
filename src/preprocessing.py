"""
Preprocessing module for EEG cognitive state classification.

Pipeline:
Current implementation note: responses are recovered from either Stimulus or
Fixation event columns, and EEG windows are aligned with Start_time_unix.
1. Load behavioral data → filter RT=0 rows → compute cognitive state labels (4 states)
2. Load EEG → window around each trial → extract features (153 EEG + optional physio)
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
        self.eeg_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45),
        }
        self.base_features_per_channel = 18
        self.time_frequency_features_per_channel = len(self.eeg_bands) * 5
        self.features_per_channel = (
            self.base_features_per_channel + self.time_frequency_features_per_channel
        )
        self.covariance_feature_pairs = [
            (left, right)
            for left in range(len(self.eeg_channels))
            for right in range(left, len(self.eeg_channels))
        ]
        self.covariance_features = len(self.covariance_feature_pairs)
        self.physio_sides = ['Left', 'Right']
        self.physio_sensors = ['EDA', 'TEMP', 'ACC', 'BVP', 'HR', 'IBI']
        self.physio_value_counts = {
            'EDA': 1,
            'TEMP': 1,
            'ACC': 3,
            'BVP': 1,
            'HR': 1,
            'IBI': 1,
        }
        self.bandpass_sos = signal.butter(
            4,
            [0.5, 45],
            btype='bandpass',
            fs=self.eeg_sampling_rate,
            output='sos'
        )
        
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
    
    def _to_float(self, value, default=np.nan):
        """Convert behavior CSV values to float while tolerating blanks."""
        try:
            if pd.isna(value) or value == '':
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _get_phase_response(self, row, phase, session_num, stimulus_onset_ms):
        """
        Return one response candidate from either the Stimulus or Fixation event.

        Some subjects respond during the Stimulus event, while others have
        responses recorded in the following Fixation event. For fixation
        responses, event RT is relative to fixation onset, so trial RT is
        computed from absolute RTTime minus the original stimulus onset.
        """
        acc_col = f"{phase}{session_num}.ACC"
        rt_col = f"{phase}{session_num}.RT"
        rt_time_col = f"{phase}{session_num}.RTTime"
        onset_col = f"{phase}{session_num}.OnsetTime"

        if rt_col not in row.index or acc_col not in row.index:
            return None

        event_rt = self._to_float(row.get(rt_col), default=0.0)
        rt_time = self._to_float(row.get(rt_time_col), default=np.nan)
        phase_onset = self._to_float(row.get(onset_col), default=np.nan)

        if event_rt <= 0 and (np.isnan(rt_time) or rt_time <= 0):
            return None

        if not np.isnan(rt_time) and rt_time > 0:
            response_time_ms = rt_time
            reaction_time = response_time_ms - stimulus_onset_ms
        elif not np.isnan(phase_onset) and event_rt > 0:
            response_time_ms = phase_onset + event_rt
            reaction_time = response_time_ms - stimulus_onset_ms
        else:
            response_time_ms = stimulus_onset_ms + event_rt
            reaction_time = event_rt

        if reaction_time <= 0:
            return None

        return {
            'phase': phase.lower(),
            'accuracy': int(self._to_float(row.get(acc_col), default=0.0)),
            'reaction_time': float(reaction_time),
            'response_time_ms': float(response_time_ms),
            'event_rt': float(event_rt),
        }

    def _session_condition(self, session_trials, session_num):
        """Return the actuation condition for one n-back session."""
        if 'Running[Block]' in session_trials.columns:
            conditions = session_trials['Running[Block]'].dropna().astype(str).unique()
            conditions = [condition for condition in conditions if condition.strip()]
            if len(conditions) == 1:
                return conditions[0]
            if len(conditions) > 1:
                return ';'.join(conditions)

        return f"session_{session_num}"

    def extract_trial_data(self, behavior_df):
        """
        Extract individual trial data from the wide-format behavioral DataFrame.

        The CSV has Stimulus and Fixation event columns for each session. Actual
        responses can appear in either event depending on timing/subject, so this
        method checks both and converts every response to stimulus-relative RT.
        """
        # Auto-detect session numbers from Stimulus/Fixation columns.
        session_nums = set()
        for col in behavior_df.columns:
            if ('Fixation' in col or 'Stimulus' in col) and '.ACC' in col:
                try:
                    num = int(col.replace('Fixation', '').replace('Stimulus', '').split('.')[0])
                    if num >= 101:  # Skip practice session 100
                        session_nums.add(num)
                except ValueError:
                    pass

        if len(session_nums) == 0:
            print("  No session data found in behavioral file")
            return None

        print(f"  Found sessions: {sorted(session_nums)}")

        behavior_start_unix = np.nan
        if 'Start_time_unix' in behavior_df.columns:
            starts = pd.to_numeric(behavior_df['Start_time_unix'], errors='coerce').dropna()
            if len(starts) > 0:
                behavior_start_unix = float(starts.iloc[0])

        trials = []
        total_with_rt = 0
        total_all = 0
        response_phase_counts = {'stimulus': 0, 'fixation': 0}

        for session_num in sorted(session_nums):
            onset_col = f"Stimulus{session_num}.OnsetTime"

            if onset_col not in behavior_df.columns:
                continue

            session_trials = behavior_df[behavior_df[onset_col].notna()].copy()
            total_all += len(session_trials)

            session_condition = self._session_condition(session_trials, session_num)

            for idx, row in session_trials.iterrows():
                stimulus_onset_ms = self._to_float(row[onset_col], default=np.nan)
                if np.isnan(stimulus_onset_ms):
                    continue

                candidates = [
                    self._get_phase_response(row, 'Stimulus', session_num, stimulus_onset_ms),
                    self._get_phase_response(row, 'Fixation', session_num, stimulus_onset_ms),
                ]
                candidates = [candidate for candidate in candidates if candidate is not None]

                if not candidates:
                    continue

                # If both phases have a response, keep the earliest valid response.
                response = min(candidates, key=lambda candidate: candidate['response_time_ms'])
                total_with_rt += 1
                response_phase_counts[response['phase']] += 1

                condition = session_condition
                if 'Running[Block]' in row.index and not pd.isna(row.get('Running[Block]')):
                    row_condition = str(row.get('Running[Block]')).strip()
                    if row_condition:
                        condition = row_condition

                trials.append({
                    'session': session_num,
                    'condition': condition,
                    'instruction': row.get('Instruction', np.nan),
                    'block': row.get('Block', np.nan),
                    'trial': row.get('Trial', np.nan),
                    'accuracy': response['accuracy'],
                    'reaction_time': response['reaction_time'],
                    'onset_time_ms': float(stimulus_onset_ms),
                    'response_time_ms': response['response_time_ms'],
                    'response_phase': response['phase'],
                    'behavior_start_unix': behavior_start_unix,
                })

        print(f"  Total non-null trials: {total_all}")
        print(f"  Trials with RT > 0: {total_with_rt} (kept)")
        print(f"  Trials with RT = 0: {total_all - total_with_rt} (filtered - no response)")
        print(f"  Response phases: {response_phase_counts}")

        return pd.DataFrame(trials)
    
    def compute_baseline_thresholds(self, trial_df):
        """
        Compute RT thresholds from the BASELINE (session 101).
        Uses only CORRECT responses to establish the "fast" cutoff.
        
        The experiment mixes n-back difficulty levels. A single subject-wide RT
        threshold can accidentally label harder tasks as "slow" cognitive states,
        so this computes instruction-specific baseline thresholds when possible.
        """
        baseline_trials = trial_df[(trial_df['session'] == 101) & 
                                   (trial_df['accuracy'] == 1)]
        
        if len(baseline_trials) == 0:
            # Fallback: use first session's correct answers
            first_session = trial_df['session'].min()
            baseline_trials = trial_df[(trial_df['session'] == first_session) & 
                                       (trial_df['accuracy'] == 1)]
            print(f"  Session 101 correct not found, using session {first_session}")
        
        global_rt_median = baseline_trials['reaction_time'].median()
        thresholds = {'__global__': float(global_rt_median)}

        if 'instruction' in baseline_trials.columns:
            for instruction, group in baseline_trials.groupby('instruction', dropna=True):
                if len(group) >= 10:
                    thresholds[str(instruction)] = float(group['reaction_time'].median())

        print(f"  Baseline correct trials: {len(baseline_trials)}")
        print(f"  Baseline median RT (correct only): {global_rt_median:.0f} ms")
        if len(thresholds) > 1:
            print(f"  Instruction-specific RT thresholds: {len(thresholds) - 1}")
        
        return thresholds
    
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
        rt_thresholds = self.compute_baseline_thresholds(trial_df)

        def row_threshold(row):
            instruction = str(row.get('instruction')) if 'instruction' in row.index else None
            if instruction in rt_thresholds:
                return rt_thresholds[instruction]
            return rt_thresholds['__global__']

        trial_df['rt_threshold'] = trial_df.apply(row_threshold, axis=1)
        
        trial_df['cognitive_state'] = trial_df.apply(
            lambda row: self.compute_cognitive_state(
                row['accuracy'], 
                row['reaction_time'],
                row['rt_threshold']
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
        
        return trial_df, rt_thresholds
    
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
    
    def get_eeg_window(self, eeg_df, onset_time_ms, behavior_start_unix=None,
                       window_before_ms=500, window_after_ms=2000):
        """
        Extract EEG segment around a trial event.
        
        onset_time_ms: Stimulus onset time in ms from experiment start
        behavior_start_unix: Absolute Unix start time from n_back_responses.csv
        window_before_ms: EEG to include BEFORE stimulus (500ms)
        window_after_ms: EEG to include AFTER stimulus (2000ms = covers response)
        """
        if behavior_start_unix is None or pd.isna(behavior_start_unix):
            # Fallback for older trial data; absolute alignment is less reliable.
            behavior_start_unix = eeg_df.iloc[0, 0]

        onset_time_sec = onset_time_ms / 1000.0 + behavior_start_unix
        
        window_start = onset_time_sec - (window_before_ms / 1000.0)
        window_end = onset_time_sec + (window_after_ms / 1000.0)
        
        timestamps = eeg_df.iloc[:, 0].values
        mask = (timestamps >= window_start) & (timestamps <= window_end)
        
        return eeg_df[mask]

    def clean_eeg_channel(self, data):
        """
        Robustly clean one short EEG window before feature extraction.

        Muse data can contain large subject/device offsets and occasional contact
        artifacts. Removing the local median, clipping extreme MAD outliers, and
        bandpassing keeps features focused on within-window spectral shape rather
        than raw amplitude offsets that do not generalize across subjects.
        """
        data = np.asarray(data, dtype=np.float64)
        data = data[np.isfinite(data)]

        if len(data) < 32:
            return data

        median = np.median(data)
        mad = np.median(np.abs(data - median))
        robust_scale = 1.4826 * mad

        if robust_scale > 1e-6:
            data = np.clip(data, median - 6 * robust_scale, median + 6 * robust_scale)

        data = signal.detrend(data - np.median(data), type='linear')

        if len(data) > 3 * self.bandpass_sos.shape[0]:
            try:
                data = signal.sosfiltfilt(self.bandpass_sos, data)
            except ValueError:
                data = signal.sosfilt(self.bandpass_sos, data)

        return data

    def _band_power(self, freqs, psd, low, high):
        """Integrate PSD values inside one EEG frequency band."""
        band_mask = (freqs >= low) & (freqs <= high)
        if not np.any(band_mask):
            return 0.0

        # NumPy 2.0+ compatibility for trapz
        try:
            trapz_func = np.trapezoid
        except AttributeError:
            trapz_func = np.trapz

        return float(trapz_func(psd[band_mask], freqs[band_mask]))

    def extract_time_frequency_features(self, data):
        """
        Extract compact STFT features from a cleaned EEG window.

        The aggregate Welch features summarize the full 2.5 second window. These
        features keep timing information by estimating band power in three
        stimulus-locked periods: pre-stimulus, early post-stimulus, and late
        post-stimulus. For each band, include the three log powers plus
        early-minus-pre and late-minus-pre changes.
        """
        if len(data) < 64:
            return [0.0] * self.time_frequency_features_per_channel

        nperseg = min(128, len(data))
        noverlap = min(64, max(0, nperseg // 2))

        try:
            freqs, times, sxx = signal.spectrogram(
                data,
                fs=self.eeg_sampling_rate,
                window='hann',
                nperseg=nperseg,
                noverlap=noverlap,
                scaling='density',
                mode='psd',
            )
        except ValueError:
            return [0.0] * self.time_frequency_features_per_channel

        if sxx.ndim != 2 or sxx.shape[1] == 0:
            return [0.0] * self.time_frequency_features_per_channel

        # get_eeg_window starts each trial 500 ms before stimulus onset.
        segment_masks = [
            times < 0.5,
            (times >= 0.5) & (times < 1.5),
            times >= 1.5,
        ]

        if any(not np.any(mask) for mask in segment_masks):
            split_indices = np.array_split(np.arange(sxx.shape[1]), 3)
            segment_masks = []
            for indices in split_indices:
                mask = np.zeros(sxx.shape[1], dtype=bool)
                mask[indices] = True
                segment_masks.append(mask)

        features = []
        for low, high in self.eeg_bands.values():
            band_mask = (freqs >= low) & (freqs <= high)
            if not np.any(band_mask):
                segment_log_powers = [0.0, 0.0, 0.0]
            else:
                segment_log_powers = []
                for time_mask in segment_masks:
                    band_slice = sxx[np.ix_(band_mask, time_mask)]
                    power = float(np.mean(band_slice)) if band_slice.size > 0 else 0.0
                    segment_log_powers.append(float(np.log1p(power)))

            pre_power, early_power, late_power = segment_log_powers
            features.extend([
                pre_power,
                early_power,
                late_power,
                early_power - pre_power,
                late_power - pre_power,
            ])

        return features

    def extract_covariance_features(self, eeg_window, channels=None):
        """
        Extract log-Euclidean covariance features across EEG channels.

        This is a lightweight Riemannian-style representation: robustly cleaned
        channels are converted to a covariance matrix, normalized by trace, then
        mapped with a matrix logarithm. The upper triangle gives compact features
        that preserve channel relationships better than per-channel PSD alone.
        """
        if channels is None:
            channels = self.eeg_channels

        cleaned_channels = []
        min_len = None
        for channel in channels:
            if channel not in eeg_window.columns:
                return [0.0] * self.covariance_features

            data = self.clean_eeg_channel(eeg_window[channel].dropna().values)
            if len(data) < 32:
                return [0.0] * self.covariance_features

            min_len = len(data) if min_len is None else min(min_len, len(data))
            cleaned_channels.append(data)

        if min_len is None or min_len < 32:
            return [0.0] * self.covariance_features

        stacked = np.vstack([data[:min_len] for data in cleaned_channels])
        cov = np.cov(stacked)
        if cov.shape != (len(channels), len(channels)):
            return [0.0] * self.covariance_features

        trace = float(np.trace(cov))
        if trace <= 1e-12 or not np.isfinite(trace):
            return [0.0] * self.covariance_features

        cov = cov / trace
        shrinkage = 1e-3
        cov = (1 - shrinkage) * cov + shrinkage * np.eye(cov.shape[0]) / cov.shape[0]

        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.clip(eigvals, 1e-8, None)
            log_cov = eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T
        except np.linalg.LinAlgError:
            return [0.0] * self.covariance_features

        return [
            float(log_cov[left, right])
            for left, right in self.covariance_feature_pairs
        ]
    
    def extract_eeg_features(self, eeg_window, channels=None):
        """
        Extract aggregate and time-frequency features from a trial EEG window.
        
        Features per channel:
        - 5 band powers: delta, theta, alpha, beta, gamma
        - 4 statistical: mean, std, skewness, kurtosis
        - 3 Hjorth: activity, mobility, complexity
        - 1 ratio: theta/beta ratio
        - 25 STFT features: pre/early/late band power plus post-pre changes
        
        Total: 4 channels x 43 = 172 channel features
        Plus: 1 frontal alpha asymmetry and 10 covariance features = 183 total
        """
        if channels is None:
            channels = self.eeg_channels
        
        available_channels = [ch for ch in channels if ch in eeg_window.columns]
        
        if len(available_channels) == 0:
            return None
        
        all_channel_features = {}
        feature_vector = []
        
        for channel in channels:
            if channel not in eeg_window.columns:
                channel_features = [0.0] * self.features_per_channel
                feature_vector.extend(channel_features)
                all_channel_features[channel] = channel_features
                continue
            
            raw_data = eeg_window[channel].dropna().values
            data = self.clean_eeg_channel(raw_data)
            
            if len(data) < 32:
                channel_features = [0.0] * self.features_per_channel
                feature_vector.extend(channel_features)
                all_channel_features[channel] = channel_features
                continue
            
            # ---- FREQUENCY DOMAIN: Band Powers ----
            freqs, psd = signal.welch(
                data, 
                fs=self.eeg_sampling_rate,
                nperseg=min(128, len(data))
            )
            
            delta_raw = self._band_power(freqs, psd, *self.eeg_bands['delta'])
            theta_raw = self._band_power(freqs, psd, *self.eeg_bands['theta'])
            alpha_raw = self._band_power(freqs, psd, *self.eeg_bands['alpha'])
            beta_raw  = self._band_power(freqs, psd, *self.eeg_bands['beta'])
            gamma_raw = self._band_power(freqs, psd, *self.eeg_bands['gamma'])

            delta = np.log1p(delta_raw)
            theta = np.log1p(theta_raw)
            alpha = np.log1p(alpha_raw)
            beta  = np.log1p(beta_raw)
            gamma = np.log1p(gamma_raw)

            total_power = delta_raw + theta_raw + alpha_raw + beta_raw + gamma_raw
            if total_power > 0:
                rel_delta = delta_raw / total_power
                rel_theta = theta_raw / total_power
                rel_alpha = alpha_raw / total_power
                rel_beta = beta_raw / total_power
                rel_gamma = gamma_raw / total_power
            else:
                rel_delta = rel_theta = rel_alpha = rel_beta = rel_gamma = 0.0
            
            # ---- TIME DOMAIN: Statistical Features ----
            mean_val = float(np.mean(data))
            std_val  = float(np.log1p(np.std(data)))
            skew_val = float(skew(data)) if len(data) > 2 else 0.0
            kurt_val = float(kurtosis(data)) if len(data) > 3 else 0.0
            
            # ---- HJORTH PARAMETERS ----
            activity = float(np.log1p(np.var(data)))
            mobility = 0.0
            complexity = 0.0
            
            raw_activity = float(np.var(data))
            if raw_activity > 0:
                diff1 = np.diff(data)
                mobility = float(np.sqrt(np.var(diff1) / raw_activity))
                
                diff1_var = np.var(diff1)
                if mobility > 0 and diff1_var > 0:
                    diff2 = np.diff(diff1)
                    complexity = float(np.sqrt(np.var(diff2) / diff1_var) / mobility)
            
            # ---- THETA/BETA RATIO (attention biomarker) ----
            tbr = np.log1p(theta_raw / beta_raw) if beta_raw > 0 else 0.0
            
            channel_features = [
                delta, theta, alpha, beta, gamma,
                rel_delta, rel_theta, rel_alpha, rel_beta, rel_gamma,
                mean_val, std_val, skew_val, kurt_val,
                activity, mobility, complexity,
                tbr
            ]

            channel_features.extend(self.extract_time_frequency_features(data))
            
            all_channel_features[channel] = channel_features
            feature_vector.extend(channel_features)
        
        # ---- FRONTAL ALPHA ASYMMETRY (emotional/motivational state) ----
        if 'AF7' in all_channel_features and 'AF8' in all_channel_features:
            af7_alpha = all_channel_features['AF7'][2]
            af8_alpha = all_channel_features['AF8'][2]

            # Band powers above are log-scaled, so use a log-power difference.
            alpha_asymmetry = af8_alpha - af7_alpha
            
            feature_vector.append(alpha_asymmetry)
        else:
            feature_vector.append(0.0)

        feature_vector.extend(self.extract_covariance_features(eeg_window, channels=channels))
        
        return np.array(feature_vector, dtype=np.float32)
    
    # ==========================================
    # PART 3: PHYSIOLOGICAL SIGNALS (OPTIONAL)
    # ==========================================
    
    def load_physiological(self, subject_id, experiment="Experiment_1",
                           sensors=None):
        """Load Empatica wristband data (heart rate, skin conductance, etc.)."""
        if sensors is None:
            sensors = self.physio_sensors

        physio_data = {}
        
        for side in self.physio_sides:
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

    def _physio_timestamps_and_values(self, df):
        """
        Convert one Empatica file to absolute timestamps and value columns.

        Empatica CSVs store start_time_unix and sampling_rate columns rather
        than one timestamp per sample. IBI is event-based and uses IBI_time
        offsets instead of a sampling rate.
        """
        if df is None or len(df) == 0:
            return None, None

        if {'IBI_time', 'IBI_intervals', 'start_time_unix'}.issubset(df.columns):
            start_time = self._to_float(df['start_time_unix'].iloc[0], default=np.nan)
            if np.isnan(start_time):
                return None, None
            offsets = pd.to_numeric(df['IBI_time'], errors='coerce').to_numpy(dtype=np.float64)
            values = pd.to_numeric(df['IBI_intervals'], errors='coerce').to_numpy(dtype=np.float64)
            timestamps = start_time + offsets
            return timestamps, values.reshape(-1, 1)

        if 'start_time_unix' not in df.columns or 'sampling_rate' not in df.columns:
            return None, None

        start_time = self._to_float(df['start_time_unix'].iloc[0], default=np.nan)
        sampling_rate = self._to_float(df['sampling_rate'].iloc[0], default=np.nan)
        if np.isnan(start_time) or np.isnan(sampling_rate) or sampling_rate <= 0:
            return None, None

        value_cols = [
            col for col in df.columns
            if col not in {'start_time_unix', 'sampling_rate'}
        ]
        if not value_cols:
            return None, None

        values = df[value_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
        timestamps = start_time + (np.arange(len(df), dtype=np.float64) / sampling_rate)

        return timestamps, values
    
    def extract_physio_features(self, physio_data, eeg_window_start, eeg_window_end):
        """Extract mean/std physiological features during an EEG-aligned window."""
        features = []
        
        for side in self.physio_sides:
            for sensor in self.physio_sensors:
                name = f"{side}_{sensor}"
                df = physio_data.get(name)
                expected_value_cols = self.physio_value_counts.get(sensor, 1)
                zero_features = [0.0, 0.0] * expected_value_cols

                if df is None:
                    features.extend(zero_features)
                    continue

                try:
                    timestamps, values = self._physio_timestamps_and_values(df)
                    if timestamps is None or values is None:
                        features.extend(zero_features)
                        continue

                    mask = (timestamps >= eeg_window_start) & (timestamps <= eeg_window_end)
                    window_values = values[mask, :]

                    if len(window_values) > 0:
                        for col_idx in range(expected_value_cols):
                            if col_idx >= window_values.shape[1]:
                                features.extend([0.0, 0.0])
                                continue

                            col_values = window_values[:, col_idx]
                            col_values = col_values[np.isfinite(col_values)]
                            if len(col_values) == 0:
                                features.extend([0.0, 0.0])
                            else:
                                features.extend([
                                    float(np.mean(col_values)),
                                    float(np.std(col_values)),
                                ])
                    else:
                        features.extend(zero_features)
                except Exception:
                    features.extend(zero_features)
        
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
                behavior_start_unix=trial.get('behavior_start_unix', None),
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
        
        used_trial_df = trial_df.loc[trial_indices].reset_index(drop=True)

        return X, y, used_trial_df


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
        Path("Data/processed").mkdir(parents=True, exist_ok=True)
        np.save("Data/processed/X_features.npy", X_combined)
        np.save("Data/processed/y_labels.npy", y_combined)
        print(f"\nSaved to Data/processed/")
    else:
        print("\nNo subjects processed successfully.")
