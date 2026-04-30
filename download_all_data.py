"""
Download ALL subject data from PhysioNet brain-wearable-monitoring dataset.
Run from project root: python download_all_data.py
"""
import urllib.request
import os
from pathlib import Path
import concurrent.futures
import time

BASE_URL = "https://physionet.org/files/brain-wearable-monitoring/1.0.0"

# Subjects in each experiment (based on the paper's supplementary materials)
EXPERIMENT_1_SUBJECTS = [f"A{i}" for i in range(1, 11)]  # A1 through A10
EXPERIMENT_2_SUBJECTS = [f"B{i}" for i in range(1, 11)]  # B1 through B10

# All possible CSV files a subject folder might contain
POSSIBLE_FILES = [
    "EEG_recording.csv",
    "n_back_responses.csv",
    "Left_EDA.csv", "Right_EDA.csv",
    "Left_HR.csv", "Right_HR.csv",
    "Left_TEMP.csv", "Right_TEMP.csv",
    "Left_BVP.csv", "Right_BVP.csv",
    "Left_ACC.csv", "Right_ACC.csv",
    "Left_IBI.csv", "Right_IBI.csv",
    "Left_tags.csv", "Right_tags.csv",
    "tags.csv",
]

def download_file(url, output_path):
    """Download a single file with retry logic."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, output_path)
            return True, output_path
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return False, str(e)

def download_subject(experiment, subject_id, output_dir, skip_existing=True):
    """
    Download all files for one subject.
    Returns: (downloaded_count, skipped_count, failed_count)
    """
    subject_path = Path(output_dir) / experiment / subject_id
    subject_path.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    print(f"\n📂 {experiment}/{subject_id}:")
    
    for filename in POSSIBLE_FILES:
        output_file = subject_path / filename
        
        if skip_existing and output_file.exists() and output_file.stat().st_size > 100:
            print(f"   ⏭️  {filename} (already exists)")
            skipped += 1
            continue
        
        url = f"{BASE_URL}/{experiment}/{subject_id}/{filename}"
        
        try:
            success, result = download_file(url, output_file)
            if success:
                size_kb = output_file.stat().st_size / 1024
                print(f"   ✅ {filename} ({size_kb:.0f} KB)")
                downloaded += 1
            else:
                # 404 means file doesn't exist for this subject - that's ok
                if "404" in str(result) or "403" in str(result):
                    print(f"   ⚠️  {filename} - not found (may not exist)")
                else:
                    print(f"   ❌ {filename} - {result}")
                failed += 1
        except Exception as e:
            print(f"   ❌ {filename} - Error: {e}")
            failed += 1
    
    return downloaded, skipped, failed

def main():
    output_dir = Path("Data/raw")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0
    
    print("=" * 60)
    print("DOWNLOADING BRAIN-WEARABLE-MONITORING DATASET")
    print("=" * 60)
    print(f"Destination: {output_dir.absolute()}")
    print(f"Expected total subjects: {len(EXPERIMENT_1_SUBJECTS) + len(EXPERIMENT_2_SUBJECTS)}")
    print()
    
    # Download Experiment 1
    print("\n🧪 EXPERIMENT 1 (Music)")
    print("-" * 40)
    for subject in EXPERIMENT_1_SUBJECTS:
        d, s, f = download_subject("Experiment_1", subject, output_dir)
        total_downloaded += d
        total_skipped += s
        total_failed += f
    
    # Download Experiment 2
    print("\n🧪 EXPERIMENT 2 (Coffee & Perfume)")
    print("-" * 40)
    for subject in EXPERIMENT_2_SUBJECTS:
        d, s, f = download_subject("Experiment_2", subject, output_dir)
        total_downloaded += d
        total_skipped += s
        total_failed += f
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Total files downloaded: {total_downloaded}")
    print(f"Total files skipped (already exist): {total_skipped}")
    print(f"Total files failed (may not exist): {total_failed}")
    print(f"\nData is in: {output_dir.absolute()}")

if __name__ == "__main__":
    main()