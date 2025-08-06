import pickle
import numpy as np
import torch
import re
from collections import defaultdict
import sys
sys.path.append('/home/zik/Research/ZJU/rPPG/rppg-work/Phase1/rPPG-Toolbox')
from evaluation.post_process import calculate_metric_per_video

# Helper functions (from your notebook)
def _reform_data_from_dict(data, flatten=True):
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)
    if flatten:
        sort_data = np.reshape(sort_data.cpu(), (-1))
    else:
        sort_data = np.array(sort_data.cpu())
    return sort_data

# ---- Main script ----

data_out_path = "/home/zik/Research/ZJU/rPPG/rppg-work/Phase1/rPPG-Toolbox/runs/exp/PURE_SizeW72_SizeH72_ClipLength180_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse/saved_test_outputs/UBFC-rPPG_TSCAN_PURE_outputs.pickle"  # Change to your actual output file

with open(data_out_path, 'rb') as f:
    data = pickle.load(f)

trial_list = list(data['predictions'].keys())
fs = data['fs']
label_type = data['label_type']
diff_flag = (label_type == 'DiffNormalized')

subject_maes = defaultdict(list)
subject_rmses = defaultdict(list)
subject_mapes = defaultdict(list)
subject_snrs = defaultdict(list)
subject_pearsons = defaultdict(list)
task_maes = defaultdict(list)
task_rmses = defaultdict(list)
task_mapes = defaultdict(list)
task_snrs = defaultdict(list)
task_pearsons = defaultdict(list)

# Store all metrics for overall statistics
all_maes = []
all_rmses = []
all_mapes = []
all_snrs = []
all_pearsons = []

for trial in trial_list:
    # Try to extract subject and task from trial name using regex
    # Example: "S01_T02_trial03" or "subject01_task02"
    match = re.search(r'(S|subject)?(\d+)[_\-\.]?(T|task)?(\d+)', trial, re.IGNORECASE)
    if match:
        subject_id = match.group(2)
        task_id = match.group(4)
    else:
        # fallback: use full trial name
        subject_id = trial
        task_id = trial

    # Get prediction and label
    pred = np.array(_reform_data_from_dict(data['predictions'][trial]))
    label = np.array(_reform_data_from_dict(data['labels'][trial]))

    # Use windowing like official metrics.py (this should improve SNR accuracy)
    video_frame_size = pred.shape[0]
    # For PURE dataset, evaluation window is typically whole video, but let's check
    # You can experiment with different window sizes (e.g., 10*fs for 10 second windows)
    USE_SMALLER_WINDOW = False  # Set to True to test windowing effect
    WINDOW_SIZE_SECONDS = 10    # seconds
    
    if USE_SMALLER_WINDOW:
        window_frame_size = WINDOW_SIZE_SECONDS * fs
        if window_frame_size > video_frame_size:
            window_frame_size = video_frame_size
    else:
        window_frame_size = video_frame_size
    
    # Process in windows like official implementation
    trial_maes = []
    trial_snrs = []
    trial_mapes = []
    trial_hr_errors = []
    
    for i in range(0, len(pred), window_frame_size):
        pred_window = pred[i:i+window_frame_size]
        label_window = label[i:i+window_frame_size]
        
        if len(pred_window) < 9:  # Skip too short windows
            continue
            
        # Use official toolbox function to calculate metrics
        gt_hr, pred_hr, SNR, macc = calculate_metric_per_video(
            pred_window, label_window, diff_flag=diff_flag, fs=fs, hr_method='FFT')

        # Compute metrics on heart rates (BPM)
        mae = abs(pred_hr - gt_hr)
        mape = abs((pred_hr - gt_hr) / gt_hr) * 100 if gt_hr != 0 else np.nan
        hr_error = pred_hr - gt_hr
        
        trial_maes.append(mae)
        trial_snrs.append(SNR)
        trial_mapes.append(mape)
        trial_hr_errors.append(hr_error)
    
    # Average metrics across windows for this trial (matching official implementation)
    mae = np.mean(trial_maes) if trial_maes else np.nan
    SNR = np.mean(trial_snrs) if trial_snrs else np.nan
    mape = np.mean(trial_mapes) if trial_mapes else np.nan
    hr_error = np.mean(trial_hr_errors) if trial_hr_errors else np.nan
    
    # Pearson correlation - need multiple samples for correlation, so skip for individual trials
    pearson = np.nan  # Individual trial correlation doesn't make sense

    subject_maes[subject_id].append(mae)
    subject_rmses[subject_id].append(hr_error)  # Store error, not squared error
    subject_mapes[subject_id].append(mape)
    subject_snrs[subject_id].append(SNR)
    subject_pearsons[subject_id].append(pearson)
    task_maes[task_id].append(mae)
    task_rmses[task_id].append(hr_error)  # Store error, not squared error
    task_mapes[task_id].append(mape)
    task_snrs[task_id].append(SNR)
    task_pearsons[task_id].append(pearson)
    
    # Store for overall statistics
    all_maes.append(mae)
    all_rmses.append(hr_error)  # Store error, not squared error
    all_mapes.append(mape)
    all_snrs.append(SNR)
    all_pearsons.append(pearson)

# Calculate RMSE properly and remove NaN values for statistics
all_maes_clean = [x for x in all_maes if not np.isnan(x)]
all_rmses_clean = [x for x in all_rmses if not np.isnan(x)]
all_mapes_clean = [x for x in all_mapes if not np.isnan(x)]
all_snrs_clean = [x for x in all_snrs if not np.isnan(x)]

# Calculate RMSE as sqrt of mean squared errors (matching official implementation)
rmse_overall = np.sqrt(np.mean(np.square(all_rmses_clean)))
rmse_std_error = np.sqrt(np.std(np.square(all_rmses_clean)) / np.sqrt(len(all_rmses_clean)))

print("=== OVERALL STATISTICS ===")
print(f"MAE (BPM): {np.mean(all_maes_clean):.4f} +/- {np.std(all_maes_clean) / np.sqrt(len(all_maes_clean)):.4f}")
print(f"RMSE (BPM): {rmse_overall:.4f} +/- {rmse_std_error:.4f}")
print(f"MAPE (%): {np.mean(all_mapes_clean):.4f} +/- {np.std(all_mapes_clean) / np.sqrt(len(all_mapes_clean)):.4f}")
print(f"SNR (dB): {np.mean(all_snrs_clean):.2f} +/- {np.std(all_snrs_clean) / np.sqrt(len(all_snrs_clean)):.2f}")
print(f"N={len(all_maes_clean)}")
print()

print("=== PER-SUBJECT METRICS ===")
for subject in sorted(subject_maes.keys()):
    maes = [x for x in subject_maes[subject] if not np.isnan(x)]
    rmse_errors = [x for x in subject_rmses[subject] if not np.isnan(x)]
    mapes = [x for x in subject_mapes[subject] if not np.isnan(x)]
    snrs = [x for x in subject_snrs[subject] if not np.isnan(x)]
    
    # Calculate RMSE properly for this subject
    rmse_subj = np.sqrt(np.mean(np.square(rmse_errors))) if rmse_errors else np.nan
    rmse_std_err = np.sqrt(np.std(np.square(rmse_errors)) / np.sqrt(len(rmse_errors))) if len(rmse_errors) > 1 else 0
    
    print(f"Subject {subject}:")
    print(f"  MAE: {np.mean(maes):.4f} +/- {np.std(maes) / np.sqrt(len(maes)) if len(maes) > 1 else 0:.4f} BPM")
    print(f"  RMSE: {rmse_subj:.4f} +/- {rmse_std_err:.4f} BPM") 
    print(f"  MAPE: {np.mean(mapes):.4f} +/- {np.std(mapes) / np.sqrt(len(mapes)) if len(mapes) > 1 else 0:.4f} %")
    print(f"  SNR: {np.mean(snrs):.2f} +/- {np.std(snrs) / np.sqrt(len(snrs)) if len(snrs) > 1 else 0:.2f} dB")
    print(f"  (N={len(maes)})")
    print()

print("=== PER-TASK METRICS ===")
for task in sorted(task_maes.keys()):
    maes = [x for x in task_maes[task] if not np.isnan(x)]
    rmse_errors = [x for x in task_rmses[task] if not np.isnan(x)]
    mapes = [x for x in task_mapes[task] if not np.isnan(x)]
    snrs = [x for x in task_snrs[task] if not np.isnan(x)]
    
    # Calculate RMSE properly for this task
    rmse_task = np.sqrt(np.mean(np.square(rmse_errors))) if rmse_errors else np.nan
    rmse_std_err = np.sqrt(np.std(np.square(rmse_errors)) / np.sqrt(len(rmse_errors))) if len(rmse_errors) > 1 else 0
    
    print(f"Task {task}:")
    print(f"  MAE: {np.mean(maes):.4f} +/- {np.std(maes) / np.sqrt(len(maes)) if len(maes) > 1 else 0:.4f} BPM")
    print(f"  RMSE: {rmse_task:.4f} +/- {rmse_std_err:.4f} BPM")
    print(f"  MAPE: {np.mean(mapes):.4f} +/- {np.std(mapes) / np.sqrt(len(mapes)) if len(mapes) > 1 else 0:.4f} %")
    print(f"  SNR: {np.mean(snrs):.2f} +/- {np.std(snrs) / np.sqrt(len(snrs)) if len(snrs) > 1 else 0:.2f} dB")
    print(f"  (N={len(maes)})")
    print()