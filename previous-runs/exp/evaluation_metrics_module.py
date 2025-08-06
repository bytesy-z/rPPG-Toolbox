import pickle
import numpy as np
import torch
import re
from collections import defaultdict
import sys
from scipy.stats import pearsonr
import os
sys.path.append('/home/zik/Research/ZJU/rPPG/rppg-work/Phase1/rPPG-Toolbox')
from evaluation.post_process import calculate_metric_per_video

def _reform_data_from_dict(data, flatten=True):
    """Helper func for calculate metrics: reformat predictions and labels from dicts."""
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    
    # Check if we have torch tensors or numpy arrays
    if len(sort_data) > 0:
        if isinstance(sort_data[0], torch.Tensor):
            # Original behavior for tensors
            sort_data = torch.cat(sort_data, dim=0)
            if flatten:
                sort_data = np.reshape(sort_data.cpu(), (-1))
            else:
                sort_data = np.array(sort_data.cpu())
        else:
            # Handle numpy arrays from unsupervised methods
            sort_data = np.concatenate(sort_data, axis=0)
            if flatten:
                sort_data = sort_data.flatten()
    else:
        # Empty data, return empty array
        sort_data = np.array([])

    return sort_data

def extract_method_info(filename):
    """Extract method name and watermark status from filename."""
    method_name = "Unknown"
    is_watermarked = False
    
    # Check for watermark status
    if "WATERMARKED" in filename.upper() or "WATERMARKING" in filename.upper():
        is_watermarked = True
    
    # Extract method name from filename
    if "POS_" in filename:
        method_name = "POS"
    elif "CHROM_" in filename:
        method_name = "CHROM"
    elif "ICA_" in filename:
        method_name = "ICA"
    elif "GREEN_" in filename:
        method_name = "GREEN"
    elif "LGI_" in filename:
        method_name = "LGI"
    elif "PBV_" in filename:
        method_name = "PBV"
    elif "OMIT_" in filename:
        method_name = "OMIT"
    # Supervised/neural methods
    elif "FactorizePhys" in filename:
        method_name = "FactorizePhys"
    elif "PhysNet" in filename:
        method_name = "PhysNet"
    elif "DeepPhys" in filename:
        method_name = "DeepPhys"
    elif "EfficientPhys" in filename:
        method_name = "EfficientPhys"
    elif "TSCAN" in filename:
        method_name = "TSCAN"
    elif "RhythmFormer" in filename:
        method_name = "RhythmFormer"
    elif "MetaPhys" in filename:
        method_name = "MetaPhys"
    elif "ViTPhys" in filename:
        method_name = "ViTPhys"
    elif "STVENet" in filename:
        method_name = "STVENet"
    elif "TSCANPlus" in filename:
        method_name = "TSCANPlus"
    elif "WaveletPhys" in filename:
        method_name = "WaveletPhys"
    elif "MambaPhys" in filename:
        method_name = "MambaPhys"
    elif "PhysFormer" in filename:
        method_name = "PhysFormer"
    elif "MTTS_CAN" in filename:
        method_name = "MTTS_CAN"
    elif "CVDPhys" in filename:
        method_name = "CVDPhys"
    elif "TransPhys" in filename:
        method_name = "TransPhys"
    elif "PPGNet" in filename:
        method_name = "PPGNet"
    elif "STMapNet" in filename:
        method_name = "STMapNet"
    elif "HRNet" in filename:
        method_name = "HRNet"
    elif "ResPhys" in filename:
        method_name = "ResPhys"
    elif "MetaPhysFormer" in filename:
        method_name = "MetaPhysFormer"
    elif "EfficientPPG" in filename:
        method_name = "EfficientPPG"
    elif "PhysNetPlus" in filename:
        method_name = "PhysNetPlus"
    elif "PhysNetV2" in filename:
        method_name = "PhysNetV2"
    elif "PhysNetV3" in filename:
        method_name = "PhysNetV3"
    elif "PhysNetV4" in filename:
        method_name = "PhysNetV4"
    elif "PhysNetV5" in filename:
        method_name = "PhysNetV5"
    elif "PhysNetV6" in filename:
        method_name = "PhysNetV6"
    
    watermark_status = "WATERMARKED" if is_watermarked else "ORIGINAL"
    return method_name, watermark_status

def evaluate_pickle_file(data_out_path):
    """
    Evaluate a single pickle file and return structured results.
    
    Returns:
        dict: Contains method info and all metrics, or None if evaluation failed
    """
    try:
        # Extract method and watermark info from file path
        filename = os.path.basename(data_out_path)
        method_name, watermark_status = extract_method_info(filename)

        with open(data_out_path, 'rb') as f:
            data = pickle.load(f)

        trial_list = list(data['predictions'].keys())
        
        if not trial_list:
            print(f"Warning: No trials found in {filename}")
            return None

        # Video Frame Rate (ensure >0)
        raw_fs = data.get('fs', None)
        if raw_fs is None or raw_fs <= 0:
            fs = 30
        else:
            fs = raw_fs

        # Label type (ensure non-empty)
        lt = data.get('label_type', None)
        if not lt:
            label_type = 'Raw'
        else:
            label_type = lt

        diff_flag = (label_type == 'DiffNormalized')

        # Initialize metric accumulators
        subject_maes = defaultdict(list)
        subject_rmses = defaultdict(list)
        subject_mapes = defaultdict(list)
        subject_snrs = defaultdict(list)
        task_maes = defaultdict(list)
        task_rmses = defaultdict(list)
        task_mapes = defaultdict(list)
        task_snrs = defaultdict(list)
        all_maes = []
        all_rmses = []
        all_mapes = []
        all_snrs = []

        # For Pearson correlation - collect all HR values across trials
        all_pred_hrs = []
        all_gt_hrs = []

        for trial in trial_list:
            # Convert trial key to string for regex
            trial_str = str(trial)
            
            # Parse subject and task for PURE dataset format
            if trial_str.isdigit():
                trial_int = int(trial_str)
                
                if trial_int >= 1001:
                    # Subject 10: format is 10TT (1001, 1002, etc.)
                    subject_id = "10"
                    task_id = str(trial_int - 1000)  # 1001 -> 1, 1002 -> 2, etc.
                elif trial_int >= 101:
                    # Subjects 1-9: format is STT (101, 102, ..., 901, 902, etc.)
                    subject_id = str(trial_int // 100)  # 101 -> 1, 201 -> 2, etc.
                    task_id = str(trial_int % 100)      # 101 -> 1, 102 -> 2, etc.
                else:
                    # Fallback for edge cases
                    subject_id = "1"
                    task_id = str(trial_int)
            elif re.match(r'^\d{2}-\d{2}$', trial_str):
                # Handle format like "10-01" -> subject 10, task 1
                parts = trial_str.split('-')
                subject_id = str(int(parts[0]))  # Remove leading zeros
                task_id = str(int(parts[1]))     # Remove leading zeros
            else:
                # Try regex pattern for other formats
                match = re.search(r'(S|subject)?(\d+)[_\-\.]?(T|task)?(\d+)', trial_str, re.IGNORECASE)
                if match:
                    subject_id = str(int(match.group(2)))  # Remove leading zeros
                    task_id = str(int(match.group(4))) if match.group(4) else "1"  # Remove leading zeros
                else:
                    # Fallback: use full trial name
                    subject_id = trial_str
                    task_id = trial_str

            # Get prediction and label
            pred = np.array(_reform_data_from_dict(data['predictions'][trial]))
            label = np.array(_reform_data_from_dict(data['labels'][trial]))

            # Use windowing like official metrics.py
            video_frame_size = pred.shape[0]
            window_frame_size = video_frame_size  # Use full video
            
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

                # Collect for overall Pearson correlation
                all_pred_hrs.append(pred_hr)
                all_gt_hrs.append(gt_hr)

                # Compute metrics on heart rates (BPM)
                mae = abs(pred_hr - gt_hr)
                mape = abs((pred_hr - gt_hr) / gt_hr) * 100 if gt_hr != 0 else np.nan
                hr_error = pred_hr - gt_hr
                
                trial_maes.append(mae)
                trial_snrs.append(SNR)
                trial_mapes.append(mape)
                trial_hr_errors.append(hr_error)
            
            # Average metrics across windows for this trial
            mae = np.mean(trial_maes) if trial_maes else np.nan
            SNR = np.mean(trial_snrs) if trial_snrs else np.nan
            mape = np.mean(trial_mapes) if trial_mapes else np.nan
            hr_error = np.mean(trial_hr_errors) if trial_hr_errors else np.nan

            subject_maes[subject_id].append(mae)
            subject_rmses[subject_id].append(hr_error)
            subject_mapes[subject_id].append(mape)
            subject_snrs[subject_id].append(SNR)
            task_maes[task_id].append(mae)
            task_rmses[task_id].append(hr_error)
            task_mapes[task_id].append(mape)
            task_snrs[task_id].append(SNR)
            
            # Store for overall statistics
            all_maes.append(mae)
            all_rmses.append(hr_error)
            all_mapes.append(mape)
            all_snrs.append(SNR)

        # Calculate RMSE properly and remove NaN values for statistics
        all_maes_clean = [x for x in all_maes if not np.isnan(x)]
        all_rmses_clean = [x for x in all_rmses if not np.isnan(x)]
        all_mapes_clean = [x for x in all_mapes if not np.isnan(x)]
        all_snrs_clean = [x for x in all_snrs if not np.isnan(x)]

        # Calculate Pearson correlation coefficient for overall predictions vs ground truth
        try:
            if len(all_pred_hrs) > 1 and len(all_gt_hrs) > 1:
                pearson_corr, pearson_p = pearsonr(all_pred_hrs, all_gt_hrs)
            else:
                pearson_corr = np.nan
        except Exception as e:
            pearson_corr = np.nan

        # Calculate RMSE as sqrt of mean squared errors
        rmse_overall = np.sqrt(np.mean(np.square(all_rmses_clean))) if all_rmses_clean else np.nan
        rmse_std_error = np.sqrt(np.std(np.square(all_rmses_clean)) / np.sqrt(len(all_rmses_clean))) if len(all_rmses_clean) > 1 else 0

        # Prepare results structure
        results = {
            'method_name': method_name,
            'watermark_status': watermark_status,
            'dataset': 'PURE',
            'num_trials': len(trial_list),
            'num_valid_trials': len(all_maes_clean),
            'overall_metrics': {
                'MAE': np.mean(all_maes_clean) if all_maes_clean else np.nan,
                'MAE_std': np.std(all_maes_clean) / np.sqrt(len(all_maes_clean)) if len(all_maes_clean) > 1 else 0,
                'RMSE': rmse_overall,
                'RMSE_std': rmse_std_error,
                'MAPE': np.mean(all_mapes_clean) if all_mapes_clean else np.nan,
                'MAPE_std': np.std(all_mapes_clean) / np.sqrt(len(all_mapes_clean)) if len(all_mapes_clean) > 1 else 0,
                'SNR': np.mean(all_snrs_clean) if all_snrs_clean else np.nan,
                'SNR_std': np.std(all_snrs_clean) / np.sqrt(len(all_snrs_clean)) if len(all_snrs_clean) > 1 else 0,
                'Pearson': pearson_corr
            },
            'subject_metrics': {},
            'task_metrics': {}
        }

        # Process per-subject metrics
        for subject in sorted(subject_maes.keys()):
            maes = [x for x in subject_maes[subject] if not np.isnan(x)]
            rmse_errors = [x for x in subject_rmses[subject] if not np.isnan(x)]
            mapes = [x for x in subject_mapes[subject] if not np.isnan(x)]
            snrs = [x for x in subject_snrs[subject] if not np.isnan(x)]
            
            rmse_subj = np.sqrt(np.mean(np.square(rmse_errors))) if rmse_errors else np.nan
            rmse_std_err = np.sqrt(np.std(np.square(rmse_errors)) / np.sqrt(len(rmse_errors))) if len(rmse_errors) > 1 else 0
            
            results['subject_metrics'][subject] = {
                'MAE': np.mean(maes) if maes else np.nan,
                'MAE_std': np.std(maes) / np.sqrt(len(maes)) if len(maes) > 1 else 0,
                'RMSE': rmse_subj,
                'RMSE_std': rmse_std_err,
                'MAPE': np.mean(mapes) if mapes else np.nan,
                'MAPE_std': np.std(mapes) / np.sqrt(len(mapes)) if len(mapes) > 1 else 0,
                'SNR': np.mean(snrs) if snrs else np.nan,
                'SNR_std': np.std(snrs) / np.sqrt(len(snrs)) if len(snrs) > 1 else 0,
                'N': len(maes)
            }

        # Process per-task metrics
        for task in sorted(task_maes.keys()):
            maes = [x for x in task_maes[task] if not np.isnan(x)]
            rmse_errors = [x for x in task_rmses[task] if not np.isnan(x)]
            mapes = [x for x in task_mapes[task] if not np.isnan(x)]
            snrs = [x for x in task_snrs[task] if not np.isnan(x)]
            
            rmse_task = np.sqrt(np.mean(np.square(rmse_errors))) if rmse_errors else np.nan
            rmse_std_err = np.sqrt(np.std(np.square(rmse_errors)) / np.sqrt(len(rmse_errors))) if len(rmse_errors) > 1 else 0
            
            results['task_metrics'][task] = {
                'MAE': np.mean(maes) if maes else np.nan,
                'MAE_std': np.std(maes) / np.sqrt(len(maes)) if len(maes) > 1 else 0,
                'RMSE': rmse_task,
                'RMSE_std': rmse_std_err,
                'MAPE': np.mean(mapes) if mapes else np.nan,
                'MAPE_std': np.std(mapes) / np.sqrt(len(mapes)) if len(mapes) > 1 else 0,
                'SNR': np.mean(snrs) if snrs else np.nan,
                'SNR_std': np.std(snrs) / np.sqrt(len(snrs)) if len(snrs) > 1 else 0,
                'N': len(maes)
            }

        return results

    except Exception as e:
        print(f"Error processing {data_out_path}: {str(e)}")
        return None
