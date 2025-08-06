#!/usr/bin/env python3
"""
Comprehensive rPPG Method Evaluation Script

This script evaluates all available rPPG methods (both original and watermarked versions)
and saves the results in a structured CSV format for easy analysis and plotting.

The evaluation order is:
1. Unsupervised methods: POS, CHROM, GREEN, ICA, LGI, PBV, OMIT (original then watermarked)
2. Supervised methods: DeepPhys, TSCAN, FactorizePhys, RhythmFormer, etc. (original then watermarked)

Output: comprehensive_evaluation_results.csv
"""

import os
import glob
import sys
import pandas as pd
from pathlib import Path
import traceback
from datetime import datetime
from tqdm import tqdm

# Add the current directory to path for our custom module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluation_metrics_module import evaluate_pickle_file, extract_method_info

def find_all_pickle_files(base_dir):
    """Find all pickle files in the experiment directories."""
    pickle_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.pickle'):
                pickle_files.append(os.path.join(root, file))
    return pickle_files

def categorize_methods(pickle_files):
    """Categorize methods into unsupervised and supervised groups."""
    
    # Define method categories and their order
    unsupervised_methods = ['POS', 'CHROM', 'GREEN', 'ICA', 'LGI', 'PBV', 'OMIT']
    supervised_methods = ['DeepPhys', 'TSCAN', 'FactorizePhys', 'RhythmFormer', 'MetaPhys', 
                         'ViTPhys', 'STVENet', 'TSCANPlus', 'WaveletPhys', 'MambaPhys', 
                         'PhysFormer', 'MTTS_CAN', 'CVDPhys', 'TransPhys', 'PPGNet', 
                         'STMapNet', 'HRNet', 'ResPhys', 'MetaPhysFormer', 'EfficientPPG',
                         'PhysNet', 'PhysNetPlus', 'PhysNetV2', 'PhysNetV3', 'PhysNetV4',
                         'PhysNetV5', 'PhysNetV6', 'EfficientPhys']
    
    # Categorize files
    categorized = {
        'unsupervised': {method: {'original': None, 'watermarked': None} for method in unsupervised_methods},
        'supervised': {method: {'original': None, 'watermarked': None} for method in supervised_methods}
    }
    
    for file_path in pickle_files:
        filename = os.path.basename(file_path)
        method_name, watermark_status = extract_method_info(filename)
        
        # Determine category
        if method_name in unsupervised_methods:
            category = 'unsupervised'
        elif method_name in supervised_methods:
            category = 'supervised'
        else:
            # Add unknown methods to supervised category
            if method_name not in categorized['supervised']:
                categorized['supervised'][method_name] = {'original': None, 'watermarked': None}
            category = 'supervised'
        
        # Store file path
        watermark_key = 'watermarked' if watermark_status == 'WATERMARKED' else 'original'
        categorized[category][method_name][watermark_key] = file_path
    
    return categorized

def create_evaluation_order(categorized_methods):
    """Create the evaluation order: original first, then watermarked for each method."""
    evaluation_order = []
    
    # Process unsupervised methods first
    for method in categorized_methods['unsupervised']:
        if categorized_methods['unsupervised'][method]['original']:
            evaluation_order.append(categorized_methods['unsupervised'][method]['original'])
        if categorized_methods['unsupervised'][method]['watermarked']:
            evaluation_order.append(categorized_methods['unsupervised'][method]['watermarked'])
    
    # Process supervised methods
    for method in categorized_methods['supervised']:
        if categorized_methods['supervised'][method]['original']:
            evaluation_order.append(categorized_methods['supervised'][method]['original'])
        if categorized_methods['supervised'][method]['watermarked']:
            evaluation_order.append(categorized_methods['supervised'][method]['watermarked'])
    
    return evaluation_order

def results_to_dataframes(all_results):
    """Convert results to pandas DataFrames for CSV export."""
    
    # Overall metrics DataFrame
    overall_data = []
    subject_data = []
    task_data = []
    
    for result in all_results:
        if result is None:
            continue
            
        method = result['method_name']
        watermark = result['watermark_status']
        
        # Overall metrics
        overall_metrics = result['overall_metrics']
        overall_data.append({
            'Method': method,
            'Watermark_Status': watermark,
            'Dataset': result['dataset'],
            'Num_Trials': result['num_trials'],
            'Num_Valid_Trials': result['num_valid_trials'],
            'MAE_Mean': overall_metrics['MAE'],
            'MAE_Std': overall_metrics['MAE_std'],
            'RMSE_Mean': overall_metrics['RMSE'],
            'RMSE_Std': overall_metrics['RMSE_std'],
            'MAPE_Mean': overall_metrics['MAPE'],
            'MAPE_Std': overall_metrics['MAPE_std'],
            'SNR_Mean': overall_metrics['SNR'],
            'SNR_Std': overall_metrics['SNR_std'],
            'Pearson_Correlation': overall_metrics['Pearson']
        })
        
        # Subject metrics
        for subject_id, metrics in result['subject_metrics'].items():
            subject_data.append({
                'Method': method,
                'Watermark_Status': watermark,
                'Subject_ID': subject_id,
                'MAE_Mean': metrics['MAE'],
                'MAE_Std': metrics['MAE_std'],
                'RMSE_Mean': metrics['RMSE'],
                'RMSE_Std': metrics['RMSE_std'],
                'MAPE_Mean': metrics['MAPE'],
                'MAPE_Std': metrics['MAPE_std'],
                'SNR_Mean': metrics['SNR'],
                'SNR_Std': metrics['SNR_std'],
                'N_Trials': metrics['N']
            })
        
        # Task metrics
        for task_id, metrics in result['task_metrics'].items():
            task_data.append({
                'Method': method,
                'Watermark_Status': watermark,
                'Task_ID': task_id,
                'MAE_Mean': metrics['MAE'],
                'MAE_Std': metrics['MAE_std'],
                'RMSE_Mean': metrics['RMSE'],
                'RMSE_Std': metrics['RMSE_std'],
                'MAPE_Mean': metrics['MAPE'],
                'MAPE_Std': metrics['MAPE_std'],
                'SNR_Mean': metrics['SNR'],
                'SNR_Std': metrics['SNR_std'],
                'N_Trials': metrics['N']
            })
    
    overall_df = pd.DataFrame(overall_data)
    subject_df = pd.DataFrame(subject_data)
    task_df = pd.DataFrame(task_data)
    
    return overall_df, subject_df, task_df

def print_summary_table(overall_df):
    """Print a nice summary table to console."""
    print("=" * 100)
    print("COMPREHENSIVE rPPG METHOD EVALUATION SUMMARY")
    print("=" * 100)
    print(f"Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total methods evaluated: {len(overall_df)}")
    print()
    
    # Create a summary table
    if not overall_df.empty:
        summary_cols = ['Method', 'Watermark_Status', 'MAE_Mean', 'RMSE_Mean', 'MAPE_Mean', 'SNR_Mean', 'Pearson_Correlation', 'Num_Valid_Trials']
        summary_df = overall_df[summary_cols].copy()
        
        # Round numerical values for display
        for col in ['MAE_Mean', 'RMSE_Mean', 'MAPE_Mean', 'SNR_Mean', 'Pearson_Correlation']:
            summary_df[col] = summary_df[col].round(4)
        
        print(summary_df.to_string(index=False))
    else:
        print("No results to display.")
    
    print("=" * 100)

def main():
    """Main evaluation function."""
    
    # Configuration
    base_exp_dir = "/home/zik/Research/ZJU/rPPG/rppg-work/Phase1/rPPG-Toolbox/runs/exp"
    output_dir = "/home/zik/Research/ZJU/rPPG/rppg-work/Phase1/rPPG-Toolbox/runs/exp"
    
    print("Starting comprehensive rPPG method evaluation...")
    print(f"Base directory: {base_exp_dir}")
    
    # Find all pickle files
    print("Searching for pickle files...")
    pickle_files = find_all_pickle_files(base_exp_dir)
    print(f"Found {len(pickle_files)} pickle files")
    
    if not pickle_files:
        print("No pickle files found. Exiting.")
        return
    
    # Categorize and order methods
    print("Categorizing methods...")
    categorized_methods = categorize_methods(pickle_files)
    evaluation_order = create_evaluation_order(categorized_methods)
    
    print(f"Evaluation order established ({len(evaluation_order)} files):")
    for i, file_path in enumerate(evaluation_order, 1):
        filename = os.path.basename(file_path)
        method, watermark = extract_method_info(filename)
        print(f"  {i:2d}. {method} ({watermark})")
    
    print("\nStarting evaluations...")
    print("-" * 80)
    
    # Evaluate all methods
    all_results = []
    successful_evaluations = 0
    failed_evaluations = 0
    
    # Use tqdm for progress bar
    for i, file_path in enumerate(tqdm(evaluation_order, desc="Evaluating methods"), 1):
        filename = os.path.basename(file_path)
        method, watermark = extract_method_info(filename)
        
        # Update tqdm description with current method
        tqdm.write(f"\n[{i:2d}/{len(evaluation_order)}] Evaluating {method} ({watermark})...")
        
        try:
            result = evaluate_pickle_file(file_path)
            if result is not None:
                all_results.append(result)
                successful_evaluations += 1
                
                # Print overall metrics for this method
                overall = result['overall_metrics']
                tqdm.write(f"    ✓ Success: {result['num_valid_trials']} valid trials")
                tqdm.write(f"      MAE: {overall['MAE']:.4f} ± {overall['MAE_std']:.4f} BPM")
                tqdm.write(f"      RMSE: {overall['RMSE']:.4f} ± {overall['RMSE_std']:.4f} BPM")
                tqdm.write(f"      MAPE: {overall['MAPE']:.4f} ± {overall['MAPE_std']:.4f} %")
                tqdm.write(f"      SNR: {overall['SNR']:.2f} ± {overall['SNR_std']:.2f} dB")
                tqdm.write(f"      Pearson: {overall['Pearson']:.4f}")
                
                # Print per-subject metrics
                tqdm.write(f"    Per-Subject MAE:")
                subject_maes = []
                for subject_id in sorted(result['subject_metrics'].keys(), key=int):
                    mae = result['subject_metrics'][subject_id]['MAE']
                    n_trials = result['subject_metrics'][subject_id]['N']
                    subject_maes.append(f"S{subject_id}:{mae:.3f}({n_trials})")
                tqdm.write(f"      {' | '.join(subject_maes)}")
                
                # Print per-task metrics
                tqdm.write(f"    Per-Task MAE:")
                task_maes = []
                for task_id in sorted(result['task_metrics'].keys(), key=int):
                    mae = result['task_metrics'][task_id]['MAE']
                    n_trials = result['task_metrics'][task_id]['N']
                    task_maes.append(f"T{task_id}:{mae:.3f}({n_trials})")
                tqdm.write(f"      {' | '.join(task_maes)}")
                
            else:
                failed_evaluations += 1
                tqdm.write(f"    ✗ Failed: No valid data found")
        except Exception as e:
            failed_evaluations += 1
            tqdm.write(f"    ✗ Failed: {str(e)}")
            # Print full traceback for debugging
            tqdm.write("    Full error trace:")
            traceback.print_exc()
    
    print("-" * 80)
    print(f"Evaluation complete: {successful_evaluations} successful, {failed_evaluations} failed")
    
    if successful_evaluations == 0:
        print("No successful evaluations. Cannot generate output files.")
        return
    
    # Convert results to DataFrames
    print("Converting results to DataFrames...")
    overall_df, subject_df, task_df = results_to_dataframes(all_results)
    
    # Save to CSV files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    overall_file = os.path.join(output_dir, f"comprehensive_evaluation_overall_{timestamp}.csv")
    subject_file = os.path.join(output_dir, f"comprehensive_evaluation_subjects_{timestamp}.csv")
    task_file = os.path.join(output_dir, f"comprehensive_evaluation_tasks_{timestamp}.csv")
    
    print(f"Saving results to CSV files...")
    overall_df.to_csv(overall_file, index=False)
    subject_df.to_csv(subject_file, index=False)
    task_df.to_csv(task_file, index=False)
    
    print(f"✓ Overall metrics saved to: {overall_file}")
    print(f"✓ Subject metrics saved to: {subject_file}")
    print(f"✓ Task metrics saved to: {task_file}")
    
    # Print summary table
    print_summary_table(overall_df)
    
    # Save a combined summary report
    summary_file = os.path.join(output_dir, f"evaluation_summary_report_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write("COMPREHENSIVE rPPG METHOD EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total files found: {len(pickle_files)}\n")
        f.write(f"Files evaluated: {len(evaluation_order)}\n")
        f.write(f"Successful evaluations: {successful_evaluations}\n")
        f.write(f"Failed evaluations: {failed_evaluations}\n")
        f.write(f"Overall metrics file: {overall_file}\n")
        f.write(f"Subject metrics file: {subject_file}\n")
        f.write(f"Task metrics file: {task_file}\n")
        f.write("\n")
        
        if not overall_df.empty:
            f.write("RESULTS SUMMARY TABLE:\n")
            f.write("-" * 40 + "\n")
            summary_cols = ['Method', 'Watermark_Status', 'MAE_Mean', 'RMSE_Mean', 'MAPE_Mean', 'SNR_Mean', 'Pearson_Correlation', 'Num_Valid_Trials']
            summary_df = overall_df[summary_cols].copy()
            for col in ['MAE_Mean', 'RMSE_Mean', 'MAPE_Mean', 'SNR_Mean', 'Pearson_Correlation']:
                summary_df[col] = summary_df[col].round(4)
            f.write(summary_df.to_string(index=False))
        
        f.write("\n\nEVALUATION ORDER:\n")
        f.write("-" * 20 + "\n")
        for i, file_path in enumerate(evaluation_order, 1):
            filename = os.path.basename(file_path)
            method, watermark = extract_method_info(filename)
            f.write(f"{i:2d}. {method} ({watermark}) - {filename}\n")
    
    print(f"✓ Summary report saved to: {summary_file}")
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()
