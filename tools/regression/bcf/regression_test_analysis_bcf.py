import pandas as pd
import os
import glob


def main():
    # Define the directory containing results
    reg_test_dir = "tools/regression/bcf/stochtree_bcf_python_results"
    
    # Get all CSV files in the directory
    reg_test_files = glob.glob(os.path.join(reg_test_dir, "*.csv"))
    
    # Read and combine all results
    reg_test_df = pd.DataFrame()
    for file in reg_test_files:
        temp_df = pd.read_csv(file)
        reg_test_df = pd.concat([reg_test_df, temp_df], ignore_index=True)
    
    # Create summary by aggregating results
    summary_df = reg_test_df.groupby([
        'n', 'p', 'num_gfr', 'num_mcmc', 'dgp_num', 'snr', 'test_set_pct', 'num_threads'
    ]).agg({
        'outcome_rmse': 'median',
        'outcome_coverage': 'median',
        'treatment_effect_rmse': 'median',
        'treatment_effect_coverage': 'median',
        'runtime': 'median'
    }).reset_index()
    
    # Save summary to CSV
    summary_file_output = os.path.join(reg_test_dir, "stochtree_bcf_python_summary.csv")
    summary_df.to_csv(summary_file_output, index=False)
    print(f"Summary saved to {summary_file_output}")
    
    # Print some basic statistics
    print(f"Total number of result files: {len(reg_test_files)}")
    print(f"Total number of iterations: {len(reg_test_df)}")
    print(f"Number of unique parameter combinations: {len(summary_df)}")
    
    # Print summary statistics
    print("\nSummary statistics:")
    print(summary_df.describe())


if __name__ == "__main__":
    main() 