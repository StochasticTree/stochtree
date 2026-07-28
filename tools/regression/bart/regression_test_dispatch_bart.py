import subprocess
import sys


def main():
    # Test case parameters
    dgps = [1, 2, 3, 4]
    ns = [1000, 10000]
    ps = [5, 20]
    threads = [-1, 1]
    
    # Create parameter grid
    varying_param_grid = []
    for dgp in dgps:
        for n in ns:
            for p in ps:
                for thread in threads:
                    varying_param_grid.append([dgp, n, p, thread])
    
    # Fixed parameters
    n_iter = 5
    num_gfr = 10
    num_mcmc = 100
    snr = 2.0
    test_set_pct = 0.2
    
    # Script path
    script_path = "tools/regression/bart/individual_regression_test_bart.py"
    
    # Run script for every case
    for i, params in enumerate(varying_param_grid):
        dgp_num, n, p, num_threads = params
        
        print(f"Running test case {i+1}/{len(varying_param_grid)}:")
        print(f"  DGP: {dgp_num}, n: {n}, p: {p}, threads: {num_threads}")
        
        # Construct command
        cmd = [
            sys.executable,  # Use current Python interpreter
            script_path,
            str(n_iter),
            str(n),
            str(p),
            str(num_gfr),
            str(num_mcmc),
            str(dgp_num),
            str(snr),
            str(test_set_pct),
            str(num_threads)
        ]
        
        # Run the command
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("  Completed successfully")
            if result.stdout:
                print(f"    Output:\n{result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"  Failed with error code {e.returncode}")
            if e.stderr:
                print(f"    Error: {e.stderr.strip()}")
            if e.stdout:
                print(f"    Output: {e.stdout.strip()}")
    
    print(f"\nCompleted {len(varying_param_grid)} test cases")


if __name__ == "__main__":
    main() 