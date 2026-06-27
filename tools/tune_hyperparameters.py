import optuna
import mlflow
import subprocess
import json
import os
import pandas as pd
import numpy as np
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.json")

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)

def run_pipeline():
    # Run the C++ pipeline binary
    app_path = os.path.join(PROJECT_ROOT, "build", "torsion_app")
    result = subprocess.run([app_path], capture_output=True, text=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print("C++ Pipeline Execution Failed!")
        print(result.stderr)
        raise RuntimeError("C++ application crashed during optimization trial.")

def get_latest_results_path():
    output_dir = os.path.join(PROJECT_ROOT, "output")
    if not os.path.exists(output_dir):
        return None
    subdirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir) 
               if os.path.isdir(os.path.join(output_dir, d)) and "_" in d]
    if not subdirs:
        return None
    # Select the most recently modified run folder
    latest_dir = max(subdirs, key=os.path.getmtime)
    return os.path.join(latest_dir, "algorithm_results.csv"), latest_dir

def calculate_metrics(results_csv):
    df = pd.read_csv(results_csv)
    if len(df) == 0:
        return float('inf'), float('inf')
    
    # Calculate circular angular errors (geodesic distance on the unit circle)
    error_rad = np.arctan2(
        np.sin(np.radians(df['algo_angle'] - df['gt_angle'])),
        np.cos(np.radians(df['algo_angle'] - df['gt_angle']))
    )
    abs_error_deg = np.abs(np.degrees(error_rad))
    maae = abs_error_deg.mean()
    
    # Calculate mean C++ algorithm-only execution time (ms)
    mean_runtime_ms = df['runtime_ms'].mean() if 'runtime_ms' in df.columns else 0.0
    return maae, mean_runtime_ms

# Set up MLflow tracking
mlflow.set_tracking_uri("sqlite:////tmp/mlflow.db")
mlflow.set_experiment("Eye_Torsion_MultiObjective_Tuning")

def objective(trial):
    # Suggest hyperparameters
    angular_bins = trial.suggest_categorical("angular_bins", [720, 1440, 2880, 3600])
    radial_bins = trial.suggest_int("radial_bins", 60, 120, step=10)
    
    glint_threshold = trial.suggest_int("glint_threshold", 110, 170)
    glint_kernel_size = trial.suggest_int("glint_kernel_size", 5, 17, step=2)  # Dilation kernel must be odd
    glint_inpaint_radius = trial.suggest_float("glint_inpaint_radius", 2.0, 8.0)
    polar_max_radius = trial.suggest_float("polar_max_radius", 65.0, 80.0)
    
    # The iris outer boundary must be greater than the inner boundary and less than radial_bins
    iris_inner_row = trial.suggest_int("iris_inner_row", 20, int(radial_bins * 0.5))
    iris_outer_row = trial.suggest_int("iris_outer_row", iris_inner_row + 10, radial_bins - 2)
    
    clahe_clip_limit = trial.suggest_float("clahe_clip_limit", 1.0, 5.0)
    clahe_grid_size = trial.suggest_int("clahe_grid_size", 4, 16, step=2)
    max_search_shift_deg = trial.suggest_int("max_search_shift_deg", 10, 30)

    # 1. Load current configuration
    config = load_config()
    
    # 2. Update configuration with trial hyperparameters
    config["algorithm"]["angular_bins"] = angular_bins
    config["algorithm"]["radial_bins"] = radial_bins
    config["algorithm"]["glint_threshold"] = glint_threshold
    config["algorithm"]["glint_kernel_size"] = glint_kernel_size
    config["algorithm"]["glint_inpaint_radius"] = glint_inpaint_radius
    config["algorithm"]["polar_max_radius"] = polar_max_radius
    config["algorithm"]["iris_inner_row"] = iris_inner_row
    config["algorithm"]["iris_outer_row"] = iris_outer_row
    config["algorithm"]["clahe_clip_limit"] = clahe_clip_limit
    config["algorithm"]["clahe_grid_size"] = clahe_grid_size
    config["algorithm"]["max_search_shift_deg"] = max_search_shift_deg
    
    save_config(config)

    # 3. Run C++ Pipeline and measure error + algorithm-only runtime
    try:
        run_pipeline()
        results_csv, latest_dir = get_latest_results_path()
        maae, runtime_ms = calculate_metrics(results_csv)
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf'), float('inf')

    # 4. Log to MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_params({
            "angular_bins": angular_bins,
            "radial_bins": radial_bins,
            "glint_threshold": glint_threshold,
            "glint_kernel_size": glint_kernel_size,
            "glint_inpaint_radius": glint_inpaint_radius,
            "polar_max_radius": polar_max_radius,
            "iris_inner_row": iris_inner_row,
            "iris_outer_row": iris_outer_row,
            "clahe_clip_limit": clahe_clip_limit,
            "clahe_grid_size": clahe_grid_size,
            "max_search_shift_deg": max_search_shift_deg
        })
        mlflow.log_metrics({
            "MAAE": maae,
            "runtime_ms": runtime_ms
        })
        mlflow.set_tag("trial_number", str(trial.number))

    print(f"Trial {trial.number} finished. MAAE: {maae:.4f}°, Algo Runtime: {runtime_ms:.2f}ms")
    return maae, runtime_ms

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuner for Eye Torsion")
    parser.add_argument("--trials", type=int, default=50, help="Number of tuning trials")
    args = parser.parse_args()
    
    n_trials = args.trials
    print(f"Starting Optuna Multi-Objective Optimization Study ({n_trials} trials)...")
    
    # Enable MLflow parent run for the entire study
    with mlflow.start_run(run_name="Optuna_MultiObjective_Parent"):
        # Minimize both MAAE (Index 0) and Runtime (Index 1)
        study = optuna.create_study(directions=["minimize", "minimize"])
        study.optimize(objective, n_trials=n_trials)
        
        # Log Pareto Front solutions
        print("\n=== Pareto Front Solutions (Optimal Trade-offs) ===")
        for i, trial in enumerate(study.best_trials):
            print(f"Solution {i}:")
            print(f"  MAAE: {trial.values[0]:.4f}°")
            print(f"  Runtime: {trial.values[1]:.3f}s")
            print("  Params:")
            for k, v in trial.params.items():
                print(f"    {k}: {v}")
        
        # Select the best trial based on a balanced metric or first Pareto point
        # For simplicity, we choose the first point of the Pareto front to write back to config
        best_trial = study.best_trials[0]
        config = load_config()
        for k, v in best_trial.params.items():
            config["algorithm"][k] = v
        save_config(config)
        print("\nSaved default Pareto solution 0 to config/config.json.")
        
        # Run final C++ execution
        run_pipeline()
        results_csv, latest_dir = get_latest_results_path()
        final_maae, final_runtime = calculate_metrics(results_csv)
        print(f"Final baseline MAAE: {final_maae:.4f}°, Runtime: {final_runtime:.2f}ms")

if __name__ == "__main__":
    main()
