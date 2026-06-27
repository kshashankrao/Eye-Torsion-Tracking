import pandas as pd
import numpy as np
import os
import argparse
import cv2
import matplotlib.pyplot as plt
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

def load_config():
    config_path = os.path.join(PROJECT_ROOT, "config", "config.json")
    with open(config_path, "r") as f:
        return json.load(f)

def generate_plots(df, processed_dir, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    sequences = df['sequence'].unique()
    print(f"Generating validation plots for {len(sequences)} sequences...")
    
    for seq in sequences:
        seq_df = df[df['sequence'] == seq]
        num_pairs = len(seq_df)
        
        fig, axes = plt.subplots(num_pairs, 2, figsize=(10, 3 * num_pairs))
        if num_pairs == 1:
            axes = np.expand_dims(axes, axis=0)
            
        for idx, (_, row) in enumerate(seq_df.iterrows()):
            prev_path = os.path.join(processed_dir, seq, row['img_prev'])
            curr_path = os.path.join(processed_dir, seq, row['img_curr'])
            
            img_prev = cv2.imread(prev_path, cv2.IMREAD_GRAYSCALE)
            img_curr = cv2.imread(curr_path, cv2.IMREAD_GRAYSCALE)
            
            gt = row['gt_angle']
            algo = row['algo_angle']
            # Circular angular difference
            diff_rad = np.arctan2(np.sin(np.radians(algo - gt)), np.cos(np.radians(algo - gt)))
            err = abs(np.degrees(diff_rad))
            
            ax_prev = axes[idx, 0]
            if img_prev is not None:
                ax_prev.imshow(img_prev, cmap='gray')
                ax_prev.set_title(f"t-1: {row['img_prev']}")
            ax_prev.axis('off')
            
            ax_curr = axes[idx, 1]
            if img_curr is not None:
                ax_curr.imshow(img_curr, cmap='gray')
                ax_curr.set_title(
                    f"t: {row['img_curr']}\n"
                    f"GT: {gt:.4f}° | Algo: {algo:.4f}°\n"
                    f"Error: {err:.4f}°",
                    fontsize=10,
                    color='green' if err < 0.1 else 'red'
                )
            ax_curr.axis('off')
            
        # Calculate MAAE for the sequence using circular diff
        diff_seq_rad = np.arctan2(
            np.sin(np.radians(seq_df['algo_angle'] - seq_df['gt_angle'])),
            np.cos(np.radians(seq_df['algo_angle'] - seq_df['gt_angle']))
        )
        maae = np.abs(np.degrees(diff_seq_rad)).mean()
        plt.suptitle(f"Sequence: {seq} | MAAE: {maae:.4f}°", fontsize=14, y=0.98)
        plt.tight_layout()
        
        save_path = os.path.join(plots_dir, f"{seq}_validation.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  Saved plot: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Validate algorithm results against ground truth.")
    parser.add_argument("--results_csv", type=str, default="", help="Path to the C++ algorithm results CSV file.")
    parser.add_argument("--visualize", action="store_true", help="Generate side-by-side verification plots for all sequences.")
    args = parser.parse_args()

    # Load paths from config
    config = load_config()
    processed_dir = os.path.join(PROJECT_ROOT, config["dataset"]["processed_images_dir"])
    
    # Locate results CSV and output plots directory
    if not args.results_csv:
        output_dir = os.path.join(PROJECT_ROOT, "output")
        if os.path.exists(output_dir):
            subdirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir) 
                       if os.path.isdir(os.path.join(output_dir, d)) and "_" in d]
            if subdirs:
                # Select the most recently modified run folder
                latest_dir = max(subdirs, key=os.path.getmtime)
                results_path = os.path.join(latest_dir, "algorithm_results.csv")
                plots_dir = os.path.join(latest_dir, "validation_plots")
                print(f"No --results_csv specified. Auto-detected latest run directory: {latest_dir}")
            else:
                results_path = os.path.join(PROJECT_ROOT, "output", "algorithm_results.csv")
                plots_dir = os.path.join(PROJECT_ROOT, "output", "validation_plots")
        else:
            results_path = os.path.join(PROJECT_ROOT, "output", "algorithm_results.csv")
            plots_dir = os.path.join(PROJECT_ROOT, "output", "validation_plots")
    else:
        results_path = os.path.abspath(args.results_csv)
        plots_dir = os.path.join(os.path.dirname(results_path), "validation_plots")
    
    if not os.path.exists(results_path):
        print(f"Results file not found at {results_path}")
        return
        
    print(f"Evaluating results from: {results_path}")
    df = pd.read_csv(results_path)
    
    # Calculate circular angular errors (geodesic distance on the unit circle)
    error_rad = np.arctan2(
        np.sin(np.radians(df['algo_angle'] - df['gt_angle'])),
        np.cos(np.radians(df['algo_angle'] - df['gt_angle']))
    )
    df['error'] = np.degrees(error_rad)
    df['abs_error'] = df['error'].abs()
    
    maae = df['abs_error'].mean()
    rmsae = np.sqrt((df['error']**2).mean())
    max_error = df['abs_error'].max()
    
    print("\n=== Eye Torsion Validation Results ===")
    print(f"Total samples evaluated: {len(df)}")
    print(f"Mean Absolute Angular Error (MAAE): {maae:.4f} degrees")
    print(f"Root Mean Squared Angular Error (RMSAE): {rmsae:.4f} degrees")
    print(f"Maximum Error: {max_error:.4f} degrees")
    print("\nSample predictions vs Ground Truth:")
    print(df[['sequence', 'img_prev', 'img_curr', 'gt_angle', 'algo_angle', 'abs_error']].head(10))
    
    # Generate visualization plots if the boolean flag is passed
    if args.visualize:
        generate_plots(df, processed_dir, plots_dir)

if __name__ == "__main__":
    main()
