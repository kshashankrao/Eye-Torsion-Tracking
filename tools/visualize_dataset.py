import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

def load_config():
    config_path = os.path.join(PROJECT_ROOT, "config", "config.json")
    with open(config_path, "r") as f:
        return json.load(f)

def main():
    config = load_config()
    
    # Default path for the first sequence
    default_seq_dir = os.path.join(PROJECT_ROOT, config["dataset"]["processed_images_dir"], "seq_00")

    parser = argparse.ArgumentParser(description="Visualize dataset and algorithm output")
    parser.add_argument('--seq_dir', type=str, default=default_seq_dir, help='Path to sequence directory containing images and ground_truth.csv')
    parser.add_argument('--algo', type=str, default='', help='Path to algorithm output CSV (optional)')
    args = parser.parse_args()

    gt_csv = os.path.join(args.seq_dir, "ground_truth.csv")

    if not os.path.exists(gt_csv):
        print(f"Ground truth not found at {gt_csv}. Please run prepare_dataset.py first.")
        return

    gt_df = pd.read_csv(gt_csv)
    
    algo_df = None
    if args.algo and os.path.exists(args.algo):
        algo_df = pd.read_csv(args.algo)

    num_to_show = min(5, len(gt_df))
    if num_to_show == 0:
        print("No ground truth records found.")
        return
        
    plt.figure(figsize=(15, 3 * num_to_show))
    
    for i in range(num_to_show):
        row = gt_df.iloc[i]
        prev_img_path = os.path.join(args.seq_dir, row['img_prev'])
        curr_img_path = os.path.join(args.seq_dir, row['img_curr'])
        
        prev_img = cv2.imread(prev_img_path, cv2.IMREAD_GRAYSCALE)
        curr_img = cv2.imread(curr_img_path, cv2.IMREAD_GRAYSCALE)
        
        gt_angle = row['angle']
        
        plt.subplot(num_to_show, 2, i * 2 + 1)
        if prev_img is not None:
            plt.imshow(prev_img, cmap='gray')
            plt.title(f"t-1: {row['img_prev']}")
        plt.axis('off')
        
        plt.subplot(num_to_show, 2, i * 2 + 2)
        if curr_img is not None:
            plt.imshow(curr_img, cmap='gray')
            title = f"t: {row['img_curr']}\nGT Angle: {gt_angle:.2f} deg"
            
            if algo_df is not None:
                algo_row = algo_df[(algo_df['img_prev'] == row['img_prev']) & 
                                   (algo_df['img_curr'] == row['img_curr'])]
                if not algo_row.empty:
                    algo_angle = algo_row.iloc[0]['angle']
                    title += f"\nAlgo Angle: {algo_angle:.2f} deg\nError: {abs(gt_angle - algo_angle):.2f} deg"
            
            plt.title(title)
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
