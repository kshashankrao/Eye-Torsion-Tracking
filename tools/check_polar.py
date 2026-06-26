import cv2
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

def main():
    prev_path = os.path.join(PROJECT_ROOT, "output", "debug_polar_prev.png")
    curr_path = os.path.join(PROJECT_ROOT, "output", "debug_polar_curr.png")
    
    if not os.path.exists(prev_path) or not os.path.exists(curr_path):
        print("Debug polar images not found!")
        return
        
    img_prev = cv2.imread(prev_path, cv2.IMREAD_GRAYSCALE)
    img_curr = cv2.imread(curr_path, cv2.IMREAD_GRAYSCALE)
    
    print("=== Debug Polar Image Diagnostics ===")
    for name, img in [("Prev", img_prev), ("Curr", img_curr)]:
        print(f"\nImage: {name}")
        print(f"Shape: {img.shape}")
        print(f"Min value: {img.min()}")
        print(f"Max value: {img.max()}")
        print(f"Mean value: {img.mean():.4f}")
        print(f"Std Dev: {img.std():.4f}")
        
        # Check how much variation there is along the columns (theta axis)
        col_means = img.mean(axis=0)
        col_std = col_means.std()
        print(f"Variation along Theta axis (Column means std): {col_std:.4f}")

if __name__ == "__main__":
    main()
