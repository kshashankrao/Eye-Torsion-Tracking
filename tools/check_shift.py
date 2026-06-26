import cv2
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

def main():
    prev_path = os.path.join(PROJECT_ROOT, "output", "debug_polar_prev.png")
    curr_path = os.path.join(PROJECT_ROOT, "output", "debug_polar_curr.png")
    
    img1 = cv2.imread(prev_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(curr_path, cv2.IMREAD_GRAYSCALE)
    
    diff = cv2.absdiff(img1, img2)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"Max Pixel Difference: {max_diff}")
    print(f"Mean Pixel Difference: {mean_diff:.4f}")
    
    # Check if they are actually shifted version of each other
    # We will shift img1 horizontally and check if the difference decreases
    best_shift = 0
    min_mean_diff = mean_diff
    
    for shift in range(-20, 21):
        shifted_img = np.roll(img1, shift, axis=1)
        d = cv2.absdiff(shifted_img, img2)
        m = d.mean()
        if m < min_mean_diff:
            min_mean_diff = m
            best_shift = shift
            
    print(f"Best horizontal shift in Python: {best_shift} pixels (with mean diff {min_mean_diff:.4f})")

if __name__ == "__main__":
    main()
