import cv2
import pandas as pd
import numpy as np
import os
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

def load_config():
    config_path = os.path.join(PROJECT_ROOT, "config", "config.json")
    with open(config_path, "r") as f:
        return json.load(f)

def get_latest_results_path():
    output_dir = os.path.join(PROJECT_ROOT, "output")
    if not os.path.exists(output_dir):
        return None
    subdirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir) 
               if os.path.isdir(os.path.join(output_dir, d)) and "_" in d]
    if not subdirs:
        return None
    latest_dir = max(subdirs, key=os.path.getmtime)
    return os.path.join(latest_dir, "algorithm_results.csv")

def remove_glints(img, threshold_val, kernel_size, inpaint_radius):
    # Threshold to find glints
    _, mask = cv2.threshold(img, threshold_val, 255, cv2.THRESH_BINARY)
    # Dilate mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_mask = cv2.dilate(mask, kernel)
    # Inpaint glints
    inpainted = cv2.inpaint(img, dilated_mask, inpaint_radius, cv2.INPAINT_TELEA)
    # Return inpainted image and inverse valid mask (1 for valid, 0 for glint)
    valid_mask = cv2.bitwise_not(dilated_mask)
    valid_mask = (valid_mask > 0).astype(np.uint8)
    return inpainted, valid_mask

def convert_to_polar(img, center, max_radius, radial_bins, angular_bins):
    # Perform polar warp
    # warpPolar maps rows -> angle, cols -> radius
    # Size parameter is (width=radial_bins, height=angular_bins)
    polar = cv2.warpPolar(
        img, 
        dsize=(radial_bins, angular_bins), 
        center=center, 
        maxRadius=max_radius, 
        flags=cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR
    )
    # Transpose so rows = radius, cols = angle
    return polar.T

def extract_features(enhanced, polar_mask, iris_inner_row, iris_outer_row, radial_bins, angular_bins):
    # Compute Sobel gradients in vertical direction (angular dimension)
    grad_y = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.abs(grad_y)
    
    # Mask out invalid pixels (glints)
    valid_mask_8u = (polar_mask * 255).astype(np.uint8)
    masked_grad = np.zeros_like(grad_mag)
    masked_grad[valid_mask_8u > 0] = grad_mag[valid_mask_8u > 0]
    
    # Threshold top 40% gradients
    _, max_val, _, _ = cv2.minMaxLoc(masked_grad)
    _, top_features = cv2.threshold(masked_grad, 0.40 * max_val, 255, cv2.THRESH_BINARY)
    top_features = top_features.astype(np.uint8)
    
    # Place within full polar image size (radial_bins x angular_bins)
    full_feature_mask = np.zeros((radial_bins, angular_bins), dtype=np.uint8)
    full_feature_mask[iris_inner_row:iris_outer_row, :] = top_features
    return full_feature_mask

def warp_back_to_cartesian(polar_mask, center, max_radius, dsize):
    # warpPolar inverse map back to Cartesian space
    cartesian = cv2.warpPolar(
        polar_mask,
        dsize=dsize,
        center=center,
        maxRadius=max_radius,
        flags=cv2.WARP_INVERSE_MAP + cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS
    )
    return cartesian

def create_video(results_csv, output_video_path):
    df = pd.read_csv(results_csv)
    run_dir = os.path.dirname(os.path.abspath(results_csv))
    
    # Video setup
    # We will build a side-by-side template: Left is Cartesian Eye (300x300), Right is Polar Warp (300x300)
    # Total width: 720 (including margins), height: 420 (with stats below)
    frame_width = 720
    frame_height = 420
    fps = 10  # Speed at which sequential frames are played
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    print(f"Creating demo video from {len(df)} pre-generated C++ frame pairs...")
    
    for idx, row in df.iterrows():
        # Load C++ output images directly
        num_str = f"{idx:02d}"
        cartesian_path = os.path.join(run_dir, f"frame_cartesian_{num_str}.png")
        polar_path = os.path.join(run_dir, f"frame_polar_{num_str}.png")
        
        if not os.path.exists(cartesian_path) or not os.path.exists(polar_path):
            print(f"Warning: Missing C++ output frame images for index {num_str}. Skipping.")
            continue
            
        cartesian_bgr = cv2.imread(cartesian_path)
        polar_bgr = cv2.imread(polar_path)
        
        # Resize to 300x300
        cartesian_disp = cv2.resize(cartesian_bgr, (300, 300), interpolation=cv2.INTER_CUBIC)
        polar_disp = cv2.resize(polar_bgr, (300, 300), interpolation=cv2.INTER_LINEAR)
        
        # Build canvas
        canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # Draw side-by-side images
        # Cartesian on the left
        canvas[40:340, 40:340] = cartesian_disp
        # Polar on the right
        canvas[40:340, 380:680] = polar_disp
        
        # Draw titles
        cv2.putText(canvas, "Cartesian (Eye Crop + Features)", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Polar Warp (Radius x Angle)", (380, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw separators
        cv2.rectangle(canvas, (40, 40), (340, 340), (100, 100, 100), 1)
        cv2.rectangle(canvas, (380, 40), (680, 340), (100, 100, 100), 1)
        
        # Draw Stats overlay
        cv2.rectangle(canvas, (40, 350), (680, 405), (30, 30, 30), -1)
        cv2.rectangle(canvas, (40, 350), (680, 405), (100, 100, 100), 1)
        
        # Text variables
        gt_angle = row['gt_angle']
        algo_angle = row['algo_angle']
        confidence = row['confidence']
        runtime = row['runtime_ms']
        fps_val = 1000.0 / runtime if runtime > 0 else 0
        error = abs(algo_angle - gt_angle)
        
        col1_text = f"Sequence: {row['sequence']} | Frame: {row['img_curr']}"
        col2_text = f"GT: {gt_angle:+.3f}  |  Est: {algo_angle:+.3f}  |  Err: {error:.3f}"
        col3_text = f"Latency: {runtime:.2f} ms ({fps_val:.1f} FPS)  |  Conf: {confidence:.2f}"
        
        cv2.putText(canvas, col1_text, (55, 368), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, col2_text, (55, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255) if error < 0.1 else (0, 165, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, col3_text, (55, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
        
        out.write(canvas)
        
    out.release()
    print(f"Video saved successfully to: {output_video_path}")

def main():
    parser = argparse.ArgumentParser(description="Create a longer demo video showing Cartesian, Polar, features, GT/Estimated angles, and FPS stats.")
    parser.add_argument("--results_csv", help="Path to algorithm results CSV. If not specified, uses the latest run.")
    parser.add_argument("--output", default="output/torsion_demo_video.mp4", help="Path to save output MP4 video.")
    args = parser.parse_args()
    
    csv_path = args.results_csv if args.results_csv else get_latest_results_path()
    if not csv_path or not os.path.exists(csv_path):
        print(f"Error: Could not locate results CSV at: {csv_path}")
        return
        
    # Ensure output parent directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    create_video(csv_path, args.output)

if __name__ == "__main__":
    main()
