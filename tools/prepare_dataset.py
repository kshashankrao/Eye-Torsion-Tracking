import cv2
import numpy as np
import pandas as pd
import os
import json
import random
import shutil
from huggingface_hub import hf_hub_download

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

def load_config():
    config_path = os.path.join(PROJECT_ROOT, "config", "config.json")
    with open(config_path, "r") as f:
        return json.load(f)

def apply_torsion(img, angle_deg):
    """Rotates the image around its center by angle_deg (simulating torsion)."""
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def main():
    config = load_config()
    processed_dir = os.path.join(PROJECT_ROOT, config["dataset"]["processed_images_dir"])
    
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir, exist_ok=True)
    
    print("Downloading dataset from Hugging Face...")
    video_path = hf_hub_download(repo_id="fabricionarcizo/eyeinfo", repo_type="dataset", filename="01_dataset/0000/00_0100x0100.mov")
    csv_path = hf_hub_download(repo_id="fabricionarcizo/eyeinfo", repo_type="dataset", filename="02_eye_feature/0000_dataset.csv")
    
    print("Loading Ground Truth CSV...")
    df = pd.read_csv(csv_path)
    
    print(f"Extracting frames from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    
    num_base_images = 10
    torsions_per_image = 5
    crop_size = 160
    
    total_pairs = 0
    
    for i in range(num_base_images):
        seq_records = []
        seq_dir = os.path.join(processed_dir, f"seq_{i:02d}")
        os.makedirs(seq_dir, exist_ok=True)
        
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        row = df[df['frame_no'] == (i + 1)]
        if row.empty:
            continue
            
        row = row.iloc[0]
        pupil_x, pupil_y = int(row['pupil_x']), int(row['pupil_y'])
        if pupil_x < 0 or pupil_y < 0:
            continue
            
        h, w = frame_gray.shape
        cy, cx = pupil_y, pupil_x
        
        y_start = max(0, cy - crop_size//2)
        y_end = y_start + crop_size
        x_start = max(0, cx - crop_size//2)
        x_end = x_start + crop_size
        
        if y_end > h: y_end, y_start = h, h - crop_size
        if x_end > w: x_end, x_start = w, w - crop_size
            
        cropped = frame_gray[y_start:y_end, x_start:x_end]
        
        # Save raw frame WITHOUT any glint removal or python preprocessing
        prev_filename = f"frame_00.png"
        cv2.imwrite(os.path.join(seq_dir, prev_filename), cropped)
        
        current_abs_angle = 0.0 
        
        for t in range(1, torsions_per_image + 1):
            delta_angle = random.uniform(-10.0, 10.0)
            current_abs_angle += delta_angle
            
            # Apply synthetic torsion to the raw cropped frame (including its original glints)
            rotated_img = apply_torsion(cropped, current_abs_angle)
            
            curr_filename = f"frame_{t:02d}.png"
            cv2.imwrite(os.path.join(seq_dir, curr_filename), rotated_img)
            
            seq_records.append({
                'img_prev': prev_filename,
                'img_curr': curr_filename,
                'angle': delta_angle
            })
            
            prev_filename = curr_filename
            total_pairs += 1

        seq_df = pd.DataFrame(seq_records)
        seq_df.to_csv(os.path.join(seq_dir, "ground_truth.csv"), index=False)

    cap.release()
    print(f"Generated {total_pairs} pairs across {num_base_images} sequences using raw cropped HF images.")

if __name__ == "__main__":
    main()
