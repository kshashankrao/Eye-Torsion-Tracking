import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

def main():
    prev_path = os.path.join(PROJECT_ROOT, "output", "debug_polar_prev.png")
    curr_path = os.path.join(PROJECT_ROOT, "output", "debug_polar_curr.png")
    
    if not os.path.exists(prev_path):
        print("Debug images not found!")
        return
        
    img1 = cv2.imread(prev_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img2 = cv2.imread(curr_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    
    # Compute 2D FFT
    f1 = np.fft.fft2(img1)
    f2 = np.fft.fft2(img2)
    
    # Cross-power spectrum
    cross = f1 * np.conj(f2)
    cross_norm = cross / (np.abs(cross) + 1e-8)
    
    # Inverse FFT to get correlation surface
    corr = np.fft.ifft2(cross_norm)
    corr_real = np.real(corr)
    
    # Shift to center for visualization
    corr_shifted = np.fft.fftshift(corr_real)
    
    # Find peak
    max_idx = np.unravel_index(np.argmax(corr_real), corr_real.shape)
    print(f"Peak index (unshifted): {max_idx}")
    
    # Convert peak index to shift
    h, w = corr_real.shape
    shift_y = max_idx[0] if max_idx[0] < h/2 else max_idx[0] - h
    shift_x = max_idx[1] if max_idx[1] < w/2 else max_idx[1] - w
    print(f"Calculated Shift: x={shift_x}, y={shift_y}")
    
    # Plot the correlation surface
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title("Polar Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(corr_shifted, cmap='hot')
    plt.colorbar()
    plt.title("Correlation Surface (Shifted)")
    plt.savefig(os.path.join(PROJECT_ROOT, "output", "debug_correlation_surface.png"))
    print("Saved debug plot to output/debug_correlation_surface.png")

if __name__ == "__main__":
    main()
