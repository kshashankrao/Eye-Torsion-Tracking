import cv2
import numpy as np

def main():
    # Create a 160x160 image with a single spoke extending to the right (theta = 0 degrees)
    img = np.zeros((160, 160), dtype=np.uint8)
    cv2.line(img, (80, 80), (160, 80), 255, 2) # Horizontal line to the right
    
    # Warp to polar coordinates: width=360, height=80
    polar = cv2.warpPolar(img, (360, 80), (80, 80), 80, cv2.WARP_POLAR_LINEAR)
    
    # Find where the white pixels are in the polar image
    y_indices, x_indices = np.where(polar > 0)
    
    print("=== OpenCV warpPolar Axis Investigation ===")
    print(f"Polar image shape: {polar.shape} (height/rows, width/cols)")
    if len(x_indices) > 0:
        print(f"White pixels found at:")
        print(f"  Rows (Y): min={y_indices.min()}, max={y_indices.max()}, unique={np.unique(y_indices)}")
        print(f"  Cols (X): min={x_indices.min()}, max={x_indices.max()}, unique={np.unique(x_indices)}")
        
        # If the white line is a horizontal row in the polar image, then the Y axis (rows) represents radius
        # If the white line is a vertical column in the polar image, then the X axis (cols) represents radius
        if y_indices.max() - y_indices.min() < x_indices.max() - x_indices.min():
            print("\nRESULT: The Row index (Y) represents THETA. The Column index (X) represents RADIUS!")
            print("This means cv::warpPolar maps RADIUS to the X-axis (width) and THETA to the Y-axis (height)!")
        else:
            print("\nRESULT: The Column index (X) represents THETA. The Row index (Y) represents RADIUS!")
            print("This means cv::warpPolar maps THETA to the X-axis (width) and RADIUS to the Y-axis (height)!")

if __name__ == "__main__":
    main()
