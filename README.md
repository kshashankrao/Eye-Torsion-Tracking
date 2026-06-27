# Eye Torsion Tracking Pipeline

A high-performance C++ and Python pipeline designed to calculate and validate eye torsion (roll) angles between sequential grayscale frames. It implements polar warp transformations and masked spatial Normalized Cross-Correlation (NCC) to measure circular shifts with sub-degree and sub-pixel accuracy, even in the presence of bright specular reflections (glints).

---

## Current Implementation Overview

The pipeline measures eye torsion (rotation of the eye around the visual axis) between a reference frame $t-1$ and a current frame $t$.

### 1. Specular Reflection (Glint) Removal
* **Thresholding**: Saturated glint pixels are identified in Cartesian space using a threshold value (default: `140`).
* **Morphological Expansion**: The mask is dilated using an 11x11 elliptical structuring element to capture the glint's transitional halos and edges.
* **Inpainting**: The target glint region is filled using **Telea's Inpainting Method** (`cv::inpaint` with a radius of 5.0).
* **Valid Mask Generation**: An inverted binary mask is created (valid pixels = `1`, glint-affected pixels = `0`) to propagate into coordinate transformations, preventing corrupted regions from impacting correlation calculations.

### 2. Cartesian to Polar Transformation
* The pupil center is designated as the origin.
* The image is warped into polar coordinates $(\rho, \theta)$ using linear polar warping up to a maximum iris boundary radius (default: `75.0` pixels).
* The resulting polar matrix is structured such that:
  * **Rows** represent the radial distance ($\rho$, 80 bins)
  * **Columns** represent the angular direction ($\theta$, 1440 bins, resolving up to $0.25^\circ$ per bin)
* This layout maps physical rotation in Cartesian space to horizontal shifting (columns) in Polar space.

### 3. Iris Annulus Extraction & Enhancement
* **Annulus Crop**: The algorithm slices the radial rows corresponding to the iris annulus (rows `35` to `70` out of `80`) to isolate the rich, fibrous patterns of the iris.
* **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization is applied to the polar slice to accentuate fine iris textures and mitigate uneven illumination.

### 4. Masked Normalized Cross-Correlation (NCC)
* A custom, single-pass spatial masked NCC is computed across all potential angular offsets in the range of $[-45^\circ, +45^\circ]$ (which translates to $\pm 180$ columns).
* Only pixels marked as valid in both the reference mask ($t-1$) and shifted current mask ($t$) contribute to the NCC score:
$$NCC(dx) = \frac{\sum (p - \bar{p})(c_{dx} - \bar{c}_{dx})}{\sqrt{\sum (p - \bar{p})^2 \sum (c_{dx} - \bar{c}_{dx})^2}}$$
* **Sub-pixel Refinement**: If the peak correlation score occurs at an internal shift (not at the boundary), a 3-point parabolic interpolation is performed on the neighboring NCC scores to calculate the sub-pixel peak shift.
* **FFT Alternative**: Standard Fourier-based Phase Correlation is supported as an alternative method (`use_fft = true`).

---

## Project Structure

```
├── CMakeLists.txt         # CMake build configuration (OpenCV, GTest, nlohmann_json)
├── config/
│   └── config.json        # Main configuration file (dataset paths, methods, etc.)
├── data/                  # Contains ground_truth.csv and sequence frames
├── include/               # Header files (.hpp)
├── src/                   # C++ Source files (.cpp)
├── tests/                 # Unit tests (GTest)
├── tools/                 # Python scripts for dataset setup and validation
└── venv/                  # Python virtual environment
```

---

## Getting Started

### Step 1: Environment Setup
A setup script is provided to automate the installation of system dependencies (C++ compilers, CMake, OpenCV) and the Python virtual environment setup.

Run the development setup script inside WSL/Linux:
```bash
chmod +x setup_dev.sh
./setup_dev.sh
```

### Step 2: Build the C++ Project
Compile the core library, main application, and test suite:
```bash
# Create build directory
cmake -B build -S .

# Build target binaries
cmake --build build -j$(nproc)
```

---

## Running the Pipeline & Validation

### 1. Run Unit Tests
Validate code correctness using GoogleTest:
```bash
./build/torsion_tests
```

### 2. Run the Main Pipeline
Process all sequential eye frames specified in `config/config.json`:
```bash
./build/torsion_app
```
*Outputs are saved under a timestamped directory inside `output/` (e.g., `output/YYYYMMDD_HHMMSS_PolarCrossCorrelation_Masked/`). This includes `algorithm_results.csv` and intermediate debug overlays.*

### 3. Run Validation and Plots
Evaluate tracking accuracy against the ground truth and plot validation graphs:
```bash
source venv/bin/activate
python3 tools/validate_results.py --visualize
```
*This outputs metrics (MAAE, RMSAE, Max Error) to the console and generates visual charts matching estimated trajectories against ground-truth angles inside the latest output directory.*
