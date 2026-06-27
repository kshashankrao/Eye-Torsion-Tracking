# Eye Torsion Tracking Pipeline

A high-performance C++ and Python pipeline designed to calculate and validate eye torsion (roll) angles between sequential grayscale frames. It implements polar warp transformations and masked spatial Normalized Cross-Correlation (NCC) to measure circular shifts with sub-degree and sub-pixel accuracy, even in the presence of bright specular reflections (glints).

* **Detailed Algorithm & Intermediate Frame Visualizations**: See the [Algorithm Documentation](docs/ALGORITHM.md).

## Demo Video

Below is the optimized tracking pipeline running on consecutive frames, showing Cartesian and Polar workspaces with green feature overlays:

<video src="docs/torsion_demo_video.mp4" width="100%" controls></video>

---

## Implementation Details

The tracking pipeline measures eye rotation around the visual axis using polar transformations and masked correlation matching. 

A comprehensive guide explaining the step-by-step math (Normalized Cross-Correlation, glint inpainting, polar warping, and parabolic fitting) along with intermediate pipeline visualizations is available in the [Algorithm Documentation](docs/ALGORITHM.md).

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
