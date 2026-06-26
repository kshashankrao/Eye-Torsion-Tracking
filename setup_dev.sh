#!/bin/bash
# setup_dev.sh - Setup developer environment in WSL2 (Ubuntu)

set -e

echo "=== 1. Updating package list and installing C++ dependencies ==="
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    libgtest-dev \
    python3-venv \
    python3-pip

# Compile and install gtest system-wide if it's not pre-compiled
if [ -d "/usr/src/gtest" ]; then
    echo "Configuring and building Google Test..."
    cd /usr/src/gtest
    sudo cmake CMakeLists.txt
    sudo make
    sudo cp lib/*.a /usr/lib
    cd -
fi

echo "=== 2. Setting up Python Virtual Environment ==="
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

echo "=== 3. Installing Python packages ==="
source venv/bin/activate
pip install --upgrade pip
pip install \
    opencv-python \
    numpy \
    pandas \
    datasets \
    matplotlib \
    fsspec \
    huggingface_hub

echo "=== Setup Complete ==="
echo "To activate the Python environment: source venv/bin/activate"
