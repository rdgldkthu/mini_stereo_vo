#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y \
  build-essential cmake ninja-build pkg-config ccache \
  git curl wget unzip \
  gdb valgrind \
  libopencv-dev libeigen3-dev \
  python3-pip python3-venv

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install evo

echo
echo "Environment ready."
echo "OpenCV version:"
pkg-config --modversion opencv4 || true
echo "CMake version:"
cmake --version
echo "Python version:"
python3 --version
