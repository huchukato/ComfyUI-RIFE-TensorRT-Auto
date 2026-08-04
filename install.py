#!/usr/bin/env python3
"""
Auto-install script for ComfyUI RIFE TensorRT Auto.
Detects the local CUDA version and installs only the matching TensorRT wheels.
The NVIDIA CUDA Toolkit must already be installed on the system.
"""

import subprocess
import sys
import os
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def run_command(cmd):
    """Run a shell command and return stdout + return code."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.returncode
    except Exception as e:
        return "", 1


def detect_cuda_version():
    """Detect CUDA version from nvcc, CUDA_PATH or CUDA_HOME."""
    print("Detecting CUDA version...")

    # Try nvcc command
    stdout, returncode = run_command("nvcc --version")
    if returncode == 0:
        match = re.search(r"release (\d+\.\d+)", stdout)
        if match:
            version = match.group(1)
            print(f"Found CUDA version: {version}")
            return version

    # Try CUDA_PATH
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        nvcc_path = os.path.join(cuda_path, "bin", "nvcc")
        if os.path.exists(nvcc_path):
            stdout, returncode = run_command(f'"{nvcc_path}" --version')
            if returncode == 0:
                match = re.search(r"release (\d+\.\d+)", stdout)
                if match:
                    version = match.group(1)
                    print(f"Found CUDA version via CUDA_PATH: {version}")
                    return version

    # Try CUDA_HOME
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
        if os.path.exists(nvcc_path):
            stdout, returncode = run_command(f'"{nvcc_path}" --version')
            if returncode == 0:
                match = re.search(r"release (\d+\.\d+)", stdout)
                if match:
                    version = match.group(1)
                    print(f"Found CUDA version via CUDA_HOME: {version}")
                    return version

    print("Could not detect CUDA version automatically.")
    print("Please ensure the NVIDIA CUDA Toolkit is installed and nvcc is in your PATH,")
    print("or set CUDA_PATH / CUDA_HOME.")
    return None


def install_requirements(cuda_version):
    """Install the correct TensorRT packages for the detected CUDA version."""
    major_version = int(cuda_version.split('.')[0])
    print(f"Installing requirements for CUDA {major_version}...")

    if major_version == 13:
        print("Using CUDA 13 TensorRT packages (RTX 50 series)")
        req_file = "requirements_cu13.txt"
    elif major_version == 12:
        print("Using CUDA 12 TensorRT packages (RTX 30/40 series)")
        req_file = "requirements_cu12.txt"
    else:
        print(f"Unsupported CUDA version: {cuda_version}")
        print("Supported versions: CUDA 12.x, CUDA 13.x")
        return False

    req_path = SCRIPT_DIR / req_file
    if not req_path.exists():
        print(f"Missing requirements file: {req_path}")
        return False

    # Install base dependencies first, then the CUDA-specific TensorRT wheels.
    # Using a single pip call per file lets pip resolve and cache dependencies efficiently.
    for req_name in ["requirements.txt", req_file]:
        req_file_path = SCRIPT_DIR / req_name
        if not req_file_path.exists():
            continue
        print(f"Installing from {req_name}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--prefer-binary", "-r", str(req_file_path)]
        )
        if result.returncode != 0:
            print(f"Failed to install {req_name}")
            return False

    print("Installation completed successfully!")
    print("You can now use the ComfyUI RIFE TensorRT Auto node.")
    return True


def main():
    cuda_version = detect_cuda_version()
    if not cuda_version:
        sys.exit(1)

    success = install_requirements(cuda_version)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
