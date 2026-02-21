#!/usr/bin/env python3
"""
Auto-install script for ComfyUI RIFE TensorRT
Detects CUDA version and installs appropriate requirements
"""

import subprocess
import sys
import os
import re

def run_command(cmd):
    """Run command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.returncode
    except Exception as e:
        return "", 1

def detect_cuda_version():
    """Detect CUDA version"""
    print("🔍 Detecting CUDA version...")
    
    # Try nvcc command
    stdout, returncode = run_command("nvcc --version")
    if returncode == 0:
        match = re.search(r"release (\d+\.\d+)", stdout)
        if match:
            version = match.group(1)
            print(f"✅ Found CUDA version: {version}")
            return version
    
    # Try CUDA_PATH
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        nvcc_path = os.path.join(cuda_path, "bin", "nvcc")
        if os.path.exists(nvcc_path):
            stdout, returncode = run_command(f"{nvcc_path} --version")
            if returncode == 0:
                match = re.search(r"release (\d+\.\d+)", stdout)
                if match:
                    version = match.group(1)
                    print(f"✅ Found CUDA version via CUDA_PATH: {version}")
                    return version
    
    # Try CUDA_HOME
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
        if os.path.exists(nvcc_path):
            stdout, returncode = run_command(f"{nvcc_path} --version")
            if returncode == 0:
                match = re.search(r"release (\d+\.\d+)", stdout)
                if match:
                    version = match.group(1)
                    print(f"✅ Found CUDA version via CUDA_HOME: {version}")
                    return version
    
    print("⚠️  Could not detect CUDA version automatically")
    print("Please ensure CUDA is installed and nvcc is in your PATH")
    print("Or set CUDA_PATH or CUDA_HOME environment variables")
    return None

def install_requirements(cuda_version):
    """Install appropriate requirements based on CUDA version"""
    major_version = int(cuda_version.split('.')[0])
    
    print(f"📦 Installing requirements for CUDA {major_version}...")
    
    if major_version == 13:
        print("🚀 Using CUDA 13 requirements (RTX 50 series)")
        requirements_file = "requirements.txt"
    elif major_version == 12:
        print("🔧 Using CUDA 12 requirements (RTX 30/40 series)")
        requirements_file = "requirements_cu12.txt"
    else:
        print(f"❌ Unsupported CUDA version: {cuda_version}")
        print("Supported versions: CUDA 12.x, CUDA 13.x")
        return False
    
    # Install requirements
    returncode = subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file]).returncode
    
    if returncode == 0:
        print("✅ Installation completed successfully!")
        print("🎯 You can now use the ComfyUI RIFE TensorRT node")
        return True
    else:
        print("❌ Installation failed!")
        return False

def main():
    """Main installation function"""
    cuda_version = detect_cuda_version()
    if not cuda_version:
        sys.exit(1)
    
    success = install_requirements(cuda_version)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
