#!/bin/bash

# Auto-install script for ComfyUI RIFE TensorRT
# Detects CUDA version and installs appropriate requirements

echo "🔍 Detecting CUDA version..."

# Try to detect CUDA version
CUDA_VERSION=""
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "✅ Found CUDA version: $CUDA_VERSION"
elif [ -n "$CUDA_PATH" ] && [ -f "$CUDA_PATH/bin/nvcc" ]; then
    CUDA_VERSION=$("$CUDA_PATH/bin/nvcc" --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "✅ Found CUDA version via CUDA_PATH: $CUDA_VERSION"
elif [ -n "$CUDA_HOME" ] && [ -f "$CUDA_HOME/bin/nvcc" ]; then
    CUDA_VERSION=$("$CUDA_HOME/bin/nvcc" --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "✅ Found CUDA version via CUDA_HOME: $CUDA_VERSION"
else
    echo "⚠️  Could not detect CUDA version automatically"
    echo "Please ensure CUDA is installed and nvcc is in your PATH"
    echo "Or set CUDA_PATH or CUDA_HOME environment variables"
    exit 1
fi

# Extract major version
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)

echo "📦 Installing requirements for CUDA $CUDA_MAJOR..."

# Install appropriate requirements based on CUDA version
if [ "$CUDA_MAJOR" = "13" ]; then
    echo "🚀 Using CUDA 13 requirements (RTX 50 series)"
    pip install -r requirements.txt
elif [ "$CUDA_MAJOR" = "12" ]; then
    echo "🔧 Using CUDA 12 requirements (RTX 30/40 series)"
    pip install -r requirements_cu12.txt
else
    echo "❌ Unsupported CUDA version: $CUDA_VERSION"
    echo "Supported versions: CUDA 12.x, CUDA 13.x"
    exit 1
fi

if [ $? -eq 0 ]; then
    echo "✅ Installation completed successfully!"
    echo "🎯 You can now use the ComfyUI RIFE TensorRT node"
else
    echo "❌ Installation failed!"
    exit 1
fi
