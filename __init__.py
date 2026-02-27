import torch
import os
import sys
import subprocess
import re
from pathlib import Path
from comfy.model_management import get_torch_device
from .vfi_utilities import preprocess_frames, postprocess_frames, generate_frames_rife, logger
from .trt_utilities import Engine
from .utilities import download_file, ColoredLogger
import folder_paths
import time

def detect_cloud_environment():
    """Detect if running in cloud environment where CUDA graph may have issues"""
    # Check for common cloud environment indicators
    cloud_indicators = [
        'RUNPOD', 'COLAB', 'KAGGLE', 'PAPERSPACE', 'GOOGLE_COLAB',
        'JUPYTER_RUNTIME', 'DOCKER', 'CONTAINER'
    ]
    
    # Check environment variables
    for indicator in cloud_indicators:
        if indicator in os.environ:
            print(f"🌩️  Detected cloud environment: {indicator}")
            return True
    
    # Check if running in Docker/container
    if os.path.exists('/.dockerenv'):
        print("🌩️  Detected Docker environment")
        return True
    
    # Check for common cloud provider files
    cloud_files = [
        '/run/cloud-init/result.json',
        '/var/lib/cloud/data/result.json'
    ]
    
    for file_path in cloud_files:
        if os.path.exists(file_path):
            print("🌩️  Detected cloud environment via cloud-init")
            return True
    
    return False

# Auto-detect CUDA and install appropriate TensorRT packages
def _auto_install_tensorrt():
    """Auto-detect CUDA version and install appropriate TensorRT packages if needed"""
    try:
        # Check if TensorRT is already installed
        try:
            import tensorrt
            print("✅ TensorRT already installed")
            return True
        except ImportError:
            print("🔍 TensorRT not found, detecting CUDA version...")
        
        # Detect CUDA version
        cuda_version = None
        
        # Try nvcc command
        try:
            result = subprocess.run("nvcc --version", shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                match = re.search(r"release (\d+\.\d+)", result.stdout)
                if match:
                    cuda_version = match.group(1)
                    print(f"✅ Detected CUDA version: {cuda_version}")
        except:
            pass
        
        # Try CUDA_PATH
        if not cuda_version and os.environ.get("CUDA_PATH"):
            nvcc_path = os.path.join(os.environ["CUDA_PATH"], "bin", "nvcc")
            if os.path.exists(nvcc_path):
                try:
                    result = subprocess.run(f"{nvcc_path} --version", shell=True, capture_output=True, text=True)
                    if result.returncode == 0:
                        match = re.search(r"release (\d+\.\d+)", result.stdout)
                        if match:
                            cuda_version = match.group(1)
                            print(f"✅ Detected CUDA via CUDA_PATH: {cuda_version}")
                except:
                    pass
        
        # Try CUDA_HOME
        if not cuda_version and os.environ.get("CUDA_HOME"):
            nvcc_path = os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc")
            if os.path.exists(nvcc_path):
                try:
                    result = subprocess.run(f"{nvcc_path} --version", shell=True, capture_output=True, text=True)
                    if result.returncode == 0:
                        match = re.search(r"release (\d+\.\d+)", result.stdout)
                        if match:
                            cuda_version = match.group(1)
                            print(f"✅ Detected CUDA via CUDA_HOME: {cuda_version}")
                except:
                    pass
        
        if not cuda_version:
            print("⚠️  Could not detect CUDA version automatically")
            print("Please run 'python install.py' manually to install TensorRT")
            return False
        
        # Install appropriate TensorRT packages
        major_version = int(cuda_version.split('.')[0])
        
        if major_version == 13:
            print("🚀 Installing CUDA 13 TensorRT packages (RTX 50 series)")
            packages = [
                "tensorrt_cu13==10.15.1.29",
                "tensorrt_cu13_bindings==10.15.1.29", 
                "tensorrt_cu13_libs==10.15.1.29",
                "cuda-toolkit>=13.0.0,<13.1.0"
            ]
        elif major_version == 12:
            print("🔧 Installing CUDA 12 TensorRT packages (RTX 30/40 series)")
            packages = [
                "tensorrt-cu12==10.13.3.9",
                "tensorrt-cu12-libs==10.13.3.9",
                "tensorrt-cu12-bindings==10.13.3.9",
                "cuda-toolkit>=12.8.0,<13.0.0"
            ]
        else:
            print(f"❌ Unsupported CUDA version: {cuda_version}")
            return False
        
        # Install packages
        for package in packages:
            print(f"Installing {package}...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", package], capture_output=True)
            if result.returncode != 0:
                print(f"❌ Failed to install {package}")
                print(f"Error: {result.stderr.decode()}")
                return False
        
        print("✅ TensorRT installation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Auto-installation failed: {e}")
        print("Please run 'python install.py' manually to install TensorRT")
        return False

# Run auto-install on module import
_auto_install_tensorrt()

# Auto-detect CUDA toolkit and add DLL path before importing polygraphy
def _setup_cuda_dll_path():
    """Auto-detect CUDA toolkit and add cudart64 DLL path on Windows."""
    if not sys.platform.startswith("win"):
        return
    
    cuda_root = None
    
    # Check for CUDA_PATH or CUDA_HOME environment variables
    cuda_root = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    
    if not cuda_root:
        # Try default Windows install location
        program_files = os.environ.get("PROGRAMFILES")
        if program_files:
            cuda_base = Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA"
            if cuda_base.exists():
                # Find highest version directory
                versions = sorted([d for d in cuda_base.iterdir() if d.is_dir()], reverse=True)
                if versions:
                    cuda_root = str(versions[0])
    
    if cuda_root:
        cuda_path = Path(cuda_root)
        # CUDA 13.0+ puts cudart64 in bin/x64 subdirectory
        cuda_bin_x64 = cuda_path / "bin" / "x64"
        if cuda_bin_x64.exists() and any(cuda_bin_x64.glob("cudart64*.dll")):
            os.add_dll_directory(str(cuda_bin_x64))
            return
        # Fallback to regular bin directory for older CUDA versions
        cuda_bin = cuda_path / "bin"
        if cuda_bin.exists() and any(cuda_bin.glob("cudart64*.dll")):
            os.add_dll_directory(str(cuda_bin))
            return
    
    # CUDA toolkit not found - print warning with download link
    print("[ComfyUI-Rife-TensorRT] WARNING: CUDA toolkit not found.")
    print("    Set CUDA_PATH environment variable or install CUDA toolkit.")
    print("    Download: https://developer.nvidia.com/cuda-13-0-2-download-archive")

_setup_cuda_dll_path()

from polygraphy import cuda
import comfy.model_management as mm
import tensorrt
import json

ENGINE_DIR = os.path.join(folder_paths.models_dir, "tensorrt", "rife")

# Default resolution profiles (fallback if config is missing)
DEFAULT_RESOLUTION_PROFILES = {
    "small": {"min": 384, "opt": 720, "max": 1080},
    "medium": {"min": 672, "opt": 1080, "max": 1312}
}

# Logger for this module
rife_logger = ColoredLogger("ComfyUI-Rife-Tensorrt")

# Function to load configuration
def load_node_config(config_filename="load_rife_config.json"):
    """Loads node configuration from a JSON file."""
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, config_filename)

    default_config = {
        "model": {
            "options": ["rife49_ensemble_True_scale_1_sim"],
            "default": "rife49_ensemble_True_scale_1_sim",
            "tooltip": "Default model (fallback from code)"
        },
        "precision": {
            "options": ["fp16", "fp32"],
            "default": "fp16",
            "tooltip": "Default precision (fallback from code)"
        }
    }

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        rife_logger.info(f"Successfully loaded configuration from {config_filename}")
        return config
    except FileNotFoundError:
        rife_logger.warning(f"Configuration file '{config_path}' not found. Using default fallback configuration.")
        return default_config
    except json.JSONDecodeError:
        rife_logger.error(f"Error decoding JSON from '{config_path}'. Using default fallback configuration.")
        return default_config
    except Exception as e:
        rife_logger.error(f"An unexpected error occurred while loading '{config_path}': {e}. Using default fallback.")
        return default_config

# Load the configuration once when the module is imported
LOAD_RIFE_NODE_CONFIG = load_node_config()


class CustomResolutionConfig:
    """Node to configure custom resolution dimensions for TensorRT engine building."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "min_dim": ("INT", {"default": 384, "min": 64, "max": 4096, "step": 8, "tooltip": "Minimum resolution dimension"}),
                "opt_dim": ("INT", {"default": 720, "min": 64, "max": 4096, "step": 8, "tooltip": "Optimal resolution dimension (most common)"}),
                "max_dim": ("INT", {"default": 1312, "min": 64, "max": 4096, "step": 8, "tooltip": "Maximum resolution dimension"}),
            }
        }

    RETURN_TYPES = ("RIFE_RESOLUTION_CONFIG",)
    RETURN_NAMES = ("resolution_config",)
    FUNCTION = "configure"
    CATEGORY = "tensorrt"
    DESCRIPTION = "Configure custom resolution dimensions for RIFE TensorRT engine."

    def configure(self, min_dim, opt_dim, max_dim):
        config = {
            "min": min_dim,
            "opt": opt_dim,
            "max": max_dim,
        }
        return (config,)


class AutoLoadRifeTensorrtModel:
    @classmethod
    def INPUT_TYPES(cls):
        # Use the pre-loaded configuration
        model_config = LOAD_RIFE_NODE_CONFIG.get("model", {})
        precision_config = LOAD_RIFE_NODE_CONFIG.get("precision", {})

        # Provide sensible defaults if keys are missing in the config
        model_options = model_config.get("options", ["rife49_ensemble_True_scale_1_sim"])
        model_default = model_config.get("default", "rife49_ensemble_True_scale_1_sim")
        model_tooltip = model_config.get("tooltip", "Select a RIFE model.")

        precision_options = precision_config.get("options", ["fp16", "fp32"])
        precision_default = precision_config.get("default", "fp16")
        precision_tooltip = precision_config.get("tooltip", "Select precision.")

        # Resolution profile configuration
        profile_config = LOAD_RIFE_NODE_CONFIG.get("resolution_profile", {})
        profile_options = profile_config.get("options", ["small", "medium"])
        # Ensure 'custom' is always available
        if "custom" not in profile_options:
            profile_options = profile_options + ["custom"]
        profile_default = profile_config.get("default", "small")
        profile_tooltip = profile_config.get("tooltip", "Resolution range for TensorRT engine. Use 'custom' with the INT inputs below.")

        return {
            "required": {
                "model": (model_options, {"default": model_default, "tooltip": model_tooltip}),
                "precision": (precision_options, {"default": precision_default, "tooltip": precision_tooltip}),
                "resolution_profile": (profile_options, {"default": profile_default, "tooltip": profile_tooltip}),
            },
            "optional": {
                "custom_config": ("RIFE_RESOLUTION_CONFIG", {"tooltip": "Custom resolution config (used when profile='custom')"}),
            }
        }

    RETURN_NAMES = ("rife_trt_model",)
    RETURN_TYPES = ("RIFE_TRT_MODEL",)
    CATEGORY = "tensorrt"
    DESCRIPTION = "Load RIFE tensorrt models, they will be built automatically if not found."
    FUNCTION = "load_rife_tensorrt_model"

    def load_rife_tensorrt_model(self, model, precision, resolution_profile, custom_config=None):
        tensorrt_models_dir = os.path.join(folder_paths.models_dir, "tensorrt", "rife")
        onnx_models_dir = os.path.join(folder_paths.models_dir, "onnx")

        os.makedirs(tensorrt_models_dir, exist_ok=True)
        os.makedirs(onnx_models_dir, exist_ok=True)

        onnx_model_path = os.path.join(onnx_models_dir, f"{model}.onnx")

        # Get resolution dimensions based on profile
        if resolution_profile == "custom":
            if custom_config is None:
                rife_logger.warning("Custom profile selected but no custom_config provided. Using defaults that cover both small and medium ranges.")
                dim_min, dim_opt, dim_max = 384, 720, 1312
            else:
                dim_min = custom_config.get("min", 384)
                dim_opt = custom_config.get("opt", 720)
                dim_max = custom_config.get("max", 1312)
            # Use dimensions in profile name for custom engines
            profile_name = f"custom_{dim_min}_{dim_opt}_{dim_max}"
        else:
            profiles = LOAD_RIFE_NODE_CONFIG.get("resolution_profiles", DEFAULT_RESOLUTION_PROFILES)
            profile = profiles.get(resolution_profile, DEFAULT_RESOLUTION_PROFILES["small"])
            dim_min = profile.get("min", 384)
            dim_opt = profile.get("opt", 720)
            dim_max = profile.get("max", 1080)
            profile_name = resolution_profile
        rife_logger.info(f"Using resolution profile '{profile_name}': min={dim_min}, opt={dim_opt}, max={dim_max}")

        # Build tensorrt model path with detailed naming (includes profile)
        engine_channel = 3
        engine_min_batch, engine_opt_batch, engine_max_batch = 1, 1, 1
        engine_min_h, engine_opt_h, engine_max_h = dim_min, dim_opt, dim_max
        engine_min_w, engine_opt_w, engine_max_w = dim_min, dim_opt, dim_max
        tensorrt_model_path = os.path.join(tensorrt_models_dir, f"{model}_{precision}_{profile_name}_{engine_min_batch}x{engine_channel}x{engine_min_h}x{engine_min_w}_{engine_opt_batch}x{engine_channel}x{engine_opt_h}x{engine_opt_w}_{engine_max_batch}x{engine_channel}x{engine_max_h}x{engine_max_w}_{tensorrt.__version__}.trt")

        if not os.path.exists(tensorrt_model_path):
            if not os.path.exists(onnx_model_path):
                onnx_model_download_url = f"https://huggingface.co/yuvraj108c/rife-onnx/resolve/main/{model}.onnx"
                rife_logger.info(f"Downloading {onnx_model_download_url}")
                download_file(url=onnx_model_download_url, save_path=onnx_model_path)
            else:
                rife_logger.info(f"ONNX model found at: {onnx_model_path}")

            rife_logger.info(f"Building TensorRT engine for {onnx_model_path}: {tensorrt_model_path}")
            mm.soft_empty_cache()
            s = time.time()
            engine = Engine(tensorrt_model_path)
            engine.build(
                onnx_path=onnx_model_path,
                fp16=True if precision == "fp16" else False,
                input_profile=[
                    {
                        "img0": [(engine_min_batch, engine_channel, engine_min_h, engine_min_w), (engine_opt_batch, engine_channel, engine_opt_h, engine_opt_w), (engine_max_batch, engine_channel, engine_max_h, engine_max_w)],
                        "img1": [(engine_min_batch, engine_channel, engine_min_h, engine_min_w), (engine_opt_batch, engine_channel, engine_opt_h, engine_opt_w), (engine_max_batch, engine_channel, engine_max_h, engine_max_w)],
                    }
                ],
            )
            e = time.time()
            rife_logger.info(f"Time taken to build: {(e-s)} seconds")

        rife_logger.info(f"Loading TensorRT engine: {tensorrt_model_path}")
        mm.soft_empty_cache()
        engine = Engine(tensorrt_model_path)
        engine.load()

        return (engine,)

class AutoRifeTensorrt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Input frames for video frame interpolation"}),
                "rife_trt_model": ("RIFE_TRT_MODEL", {"tooltip": "Tensorrt model built and loaded"}),
                "clear_cache_after_n_frames": ("INT", {"default": 100, "min": 1, "max": 1000, "tooltip": "Clear CUDA cache after processing this many frames"}),
                "multiplier": ("INT", {"default": 2, "min": 1, "tooltip": "Frame interpolation multiplier"}),
                "use_cuda_graph": ("BOOLEAN", {"default": not detect_cloud_environment(), "tooltip": "Use CUDA graph for better performance (auto-disabled in cloud environments)"}),
                "keep_model_loaded": ("BOOLEAN", {"default": False, "tooltip": "Keep model loaded in memory after processing"}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "tensorrt"
    OUTPUT_NODE=True

    def vfi(
        self,
        frames,
        rife_trt_model,
        clear_cache_after_n_frames=100,
        multiplier=2,
        use_cuda_graph=True,
        keep_model_loaded=False,
    ):
        B, H, W, C = frames.shape
        shape_dict = {
            "img0": {"shape": (1, 3, H, W)},
            "img1": {"shape": (1, 3, H, W)},
            "output": {"shape": (1, 3, H, W)},
        }

        cudaStream = cuda.Stream()

        # Use the provided model directly
        engine = rife_trt_model
        logger(f"Using loaded TensorRT engine")

        # Activate and allocate buffers for the engine
        engine.activate()
        engine.allocate_buffers(shape_dict=shape_dict)

        frames = preprocess_frames(frames)

        def return_middle_frame(frame_0, frame_1, timestep):
            timestep_t = torch.tensor([timestep], dtype=torch.float32).to(get_torch_device())
            # s = time.time()
            output = engine.infer({"img0": frame_0, "img1": frame_1, "timestep": timestep_t}, cudaStream, use_cuda_graph)
            # e = time.time()
            # print(f"Time taken to infer: {(e-s)*1000} ms")

            result = output['output']
            return result

        result = generate_frames_rife(frames, clear_cache_after_n_frames, multiplier, return_middle_frame)
        out = postprocess_frames(result)

        if not keep_model_loaded:
            engine.reset()

        return (out,)


NODE_CLASS_MAPPINGS = {
    "AutoRifeTensorrt": AutoRifeTensorrt,
    "AutoLoadRifeTensorrtModel": AutoLoadRifeTensorrtModel,
    "CustomResolutionConfig": CustomResolutionConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoRifeTensorrt": "Auto RIFE TensorRT",
    "AutoLoadRifeTensorrtModel": "(Down)load RIFE TensorRT Model",
    "CustomResolutionConfig": "RIFE Custom Resolution Config",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

