import torch
import os
import sys
import subprocess
import re
from pathlib import Path
from comfy.model_management import get_torch_device
from .vfi_utilities import preprocess_frames, postprocess_frames, generate_frames_rife, logger, InterpolationStateList, infer_tiled
from .trt_utilities import Engine
from .utilities import download_file, ColoredLogger
import folder_paths
import time

def detect_cloud_environment():
    """Detect if running in cloud environment with CUDA 12 where CUDA graph may have issues"""
    # Check for common cloud environment indicators
    cloud_indicators = [
        'RUNPOD', 'COLAB', 'KAGGLE', 'PAPERSPACE', 'GOOGLE_COLAB',
        'JUPYTER_RUNTIME', 'DOCKER', 'CONTAINER'
    ]
    
    # Check environment variables
    for indicator in cloud_indicators:
        if indicator in os.environ:
            print(f"🌩️  Detected cloud environment: {indicator}")
            
            # Check CUDA version - only disable CUDA graph for CUDA 12
            try:
                import torch
                cuda_version = torch.version.cuda
                if cuda_version and cuda_version.startswith('12'):
                    print(f"🌩️  CUDA {cuda_version} detected in cloud environment - disabling CUDA graph for stability")
                    return True
                else:
                    print(f"✅ CUDA {cuda_version} detected - keeping CUDA graph enabled")
                    return False
            except:
                print("⚠️  Could not detect CUDA version, disabling CUDA graph for safety")
                return True
    
    # Check if running in Docker/container
    if os.path.exists('/.dockerenv'):
        print("🌩️  Detected Docker environment")
        try:
            import torch
            cuda_version = torch.version.cuda
            if cuda_version and cuda_version.startswith('12'):
                print(f"🌩️  CUDA {cuda_version} in Docker - disabling CUDA graph")
                return True
            else:
                print(f"✅ CUDA {cuda_version} in Docker - keeping CUDA graph enabled")
                return False
        except:
            print("⚠️  Could not detect CUDA version in Docker, disabling CUDA graph")
            return True
    
    # Check for common cloud provider files
    cloud_files = [
        '/run/cloud-init/result.json',
        '/var/lib/cloud/data/result.json'
    ]
    
    for file_path in cloud_files:
        if os.path.exists(file_path):
            print("🌩️  Detected cloud environment via cloud-init")
            try:
                import torch
                cuda_version = torch.version.cuda
                if cuda_version and cuda_version.startswith('12'):
                    print(f"🌩️  CUDA {cuda_version} in cloud - disabling CUDA graph")
                    return True
                else:
                    print(f"✅ CUDA {cuda_version} in cloud - keeping CUDA graph enabled")
                    return False
            except:
                print("⚠️  Could not detect CUDA version, disabling CUDA graph for safety")
                return True
    
    return False

# Auto-detect CUDA and install appropriate TensorRT packages
def _auto_install_tensorrt():
    """Auto-detect CUDA version and install the matching TensorRT wheels.

    The NVIDIA CUDA Toolkit must already be installed on the system.
    This function installs only the TensorRT packages via pip.
    A marker file prevents repeated install attempts on every ComfyUI startup.
    """
    disable_auto_install = os.environ.get("DISABLE_TENSORRT_AUTO_INSTALL", "false").lower() == "true"
    if disable_auto_install:
        print("[ComfyUI-RIFE-TensorRT] Auto-installation disabled via DISABLE_TENSORRT_AUTO_INSTALL")
        return True

    node_dir = Path(__file__).resolve().parent
    installed_marker = node_dir / ".tensorrt_auto_installed"
    failed_marker = node_dir / ".tensorrt_auto_install_failed"

    # Skip if we already installed successfully in a previous run.
    if installed_marker.exists():
        return True

    # Avoid retrying too often after a failure (do not block every startup).
    if failed_marker.exists():
        try:
            last_fail = failed_marker.stat().st_mtime
            if time.time() - last_fail < 3600:
                print("[ComfyUI-RIFE-TensorRT] Recent failed install attempt; skipping auto-install.")
                return False
        except Exception:
            pass

    try:
        # Check if TensorRT is already installed.
        try:
            import tensorrt
            print(f"[ComfyUI-RIFE-TensorRT] TensorRT already installed (version: {tensorrt.__version__})")
            installed_marker.touch()
            return True
        except ImportError:
            print("[ComfyUI-RIFE-TensorRT] TensorRT not found, detecting CUDA version...")

        # Detect CUDA version
        cuda_version = None

        # Try PyTorch CUDA version first (most reliable on cloud platforms)
        try:
            if hasattr(torch, 'version') and hasattr(torch.version, 'cuda'):
                pytorch_cuda = torch.version.cuda
                if pytorch_cuda:
                    cuda_version = pytorch_cuda
                    print(f"[ComfyUI-RIFE-TensorRT] Detected CUDA version from PyTorch: {cuda_version}")
        except Exception as e:
            print(f"[ComfyUI-RIFE-TensorRT] Could not detect CUDA from PyTorch: {e}")

        # Try nvcc command as fallback
        if not cuda_version:
            try:
                result = subprocess.run("nvcc --version", shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    match = re.search(r"release (\d+\.\d+)", result.stdout)
                    if match:
                        cuda_version = match.group(1)
                        print(f"[ComfyUI-RIFE-TensorRT] Detected CUDA version from nvcc: {cuda_version}")
            except Exception:
                pass

        # Try CUDA_PATH
        if not cuda_version and os.environ.get("CUDA_PATH"):
            nvcc_path = os.path.join(os.environ["CUDA_PATH"], "bin", "nvcc")
            if os.path.exists(nvcc_path):
                try:
                    result = subprocess.run(f'"{nvcc_path}" --version', shell=True, capture_output=True, text=True)
                    if result.returncode == 0:
                        match = re.search(r"release (\d+\.\d+)", result.stdout)
                        if match:
                            cuda_version = match.group(1)
                            print(f"[ComfyUI-RIFE-TensorRT] Detected CUDA via CUDA_PATH: {cuda_version}")
                except Exception:
                    pass

        # Try CUDA_HOME
        if not cuda_version and os.environ.get("CUDA_HOME"):
            nvcc_path = os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc")
            if os.path.exists(nvcc_path):
                try:
                    result = subprocess.run(f'"{nvcc_path}" --version', shell=True, capture_output=True, text=True)
                    if result.returncode == 0:
                        match = re.search(r"release (\d+\.\d+)", result.stdout)
                        if match:
                            cuda_version = match.group(1)
                            print(f"[ComfyUI-RIFE-TensorRT] Detected CUDA via CUDA_HOME: {cuda_version}")
                except Exception:
                    pass

        if not cuda_version:
            print("[ComfyUI-RIFE-TensorRT] WARNING: Could not detect CUDA version automatically.")
            print("The NVIDIA CUDA Toolkit must be installed before TensorRT can work.")
            print("Please run 'python install.py' manually after installing CUDA.")
            failed_marker.touch()
            return False

        major_version = int(cuda_version.split('.')[0])

        if major_version == 13:
            print("[ComfyUI-RIFE-TensorRT] Installing CUDA 13 TensorRT packages (RTX 50 series)")
            req_file = "requirements_cu13.txt"
        elif major_version == 12:
            print("[ComfyUI-RIFE-TensorRT] Installing CUDA 12 TensorRT packages (RTX 30/40 series)")
            req_file = "requirements_cu12.txt"
        else:
            print(f"[ComfyUI-RIFE-TensorRT] Unsupported CUDA version: {cuda_version}")
            failed_marker.touch()
            return False

        req_path = node_dir / req_file
        if not req_path.exists():
            print(f"[ComfyUI-RIFE-TensorRT] Missing requirements file: {req_path}")
            failed_marker.touch()
            return False

        # Install base dependencies first, then the CUDA-specific TensorRT wheels.
        for req_name in ["requirements.txt", req_file]:
            req_file_path = node_dir / req_name
            if not req_file_path.exists():
                continue
            print(f"[ComfyUI-RIFE-TensorRT] Installing from {req_name}...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--prefer-binary", "-r", str(req_file_path)],
                capture_output=True
            )
            if result.returncode != 0:
                print(f"[ComfyUI-RIFE-TensorRT] Failed to install {req_name}")
                print(result.stderr.decode(errors="replace"))
                failed_marker.touch()
                return False

        installed_marker.touch()
        print("[ComfyUI-RIFE-TensorRT] TensorRT installation completed successfully!")
        return True

    except Exception as e:
        print(f"[ComfyUI-RIFE-TensorRT] Auto-installation failed: {e}")
        print("Please run 'python install.py' manually to install TensorRT")
        try:
            failed_marker.touch()
        except Exception:
            pass
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
    CATEGORY = "⚡️ TensorRT/RIFE"
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
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1,
                                       "tooltip": "Max batch size for the engine profile. "
                                                  "Higher values allow batched inference (multiple frame pairs per GPU call) "
                                                  "for better throughput, but use more VRAM and require a rebuild. "
                                                  "Set to 1 for the original behaviour."}),
            },
            "optional": {
                "custom_config": ("RIFE_RESOLUTION_CONFIG", {"tooltip": "Custom resolution config (used when profile='custom')"}),
            }
        }

    RETURN_NAMES = ("rife_trt_model",)
    RETURN_TYPES = ("RIFE_TRT_MODEL",)
    CATEGORY = "⚡️ TensorRT/RIFE"
    DESCRIPTION = "Load RIFE tensorrt models, they will be built automatically if not found."
    FUNCTION = "load_rife_tensorrt_model"

    def load_rife_tensorrt_model(self, model, precision, resolution_profile, batch_size=1, custom_config=None):
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
        engine_min_batch = 1
        engine_opt_batch = max(1, batch_size)
        engine_max_batch = max(1, batch_size)
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


class MakeInterpolationStateList:
    """Build an InterpolationStateList to skip or keep specific frame pairs.
    Ported from Fannovel16/ComfyUI-Frame-Interpolation.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frame_indices": ("STRING", {"multiline": True, "default": "1,2,3",
                                             "tooltip": "Comma-separated list of frame-pair indices (0-based). "
                                                        "Pair i is formed by frame i and frame i+1."}),
                "is_skip_list": ("BOOLEAN", {"default": True,
                                             "tooltip": "True = skip interpolation for the listed pairs. "
                                                        "False = interpolate ONLY the listed pairs."}),
            },
        }

    RETURN_TYPES = ("INTERPOLATION_STATES",)
    FUNCTION = "create_options"
    CATEGORY = "⚡️ TensorRT/RIFE"
    DESCRIPTION = "Build a skip/keep list controlling which frame pairs get interpolated by RIFE."

    def create_options(self, frame_indices: str, is_skip_list: bool):
        frame_indices_list = [int(item.strip()) for item in frame_indices.split(',') if item.strip() != '']
        interpolation_state_list = InterpolationStateList(
            frame_indices=frame_indices_list,
            is_skip_list=is_skip_list,
        )
        return (interpolation_state_list,)


class AutoRifeTensorrt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Input frames for video frame interpolation"}),
                "rife_trt_model": ("RIFE_TRT_MODEL", {"tooltip": "Tensorrt model built and loaded"}),
                "clear_cache_after_n_frames": ("INT", {"default": 100, "min": 1, "max": 1000, "tooltip": "Clear CUDA cache after processing this many frames"}),
                "multiplier": ("INT", {"default": 2, "min": 1, "tooltip": "Frame interpolation multiplier (uniform). Ignored if multiplier_list is connected."}),
                "keep_model_loaded": ("BOOLEAN", {"default": False, "tooltip": "Keep model loaded in memory after processing"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16,
                                       "tooltip": "Number of interpolation tasks per GPU call. "
                                                  "Higher values improve throughput but use more VRAM. "
                                                  "Must be <= the batch_size used when building the engine. "
                                                  "Set to 1 for the safest behaviour."}),
            },
            "optional": {
                "multiplier_list": ("STRING", {"multiline": True, "default": "",
                                               "tooltip": "Per-pair multiplier schedule, comma-separated (e.g. '2,2,4,4,2'). "
                                                          "When connected, overrides 'multiplier'. Missing trailing pairs default to 2."}),
                "interpolation_states": ("INTERPOLATION_STATES", {"tooltip": "Optional skip/keep list from MakeInterpolationStateList"}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "⚡️ TensorRT/RIFE"
    OUTPUT_NODE=True

    def vfi(
        self,
        frames,
        rife_trt_model,
        clear_cache_after_n_frames=100,
        multiplier=2,
        keep_model_loaded=False,
        batch_size=1,
        multiplier_list="",
        interpolation_states=None,
    ):
        B, H, W, C = frames.shape
        shape_dict = {
            "img0": {"shape": (1, 3, H, W)},
            "img1": {"shape": (1, 3, H, W)},
            "output": {"shape": (1, 3, H, W)},
        }

        cudaStream = cuda.Stream()

        # Auto-detect CUDA graph usage based on CUDA version
        use_cuda_graph = False
        try:
            import torch
            cuda_version = torch.version.cuda
            if cuda_version and cuda_version.startswith('13'):
                use_cuda_graph = True
                print(f"✅ CUDA {cuda_version} detected - enabling CUDA graph for better performance")
            else:
                use_cuda_graph = False
                print(f"⚠️  CUDA {cuda_version} detected - disabling CUDA graph for stability")
        except:
            use_cuda_graph = False
            print("⚠️  Could not detect CUDA version - disabling CUDA graph for safety")

        # Use the provided model directly
        engine = rife_trt_model
        logger(f"Using loaded TensorRT engine")

        # Activate engine and retrieve profile bounds
        engine.activate()
        bounds = engine.get_input_profile_bounds()

        # Determine if tiling/padding is needed
        needs_tiling = False
        min_hw = (1, 1)
        max_hw = (4096, 4096)
        if bounds and 'img0' in bounds:
            min_shape = bounds['img0'][0]
            max_shape = bounds['img0'][2]
            min_hw = (min_shape[2], min_shape[3])
            max_hw = (max_shape[2], max_shape[3])
            needs_tiling = H > max_hw[0] or W > max_hw[1] or H < min_hw[0] or W < min_hw[1]
            if needs_tiling:
                logger(f"Frame {H}x{W} outside profile range "
                       f"[{min_hw[0]}-{max_hw[0]}]x[{min_hw[1]}-{max_hw[1]}] — enabling tiling/padding")

        frames = preprocess_frames(frames)

        if needs_tiling:
            # Disable CUDA graph when tiling (buffers are reallocated per tile)
            use_cuda_graph = False
            logger("Tiling mode: CUDA graph disabled")

            def _raw_infer(f0, f1, t):
                """Per-tile inference: reallocate buffers for the tile shape."""
                _, _, th, tw = f0.shape
                tile_shape_dict = {
                    "img0": {"shape": (1, 3, th, tw)},
                    "img1": {"shape": (1, 3, th, tw)},
                    "output": {"shape": (1, 3, th, tw)},
                }
                engine.allocate_buffers(shape_dict=tile_shape_dict)
                timestep_t = torch.tensor([t], dtype=torch.float32).to(get_torch_device())
                output = engine.infer({"img0": f0, "img1": f1, "timestep": timestep_t}, cudaStream, use_cuda_graph)
                return output['output']

            def return_middle_frame(frame_0, frame_1, timestep):
                return infer_tiled(frame_0, frame_1, timestep, _raw_infer, min_hw, max_hw)

            # Batch inference is not compatible with tiling (each tile has different shape)
            batch_infer_fn = None
            effective_batch_size = 1
        else:
            # Fast path: single allocation, original behaviour
            engine.allocate_buffers(shape_dict=shape_dict)

            def return_middle_frame(frame_0, frame_1, timestep):
                timestep_t = torch.tensor([timestep], dtype=torch.float32).to(get_torch_device())
                output = engine.infer({"img0": frame_0, "img1": frame_1, "timestep": timestep_t}, cudaStream, use_cuda_graph)
                result = output['output']
                return result

            # Batch inference: allocate buffers for the requested batch size
            if batch_size > 1:
                batch_shape_dict = {
                    "img0": {"shape": (batch_size, 3, H, W)},
                    "img1": {"shape": (batch_size, 3, H, W)},
                    "output": {"shape": (batch_size, 3, H, W)},
                }
                # Check if the engine profile supports this batch size
                engine_max_batch = max_hw[0]  # placeholder, real check below
                if bounds and 'img0' in bounds:
                    engine_max_batch = bounds['img0'][2][0]  # max batch dim
                if engine_max_batch >= batch_size:
                    logger(f"Batch inference enabled: batch_size={batch_size}")
                    # Pre-allocate buffers for the full batch
                    engine.allocate_buffers(shape_dict=batch_shape_dict)

                    def batch_infer_fn(frame0_batch, frame1_batch, timestep_batch):
                        B_actual = frame0_batch.shape[0]
                        # If the actual batch is smaller than the allocated batch,
                        # we still pass the full buffer but only use the first B_actual rows.
                        # TensorRT requires the input shape to match what was set in allocate_buffers.
                        # So we pad to the full batch size if needed.
                        if B_actual < batch_size:
                            pad_shape = (batch_size - B_actual,) + frame0_batch.shape[1:]
                            pad0 = torch.zeros(pad_shape, dtype=frame0_batch.dtype, device=frame0_batch.device)
                            pad1 = torch.zeros(pad_shape, dtype=frame1_batch.dtype, device=frame1_batch.device)
                            frame0_batch = torch.cat([frame0_batch, pad0], dim=0)
                            frame1_batch = torch.cat([frame1_batch, pad1], dim=0)
                            timestep_batch = torch.cat([timestep_batch, torch.zeros(batch_size - B_actual, dtype=timestep_batch.dtype, device=timestep_batch.device)])

                        timestep_t = timestep_batch.to(get_torch_device())
                        output = engine.infer({"img0": frame0_batch, "img1": frame1_batch, "timestep": timestep_t}, cudaStream, use_cuda_graph)
                        return output['output'][:B_actual]

                    effective_batch_size = batch_size
                else:
                    logger(f"⚠️  Engine max_batch={engine_max_batch} < requested batch_size={batch_size}. "
                           "Falling back to sequential inference. Rebuild engine with batch_size >= "
                           f"{batch_size} to enable batched inference.")
                    batch_infer_fn = None
                    effective_batch_size = 1
            else:
                batch_infer_fn = None
                effective_batch_size = 1

        # Resolve multiplier: per-pair schedule (list) overrides uniform int
        effective_multiplier = multiplier
        if multiplier_list and multiplier_list.strip():
            try:
                effective_multiplier = [int(x.strip()) for x in multiplier_list.split(',') if x.strip() != '']
                logger(f"Using per-pair multiplier schedule: {effective_multiplier}")
            except ValueError as e:
                logger(f"Invalid multiplier_list '{multiplier_list}': {e}. Falling back to uniform multiplier={multiplier}")
                effective_multiplier = multiplier

        result = generate_frames_rife(
            frames,
            clear_cache_after_n_frames,
            effective_multiplier,
            return_middle_frame,
            interpolation_states=interpolation_states,
            batch_size=effective_batch_size,
            batch_infer_function=batch_infer_fn,
        )
        out = postprocess_frames(result)

        if not keep_model_loaded:
            engine.reset()

        return (out,)


NODE_CLASS_MAPPINGS = {
    "AutoRifeTensorrt": AutoRifeTensorrt,
    "AutoLoadRifeTensorrtModel": AutoLoadRifeTensorrtModel,
    "CustomResolutionConfig": CustomResolutionConfig,
    "MakeInterpolationStateList": MakeInterpolationStateList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoRifeTensorrt": "Auto RIFE TensorRT",
    "AutoLoadRifeTensorrtModel": "(Down)load RIFE TensorRT Model",
    "CustomResolutionConfig": "RIFE Custom Resolution Config",
    "MakeInterpolationStateList": "RIFE Interpolation State List",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

