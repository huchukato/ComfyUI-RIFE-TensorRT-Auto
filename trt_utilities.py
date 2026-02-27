import torch
from torch.cuda import nvtx
from collections import OrderedDict
import numpy as np
from polygraphy.backend.common import bytes_from_path
from polygraphy import util
from polygraphy.backend.trt import ModifyNetworkOutputs, Profile
from polygraphy.backend.trt import (
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from polygraphy.logger import G_LOGGER
from logging import error, warning
from tqdm import tqdm
import copy
import cuda.bindings.runtime as cudart
import subprocess
import sys

# Lazy import tensorrt to avoid import conflicts
_trt = None
def get_trt():
    global _trt
    if _trt is None:
        try:
            import tensorrt as trt
            _trt = trt
        except ImportError:
            raise ImportError("TensorRT is not installed. Please install it first.")
    return _trt

def diagnose_cuda_environment():
    """Diagnose CUDA environment and provide helpful error messages"""
    print("🔍 Diagnosing CUDA environment...")
    
    # Check CUDA availability
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA is not available in PyTorch")
            print("Please check:")
            print("1. CUDA is properly installed")
            print("2. PyTorch was installed with CUDA support")
            print("3. NVIDIA drivers are up to date")
            return False
        
        print(f"✅ PyTorch CUDA version: {torch.version.cuda}")
        print(f"✅ CUDA device count: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"✅ Current CUDA device: {torch.cuda.get_device_name()}")
            print(f"✅ CUDA device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except Exception as e:
        print(f"❌ Error checking PyTorch CUDA: {e}")
        return False
    
    # Check TensorRT CUDA compatibility
    try:
        trt = get_trt()
        print(f"✅ TensorRT version: {trt.__version__}")
        
        # Try to create a simple builder to test CUDA initialization
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            print("✅ TensorRT CUDA initialization successful")
            return True
        except Exception as e:
            print(f"❌ TensorRT CUDA initialization failed: {e}")
            print("Possible solutions:")
            print("1. Restart your system to clear CUDA state")
            print("2. Check NVIDIA driver version compatibility")
            print("3. Verify CUDA toolkit installation")
            print("4. Try: nvidia-smi to check GPU status")
            return False
    except ImportError as e:
        print(f"❌ TensorRT import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking TensorRT: {e}")
        return False

def check_nvidia_driver():
    """Check NVIDIA driver status"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA driver is working")
            # Extract driver version
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line:
                    print(f"✅ {line.strip()}")
                    break
            return True
        else:
            print("❌ nvidia-smi command failed")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found - NVIDIA drivers may not be installed")
        return False
    except Exception as e:
        print(f"❌ Error running nvidia-smi: {e}")
        return False

def get_trt_logger():
    trt = get_trt()
    return trt.Logger(trt.Logger.ERROR)

TRT_LOGGER = get_trt_logger()
G_LOGGER.module_severity = G_LOGGER.ERROR

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {
    value: key for (key, value) in numpy_to_torch_dtype_dict.items()
}

# https://github.com/Jeff-LiangF/streamv2v/blob/18c1a3bd56ff348d54a3300605936980bb13b03c/src/streamv2v/acceleration/tensorrt/utilities.py
def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        # Special handling for CUDA ERROR 35 (CUDA_ERROR_NO_DEVICE)
        if err == 35:
            raise RuntimeError(
                f"CUDA ERROR: {err} (Device not available or CUDA state corrupted). "
                f"Try: 1) Restart ComfyUI, 2) Check GPU availability with nvidia-smi, "
                f"3) Ensure no other processes are using the GPU. "
                f"Error reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
            )
        else:
            raise RuntimeError(
                f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
            )
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None

def safe_cuda_call(func, *args, max_retries=3, **kwargs):
    """Wrapper for CUDA calls with retry mechanism for ERROR 35"""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "CUDA ERROR: 35" in str(e) and attempt < max_retries - 1:
                print(f"⚠️  CUDA ERROR 35 detected, attempt {attempt + 1}/{max_retries}")
                print("Trying to reset CUDA context...")
                
                # Try to reset CUDA context
                try:
                    import torch
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print("✅ CUDA cache cleared and synchronized")
                except Exception as reset_error:
                    print(f"⚠️  Could not reset CUDA: {reset_error}")
                
                # Small delay before retry
                import time
                time.sleep(2)  # Increased delay for cloud environments
                
                if attempt == max_retries - 2:  # Last attempt
                    print("🔄 Final retry attempt...")
            else:
                # Re-raise the error if it's not ERROR 35 or we've exhausted retries
                raise e

def safe_cuda_call_with_graph_fallback(func, *args, **kwargs):
    """Wrapper with CUDA graph fallback for persistent ERROR 35"""
    try:
        # First try with CUDA graph enabled
        return safe_cuda_call(func, *args, **kwargs)
    except RuntimeError as e:
        if "CUDA ERROR: 35" in str(e):
            print("⚠️  CUDA ERROR 35 persists, trying without CUDA graph...")
            # Force disable CUDA graph and retry
            if 'use_cuda_graph' in kwargs:
                kwargs['use_cuda_graph'] = False
                print("🔄 Retrying inference without CUDA graph optimization...")
                return safe_cuda_call(func, *args, max_retries=2, **kwargs)
        raise e

class TQDMProgressMonitor:
    def __init__(self):
        trt = get_trt()
        trt.IProgressMonitor.__init__(self)
        self._active_phases = {}
        self._step_result = True
        self.max_indent = 5

    def phase_start(self, phase_name, parent_phase, num_steps):
        leave = False
        try:
            if parent_phase is not None:
                nbIndents = (
                    self._active_phases.get(parent_phase, {}).get(
                        "nbIndents", self.max_indent
                    )
                    + 1
                )
                if nbIndents >= self.max_indent:
                    return
            else:
                nbIndents = 0
                leave = True
            self._active_phases[phase_name] = {
                "tq": tqdm(
                    total=num_steps, desc=phase_name, leave=leave, position=nbIndents
                ),
                "nbIndents": nbIndents,
                "parent_phase": parent_phase,
            }
        except KeyboardInterrupt:
            # The phase_start callback cannot directly cancel the build, so request the cancellation from within step_complete.
            _step_result = False

    def phase_finish(self, phase_name):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    self._active_phases[phase_name]["tq"].total
                    - self._active_phases[phase_name]["tq"].n
                )

                parent_phase = self._active_phases[phase_name].get("parent_phase", None)
                while parent_phase is not None:
                    self._active_phases[parent_phase]["tq"].refresh()
                    parent_phase = self._active_phases[parent_phase].get(
                        "parent_phase", None
                    )
                if (
                    self._active_phases[phase_name]["parent_phase"]
                    in self._active_phases.keys()
                ):
                    self._active_phases[
                        self._active_phases[phase_name]["parent_phase"]
                    ]["tq"].refresh()
                del self._active_phases[phase_name]
            pass
        except KeyboardInterrupt:
            _step_result = False

    def step_complete(self, phase_name, step):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    step - self._active_phases[phase_name]["tq"].n
                )
            return self._step_result
        except KeyboardInterrupt:
            # There is no need to propagate this exception to TensorRT. We can simply cancel the build.
            return False

class Engine:
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.inputs = {}  # Initialize here
        self.outputs = {}  # Initialize here
        self.cuda_graph_instance = None  # cuda graph
        self.graph = None

    def __del__(self):
        # Clean up CUDA graph resources
        if hasattr(self, 'cuda_graph_instance') and self.cuda_graph_instance is not None:
            try:
                cudart.cudaGraphDestroy(self.cuda_graph_instance)
            except:
                pass
        if hasattr(self, 'graph') and self.graph is not None:
            try:
                cudart.cudaGraphDestroy(self.graph)
            except:
                pass

        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def reset(self, engine_path=None):
        # Clean up CUDA graph resources first
        if hasattr(self, 'cuda_graph_instance') and self.cuda_graph_instance is not None:
            try:
                cudart.cudaGraphDestroy(self.cuda_graph_instance)
            except:
                pass
            self.cuda_graph_instance = None
        if hasattr(self, 'graph') and self.graph is not None:
            try:
                cudart.cudaGraphDestroy(self.graph)
            except:
                pass
            self.graph = None

        if hasattr(self, 'engine') and self.engine is not None:
            del self.engine
        if hasattr(self, 'context') and self.context is not None:
            del self.context
        if hasattr(self, 'buffers'):
            del self.buffers
        if hasattr(self, 'tensors'):
            del self.tensors

        self.engine = None
        self.context = None
        self.engine_path = engine_path if engine_path else self.engine_path

        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.inputs = {}
        self.outputs = {}

    def build(
        self,
        onnx_path,
        fp16,
        input_profile=None,
        enable_refit=False,
        enable_preview=False,
        enable_all_tactics=False,
        timing_cache=None,
        update_output_names=None,
    ):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        
        # Diagnose CUDA environment before building
        print("🔍 Checking CUDA environment before TensorRT build...")
        if not diagnose_cuda_environment():
            print("❌ CUDA environment check failed. Cannot build TensorRT engine.")
            print("Please resolve the CUDA issues above and try again.")
            raise RuntimeError("CUDA environment check failed - see diagnostic output above")
        
        # Additional driver check
        if not check_nvidia_driver():
            print("⚠️  NVIDIA driver check failed, but attempting to continue...")
        
        print("✅ CUDA environment looks good, proceeding with TensorRT build...")
        
        p = [Profile()]
        if input_profile:
            p = [Profile() for i in range(len(input_profile))]
            for _p, i_profile in zip(p, input_profile):
                for name, dims in i_profile.items():
                    assert len(dims) == 3
                    _p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        config_kwargs = {}
        if not enable_all_tactics:
            config_kwargs["tactic_sources"] = []

        try:
            trt = get_trt()
            network = network_from_onnx_path(
                onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]
            )
        except Exception as e:
            print(f"❌ Failed to load ONNX network: {e}")
            raise RuntimeError(f"ONNX loading failed: {e}")
        
        if update_output_names:
            print(f"Updating network outputs to {update_output_names}")
            network = ModifyNetworkOutputs(network, update_output_names)

        builder = network[0]
        config = builder.create_builder_config()
        config.progress_monitor = TQDMProgressMonitor()

        trt = get_trt()
        config.set_flag(trt.BuilderFlag.FP16) if fp16 else None
        config.set_flag(trt.BuilderFlag.REFIT) if enable_refit else None

        profiles = copy.deepcopy(p)
        for profile in profiles:
            # Last profile is used for set_calibration_profile.
            calib_profile = profile.fill_defaults(network[1]).to_trt(
                builder, network[1]
            )
            config.add_optimization_profile(calib_profile)

        try:
            engine = engine_from_network(
                network,
                config,
            )
        except Exception as e:
            error(f"Failed to build engine: {e}")
            return 1
        try:
            save_engine(engine, path=self.engine_path)
        except Exception as e:
            error(f"Failed to save engine: {e}")
            return 1
        return 0

    def load(self):
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, reuse_device_memory=None):
        # If engine was reset, reload it
        if self.engine is None:
            self.load()

        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
        #    self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        # Clean up CUDA graph resources since tensors will be recreated
        if hasattr(self, 'cuda_graph_instance') and self.cuda_graph_instance is not None:
            try:
                cudart.cudaGraphDestroy(self.cuda_graph_instance)
            except:
                pass
            self.cuda_graph_instance = None
        if hasattr(self, 'graph') and self.graph is not None:
            try:
                cudart.cudaGraphDestroy(self.graph)
            except:
                pass
            self.graph = None

        nvtx.range_push("allocate_buffers")
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            binding = self.engine[idx]
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding]["shape"]
            else:
                shape = self.context.get_tensor_shape(name)

            trt_instance = get_trt()
            dtype = trt_instance.nptype(self.engine.get_tensor_dtype(name))
            if self.engine.get_tensor_mode(name) == trt_instance.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)
            tensor = torch.empty(
                tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]
            ).to(device)
            
            self.buffers[name] = tensor
            self.tensors[name] = tensor
            
            if self.engine.get_tensor_mode(name) == trt_instance.TensorIOMode.INPUT:
                self.inputs[name] = tensor
            else:
                self.outputs[name] = tensor
        nvtx.range_pop()

    def infer(self, feed_dict, stream, use_cuda_graph=False):
        """Inference with CUDA error recovery and graph fallback"""
        def _do_inference():
            for name, buf in feed_dict.items():
                self.tensors[name].copy_(buf)

            nvtx.range_push("TensorRT Inference")
            
            # Set tensor addresses explicitly for all bindings
            for name, tensor in self.tensors.items():
                self.context.set_tensor_address(name, tensor.data_ptr())
            
            if use_cuda_graph:
                if self.cuda_graph_instance is not None:
                    CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream.ptr))
                    CUASSERT(cudart.cudaStreamSynchronize(stream.ptr))
                else:
                    # do inference before CUDA graph capture
                    noerror = self.context.execute_async_v3(stream.ptr)
                    if not noerror:
                        raise ValueError("ERROR: inference failed.")
                    # capture cuda graph
                    CUASSERT(
                        cudart.cudaStreamBeginCapture(stream.ptr, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
                    )
                    self.context.execute_async_v3(stream.ptr)
                    self.graph = CUASSERT(cudart.cudaStreamEndCapture(stream.ptr))
                    self.cuda_graph_instance = CUASSERT(cudart.cudaGraphInstantiate(self.graph, 0))
            else:
                noerror = self.context.execute_async_v3(stream.ptr)
                if not noerror:
                    raise ValueError("ERROR: inference failed.")

            nvtx.range_pop()
            return self.tensors
        
        # Use safe wrapper with CUDA graph fallback for cloud environments
        return safe_cuda_call_with_graph_fallback(_do_inference, use_cuda_graph=use_cuda_graph)
