# Update Log

## Version 1.2.2

- Removed the bogus `requires-comfyui >=1.0.0` constraint — ComfyUI versions are 0.x, and the mismatch was disabling the node pack in ComfyUI Manager.

## Version 1.2.1

- CUDA detection now invokes `nvcc --version` via argv lists instead of `shell=True` (registry `python_command_injection_risk` false positive; also general hardening).

## Version 1.2.0

- Fixed auto-install reliability: TensorRT engine build and dependency setup are more robust on fresh pods.
- Reduced install time and fixed registry metadata.
- Updated Comfy Node Registry metadata (publisher, description, icon, banner).
