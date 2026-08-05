# https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/blob/main/vfi_utils.py

import os
import gc
import torch
import typing
import einops
from comfy.model_management import soft_empty_cache, get_torch_device
import numpy as np
from comfy.utils import ProgressBar
from colored import Fore, Back, Style

DEVICE = get_torch_device()


class InterpolationStateList():
    """Carries a list of frame indices and a flag indicating whether the list
    is a skip-list (frames to skip) or a keep-list (frames to interpolate).
    Ported from Fannovel16/ComfyUI-Frame-Interpolation.
    """
    def __init__(self, frame_indices: typing.List[int], is_skip_list: bool):
        self.frame_indices = frame_indices
        self.is_skip_list = is_skip_list

    def is_frame_skipped(self, frame_index):
        is_frame_in_list = frame_index in self.frame_indices
        return self.is_skip_list and is_frame_in_list or not self.is_skip_list and not is_frame_in_list

def load_file_from_github_release(model_type, ckpt_name):
    error_strs = []
    for i, base_model_download_url in enumerate(BASE_MODEL_DOWNLOAD_URLS):
        try:
            return load_file_from_url(base_model_download_url + ckpt_name, get_ckpt_container_path(model_type))
        except Exception:
            traceback_str = traceback.format_exc()
            if i < len(BASE_MODEL_DOWNLOAD_URLS) - 1:
                print("Failed! Trying another endpoint.")
            error_strs.append(f"Error when downloading from: {base_model_download_url + ckpt_name}\n\n{traceback_str}")

    error_str = '\n\n'.join(error_strs)
    raise Exception(f"Tried all GitHub base urls to download {ckpt_name} but no suceess. Below is the error log:\n\n{error_str}")

def logger(msg):
    print(f'{Style.reset}{Fore.cyan}⚡ [Rife Tensorrt] - {msg}{Style.reset}')

def preprocess_frames(frames):
    return einops.rearrange(frames[..., :3], "n h w c -> n c h w")

def postprocess_frames(frames):
    # Always return float32 — numpy and all downstream ComfyUI nodes require it.
    # Engine output may be fp16 when built with precision="fp16".
    return einops.rearrange(frames, "n c h w -> n h w c")[..., :3].to(torch.float32).cpu()


def _make_feather_window(tile_h, tile_w, overlap):
    """Build a 2D cosine-feathered weight window for blending tiles."""
    def _ramp(size):
        win = torch.ones(size)
        ramp = min(overlap, size // 2)
        if ramp > 0:
            t = torch.linspace(0, 1, ramp)
            fade = 0.5 * (1 - torch.cos(torch.pi * t))
            win[:ramp] = fade
            win[-ramp:] = fade.flip(0)
        return win
    win_h = _ramp(tile_h)
    win_w = _ramp(tile_w)
    return win_h.view(1, 1, -1, 1) * win_w.view(1, 1, 1, -1)


def _compute_tile_positions(total, tile_size, overlap):
    """Return list of start positions covering [0, total) with given tile size and overlap."""
    if total <= tile_size:
        return [0]
    stride = tile_size - overlap
    positions = list(range(0, total - tile_size + 1, stride))
    # Ensure the last tile reaches the end
    if positions[-1] + tile_size < total:
        positions.append(total - tile_size)
    # Deduplicate (can happen when total is only slightly larger than tile_size)
    return sorted(set(positions))


def infer_tiled(frame_0, frame_1, timestep, infer_fn, min_hw, max_hw, overlap=128):
    """Run inference with automatic tiling/padding when the frame dimensions
    fall outside the TensorRT engine's optimization profile.

    Args:
        frame_0, frame_1: [1, C, H, W] tensors (CPU or CUDA).
        timestep: float.
        infer_fn: callable(f0, f1, t) -> [1, C, H, W] (the raw engine.infer wrapper).
        min_hw: (min_h, min_w) from the engine profile.
        max_hw: (max_h, max_w) from the engine profile.
        overlap: feathering overlap in pixels between adjacent tiles.

    Returns:
        [1, C, H, W] interpolated frame at the original resolution (on CPU,
        ready for generate_frames_rife which does .detach().cpu()).
    """
    # Move inputs to CUDA — the engine inference requires GPU tensors.
    device = get_torch_device()
    frame_0 = frame_0.to(device)
    frame_1 = frame_1.to(device)

    _, C, H, W = frame_0.shape
    min_h, min_w = min_hw
    max_h, max_w = max_hw

    needs_tile = H > max_h or W > max_w
    needs_pad = H < min_h or W < min_w

    # Fast path: dimensions are within profile
    if not needs_tile and not needs_pad:
        return infer_fn(frame_0, frame_1, timestep)

    # --- Padding (under-min) ---
    pad_h_before = pad_h_after = pad_w_before = pad_w_after = 0
    if needs_pad:
        pad_h_before = max(0, (min_h - H + 1) // 2)
        pad_h_after = max(0, min_h - H - pad_h_before)
        pad_w_before = max(0, (min_w - W + 1) // 2)
        pad_w_after = max(0, min_w - W - pad_w_before)
        frame_0 = torch.nn.functional.pad(frame_0, (pad_w_before, pad_w_after, pad_h_before, pad_h_after), mode='replicate')
        frame_1 = torch.nn.functional.pad(frame_1, (pad_w_before, pad_w_after, pad_h_before, pad_h_after), mode='replicate')
        _, _, H_pad, W_pad = frame_0.shape
    else:
        H_pad, W_pad = H, W

    # --- Tiling (over-max) ---
    if H_pad > max_h or W_pad > max_w:
        tile_h = min(max_h, H_pad)
        tile_w = min(max_w, W_pad)
        h_positions = _compute_tile_positions(H_pad, tile_h, overlap)
        w_positions = _compute_tile_positions(W_pad, tile_w, overlap)

        window = _make_feather_window(tile_h, tile_w, overlap).to(device)

        output = torch.zeros(1, C, H_pad, W_pad, device=device, dtype=frame_0.dtype)
        weight = torch.zeros(1, 1, H_pad, W_pad, device=device, dtype=frame_0.dtype)

        for h0 in h_positions:
            for w0 in w_positions:
                h1, w1 = h0 + tile_h, w0 + tile_w
                t0 = frame_0[:, :, h0:h1, w0:w1]
                t1 = frame_1[:, :, h0:h1, w0:w1]
                tile_out = infer_fn(t0, t1, timestep)
                output[:, :, h0:h1, w0:w1] += tile_out * window
                weight[:, :, h0:h1, w0:w1] += window

        result = output / weight.clamp(min=1e-8)
    else:
        result = infer_fn(frame_0, frame_1, timestep)

    # --- Crop back if we padded ---
    if needs_pad:
        result = result[:, :, pad_h_before:pad_h_before + H, pad_w_before:pad_w_before + W]

    return result

def generate_frames_rife(
        frames,
        clear_cache_after_n_frames,
        multiplier,
        return_middle_frame_function,
        interpolation_states: InterpolationStateList = None,
        batch_size: int = 1,
        batch_infer_function=None,
        ):
    """Interpolate frames.

    Args:
        frames: input tensor [N, C, H, W] (already preprocessed).
        clear_cache_after_n_frames: clear CUDA cache every N interpolated frames.
        multiplier: int (uniform) or list[int] (per-pair schedule). A list is
            padded with 2 for any missing trailing pairs, matching Fannovel16.
        return_middle_frame_function: callable(frame0, frame1, timestep) -> middle.
            Used when batch_size == 1 (or when batch_infer_function is None).
        interpolation_states: optional InterpolationStateList to skip specific pairs.
        batch_size: number of (pair, timestep) tasks per GPU call. >1 enables
            batched inference (requires batch_infer_function and an engine built
            with max_batch >= batch_size).
        batch_infer_function: optional callable(frame0_batch, frame1_batch, timestep_batch)
            -> middle_batch. frame0_batch/frame1_batch are [B,C,H,W], timestep_batch
            is [B]. Returns [B,C,H,W]. Required when batch_size > 1.
    """

    n_pairs = len(frames) - 1

    # Normalise multiplier to a per-pair list
    if isinstance(multiplier, int):
        multipliers = [int(multiplier)] * n_pairs
    else:
        multipliers = list(map(int, multiplier))
        multipliers += [2] * (n_pairs - len(multipliers))

    # Upper bound on output length (sum of per-pair multipliers + final frame)
    total_out = sum(multipliers) + 1
    output_frames = torch.zeros(total_out, *frames.shape[1:], device="cpu")
    out_len = 0

    number_of_frames_processed_since_last_cleared_cuda_cache = 0

    # Decide whether to use batched inference
    use_batch = batch_size > 1 and batch_infer_function is not None

    if use_batch:
        # Build a flat list of all (pair_idx, timestep) tasks, skipping excluded pairs.
        # Each task produces exactly one intermediate frame.
        tasks = []
        for pair_idx in range(n_pairs):
            if interpolation_states is not None and interpolation_states.is_frame_skipped(pair_idx):
                continue
            m = multipliers[pair_idx]
            for step in range(1, m):
                tasks.append((pair_idx, step / m))

        # Storage for intermediate frames, keyed by pair index
        results = {i: [] for i in range(n_pairs)}

        pos = 0
        while pos < len(tasks):
            batch_tasks = tasks[pos : pos + batch_size]
            B = len(batch_tasks)

            frame0_list = [frames[pair_idx : pair_idx + 1] for pair_idx, _ in batch_tasks]
            frame1_list = [frames[pair_idx + 1 : pair_idx + 2] for pair_idx, _ in batch_tasks]
            timestep_list = [dt for _, dt in batch_tasks]

            frame0_batch = torch.cat(frame0_list, dim=0)
            frame1_batch = torch.cat(frame1_list, dim=0)
            timestep_batch = torch.tensor(timestep_list, dtype=torch.float32)

            middle_batch = batch_infer_function(frame0_batch, frame1_batch, timestep_batch).detach().cpu()

            for idx, (pair_idx, _) in enumerate(batch_tasks):
                results[pair_idx].append(middle_batch[idx : idx + 1])
                number_of_frames_processed_since_last_cleared_cuda_cache += 1
                if number_of_frames_processed_since_last_cleared_cuda_cache >= clear_cache_after_n_frames:
                    soft_empty_cache()
                    gc.collect()
                    number_of_frames_processed_since_last_cleared_cuda_cache = 0
                    logger("Clearing cache...")

            pos += B

        # Assemble output: each original frame followed by its interpolated frames
        for pair_idx in range(n_pairs):
            output_frames[out_len] = frames[pair_idx : pair_idx + 1]
            out_len += 1
            for mid in results[pair_idx]:
                output_frames[out_len] = mid
                out_len += 1
    else:
        # Sequential path (original behaviour, batch_size == 1)
        for frame_itr in range(n_pairs):
            frame_0 = frames[frame_itr:frame_itr+1]
            frame_1 = frames[frame_itr+1:frame_itr+2]
            output_frames[out_len] = frame_0
            out_len += 1

            if interpolation_states is not None and interpolation_states.is_frame_skipped(frame_itr):
                continue

            m = multipliers[frame_itr]
            for middle_i in range(1, m):
                timestep = middle_i / m
                middle_frame = return_middle_frame_function(frame_0, frame_1, timestep).detach().cpu()

                output_frames[out_len] = middle_frame
                out_len += 1

                number_of_frames_processed_since_last_cleared_cuda_cache += 1
                if number_of_frames_processed_since_last_cleared_cuda_cache >= clear_cache_after_n_frames:
                    soft_empty_cache()
                    gc.collect()
                    number_of_frames_processed_since_last_cleared_cuda_cache = 0
                    logger("Clearing cache...")

    # Append final frame
    output_frames[out_len] = frames[-1:]
    new_frames_count = sum(max(m - 1, 0) for m in multipliers)
    logger(f"done! - {new_frames_count} new frames generated at resolution: {output_frames[0].shape}")
    out_len += 1

    soft_empty_cache()
    gc.collect()
    logger("Final clearing cache done ...")

    res = output_frames[:out_len]
    return res
