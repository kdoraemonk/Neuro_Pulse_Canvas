# -*- coding: utf-8 -*-
# Copyright (c) 2025 Doraemon
# Released under the MIT license
# https://opensource.org/licenses/MIT
"""
LFP 数据预处理脚本

本脚本负责将原始的神经数据（NWB格式）预处理成适用于交互式仪表盘的高效帧格式。
其核心功能是加载指定的局部场电位（LFP）数据，并将其转换为“事件相机式”的能量视图。

处理流程:
1.  **加载数据**: 使用 AllenSDK 从缓存目录加载指定的实验会话 (Session) 和探针 (Probe) 的 LFP 数据。
2.  **滤波**: 对 LFP 信号进行带通滤波，以提取特定的脑波频段（如 Gamma: 30-80Hz）。
3.  **能量计算**: 通过希尔伯特变换 (Hilbert Transform) 计算滤波后信号的瞬时能量。
4.  **帧生成**: 将连续的能量信号分割成固定长度的窗口，并根据设定的帧率进行积分（或取平均/最大值），
    生成一系列代表能量变化的二维“能量帧”。
5.  **分块处理**: 为了有效管理内存，整个过程以分块 (Chunk) 的方式进行，避免一次性加载全部数据。
6.  **保存结果**: 将最终生成的能量帧数组以及相关的元数据（如帧率、频率范围等）
    保存到一个压缩的 NumPy 文件 (`.npz`) 中，供 `dashboard_app.py` 加载使用。

运行此脚本是启动仪表盘前的必要步骤。
"""
import os
import numpy as np
import json
from scipy.signal import butter, filtfilt, hilbert
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from tqdm import tqdm
import warnings

# ===============================================================
#               CONFIGURATION PARAMETERS
# ===============================================================
SESSION_ID = 847657808
PROBE_ID = 848037574
# Get the directory where the script is located, which is the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, 'ecephys_cache')
OUTPUT_NPZ = os.path.join(SCRIPT_DIR, 'lfp_frames_gamma.npz')

# --- Frame Generation Parameters (from generate_lfp_video.py) ---
# These parameters are critical for ensuring data compatibility with the dashboard.
INTEGRATION_WINDOW_MS = 100  # Integration time for each frame
FRAME_RATE = 30              # Frame rate for the dashboard's video player
FREQ_BAND = [30, 80]         # Gamma band
ANALYSIS_MODE = 'mean'       # How to aggregate power in the window

# ===============================================================
#               MEMORY MANAGEMENT
# ===============================================================
CHUNK_DURATION_S = 300 # (seconds) Process data in chunks to conserve RAM.
# ===============================================================

def precompute_lfp_frames_from_video_logic():
    """
    Precomputes LFP frames for the dashboard using the exact data processing 
    logic from the original 'generate_lfp_video.py' script.
    """
    # --- 1. Load LFP Metadata and Channel Info ---
    print("Step 1/4: Loading LFP metadata and channel information...")
    manifest_path = os.path.join(CACHE_DIR, 'manifest.json')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found at {manifest_path}")

    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    session = cache.get_session_data(SESSION_ID)
    lfp = session.get_lfp(PROBE_ID)
    lfp_sampling_rate = 1 / np.median(np.diff(lfp.time.values))
    total_duration_s = lfp.time.values[-1]
    
    channels_df = session.channels
    probe_channels = channels_df.loc[lfp.channel.values]
    sorted_channels = probe_channels.sort_values("probe_vertical_position")
    channel_count = len(sorted_channels)
    
    # --- 2. Process LFP Data in Chunks ---
    print(f"Step 2/4: Preparing to process {total_duration_s:.2f}s of data in {CHUNK_DURATION_S}s chunks...")
    
    all_video_frames = []
    low, high = FREQ_BAND
    nyquist = 0.5 * lfp_sampling_rate
    b, a = butter(3, [low / nyquist, high / nyquist], btype='band')

    window_size_samples = int(lfp_sampling_rate * (INTEGRATION_WINDOW_MS / 1000.0))
    step_size_samples = int(lfp_sampling_rate / FRAME_RATE)

    chunk_start_times = range(0, int(total_duration_s), CHUNK_DURATION_S)
    for start_time_s in tqdm(chunk_start_times, desc="Processing Chunks"):
        end_time_s = min(start_time_s + CHUNK_DURATION_S, total_duration_s)

        lfp_chunk_data = lfp.sel(
            time=slice(start_time_s, end_time_s)
        ).sel(channel=sorted_channels.index.values).values

        if lfp_chunk_data.shape[0] < window_size_samples:
            continue

        filtered_chunk = filtfilt(b, a, lfp_chunk_data, axis=0)
        analytic_signal_chunk = hilbert(filtered_chunk, axis=0)
        power_chunk = np.abs(analytic_signal_chunk)**2

        num_frames_in_chunk = int((power_chunk.shape[0] - window_size_samples) / step_size_samples)
        if num_frames_in_chunk <= 0:
            continue

        shape = (num_frames_in_chunk, window_size_samples, channel_count)
        strides = (step_size_samples * power_chunk.strides[0], power_chunk.strides[0], power_chunk.strides[1])
        sliding_windows = np.lib.stride_tricks.as_strided(power_chunk, shape=shape, strides=strides)

        analysis_func = getattr(np, f"nan{ANALYSIS_MODE}", None)
        if analysis_func is None:
            raise ValueError(f"Invalid analysis mode: {ANALYSIS_MODE}")
        
        chunk_frames = analysis_func(sliding_windows, axis=1)
        
        # Replace NaN values with 0 to maintain frame count, instead of dropping them
        np.nan_to_num(chunk_frames, nan=0.0, copy=False)

        all_video_frames.append(chunk_frames)

    # --- 3. Concatenate Frames and Prepare for Saving ---
    print("Step 3/4: All chunks processed. Concatenating frames...")
    if not all_video_frames:
        raise RuntimeError("No video frames were generated. The data might be too short or parameters are incorrect.")
        
    final_frames = np.concatenate(all_video_frames, axis=0)
    print(f"Final frames array shape: {final_frames.shape}")

    # --- 4. Save Frames and Metadata to NPZ file ---
    print(f"Step 4/4: Saving frames and metadata to {OUTPUT_NPZ}...")
    
    # Create metadata compatible with dashboard_app.py
    metadata = {
        'session_id': SESSION_ID,
        'probe_id': PROBE_ID,
        'structure': 'CA1', # This was hardcoded in the old script, preserving for compatibility
        'frame_rate': FRAME_RATE,
        'lfp_sampling_rate': lfp_sampling_rate,
        'integration_window_ms': INTEGRATION_WINDOW_MS,
        'freq_band': FREQ_BAND
    }

    # The dashboard expects metadata as a JSON string
    metadata_json_string = json.dumps(metadata)

    save_dict = {
        'frames': final_frames,
        'metadata': metadata_json_string
    }

    np.savez_compressed(OUTPUT_NPZ, **save_dict)
    
    print("\nLFP frame precomputation complete.")
    print(f"Data saved to: {os.path.abspath(OUTPUT_NPZ)}")

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning) # Suppress AllenSDK version warnings
        precompute_lfp_frames_from_video_logic()