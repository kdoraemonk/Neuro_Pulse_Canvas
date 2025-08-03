# -*- coding: utf-8 -*-
# Copyright (c) 2025 Doraemon
# Released under the MIT license
# https://opensource.org/licenses/MIT
"""
Spike 数据预处理脚本

本脚本负责将原始的神经脉冲 (Spike) 数据预处理成适用于交互式仪表盘的
“事件相机式”可视化帧。它为每个脑区创建一个独立的动态“星空图”。

处理流程:
1.  **加载数据与布局准备**: 使用 AllenSDK 加载指定实验会话的所有神经元 (units) 的
    脉冲时间戳和元数据。根据神经元的物理位置和所属脑区 (brain region)，
    为每个脑区计算出一个二维的面板 (panel) 布局，保留神经元在探针上的相对空间关系。
2.  **帧生成**: 模拟一个随时间演变的“亮度矩阵”。
    -   **初始化**: 为每个脑区创建一个代表其面板的、初始为全黑的亮度矩阵。
    -   **时间演化**: 按固定的帧率遍历总时长。在每一帧：
        a.  **亮度衰减**: 所有像素的亮度乘以一个衰减因子，模拟光亮的自然消退。
        b.  **脉冲激活**: 检查在当前时间窗口内是否有神经元放电。若有，则将其在
            亮度矩阵中对应位置的像素亮度设置为最大值 (1.0)。
        c.  **帧保存**: 将当前帧的亮度矩阵保存下来。
3.  **分区域保存**: 为了优化加载和渲染，每个脑区的动画帧序列被分别存储。
4.  **保存结果**: 将所有脑区的帧序列以及布局、元数据等信息，统一保存到一个
    压缩的 NumPy 文件 (`.npz`) 中，供 `dashboard_app.py` 使用。

运行此脚本是启动仪表盘前的必要步骤。
"""
import os
import numpy as np
import json
import cv2
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from tqdm import tqdm
import warnings

# ===============================================================
#               CONFIGURATION PARAMETERS
# ===============================================================
SESSION_ID = 847657808
PROBE_IDS = sorted([848037574, 848037576, 848037578, 848037568])
# Get the directory where the script is located, which is the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, 'ecephys_cache')
OUTPUT_NPZ = os.path.join(SCRIPT_DIR, 'spike_frames.npz')

# --- DEBUG SETTINGS ---
IS_DEBUG = True
DEBUG_DURATION_S = 60 # Process only 60 seconds for a quick debug run

# --- Frame Generation Parameters ---
FRAME_PARAMS = {
    'fps': 30,
    'decay_factor': 0.90,
    'spike_size': 8,
    'panel_spacing': 50,
    'canvas_padding': 20
}

# ===============================================================

def make_even(n):
    return int(n if n % 2 == 0 else n + 1)

def get_session_data(session_id):
    manifest_path = os.path.join(CACHE_DIR, 'manifest.json')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found at {manifest_path}")
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    return cache.get_session_data(session_id)

def prepare_layout_and_spike_data(session, probe_ids):
    print("Step 1/4: Preparing region-specific layouts and spike data...")
    units = session.units[session.units['probe_id'].isin(probe_ids)]
    channels = session.channels

    all_regions = units['structure_acronym'].unique()
    region_depths = {r: channels.loc[units[units['structure_acronym'] == r]['peak_channel_id']]['probe_vertical_position'].mean() for r in all_regions if not units[units['structure_acronym'] == r].empty}
    sorted_regions = sorted(region_depths.keys(), key=lambda r: region_depths[r])

    layout = {}
    for region in sorted_regions:
        region_units_df = units[units['structure_acronym'] == region]
        if region_units_df.empty: continue

        region_channel_ids = region_units_df['peak_channel_id'].unique()
        region_channels_df = channels.loc[region_channel_ids]
        if region_channels_df.empty: continue

        min_x = region_channels_df['probe_horizontal_position'].min()
        max_x = region_channels_df['probe_horizontal_position'].max()
        min_y = region_channels_df['probe_vertical_position'].min()
        max_y = region_channels_df['probe_vertical_position'].max()

        panel_width = make_even(int(max_x - min_x) + FRAME_PARAMS['spike_size'] + 2 * FRAME_PARAMS['canvas_padding'])
        panel_height = make_even(int(max_y - min_y) + FRAME_PARAMS['spike_size'] + 2 * FRAME_PARAMS['canvas_padding'])

        unit_coords = {}
        for unit_id, unit in region_units_df.iterrows():
            channel = channels.loc[unit['peak_channel_id']]
            x_coord = int(channel['probe_horizontal_position'] - min_x) + FRAME_PARAMS['canvas_padding']
            y_coord = int(channel['probe_vertical_position'] - min_y) + FRAME_PARAMS['canvas_padding']
            unit_coords[unit_id] = (x_coord, y_coord)

        layout[region] = {
            'units': list(region_units_df.index.astype(int)),
            'panel_size': [panel_width, panel_height],
            'unit_coords': {str(k): v for k, v in unit_coords.items()}
        }
        print(f"  - Region: {region.ljust(8)} Panel Size: {str(panel_width).ljust(4)}x {str(panel_height).ljust(4)} Units: {len(region_units_df)}")

    spike_times_by_unit = session.spike_times
    print("Layout preparation complete.")
    return layout, spike_times_by_unit

def precompute_spike_frames_memory_efficient():
    session = get_session_data(SESSION_ID)
    layout, spike_times_by_unit = prepare_layout_and_spike_data(session, PROBE_IDS)

    total_duration = np.max([times.max() for times in spike_times_by_unit.values() if len(times) > 0])
    if IS_DEBUG:
        total_duration = min(total_duration, DEBUG_DURATION_S)
        print(f"--- DEBUG MODE: Processing first {total_duration:.2f} seconds. ---")

    num_frames = int(total_duration * FRAME_PARAMS['fps'])
    time_step = 1.0 / FRAME_PARAMS['fps']

    print("\nStep 2/4: Initializing brightness matrices and frame lists (memory efficient)...")
    brightness_matrices = {r: np.zeros(d['panel_size'][::-1], dtype=np.float32) for r, d in layout.items()}
    region_frame_lists = {r: [] for r in layout.keys()}

    print("Step 3/4: Generating frames for all regions...")
    for i in tqdm(range(num_frames), desc="Generating Frames"):
        for region, matrix in brightness_matrices.items():
            matrix *= FRAME_PARAMS['decay_factor']
            current_time = i * time_step
            region_data = layout[region]

            for unit_id in region_data['units']:
                spikes = spike_times_by_unit.get(int(unit_id))
                if spikes is None: continue
                
                if np.any((spikes >= current_time) & (spikes < current_time + time_step)):
                    px, py = region_data['unit_coords'][str(unit_id)]
                    # Draw square instead of circle
                    half_size = FRAME_PARAMS['spike_size'] // 2
                    top_left = (px - half_size, py - half_size)
                    bottom_right = (px + half_size, py + half_size)
                    cv2.rectangle(matrix, top_left, bottom_right, 1.0, -1)
            
            np.clip(matrix, 0, 1.0, out=matrix)
            region_frame_lists[region].append(matrix.copy().astype(np.float16))

    print("\nStep 4/4: Converting lists to numpy arrays and saving to NPZ file...")
    metadata = {
        'session_id': SESSION_ID,
        'probe_ids': PROBE_IDS,
        'frame_params': FRAME_PARAMS,
        'total_duration_s': total_duration,
        'num_frames': num_frames,
        'is_debug': IS_DEBUG,
        'layout': layout,
    }
    
    save_dict = {'metadata': json.dumps(metadata)}

    for region, frame_list in region_frame_lists.items():
        final_frame_array = np.array(frame_list)
        save_key = f'frames_{region}'
        save_dict[save_key] = final_frame_array
        print(f"  - Adding '{save_key}' with final shape {final_frame_array.shape}")

    np.savez_compressed(OUTPUT_NPZ, **save_dict)
    
    print(f"\nPrecomputation complete. Data saved to {os.path.abspath(OUTPUT_NPZ)}")

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        precompute_spike_frames_memory_efficient()