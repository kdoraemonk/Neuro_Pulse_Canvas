# -*- coding: utf-8 -*-
# Copyright (c) 2025 Doraemon
# Released under the MIT license
# https://opensource.org/licenses/MIT
"""
神经事件可视化仪表盘 - 主应用程序

本脚本启动用于可视化神经数据的主交互式仪表盘。
它使用 PyQt5 和 pyqtgraph 构建一个功能丰富的图形用户界面，用以展示
来自艾伦研究所 (Allen Institute) Ecephys 数据集的局部场电位 (LFP) 和神经脉冲 (Spike) 数据。

仪表盘包含两个主要标签页：
1. LFP 信号仪表盘: 展示传统的 LFP 热图，以及一个“事件相机式”视图，
   该视图可视化了特定频段（如 Gamma 波）的能量强度。
2. Spike 信号仪表盘: 展示经典的脉冲光栅图，以及一个“事件相机式”视图，
   其中每次脉冲放电都以其神经元物理位置为中心（按脑区组织）的一次“闪光”来表示。

该应用程序负责处理数据加载、UI 组件设置、实时绘图以及用户交互
（如缩放、平移和调整时间窗口）。为了实现高效流畅的性能，它依赖于
由 `precompute_lfp_frames.py` 和 `precompute_spike_frames.py` 脚本预先计算好的数据帧。

运行方式:
1. 确保已安装 `requirements.txt` 中的所有依赖项。
2. 运行预处理脚本 (`precompute_lfp_frames.py` 和 `precompute_spike_frames.py`)。
3. 执行此脚本: `python dashboard_app.py`
"""
import sys
import os
import json
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QLabel, QLineEdit, QGridLayout, QGraphicsView, QGraphicsScene
)
from PyQt5.QtCore import Qt, QTimer, QRectF, QObject, QThread, pyqtSignal
import pyqtgraph as pg
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# --- Configuration ---
# Get the directory where the script is located, which is the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, 'ecephys_cache')
SESSION_ID = 847657808
PROBE_ID = 848037574
LFP_DATA_PATH = os.path.join(SCRIPT_DIR, 'lfp_frames_gamma.npz')
SPIKE_DATA_PATH = os.path.join(SCRIPT_DIR, 'spike_frames.npz')

class DataManager:
    """Loads and holds all necessary data to prevent reloading."""
    def __init__(self):
        print("Loading all data. This may take a moment...")
        # LFP Data
        lfp_npz = np.load(LFP_DATA_PATH, allow_pickle=True)
        self.lfp_frames = lfp_npz['frames']
        self.lfp_meta = json.loads(str(lfp_npz['metadata']))
        finite_lfp = self.lfp_frames[np.isfinite(self.lfp_frames)]
        self.lfp_vmax = np.percentile(finite_lfp, 99) if finite_lfp.size > 0 else 1.0

        # Spike Data
        spike_npz = np.load(SPIKE_DATA_PATH, allow_pickle=True)
        self.spike_meta = json.loads(str(spike_npz['metadata']))
        self.spike_frames_by_region = {key.replace('frames_', ''): spike_npz[key] for key in spike_npz.files if key.startswith('frames_')}
        self.spike_layout = self.spike_meta['layout']

        # AllenSDK Session Data
        manifest_path = os.path.join(CACHE_DIR, 'manifest.json')
        cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
        self.session = cache.get_session_data(SESSION_ID)
        
        # LFP standard plot data
        lfp_raw = self.session.get_lfp(PROBE_ID)
        self.lfp_time = lfp_raw.time.values
        channels_df = self.session.channels
        lfp_channel_ids = lfp_raw.channel.values
        probe_channels = channels_df.loc[lfp_channel_ids]
        self.sorted_channels = probe_channels.sort_values("probe_vertical_position")
        # Data is loaded as (time, channels)
        self.lfp_sorted_data = lfp_raw.sel(channel=self.sorted_channels.index.values).values
        abs_lfp = np.abs(self.lfp_sorted_data)
        self.lfp_raw_vmax = np.nanpercentile(abs_lfp[np.isfinite(abs_lfp)], 99) if np.any(np.isfinite(abs_lfp)) else 1.0

        # Spike standard plot data
        units = self.session.units
        channels = self.session.channels
        channel_depths = channels['probe_vertical_position']
        units_with_depth = units.join(channel_depths.rename('depth'), on='peak_channel_id')
        self.sorted_units = units_with_depth.sort_values(by='depth', ascending=True).dropna(subset=['depth'])
        self.sorted_spike_times = [self.session.spike_times[unit_id] for unit_id in self.sorted_units.index]
        
        # Brain Region Labels and Colors
        self.all_structure_acronyms = self.sorted_units["structure_acronym"].unique()
        self.structure_colors = {s: pg.intColor(i, len(self.all_structure_acronyms)) for i, s in enumerate(self.all_structure_acronyms)}
        
        boundaries, region_labels = [], []
        last_acronym = None
        for i, acronym in enumerate(self.sorted_channels["structure_acronym"]):
            if acronym != last_acronym:
                if last_acronym is not None:
                    boundaries.append(i - 0.5)
                    region_labels[-1] = ((region_labels[-1][0] + i - 1) / 2, region_labels[-1][1])
                region_labels.append((i, acronym))
                last_acronym = acronym
        region_labels[-1] = ((region_labels[-1][0] + len(self.sorted_channels) - 1) / 2, region_labels[-1][1])
        self.region_labels = region_labels
        self.boundaries = boundaries

        print("Data loading complete.")

class SpikeEventView(pg.GraphicsView):
    def __init__(self, data_manager):
        super().__init__()
        self.data = data_manager
        self.vb = pg.ViewBox()
        self.setCentralItem(self.vb)
        self.vb.setAspectLocked()
        self.vb.invertY()
        
        self.image_items = {}
        self.text_items = {}
        self.setup_panels()

    def setup_panels(self):
        x_cursor = 0
        max_h = 0
        sorted_regions = sorted(self.data.spike_layout.keys(), key=lambda r: self.data.session.units[self.data.session.units['structure_acronym'] == r]['probe_vertical_position'].mean())

        for region in sorted_regions:
            panel_w, panel_h = self.data.spike_layout[region]['panel_size']
            
            img_item = pg.ImageItem(np.zeros((panel_h, panel_w)))
            self.vb.addItem(img_item)
            img_item.setRect(QRectF(x_cursor, 0, panel_w, panel_h))
            self.image_items[region] = img_item

            text_item = pg.TextItem(region, color='w')
            self.vb.addItem(text_item)
            text_item.setPos(x_cursor + panel_w / 2 - text_item.boundingRect().width() / 2, -90)
            self.text_items[region] = text_item

            x_cursor += panel_w + self.data.spike_meta['frame_params']['panel_spacing']
            max_h = max(max_h, panel_h)
        
        self.vb.autoRange()

    def update_frame(self, frame_idx):
        for region, img_item in self.image_items.items():
            if region in self.data.spike_frames_by_region and 0 <= frame_idx < len(self.data.spike_frames_by_region[region]):
                frame_data = self.data.spike_frames_by_region[region][frame_idx]
                img_item.setImage(frame_data, levels=(0, 1.0), autoLevels=False)

class Worker(QObject):
    lfp_data_updated = pyqtSignal(object, object, object)
    spike_data_updated = pyqtSignal(object, object)

    def __init__(self, data_manager):
        super().__init__()
        self.data = data_manager

    def update_lfp_views(self, start, end):
        time_mask = (self.data.lfp_time >= start) & (self.data.lfp_time <= end)
        lfp_slice = self.data.lfp_sorted_data[time_mask, :]

        frame_rate = self.data.lfp_meta['frame_rate']
        frame_idx = int(start * frame_rate)
        frame_data = None
        if 0 <= frame_idx < len(self.data.lfp_frames):
            frame_data = self.data.lfp_frames[frame_idx]
        
        self.lfp_data_updated.emit(lfp_slice, frame_data, (start, end))

    def update_spike_views(self, start, end):
        all_x, all_y, all_pens = [], [], []
        for i, unit_id in enumerate(self.data.sorted_units.index):
            times = self.data.sorted_spike_times[i]
            spikes_in_window = times[(times >= start) & (times <= end)]
            if len(spikes_in_window) > 0:
                all_x.extend(spikes_in_window)
                all_y.extend(np.full(len(spikes_in_window), i))
                structure = self.data.sorted_units.loc[unit_id]['structure_acronym']
                color = self.data.structure_colors[structure]
                all_pens.extend([color] * len(spikes_in_window))
        
        scatter_data = {'x': np.array(all_x), 'y': np.array(all_y), 'pen': all_pens}
        self.spike_data_updated.emit(scatter_data, (start, end))

class DashboardApp(QMainWindow):
    request_lfp_update = pyqtSignal(float, float)
    request_spike_update = pyqtSignal(float, float)

    def __init__(self, data_manager):
        super().__init__()
        self.data = data_manager
        self.setWindowTitle("Neuro-Event Dashboard v4.0 - Final")
        self.setGeometry(100, 100, 1920, 1080)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.lfp_page = QWidget()
        self.spike_page = QWidget()
        self.tabs.addTab(self.lfp_page, "LFP Signal Dashboard")
        self.tabs.addTab(self.spike_page, "Spike Signal Dashboard")

        self.lfp_is_dragging = False
        self.spike_is_dragging = False

        # Timers and state for animations
        self.lfp_animation_timer = QTimer(self)
        self.spike_animation_timer = QTimer(self)
        self.lfp_animation_time_cursor = 0.0
        self.spike_animation_time_cursor = 0.0

        # Debounce timer for handling region changes
        self.debounce_timer = QTimer(self)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.handle_region_change_finished)
        self.current_mode_for_debounce = 'lfp' # To know which animation to restart

        self.setup_worker_thread()
        self.setup_page(self.lfp_page, 'lfp')
        self.setup_page(self.spike_page, 'spike')

    def setup_worker_thread(self):
        self.thread = QThread()
        self.worker = Worker(self.data)
        self.worker.moveToThread(self.thread)

        # Connect worker signals to main thread slots
        self.worker.lfp_data_updated.connect(self.apply_lfp_update)
        self.worker.spike_data_updated.connect(self.apply_spike_update)

        # Connect main thread signals to worker slots
        self.request_lfp_update.connect(self.worker.update_lfp_views)
        self.request_spike_update.connect(self.worker.update_spike_views)

        self.thread.start()

    def setup_page(self, page, mode):
        page_layout = QVBoxLayout(page)
        views_layout = QHBoxLayout()
        controls_layout = QGridLayout()

        std_plot = pg.PlotWidget()
        overview_plot = pg.PlotWidget()

        if mode == 'lfp':
            event_plot = pg.PlotWidget()
            region_axis_widget = pg.PlotWidget()
            region_axis_widget.setMaximumWidth(120)
            y_axis = region_axis_widget.getAxis('left')
            y_axis.setTicks([[(val, label) for val, label in self.data.region_labels]])
            region_axis_widget.getAxis('bottom').hide()
            for boundary in self.data.boundaries:
                std_plot.addItem(pg.InfiniteLine(pos=boundary, angle=0, pen='w'))
                event_plot.addItem(pg.InfiniteLine(pos=boundary, angle=0, pen='w'))
            std_plot.setYLink(region_axis_widget)
            event_plot.setYLink(region_axis_widget)
            region_axis_widget.invertY(True)
            views_layout.addWidget(std_plot, 4)
            views_layout.addWidget(region_axis_widget, 1)
            views_layout.addWidget(event_plot, 4)
            self.lfp_event_plot = event_plot
        else: # spike
            self.spike_event_view = SpikeEventView(self.data)
            views_layout.addWidget(std_plot, 1)
            views_layout.addWidget(self.spike_event_view, 1)

        page_layout.addLayout(views_layout, 10)
        page_layout.addLayout(controls_layout, 1)

        overview_plot.getAxis('left').hide()
        overview_plot.setLabel('bottom', 'Time (s)')
        linear_region = pg.LinearRegionItem(movable=True, brush=pg.mkBrush(255, 0, 0, 50))
        overview_plot.addItem(linear_region)
        
        controls_layout.addWidget(overview_plot, 0, 0, 2, 1)
        controls_layout.addWidget(QLabel("Window Width (s):"), 0, 1)
        width_box = QLineEdit("0.5")
        width_box.setMaximumWidth(100)
        controls_layout.addWidget(width_box, 1, 1)
        controls_layout.setColumnStretch(0, 1)

        # Store references to page-specific widgets
        if mode == 'lfp':
            self.lfp_linear_region = linear_region
            self.lfp_animation_timer.timeout.connect(lambda: self.update_animation_frame('lfp'))
        else:
            self.spike_linear_region = linear_region
            self.spike_animation_timer.timeout.connect(lambda: self.update_animation_frame('spike'))

        if mode == 'lfp':
            self.lfp_std_img = pg.ImageItem()
            std_plot.addItem(self.lfp_std_img)
            std_plot.setTitle('Standard View: LFP Heatmap')
            std_plot.setLabel('left', 'Channel Index')
            
            self.lfp_event_img = pg.ImageItem()
            self.lfp_event_plot.addItem(self.lfp_event_img)
            self.lfp_event_plot.setTitle('Event-Camera-like View: LFP Power')
            self.lfp_event_plot.setLabel('left', 'Channel Index')

            coolwarm_pos = np.array([0.0, 0.5, 1.0])
            coolwarm_color = np.array([[0, 0, 255, 255], [255, 255, 255, 255], [255, 0, 0, 255]], dtype=np.ubyte)
            coolwarm_map = pg.ColorMap(coolwarm_pos, coolwarm_color)
            self.lfp_std_img.setLookupTable(coolwarm_map.getLookupTable())

            magma_pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            magma_color = np.array([[0,0,3,255], [87,31,110,255], [187,55,84,255], [251,135,43,255], [252,253,191,255]], dtype=np.ubyte)
            magma_map = pg.ColorMap(magma_pos, magma_color)
            self.lfp_event_img.setLookupTable(magma_map.getLookupTable())

            overview_signal = np.nanmean(np.abs(self.data.lfp_sorted_data), axis=1)
            overview_plot.plot(self.data.lfp_time, overview_signal, pen='w')
            overview_plot.setXRange(self.data.lfp_time[0], self.data.lfp_time[-1])
            width_box.setText("10.0")
            linear_region.setRegion([0, 10.0])
            
            # --- New Interaction Logic ---
            linear_region.sigRegionChanged.connect(lambda: self.handle_region_change('lfp'))
            width_box.returnPressed.connect(lambda: self.set_region_width(linear_region, width_box, 'lfp'))
            self.handle_region_change('lfp') # Initial update
            self.handle_region_change_finished() # Start initial animation

        elif mode == 'spike':
            std_plot.setTitle('Standard View: Spike Raster')
            std_plot.setLabel('left', 'Neuron ID')
            std_plot.invertY(True)
            self.spike_std_plot_item = pg.ScatterPlotItem(size=8, symbol='|')
            std_plot.addItem(self.spike_std_plot_item)
            self.spike_std_plot = std_plot

            all_spike_times = np.concatenate(self.data.sorted_spike_times)
            total_duration = self.data.spike_meta['total_duration_s']
            bins = np.arange(0, total_duration, 1.0)
            pop_rate, _ = np.histogram(all_spike_times, bins=bins)
            overview_plot.plot(bins[:-1], pop_rate, pen='w')
            overview_plot.setXRange(0, total_duration)
            width_box.setText("0.5")
            linear_region.setRegion([0, 0.5])

            # --- New Interaction Logic ---
            linear_region.sigRegionChanged.connect(lambda: self.handle_region_change('spike'))
            width_box.returnPressed.connect(lambda: self.set_region_width(linear_region, width_box, 'spike'))
            self.handle_region_change('spike') # Initial update
            # self.handle_region_change_finished() # Don't auto-start spike animation

    def handle_region_change(self, mode):
        self.lfp_animation_timer.stop()
        self.spike_animation_timer.stop()
        
        self.current_mode_for_debounce = mode
        if mode == 'lfp':
            region = self.lfp_linear_region
            start, end = region.getRegion()
            self.request_lfp_update.emit(start, end)
        else:
            region = self.spike_linear_region
            start, end = region.getRegion()
            self.request_spike_update.emit(start, end)

        self.debounce_timer.start(250) # ms

    def handle_region_change_finished(self):
        mode = self.current_mode_for_debounce
        if mode == 'lfp':
            region = self.lfp_linear_region
            timer = self.lfp_animation_timer
            frame_rate = self.data.lfp_meta['frame_rate']
            start, _ = region.getRegion()
            self.lfp_animation_time_cursor = start
            timer.start(1000 // frame_rate)
        else:
            region = self.spike_linear_region
            timer = self.spike_animation_timer
            frame_rate = self.data.spike_meta['frame_params']['fps']
            start, _ = region.getRegion()
            self.spike_animation_time_cursor = start
            timer.start(1000 // frame_rate)

    def update_animation_frame(self, mode):
        if mode == 'lfp':
            region = self.lfp_linear_region
            start, end = region.getRegion()
            frame_rate = self.data.lfp_meta['frame_rate']
            time_step = 1.0 / frame_rate

            self.lfp_animation_time_cursor += time_step
            if self.lfp_animation_time_cursor > end:
                self.lfp_animation_time_cursor = start

            frame_idx = int(self.lfp_animation_time_cursor * frame_rate)
            if 0 <= frame_idx < len(self.data.lfp_frames):
                frame_data = self.data.lfp_frames[frame_idx]
                if frame_data is not None and hasattr(frame_data, 'ndim') and frame_data.ndim == 1:
                    frame_2d = frame_data[:, np.newaxis]
                    self.lfp_event_img.setImage(frame_2d, levels=(0, self.data.lfp_vmax), autoLevels=False)
                else:
                    self.lfp_event_img.clear()

        else: # spike
            region = self.spike_linear_region
            start, end = region.getRegion()
            frame_rate = self.data.spike_meta['frame_params']['fps']
            time_step = 1.0 / frame_rate

            self.spike_animation_time_cursor += time_step
            if self.spike_animation_time_cursor > end:
                self.spike_animation_time_cursor = start

            frame_idx = int(self.spike_animation_time_cursor * frame_rate)
            if 0 <= frame_idx < len(self.data.spike_frames_by_region.get('CA1', [])): # Safe get
                for region_name, img_item in self.spike_event_view.image_items.items():
                    if region_name in self.data.spike_frames_by_region and 0 <= frame_idx < len(self.data.spike_frames_by_region[region_name]):
                        frame_data = self.data.spike_frames_by_region[region_name][frame_idx]
                        img_item.setImage(frame_data, levels=(0, 1.0), autoLevels=False)

    def apply_lfp_update(self, lfp_slice, frame_data, region):
        start, end = region
        if lfp_slice is not None and lfp_slice.size > 0:
            self.lfp_std_img.setImage(lfp_slice.T, levels=(-self.data.lfp_raw_vmax, self.data.lfp_raw_vmax), autoLevels=False)
            self.lfp_std_img.setRect(pg.QtCore.QRectF(start, 0, end - start, self.data.lfp_sorted_data.shape[1]))
        else:
            self.lfp_std_img.clear()
        
        if frame_data is not None and hasattr(frame_data, 'ndim') and frame_data.ndim == 1:
            frame_2d = frame_data[:, np.newaxis]
            self.lfp_event_img.setImage(frame_2d, levels=(0, self.data.lfp_vmax), autoLevels=False)
            h, w = frame_2d.shape
            self.lfp_event_img.setRect(QRectF(0, 0, w, h))
            self.lfp_event_plot.autoRange()
        else:
            self.lfp_event_img.clear()

    def apply_spike_update(self, scatter_data, region):
        start, end = region
        self.spike_std_plot.setXRange(start, end, padding=0)
        self.spike_std_plot.setYRange(0, len(self.data.sorted_units), padding=0)
        if scatter_data and scatter_data['x'].size > 0:
            self.spike_std_plot_item.setData(x=scatter_data['x'], y=scatter_data['y'], pen=scatter_data['pen'])
        else:
            self.spike_std_plot_item.clear()

    def set_region_width(self, region, width_box, mode):
        start, _ = region.getRegion()
        try:
            width = float(width_box.text())
            if width > 0:
                region.setRegion([start, start + width])
                # Manually trigger the change handler after width change
                self.handle_region_change(mode)
        except ValueError:
            pass

    def update_playback(self, region, mode):
        is_dragging = self.lfp_is_dragging if mode == 'lfp' else self.spike_is_dragging
        if not is_dragging:
            if mode == 'lfp':
                time_step = 1.0 / self.data.lfp_meta['frame_rate']
                max_time = self.data.lfp_time[-1]
                start, end = region.getRegion()
                width = end - start
                new_start = start + time_step
                if new_start + width > max_time:
                    new_start = 0
                region.setRegion([new_start, new_start + width])
            elif mode == 'spike':
                start, end = region.getRegion()
                width = end - start
                frame_rate = self.data.spike_meta['frame_params']['fps']
                time_step = 1.0 / frame_rate
                max_time = self.data.spike_meta['total_duration_s']

                new_start = start + time_step
                if new_start + width > max_time:
                    new_start = 0
                region.setRegion([new_start, new_start + width])

                # Directly calculate frame index from the region's start time
                actual_frame_idx = int(new_start * frame_rate)
                self.spike_event_view.update_frame(actual_frame_idx)
    
    def closeEvent(self, event):
        print("Closing application and stopping threads...")
        self.thread.quit()
        self.thread.wait()
        event.accept()

if __name__ == '__main__':
    pg.setConfigOptions(imageAxisOrder='row-major')
    app = QApplication(sys.argv)
    data_manager = DataManager()
    main_win = DashboardApp(data_manager)
    main_win.show()
    sys.exit(app.exec_())
