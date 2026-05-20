"""Analytics visualization and plot generation - Clean white theme."""

import json
import os
import logging
from typing import List, Optional
from dataclasses import asdict
from collections import deque

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as _plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from .models import FrameMetrics, BioFrame, MATSummary
from .math_utils import clean_nans
from .types import ExportFormat, benchmark_method, PerformanceTimer
from .cv_wrapper import cv2

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# CLEAN WHITE THEME - Professional, crisp, publication-ready
# ──────────────────────────────────────────────────────────────────────────────

class PlotColors:
    """Clean white theme for professional publications."""
    
    # Enable colorblind-friendly mode via environment variable
    COLORBLIND_MODE = os.getenv("ANALYTICS_COLORBLIND", "0") == "1"
    
    if COLORBLIND_MODE:
        # Colorblind-friendly palette (Okabe-Ito)
        PRIMARY_BLUE = "#0072B2"      # Blue
        PRIMARY_ORANGE = "#D55E00"    # Vermilion
        PRIMARY_GREEN = "#009E73"     # Bluish green
        PRIMARY_RED = "#CC79A7"       # Reddish purple
        COLOR_LEFT = "#E69F00"        # Orange
        COLOR_RIGHT = "#56B4E9"       # Sky blue
        ACCENT_WARNING = "#D55E00"    # Vermilion
        ACCENT_ALERT = "#E69F00"      # Orange
        ACCENT_SUCCESS = "#009E73"    # Bluish green
    else:
        # Standard palette
        PRIMARY_BLUE = "#2E86AB"      # Calm blue for primary data
        PRIMARY_ORANGE = "#F18F01"    # Warm orange for secondary data
        PRIMARY_GREEN = "#14A085"     # Mint green for positive metrics
        PRIMARY_RED = "#D64933"       # Clean red for warnings/alerts
        COLOR_LEFT = "#E66A2C"        # Terracotta orange - left side
        COLOR_RIGHT = "#2C7DA0"       # Ocean blue - right side
        ACCENT_WARNING = "#E63946"    # Coral red for thresholds
        ACCENT_ALERT = "#F4A261"      # Warm orange for attention
        ACCENT_SUCCESS = "#2A9D8F"    # Teal for positive metrics
    
    # Background & base
    BACKGROUND = "#FFFFFF"            # Pure white background
    TEXT = "#2C3E50"                 # Dark slate for text
    GRID = "#E8ECEF"                 # Very light gray for grid lines
    
    # Opacity
    FILL_ALPHA = 0.12
    GRID_ALPHA = 0.5
    REF_ALPHA = 0.6


class PlotStyle:
    """Professional typography and styling configuration."""
    
    # Font configuration - use system-safe fonts to avoid findfont warnings
    FONT_FAMILY = ["Segoe UI", "Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
    TITLE_SIZE = 12
    SUBTITLE_SIZE = 10
    LABEL_SIZE = 10
    LEGEND_SIZE = 9
    TICK_SIZE = 8
    
    # Line styling - extra smooth
    LINE_WIDTH_MAIN = 1.8
    LINE_WIDTH_SECONDARY = 1.2
    LINE_WIDTH_REF = 1.0
    LINE_STYLE_REF = "--"
    
    # Figure sizing
    FIG_WIDTH_SINGLE = 12
    FIG_HEIGHT_SINGLE = 4
    FIG_HEIGHT_DOUBLE = 5
    FIG_HEIGHT_TRIPLE = 8
    DPI_SAVE = 300
    
    @classmethod
    def apply_global_style(cls):
        """Apply clean white theme to all matplotlib plots."""
        if not HAS_MPL:
            return
        import matplotlib
        import matplotlib.pyplot as plt

        # Silence persistent 'findfont: Font family not found' warnings
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
        
        # Set font defaults
        matplotlib.rcParams['font.family'] = cls.FONT_FAMILY
        matplotlib.rcParams['font.size'] = cls.TICK_SIZE
        
        # Title and labels
        matplotlib.rcParams['axes.titlesize'] = cls.TITLE_SIZE
        matplotlib.rcParams['axes.labelsize'] = cls.LABEL_SIZE
        matplotlib.rcParams['axes.titleweight'] = 'normal'
        matplotlib.rcParams['axes.labelweight'] = 'normal'
        
        # Legend
        matplotlib.rcParams['legend.fontsize'] = cls.LEGEND_SIZE
        matplotlib.rcParams['legend.frameon'] = False
        
        # Figure defaults
        matplotlib.rcParams['figure.dpi'] = 100
        matplotlib.rcParams['savefig.dpi'] = cls.DPI_SAVE
        matplotlib.rcParams['savefig.bbox'] = 'tight'
        matplotlib.rcParams['savefig.facecolor'] = PlotColors.BACKGROUND
        
        # Lines - smooth
        matplotlib.rcParams['lines.linewidth'] = cls.LINE_WIDTH_MAIN
        matplotlib.rcParams['lines.antialiased'] = True
        
        # Grid - subtle
        matplotlib.rcParams['grid.linewidth'] = 0.8
        matplotlib.rcParams['grid.linestyle'] = '--'
        matplotlib.rcParams['grid.alpha'] = 0.4


class AnalyticsPlotter:
    """
    Professional plot generator with clean white theme.
    All plots saved as 300 DPI PNG.
    """

    def __init__(self, results_dir: str, player_id: int = 1):
        self.results_dir = results_dir
        self.player_id = player_id
        os.makedirs(results_dir, exist_ok=True)
        if HAS_MPL:
            import matplotlib
            try:
                matplotlib.use("Agg")
            except Exception:
                pass
            PlotStyle.apply_global_style()

    def _save(self, fig, name: str):
        """Save figure as a single PNG (300 DPI)."""
        import matplotlib.pyplot as _plt
        base = os.path.join(self.results_dir, name)
        fig.savefig(base + ".png", dpi=PlotStyle.DPI_SAVE, bbox_inches="tight", 
                    facecolor=PlotColors.BACKGROUND, edgecolor='none')
        _plt.close(fig)
        logger.info(f"[PLOT] Saved → {base}.png")

    def _setup_figure(self, figsize: tuple, title: str = None):
        """Create clean white figure with consistent styling."""
        import matplotlib.pyplot as _plt
        fig, ax = _plt.subplots(figsize=figsize, facecolor=PlotColors.BACKGROUND)
        if title:
            fig.suptitle(title, fontsize=PlotStyle.TITLE_SIZE, 
                        color=PlotColors.TEXT, ha='center', y=0.98)
        ax.set_facecolor(PlotColors.BACKGROUND)
        ax.tick_params(colors=PlotColors.TEXT, labelsize=PlotStyle.TICK_SIZE)
        # Clean spines
        for spine in ax.spines.values():
            spine.set_color(PlotColors.TEXT)
            spine.set_linewidth(0.8)
        return fig, ax

    def _style_ax(self, ax, xlabel: str = None, ylabel: str = None, grid: bool = True):
        """Apply clean styling to axes."""
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=PlotStyle.LABEL_SIZE, 
                         color=PlotColors.TEXT, labelpad=8)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=PlotStyle.LABEL_SIZE, 
                         color=PlotColors.TEXT, labelpad=8)
        if grid:
            ax.grid(True, alpha=PlotColors.GRID_ALPHA, linestyle='--', linewidth=0.6)
        ax.set_facecolor(PlotColors.BACKGROUND)

    def plot_speed_profile(self, frame_metrics: List[FrameMetrics]):
        """Plot speed and acceleration over time."""
        if not frame_metrics or not HAS_MPL:
            return
        import matplotlib.pyplot as _plt
        
        ts = [f.timestamp for f in frame_metrics]
        speed = [f.speed for f in frame_metrics]
        accel = [f.acceleration for f in frame_metrics]

        fig = _plt.figure(figsize=(PlotStyle.FIG_WIDTH_SINGLE, PlotStyle.FIG_HEIGHT_DOUBLE),
                          facecolor=PlotColors.BACKGROUND)
        gs = fig.add_gridspec(2, 1, hspace=0.12, height_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        # Speed plot
        ax1.plot(ts, speed, color=PlotColors.PRIMARY_BLUE, 
                linewidth=PlotStyle.LINE_WIDTH_MAIN, label="Speed", solid_capstyle='round')
        ax1.fill_between(ts, speed, alpha=PlotColors.FILL_ALPHA, color=PlotColors.PRIMARY_BLUE)
        ax1.set_ylabel("Speed (m/s)")
        ax1.legend(loc="upper right", fontsize=PlotStyle.LEGEND_SIZE)
        self._style_ax(ax1)

        # Acceleration plot
        ax2.plot(ts, accel, color=PlotColors.ACCENT_WARNING,
                linewidth=PlotStyle.LINE_WIDTH_MAIN, label="Acceleration", solid_capstyle='round')
        ax2.axhline(0, color=PlotColors.TEXT, linewidth=PlotStyle.LINE_WIDTH_REF,
                   linestyle='-', alpha=0.3)
        ax2.fill_between(ts, 0, accel, alpha=PlotColors.FILL_ALPHA, 
                        color=PlotColors.ACCENT_WARNING)
        ax2.set_ylabel("Acceleration (m/s²)")
        ax2.set_xlabel("Time (s)")
        ax2.legend(loc="upper right", fontsize=PlotStyle.LEGEND_SIZE)
        self._style_ax(ax2)

        _plt.tight_layout()
        self._save(fig, "speed_acceleration_profile")

    def plot_joint_angles(self, frame_metrics: List[FrameMetrics]):
        """Plot knee, hip, and trunk angles over time."""
        if not frame_metrics or not HAS_MPL:
            return
        import matplotlib.pyplot as _plt
        
        ts = [f.timestamp for f in frame_metrics]
        lk = [f.left_knee_angle for f in frame_metrics]
        rk = [f.right_knee_angle for f in frame_metrics]
        lh = [f.left_hip_angle for f in frame_metrics]
        rh = [f.right_hip_angle for f in frame_metrics]
        trl = [getattr(f, 'trunk_lateral_lean', getattr(f, 'trunk_lean', 0)) for f in frame_metrics]

        fig = _plt.figure(figsize=(PlotStyle.FIG_WIDTH_SINGLE, PlotStyle.FIG_HEIGHT_TRIPLE),
                          facecolor=PlotColors.BACKGROUND)
        gs = fig.add_gridspec(3, 1, hspace=0.15)
        ax1, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

        fig.suptitle(f"Player #{self.player_id} — Joint Angle Timeseries", 
                    fontsize=PlotStyle.TITLE_SIZE, color=PlotColors.TEXT)

        # Knee flexion
        ax1.plot(ts, lk, color=PlotColors.COLOR_LEFT, linewidth=PlotStyle.LINE_WIDTH_MAIN,
                label="Left Knee", solid_capstyle='round')
        ax1.plot(ts, rk, color=PlotColors.COLOR_RIGHT, linewidth=PlotStyle.LINE_WIDTH_MAIN,
                label="Right Knee", solid_capstyle='round')
        ax1.axhline(120, color=PlotColors.ACCENT_WARNING, linewidth=PlotStyle.LINE_WIDTH_REF,
                   linestyle='--', alpha=PlotColors.REF_ALPHA, label="Risk threshold (120°)")
        ax1.set_ylabel("Knee Flexion (°)")
        ax1.legend(loc="upper right", fontsize=PlotStyle.LEGEND_SIZE)
        self._style_ax(ax1)

        # Hip flexion
        ax2.plot(ts, lh, color=PlotColors.COLOR_LEFT, linewidth=PlotStyle.LINE_WIDTH_MAIN,
                label="Left Hip", solid_capstyle='round')
        ax2.plot(ts, rh, color=PlotColors.COLOR_RIGHT, linewidth=PlotStyle.LINE_WIDTH_MAIN,
                label="Right Hip", solid_capstyle='round')
        ax2.set_ylabel("Hip Flexion (°)")
        ax2.legend(loc="upper right", fontsize=PlotStyle.LEGEND_SIZE)
        self._style_ax(ax2)

        # Trunk lean
        ax3.plot(ts, trl, color=PlotColors.PRIMARY_BLUE, linewidth=PlotStyle.LINE_WIDTH_MAIN,
                label="Trunk Lateral Lean", solid_capstyle='round')
        ax3.axhline(0, color=PlotColors.TEXT, linewidth=PlotStyle.LINE_WIDTH_REF,
                   linestyle='-', alpha=0.3)
        ax3.axhline(15, color=PlotColors.ACCENT_ALERT, linewidth=PlotStyle.LINE_WIDTH_REF,
                   linestyle='--', alpha=PlotColors.REF_ALPHA, label="Moderate (15°)")
        ax3.axhline(25, color=PlotColors.ACCENT_WARNING, linewidth=PlotStyle.LINE_WIDTH_REF,
                   linestyle='--', alpha=PlotColors.REF_ALPHA, label="High (25°)")
        ax3.set_ylabel("Trunk Lean (°)")
        ax3.set_xlabel("Time (s)")
        ax3.legend(loc="upper right", fontsize=PlotStyle.LEGEND_SIZE)
        self._style_ax(ax3)

        _plt.tight_layout()
        self._save(fig, "joint_angles_timeseries")

    def plot_biomechanics(self, bio_engine: "BiomechanicsEngine"):
        """Plot comprehensive biomechanics data."""
        if not bio_engine or not bio_engine.frames or not HAS_MPL:
            return
        import matplotlib.pyplot as _plt
        
        frames = bio_engine.frames
        ts = [f.timestamp for f in frames]

        # ── Knee flexion ──────────────────────────────────────────────────────
        fig, ax = self._setup_figure((PlotStyle.FIG_WIDTH_SINGLE, PlotStyle.FIG_HEIGHT_SINGLE),
                                      f"Player #{self.player_id} — Knee Flexion")
        ax.plot(ts, [f.left_knee_flexion for f in frames], color=PlotColors.COLOR_LEFT,
                linewidth=PlotStyle.LINE_WIDTH_MAIN, label="Left Knee", solid_capstyle='round')
        ax.plot(ts, [f.right_knee_flexion for f in frames], color=PlotColors.COLOR_RIGHT,
                linewidth=PlotStyle.LINE_WIDTH_MAIN, label="Right Knee", solid_capstyle='round')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (°)")
        ax.legend(loc="upper right", fontsize=PlotStyle.LEGEND_SIZE)
        self._style_ax(ax)
        _plt.tight_layout()
        self._save(fig, "knee_flexion")

        # ── Valgus (clinical) ────────────────────────────────────────────────
        fig, ax = self._setup_figure((PlotStyle.FIG_WIDTH_SINGLE, PlotStyle.FIG_HEIGHT_SINGLE),
                                      f"Player #{self.player_id} — Clinical Knee Valgus/Varus")
        ax.plot(ts, [f.left_valgus_clinical for f in frames], color=PlotColors.COLOR_LEFT,
                linewidth=PlotStyle.LINE_WIDTH_MAIN, label="Left Valgus", solid_capstyle='round')
        ax.plot(ts, [f.right_valgus_clinical for f in frames], color=PlotColors.COLOR_RIGHT,
                linewidth=PlotStyle.LINE_WIDTH_MAIN, label="Right Valgus", solid_capstyle='round')
        
        ax.axhline(10, color=PlotColors.ACCENT_WARNING, linewidth=PlotStyle.LINE_WIDTH_REF,
                   linestyle='--', alpha=PlotColors.REF_ALPHA, label="±10° risk zone")
        ax.axhline(-10, color=PlotColors.ACCENT_WARNING, linewidth=PlotStyle.LINE_WIDTH_REF,
                   linestyle='--', alpha=PlotColors.REF_ALPHA)
        ax.axhline(0, color=PlotColors.TEXT, linewidth=0.8, alpha=0.3)
        
        ax.fill_between(ts, 10, -10, alpha=0.05, color=PlotColors.ACCENT_WARNING)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (°)")
        ax.legend(loc="upper right", fontsize=PlotStyle.LEGEND_SIZE)
        self._style_ax(ax)
        _plt.tight_layout()
        self._save(fig, "clinical_valgus")

        # ── Hip & ankle ───────────────────────────────────────────────────────
        fig = _plt.figure(figsize=(PlotStyle.FIG_WIDTH_SINGLE, PlotStyle.FIG_HEIGHT_DOUBLE),
                          facecolor=PlotColors.BACKGROUND)
        gs = fig.add_gridspec(2, 1, hspace=0.15)
        ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        fig.suptitle(f"Player #{self.player_id} — Hip & Ankle Kinematics",
                    fontsize=PlotStyle.TITLE_SIZE, color=PlotColors.TEXT)

        ax1.plot(ts, [f.left_hip_flexion for f in frames], color=PlotColors.COLOR_LEFT,
                linewidth=PlotStyle.LINE_WIDTH_MAIN, label="Left Hip", solid_capstyle='round')
        ax1.plot(ts, [f.right_hip_flexion for f in frames], color=PlotColors.COLOR_RIGHT,
                linewidth=PlotStyle.LINE_WIDTH_MAIN, label="Right Hip", solid_capstyle='round')
        ax1.set_ylabel("Hip Flexion (°)")
        ax1.legend(loc="upper right", fontsize=PlotStyle.LEGEND_SIZE)
        self._style_ax(ax1)

        ax2.plot(ts, [f.left_ankle_dorsiflexion for f in frames], color=PlotColors.COLOR_LEFT,
                linewidth=PlotStyle.LINE_WIDTH_MAIN, label="Left Ankle", solid_capstyle='round')
        ax2.plot(ts, [f.right_ankle_dorsiflexion for f in frames], color=PlotColors.COLOR_RIGHT,
                linewidth=PlotStyle.LINE_WIDTH_MAIN, label="Right Ankle", solid_capstyle='round')
        ax2.set_ylabel("Ankle Dorsiflexion (°)")
        ax2.set_xlabel("Time (s)")
        ax2.legend(loc="upper right", fontsize=PlotStyle.LEGEND_SIZE)
        self._style_ax(ax2)

        _plt.tight_layout()
        self._save(fig, "hip_ankle_kinematics")

        # ── Angular velocities ────────────────────────────────────────────────
        fig, ax = self._setup_figure((PlotStyle.FIG_WIDTH_SINGLE, PlotStyle.FIG_HEIGHT_SINGLE),
                                      f"Player #{self.player_id} — Joint Angular Velocities")
        ax.plot(ts, [f.left_knee_ang_vel for f in frames], color=PlotColors.COLOR_LEFT,
                linewidth=PlotStyle.LINE_WIDTH_MAIN, label="Left Knee", solid_capstyle='round')
        ax.plot(ts, [f.right_knee_ang_vel for f in frames], color=PlotColors.COLOR_RIGHT,
                linewidth=PlotStyle.LINE_WIDTH_MAIN, label="Right Knee", solid_capstyle='round')
        ax.plot(ts, [f.left_hip_ang_vel for f in frames], color=PlotColors.PRIMARY_GREEN,
                linewidth=PlotStyle.LINE_WIDTH_SECONDARY, linestyle=':', label="Left Hip")
        ax.plot(ts, [f.right_hip_ang_vel for f in frames], color=PlotColors.PRIMARY_BLUE,
                linewidth=PlotStyle.LINE_WIDTH_SECONDARY, linestyle=':', label="Right Hip")
        ax.axhline(0, color=PlotColors.TEXT, linewidth=PlotStyle.LINE_WIDTH_REF,
                   linestyle='-', alpha=0.3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angular Velocity (°/s)")
        ax.legend(loc="upper right", fontsize=PlotStyle.LEGEND_SIZE, ncol=2)
        self._style_ax(ax)
        _plt.tight_layout()
        self._save(fig, "angular_velocities")

        # ── Gait events ───────────────────────────────────────────────────────
        fig, ax = self._setup_figure((PlotStyle.FIG_WIDTH_SINGLE, 3),
                                      f"Player #{self.player_id} — Gait Events")
        
        lhs_ts = [frames[i].timestamp for i in bio_engine.lhs if i < len(frames)]
        rhs_ts = [frames[i].timestamp for i in bio_engine.rhs if i < len(frames)]
        lto_ts = [frames[i].timestamp for i in bio_engine.lto if i < len(frames)]
        rto_ts = [frames[i].timestamp for i in bio_engine.rto if i < len(frames)]
        
        for t in lhs_ts:
            ax.axvline(t, color=PlotColors.COLOR_LEFT, linewidth=1.2, alpha=0.7,
                       label="L Heel Strike" if t == lhs_ts[0] else "")
        for t in rhs_ts:
            ax.axvline(t, color=PlotColors.COLOR_RIGHT, linewidth=1.2, alpha=0.7,
                       label="R Heel Strike" if t == rhs_ts[0] else "")
        for t in lto_ts:
            ax.axvline(t, color=PlotColors.COLOR_LEFT, linewidth=0.8, linestyle="--", alpha=0.5,
                       label="L Toe Off" if t == lto_ts[0] else "")
        for t in rto_ts:
            ax.axvline(t, color=PlotColors.COLOR_RIGHT, linewidth=0.8, linestyle="--", alpha=0.5,
                       label="R Toe Off" if t == rto_ts[0] else "")
        
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        
        handles, labels = ax.get_legend_handles_labels()
        unique = {}
        for h, l in zip(handles, labels):
            if l not in unique:
                unique[l] = h
        ax.legend(unique.values(), unique.keys(), loc="upper right",
                  fontsize=PlotStyle.LEGEND_SIZE, ncol=2)
        
        self._style_ax(ax)
        _plt.tight_layout()
        self._save(fig, "gait_events")

        # ── Arm swing ─────────────────────────────────────────────────────────
        fig, ax = self._setup_figure((PlotStyle.FIG_WIDTH_SINGLE, PlotStyle.FIG_HEIGHT_SINGLE),
                                      f"Player #{self.player_id} — Arm Swing Excursion")
        ax.plot(ts, [f.left_arm_swing for f in frames], color=PlotColors.COLOR_LEFT,
                linewidth=PlotStyle.LINE_WIDTH_MAIN, label="Left Arm", solid_capstyle='round')
        ax.plot(ts, [f.right_arm_swing for f in frames], color=PlotColors.COLOR_RIGHT,
                linewidth=PlotStyle.LINE_WIDTH_MAIN, label="Right Arm", solid_capstyle='round')
        ax.plot(ts, [f.arm_swing_asymmetry for f in frames], color=PlotColors.ACCENT_WARNING,
                linewidth=PlotStyle.LINE_WIDTH_SECONDARY, linestyle='--', label="Asymmetry")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (°)")
        ax.legend(loc="upper right", fontsize=PlotStyle.LEGEND_SIZE)
        self._style_ax(ax)
        _plt.tight_layout()
        self._save(fig, "arm_swing")

    def plot_risk_scores(self, frame_metrics: List[FrameMetrics]):
        """Plot risk scores over time."""
        if not frame_metrics or not HAS_MPL:
            return
        import matplotlib.pyplot as _plt
        
        ts = [f.timestamp for f in frame_metrics]
        risk = [f.risk_score for f in frame_metrics]
        inj = [f.injury_risk for f in frame_metrics]
        joint_s = [f.joint_stress for f in frame_metrics]
        fatigue = [f.fatigue_index for f in frame_metrics]

        fig = _plt.figure(figsize=(PlotStyle.FIG_WIDTH_SINGLE, PlotStyle.FIG_HEIGHT_DOUBLE),
                          facecolor=PlotColors.BACKGROUND)
        gs = fig.add_gridspec(2, 1, hspace=0.15)
        ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

        fig.suptitle(f"Player #{self.player_id} — Risk Indicators",
                    fontsize=PlotStyle.TITLE_SIZE, color=PlotColors.TEXT)

        # Composite risk score
        ax1.plot(ts, risk, color=PlotColors.ACCENT_WARNING, linewidth=PlotStyle.LINE_WIDTH_MAIN,
                label="Composite Risk Score", solid_capstyle='round')
        ax1.fill_between(ts, risk, alpha=PlotColors.FILL_ALPHA, color=PlotColors.ACCENT_WARNING)
        
        ax1.axhline(25, color=PlotColors.ACCENT_SUCCESS, linewidth=PlotStyle.LINE_WIDTH_REF,
                   linestyle='--', alpha=PlotColors.REF_ALPHA, label="Low (<25)")
        ax1.axhline(50, color=PlotColors.ACCENT_ALERT, linewidth=PlotStyle.LINE_WIDTH_REF,
                   linestyle='--', alpha=PlotColors.REF_ALPHA, label="Moderate (25–50)")
        ax1.axhline(75, color=PlotColors.ACCENT_WARNING, linewidth=PlotStyle.LINE_WIDTH_REF,
                   linestyle='--', alpha=PlotColors.REF_ALPHA, label="High (>50)")
        
        ax1.fill_between(ts, 0, 25, alpha=0.04, color=PlotColors.ACCENT_SUCCESS)
        ax1.fill_between(ts, 25, 50, alpha=0.04, color=PlotColors.ACCENT_ALERT)
        ax1.fill_between(ts, 50, 100, alpha=0.06, color=PlotColors.ACCENT_WARNING)
        
        ax1.set_ylabel("Risk Score (0–100)")
        ax1.set_ylim(0, 105)
        ax1.legend(loc="upper left", fontsize=PlotStyle.LEGEND_SIZE)
        self._style_ax(ax1)

        # Sub-scores
        ax2.plot(ts, [v * 100 for v in inj], color=PlotColors.ACCENT_WARNING,
                linewidth=PlotStyle.LINE_WIDTH_MAIN, label="Acute Injury Risk", solid_capstyle='round')
        ax2.plot(ts, [v * 100 for v in joint_s], color=PlotColors.PRIMARY_BLUE,
                linewidth=PlotStyle.LINE_WIDTH_MAIN, label="Joint Stress", solid_capstyle='round')
        ax2.plot(ts, [v * 100 for v in fatigue], color=PlotColors.PRIMARY_GREEN,
                linewidth=PlotStyle.LINE_WIDTH_MAIN, label="Fatigue Index", solid_capstyle='round')
        ax2.set_ylabel("Sub-scores (%)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylim(0, 105)
        ax2.legend(loc="upper right", fontsize=PlotStyle.LEGEND_SIZE)
        self._style_ax(ax2)

        _plt.tight_layout()
        self._save(fig, "risk_scores")

    def plot_energy(self, frame_metrics: List[FrameMetrics]):
        """Plot energy expenditure over time."""
        if not frame_metrics or not HAS_MPL:
            return
        import matplotlib.pyplot as _plt
        
        ts = [f.timestamp for f in frame_metrics]
        energy = [f.energy_expenditure for f in frame_metrics]
        
        fig, ax = self._setup_figure((PlotStyle.FIG_WIDTH_SINGLE, PlotStyle.FIG_HEIGHT_SINGLE),
                                      f"Player #{self.player_id} — Energy Expenditure")
        ax.plot(ts, energy, color=PlotColors.PRIMARY_ORANGE, linewidth=PlotStyle.LINE_WIDTH_MAIN,
                label="Energy (kcal/min)", solid_capstyle='round')
        ax.fill_between(ts, energy, alpha=PlotColors.FILL_ALPHA, color=PlotColors.PRIMARY_ORANGE)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Energy Expenditure (kcal/min)")
        ax.legend(loc="upper right", fontsize=PlotStyle.LEGEND_SIZE)
        self._style_ax(ax)
        _plt.tight_layout()
        self._save(fig, "energy_expenditure")

        fig, ax = self._setup_figure((PlotStyle.FIG_WIDTH_SINGLE, PlotStyle.FIG_HEIGHT_SINGLE),
                                      f"Player #{self.player_id} — Metabolic Power")
        
        ax.plot(ts, energy, color=PlotColors.ACCENT_ALERT, linewidth=PlotStyle.LINE_WIDTH_MAIN,
               label="Metabolic Power", solid_capstyle='round')
        ax.fill_between(ts, energy, alpha=PlotColors.FILL_ALPHA, color=PlotColors.ACCENT_ALERT)
        
        avg_power = np.mean(energy) if energy else 0
        ax.axhline(avg_power, color=PlotColors.PRIMARY_BLUE, linewidth=PlotStyle.LINE_WIDTH_REF,
                   linestyle=':', alpha=PlotColors.REF_ALPHA, label=f"Mean: {avg_power:.0f} W")
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power (W)")
        ax.legend(loc="upper right", fontsize=PlotStyle.LEGEND_SIZE)
        self._style_ax(ax)
        _plt.tight_layout()
        self._save(fig, "metabolic_power")

    def plot_mat_results(self, mat_summary: "MATSummary"):
        """Plot MAT (Movement Assessment Tool) results dashboard."""
        if not mat_summary or not mat_summary.events or not HAS_MPL:
            return
        import matplotlib.pyplot as _plt
        
        n_events = len(mat_summary.events)
        
        # Create dashboard with subplots for each event
        fig, axes = _plt.subplots(n_events, 3, 
                                   figsize=(PlotStyle.FIG_WIDTH_SINGLE, 
                                           PlotStyle.FIG_HEIGHT_DOUBLE * max(1, n_events//2)),
                                   facecolor=PlotColors.BACKGROUND)
        
        # Handle single event case
        if n_events == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f"MAT Analysis — {mat_summary.protocol_id.replace('_', ' ').upper()}", 
                    fontsize=PlotStyle.TITLE_SIZE + 2, color=PlotColors.TEXT, y=1.02)
        
        for idx, event in enumerate(mat_summary.events):
            ax1, ax2, ax3 = axes[idx]
            
            # 1. Landing Valgus
            valgus_val = event.landing_valgus_left
            color = PlotColors.PRIMARY_RED if abs(valgus_val) > 10 else PlotColors.PRIMARY_GREEN
            ax1.bar(['Valgus'], [valgus_val], color=color, width=0.5, alpha=0.8)
            ax1.axhline(10, color=PlotColors.ACCENT_WARNING, linestyle='--', linewidth=1, alpha=0.6)
            ax1.set_ylabel("Degrees (°)")
            ax1.set_title(f"Event #{idx+1}: Knee Stability", fontsize=PlotStyle.SUBTITLE_SIZE)
            self._style_ax(ax1, grid=False)
            ax1.set_ylim(0, max(15, valgus_val + 5))
            
            # 2. Peak Flexion
            bend_val = 180 - event.peak_knee_flexion_landing
            ax2.bar(['Flexion'], [bend_val], color=PlotColors.PRIMARY_BLUE, width=0.5, alpha=0.8)
            ax2.set_ylim(0, 90)
            ax2.set_ylabel("Degrees of Bend (°)")
            ax2.set_title("Impact Absorption", fontsize=PlotStyle.SUBTITLE_SIZE)
            self._style_ax(ax2, grid=False)
            
            # 3. Flight Time & TTS
            metrics = ['Flight Time', 'TTS']
            m_vals = [event.flight_time, event.time_to_stabilization]
            ax3.bar(metrics, m_vals, color=PlotColors.PRIMARY_ORANGE, width=0.5, alpha=0.8)
            ax3.set_ylim(0, max(1.2, max(m_vals) * 1.2))
            ax3.set_ylabel("Time (s)")
            ax3.set_title("Dynamic Balance", fontsize=PlotStyle.SUBTITLE_SIZE)
            self._style_ax(ax3, grid=False)
        
        _plt.tight_layout()
        self._save(fig, "mat_performance_dashboard")

    def plot_gait_cycle(self, frame_metrics: List[FrameMetrics], bio_engine: "BiomechanicsEngine"):
        """Plot normalized gait cycle for left and right legs."""
        if not frame_metrics or not bio_engine or not HAS_MPL:
            return
        import matplotlib.pyplot as _plt
        
        # Extract gait cycles
        lhs_indices = bio_engine.lhs
        if len(lhs_indices) < 2:
            return
        
        # Get knee angles for full cycles
        knee_angles_left = [f.left_knee_angle for f in frame_metrics]
        knee_angles_right = [f.right_knee_angle for f in frame_metrics]
        
        # Interpolate to 100 points per cycle
        cycles_left = []
        cycles_right = []
        
        for start, end in zip(lhs_indices[:-1], lhs_indices[1:]):
            if end - start > 10:  # Valid cycle
                cycle_left = knee_angles_left[start:end]
                cycle_right = knee_angles_right[start:end]
                
                # Interpolate to 100 points
                x_old = np.linspace(0, 100, len(cycle_left))
                x_new = np.linspace(0, 100, 100)
                cycles_left.append(np.interp(x_new, x_old, cycle_left))
                cycles_right.append(np.interp(x_new, x_old, cycle_right))
        
        if not cycles_left:
            return
        
        # Calculate mean and std
        mean_left = np.mean(cycles_left, axis=0)
        std_left = np.std(cycles_left, axis=0)
        mean_right = np.mean(cycles_right, axis=0)
        std_right = np.std(cycles_right, axis=0)
        
        fig, ax = self._setup_figure((PlotStyle.FIG_WIDTH_SINGLE, PlotStyle.FIG_HEIGHT_SINGLE),
                                      f"Player #{self.player_id} — Gait Cycle (Normalized)")
        
        x = np.linspace(0, 100, 100)
        ax.plot(x, mean_left, color=PlotColors.COLOR_LEFT, linewidth=PlotStyle.LINE_WIDTH_MAIN,
                label="Left Knee")
        ax.fill_between(x, mean_left - std_left, mean_left + std_left,
                        alpha=PlotColors.FILL_ALPHA, color=PlotColors.COLOR_LEFT)
        
        ax.plot(x, mean_right, color=PlotColors.COLOR_RIGHT, linewidth=PlotStyle.LINE_WIDTH_MAIN,
                label="Right Knee")
        ax.fill_between(x, mean_right - std_right, mean_right + std_right,
                        alpha=PlotColors.FILL_ALPHA, color=PlotColors.COLOR_RIGHT)
        
        ax.set_xlabel("Gait Cycle (%)")
        ax.set_ylabel("Knee Flexion (°)")
        ax.legend(loc="upper right", fontsize=PlotStyle.LEGEND_SIZE)
        self._style_ax(ax)
        _plt.tight_layout()
        self._save(fig, "gait_cycle_normalized")

    def plot_mat_radar(self, mat_summary: "MATSummary"):
        """Create radar chart for MAT performance metrics."""
        if not mat_summary or not mat_summary.events or not HAS_MPL:
            return
        
        import matplotlib.pyplot as _plt
        import numpy as np
        
        # Aggregate metrics across events
        n_events = len(mat_summary.events)
        metrics = {
            "Valgus\nControl": np.mean([abs(e.landing_valgus_left) for e in mat_summary.events]),
            "Impact\nAbsorption": np.mean([180 - e.peak_knee_flexion_landing for e in mat_summary.events]),
            "Flight\nEfficiency": np.mean([e.flight_time for e in mat_summary.events]),
            "Stabilization": np.mean([e.time_to_stabilization for e in mat_summary.events]),
            "LSI": mat_summary.limb_symmetry_index,
        }
        
        # Normalize metrics
        max_vals = {"Valgus\nControl": 15, "Impact\nAbsorption": 90, 
                    "Flight\nEfficiency": 1.0, "Stabilization": 1.0, "LSI": 100}
        normalized = {k: min(1.0, metrics[k] / max_vals[k]) for k in metrics}
        
        # Create radar chart
        categories = list(normalized.keys())
        values = list(normalized.values())
        values += values[:1]  # Close the loop
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = _plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'},
                                facecolor=PlotColors.BACKGROUND)
        
        ax.plot(angles, values, 'o-', linewidth=PlotStyle.LINE_WIDTH_MAIN, 
                color=PlotColors.PRIMARY_BLUE)
        ax.fill(angles, values, alpha=PlotColors.FILL_ALPHA, color=PlotColors.PRIMARY_BLUE)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=PlotStyle.LABEL_SIZE)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=PlotStyle.TICK_SIZE)
        
        ax.set_title(f"MAT Performance — {mat_summary.protocol_id}", 
                    fontsize=PlotStyle.TITLE_SIZE, color=PlotColors.TEXT, pad=20)
        
        _plt.tight_layout()
        self._save(fig, "mat_radar_chart")

    def plot_video_overlay(self, frame: np.ndarray, metrics: FrameMetrics, 
                           kp: "PoseKeypoints", save_name: str = "overlay"):
        """Create annotated frame overlay with metrics."""
        if not HAS_MPL:
            return
        
        import matplotlib.pyplot as _plt
        
        fig, (ax_img, ax_metrics) = _plt.subplots(1, 2, 
                                                    figsize=(14, 5),
                                                    facecolor=PlotColors.BACKGROUND)
        
        # Display frame
        ax_img.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax_img.set_title(f"Frame at {metrics.timestamp:.2f}s", fontsize=PlotStyle.SUBTITLE_SIZE)
        ax_img.axis('off')
        
        # Display metrics as text
        metrics_text = f"""
    Speed: {metrics.speed:.2f} m/s
    Risk Score: {metrics.risk_score:.1f}
    Knee L/R: {metrics.left_knee_angle:.0f}° / {metrics.right_knee_angle:.0f}°
    Fatigue: {metrics.fatigue_index:.2f}
        """
        ax_metrics.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
                       transform=ax_metrics.transAxes, fontfamily='monospace')
        ax_metrics.axis('off')
        ax_metrics.set_facecolor(PlotColors.BACKGROUND)
        
        _plt.tight_layout()
        self._save(fig, save_name)

    @benchmark_method(threshold_ms=200.0)
    def generate_all(self, frame_metrics: List[FrameMetrics], bio_engine: Optional["BiomechanicsEngine"] = None,
                     mat_summary: Optional["MATSummary"] = None):
        """Generate and save all standard plots with performance monitoring."""
        logger.info("[PLOT] Visualization generation is disabled.")
        return

        # --- Preserved implementation (re-enable by removing early return above) ---
        # if not HAS_MPL:
        #     print("[PLOT] matplotlib not installed — skipping plot generation.")
        #     print("       Run: pip install matplotlib")
        #     return
        #
        # print(f"[PLOT] Generating plots with clean white theme...")
        #
        # plot_methods = [
        #     ("speed_profile", lambda: self.plot_speed_profile(frame_metrics)),
        #     ("joint_angles", lambda: self.plot_joint_angles(frame_metrics)),
        #     ("risk_scores", lambda: self.plot_risk_scores(frame_metrics)),
        #     ("energy", lambda: self.plot_energy(frame_metrics)),
        # ]
        #
        # for plot_name, plot_func in plot_methods:
        #     with PerformanceTimer(f"plot_{plot_name}") as timer:
        #         plot_func()
        #     if timer.result:
        #         logger.debug(f"📊 {timer.result}")
        #
        # if bio_engine and bio_engine.frames:
        #     with PerformanceTimer("plot_biomechanics") as timer:
        #         self.plot_biomechanics(bio_engine)
        #     if timer.result:
        #         logger.debug(f"📊 {timer.result}")
        #
        #     with PerformanceTimer("plot_gait_cycle") as timer:
        #         self.plot_gait_cycle(frame_metrics, bio_engine)
        #     if timer.result:
        #         logger.debug(f"📊 {timer.result}")
        #
        # if mat_summary and mat_summary.events:
        #     with PerformanceTimer("plot_mat_results") as timer:
        #         self.plot_mat_results(mat_summary)
        #     if timer.result:
        #         logger.debug(f"📊 {timer.result}")
        #
        #     with PerformanceTimer("plot_mat_radar") as timer:
        #         self.plot_mat_radar(mat_summary)
        #     if timer.result:
        #         logger.debug(f"📊 {timer.result}")
        #
        # print(f"[PLOT] All plots saved to: {self.results_dir}")