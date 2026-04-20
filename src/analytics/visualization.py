"""Analytics visualization and plot generation."""

from .core import *  # noqa: F401,F403

class AnalyticsPlotter:
    """
    Generates and saves high-resolution analytical plots to a /results directory.
    All plots are saved as 300 DPI PNG and SVG for publication / external use.
    """

    def __init__(self, results_dir: str, player_id: int = 1):
        self.results_dir = results_dir
        self.player_id   = player_id
        os.makedirs(results_dir, exist_ok=True)
        if HAS_MPL:
            import matplotlib
            try:
                matplotlib.use("Agg")
            except Exception:
                pass  # backend already set; acceptable if Sports2D already switched it

    def _save(self, fig, name: str):
        """Save figure as both PNG (300 DPI) and SVG."""
        import matplotlib.pyplot as _plt
        base = os.path.join(self.results_dir, name)
        fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(base + ".svg", bbox_inches="tight")
        _plt.close(fig)
        print(f"[PLOT] Saved → {base}.png / .svg")

    def plot_speed_profile(self, frame_metrics: List[FrameMetrics]):
        if not frame_metrics or not HAS_MPL:
            return
        ts    = [f.timestamp for f in frame_metrics]
        speed = [f.speed     for f in frame_metrics]
        accel = [f.acceleration for f in frame_metrics]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        fig.suptitle(f"Player #{self.player_id} — Speed & Acceleration Profile", fontsize=13)

        ax1.plot(ts, speed, color="#00C8A0", linewidth=1.5, label="Speed (m/s)")
        ax1.fill_between(ts, speed, alpha=0.15, color="#00C8A0")
        ax1.set_ylabel("Speed (m/s)")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        ax2.plot(ts, accel, color="#FF6B35", linewidth=1.2, label="Acceleration (m/s²)")
        ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax2.set_ylabel("Acceleration (m/s²)")
        ax2.set_xlabel("Time (s)")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save(fig, "speed_acceleration_profile")

    def plot_joint_angles(self, frame_metrics: List[FrameMetrics]):
        if not frame_metrics or not HAS_MPL:
            return
        ts  = [f.timestamp        for f in frame_metrics]
        lk  = [f.left_knee_angle  for f in frame_metrics]
        rk  = [f.right_knee_angle for f in frame_metrics]
        lh  = [f.left_hip_angle   for f in frame_metrics]
        rh  = [f.right_hip_angle  for f in frame_metrics]
        trl = [f.trunk_lean       for f in frame_metrics]

        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        fig.suptitle(f"Player #{self.player_id} — Joint Angle Timeseries", fontsize=13)

        axes[0].plot(ts, lk, label="Left Knee",  color="#FFB300", linewidth=1.4)
        axes[0].plot(ts, rk, label="Right Knee", color="#0088FF", linewidth=1.4)
        axes[0].set_ylabel("Knee Flexion (°)")
        axes[0].axhline(120, color="red", linewidth=0.7, linestyle="--", label="Risk threshold 120°")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ts, lh, label="Left Hip",  color="#FFB300", linewidth=1.4)
        axes[1].plot(ts, rh, label="Right Hip", color="#0088FF", linewidth=1.4)
        axes[1].set_ylabel("Hip Flexion (°)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(ts, trl, label="Trunk Lean", color="#AA44FF", linewidth=1.4)
        axes[2].set_ylabel("Trunk Lean (°)")
        axes[2].set_xlabel("Time (s)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        self._save(fig, "joint_angles_timeseries")

    def plot_biomechanics(self, bio_engine: "BiomechanicsEngine"):
        if not bio_engine or not bio_engine.frames or not HAS_MPL:
            return
        frames = bio_engine.frames
        ts     = [f.timestamp for f in frames]

        # ── Knee flexion ──────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(ts, [f.left_knee_flexion  for f in frames], label="L Knee Flexion",
                color="#FFB300", linewidth=1.4)
        ax.plot(ts, [f.right_knee_flexion for f in frames], label="R Knee Flexion",
                color="#0088FF", linewidth=1.4)
        ax.set_title(f"Player #{self.player_id} — Knee Flexion (BiomechanicsEngine)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (°)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig, "knee_flexion")

        # ── Valgus (clinical) ────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(ts, [f.left_valgus_clinical  for f in frames], label="L Valgus",
                color="#FF6B35", linewidth=1.4)
        ax.plot(ts, [f.right_valgus_clinical for f in frames], label="R Valgus",
                color="#35C2FF", linewidth=1.4)
        ax.axhline( 10, color="red",    linewidth=0.8, linestyle="--", label="±10° risk")
        ax.axhline(-10, color="red",    linewidth=0.8, linestyle="--")
        ax.axhline( 0,  color="gray",   linewidth=0.6, linestyle="-")
        ax.set_title(f"Player #{self.player_id} — Clinical Knee Valgus/Varus")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (°)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig, "clinical_valgus")

        # ── Hip & ankle ───────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        axes[0].plot(ts, [f.left_hip_flexion  for f in frames], label="L Hip", color="#FFB300", linewidth=1.4)
        axes[0].plot(ts, [f.right_hip_flexion for f in frames], label="R Hip", color="#0088FF", linewidth=1.4)
        axes[0].set_ylabel("Hip Flexion (°)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ts, [f.left_ankle_dorsiflexion  for f in frames], label="L Ankle", color="#FFB300", linewidth=1.4)
        axes[1].plot(ts, [f.right_ankle_dorsiflexion for f in frames], label="R Ankle", color="#0088FF", linewidth=1.4)
        axes[1].set_ylabel("Ankle Dorsiflexion (°)")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"Player #{self.player_id} — Hip & Ankle Kinematics")
        plt.tight_layout()
        self._save(fig, "hip_ankle_kinematics")

        # ── Angular velocities ────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(ts, [f.left_knee_ang_vel  for f in frames], label="L Knee ω",  color="#FFB300", linewidth=1.2)
        ax.plot(ts, [f.right_knee_ang_vel for f in frames], label="R Knee ω",  color="#0088FF", linewidth=1.2)
        ax.plot(ts, [f.left_hip_ang_vel   for f in frames], label="L Hip ω",   color="#FF8800", linewidth=1.0, linestyle="--")
        ax.plot(ts, [f.right_hip_ang_vel  for f in frames], label="R Hip ω",   color="#0055CC", linewidth=1.0, linestyle="--")
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.set_title(f"Player #{self.player_id} — Joint Angular Velocities")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angular Velocity (°/s)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig, "angular_velocities")

        # ── Gait events ───────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.set_title(f"Player #{self.player_id} — Gait Events (Heel Strikes & Toe Offs)")
        lhs_ts = [frames[i].timestamp for i in bio_engine.lhs if i < len(frames)]
        rhs_ts = [frames[i].timestamp for i in bio_engine.rhs if i < len(frames)]
        lto_ts = [frames[i].timestamp for i in bio_engine.lto if i < len(frames)]
        rto_ts = [frames[i].timestamp for i in bio_engine.rto if i < len(frames)]
        for t in lhs_ts:
            ax.axvline(t, color="#FFB300", linewidth=1.2, alpha=0.8, label="L Heel Strike" if t == lhs_ts[0] else "")
        for t in rhs_ts:
            ax.axvline(t, color="#0088FF", linewidth=1.2, alpha=0.8, label="R Heel Strike" if t == rhs_ts[0] else "")
        for t in lto_ts:
            ax.axvline(t, color="#FFB300", linewidth=0.8, linestyle="--", alpha=0.6, label="L Toe Off" if t == lto_ts[0] else "")
        for t in rto_ts:
            ax.axvline(t, color="#0088FF", linewidth=0.8, linestyle="--", alpha=0.6, label="R Toe Off" if t == rto_ts[0] else "")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        unique_handles, unique_labels = [], []
        for h, l in zip(handles, labels):
            if l and l not in seen:
                seen[l] = True
                unique_handles.append(h)
                unique_labels.append(l)
        ax.legend(unique_handles, unique_labels, loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        self._save(fig, "gait_events")

        # ── Arm swing ─────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(ts, [f.left_arm_swing  for f in frames], label="L Arm Swing",  color="#FFB300", linewidth=1.4)
        ax.plot(ts, [f.right_arm_swing for f in frames], label="R Arm Swing",  color="#0088FF", linewidth=1.4)
        ax.plot(ts, [f.arm_swing_asymmetry for f in frames], label="Asymmetry", color="red",    linewidth=1.0, linestyle="--")
        ax.set_title(f"Player #{self.player_id} — Arm Swing Excursion")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (°)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig, "arm_swing")

    def plot_risk_scores(self, frame_metrics: List[FrameMetrics]):
        if not frame_metrics or not HAS_MPL:
            return
        ts       = [f.timestamp    for f in frame_metrics]
        risk     = [f.risk_score   for f in frame_metrics]
        inj      = [f.injury_risk  for f in frame_metrics]
        joint_s  = [f.joint_stress for f in frame_metrics]
        fatigue  = [f.fatigue_index for f in frame_metrics]

        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        fig.suptitle(f"Player #{self.player_id} — Risk Indicators", fontsize=13)

        axes[0].plot(ts, risk, color="#FF3333", linewidth=1.5, label="Composite Risk Score")
        axes[0].fill_between(ts, risk, alpha=0.12, color="#FF3333")
        axes[0].axhline(50, color="orange", linewidth=0.8, linestyle="--", label="Moderate threshold (50)")
        axes[0].axhline(75, color="red",    linewidth=0.8, linestyle="--", label="High threshold (75)")
        axes[0].set_ylabel("Risk Score (0–100)")
        axes[0].set_ylim(0, 105)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ts, [v * 100 for v in inj],     label="Acute Injury Risk", color="#FF6B35", linewidth=1.3)
        axes[1].plot(ts, [v * 100 for v in joint_s], label="Joint Stress",      color="#9B59B6", linewidth=1.3)
        axes[1].plot(ts, [v * 100 for v in fatigue], label="Fatigue Index",     color="#2ECC71", linewidth=1.3)
        axes[1].set_ylabel("Sub-scores (%)")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        self._save(fig, "risk_scores")

    def plot_energy(self, frame_metrics: List[FrameMetrics]):
        if not frame_metrics or not HAS_MPL:
            return
        ts     = [f.timestamp          for f in frame_metrics]
        energy = [f.energy_expenditure for f in frame_metrics]

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(ts, energy, color="#F39C12", linewidth=1.5, label="Metabolic Power (W)")
        ax.fill_between(ts, energy, alpha=0.12, color="#F39C12")
        ax.set_title(f"Player #{self.player_id} — Estimated Metabolic Power (Minetti Model)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power (W)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig, "metabolic_power")

    def generate_all(self, frame_metrics: List[FrameMetrics],
                     bio_engine: Optional["BiomechanicsEngine"]):
        """Generate and save all standard plots."""
        if not HAS_MPL:
            print("[PLOT] matplotlib not installed — skipping plot generation.")
            print("       Run: pip install matplotlib")
            return
        self.plot_speed_profile(frame_metrics)
        self.plot_joint_angles(frame_metrics)
        self.plot_risk_scores(frame_metrics)
        self.plot_energy(frame_metrics)
        if bio_engine and bio_engine.frames:
            self.plot_biomechanics(bio_engine)
        print(f"[PLOT] All plots saved to: {self.results_dir}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
