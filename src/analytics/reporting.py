from .core import HAS_SPORTS2D, HAS_SCIPY

def generate_report(s, bio_engine=None) -> str:
    """Generates a detailed summary report string."""
    dm  = "YOLO" # Default if not passed, but we can refine this
    bio = "sports2d" if HAS_SPORTS2D else "scipy" if HAS_SCIPY else "numpy"
    W   = 70
    lines = ["=" * W,
             f"SPORTS ANALYTICS v6 — Player #{s.player_id}".center(W),
             "=" * W, "",
             "SESSION OVERVIEW", "-" * W,
             f"  Duration        : {s.duration_seconds:>6.1f} s",
             f"  Total Frames    : {s.total_frames:>6}",
             f"  Total Distance  : {s.total_distance_m:>6.1f} m",
             f"  Angle Backend   : {bio}", "",
             "PLAYER METRICS", "-" * W,
             f"  Avg Speed       : {s.avg_speed:>6.2f} m/s",
             f"  Max Speed       : {s.max_speed:>6.2f} m/s",
             f"  Avg Stride      : {s.avg_stride_length:>6.2f} m",
             f"  Avg Cadence     : {s.avg_cadence:>6.0f} strides/min",
             f"  Avg Step Time   : {s.avg_step_time:>6.2f} s",
             f"  Avg Flight Time : {s.avg_flight_time:>6.2f} s",
             f"  Changes/Min     : {s.direction_change_freq:>6.1f}",
             f"  Energy (avg)    : {s.estimated_energy_kcal_hr:>6.0f} kcal/hr"]

    if bio_engine and bio_engine.frames:
        bd = bio_engine.summary_dict()
        lines += ["", "BIOMECHANICS  (Butterworth 6 Hz)", "-" * W,
                  f"  L Knee flexion  : {bd.get('left_knee_flexion_mean',0):>6.1f}° avg  {bd.get('left_knee_flexion_std',0):.1f}° sd",
                  f"  R Knee flexion  : {bd.get('right_knee_flexion_mean',0):>6.1f}° avg  {bd.get('right_knee_flexion_std',0):.1f}° sd",
                  f"  L Hip flexion   : {bd.get('left_hip_flexion_mean',0):>6.1f}°",
                  f"  R Hip flexion   : {bd.get('right_hip_flexion_mean',0):>6.1f}°",
                  f"  L Ankle dorsi   : {bd.get('left_ankle_dorsiflexion_mean',0):>6.1f}°",
                  f"  R Ankle dorsi   : {bd.get('right_ankle_dorsiflexion_mean',0):>6.1f}°",
                  f"  Trunk lat lean  : {bd.get('trunk_lateral_lean_mean',0):>6.1f}°",
                  f"  Pelvis obliquity: {bd.get('pelvis_obliquity_mean',0):>6.1f}°",
                  f"  Arm swing asym  : {bd.get('arm_swing_asymmetry_mean',0):>6.1f}°",
                  f"  Double support  : {bd.get('double_support_pct',0):>6.1f}%",
                  f"  Heel strikes L/R: {bd.get('lhs_count',0)} / {bd.get('rhs_count',0)}"]
        lvc = bd.get('left_valgus_clinical_mean',  0)
        rvc = bd.get('right_valgus_clinical_mean', 0)
        lines.append(f"  L Valgus (clin) : {lvc:>+6.1f}°{'  ⚠ VALGUS' if abs(lvc)>10 else ''}")
        lines.append(f"  R Valgus (clin) : {rvc:>+6.1f}°{'  ⚠ VALGUS' if abs(rvc)>10 else ''}")

    lines += ["", "RISK INDICATORS", "-" * W,
              f"  Peak Risk Score : {s.peak_risk_score:>6.0f} / 100",
              f"  Gait Symmetry   : {s.gait_symmetry_pct:>6.1f} %",
              f"  Acute Inj. Risk : {s.injury_risk_label}",
              f"  Body Stress     : {s.body_stress_label}",
              f"  Fatigue Level   : {s.fatigue_label}",
              f"  Risk Detail     : {s.injury_risk_detail}",
              "", "=" * W]
    return "\n".join(lines)
