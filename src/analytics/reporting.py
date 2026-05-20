"""Reporting module for sports analytics."""

from datetime import datetime
from typing import Optional
from .core import HAS_SPORTS2D, HAS_SCIPY
from .models import MATSummary, PlayerSummary
from .biomechanics import BiomechanicsEngine
# from .scoring import MATSummary


def generate_report(
    summary: PlayerSummary,
    bio_engine: Optional[BiomechanicsEngine] = None,
    mat_summary: Optional[MATSummary] = None,
    width: int = 70
) -> str:
    """Generate a detailed performance report.
    
    Args:
        summary: Player summary metrics
        bio_engine: Optional biomechanics data
        mat_summary: Optional MAT jump analysis
        width: Report line width
    
    Returns:
        Formatted text report
    """
    if summary is None:
        return "ERROR: No summary data available"
    
    W = width
    bio_backend = "sports2d" if HAS_SPORTS2D else "scipy" if HAS_SCIPY else "numpy"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    lines = [
        "=" * W,
        f"SPORTS ANALYTICS v6 — Player #{summary.player_id}".center(W),
        "=" * W,
        f"Generated: {timestamp}".center(W),
        "",
        "SESSION OVERVIEW",
        "-" * W,
        f"  Duration        : {getattr(summary, 'duration_seconds', 0):>6.1f} s",
        f"  Total Frames    : {getattr(summary, 'total_frames', 0):>6}",
        f"  Total Distance  : {getattr(summary, 'total_distance_m', 0):>6.1f} m",
        f"  Angle Backend   : {bio_backend}",
        ""
    ]
    
    # MAT Protocol Results
    if mat_summary:
        lines.extend([
            "MAT PROTOCOL RESULTS",
            "-" * W,
            f"  Protocol        : {getattr(mat_summary, 'protocol_id', 'N/A')}",
            f"  LSI Symmetry Index: {getattr(mat_summary, 'limb_symmetry_index', 0):>6.1f} %",
            ""
        ])
        
        for i, ev in enumerate(getattr(mat_summary, 'events', [])):
            lines.extend([
                f"  Event #{i+1} ({getattr(ev, 'event_type', 'Unknown')})",
                f"    Flight Time   : {getattr(ev, 'flight_time', 0):>6.2f} s",
                f"    Landing Valgus: {getattr(ev, 'landing_valgus_left', 0):>6.1f}°",
                f"    Peak Flexion  : {getattr(ev, 'peak_knee_flexion_landing', 0):>6.1f}°",
                f"    Stabilization : {getattr(ev, 'time_to_stabilization', 0):>6.2f} s",
                f"    Hop Distance  : {getattr(ev, 'hop_distance_m', 0):>6.2f} m",
                ""
            ])
    
    # Player Metrics
    lines.extend([
        "PLAYER METRICS",
        "-" * W,
        f"  Avg Speed       : {getattr(summary, 'avg_speed', 0):>6.2f} m/s",
        f"  Max Speed       : {getattr(summary, 'max_speed', 0):>6.2f} m/s",
        f"  Avg Stride      : {getattr(summary, 'avg_stride_length', 0):>6.2f} m",
        f"  Avg Cadence     : {getattr(summary, 'avg_cadence', 0):>6.0f} strides/min",
        f"  Avg Step Time   : {getattr(summary, 'avg_step_time', 0):>6.2f} s",
        f"  Avg Flight Time : {getattr(summary, 'avg_flight_time', 0):>6.2f} s",
        f"  Changes/Min     : {getattr(summary, 'direction_change_freq', 0):>6.1f}",
        f"  Energy (avg)    : {getattr(summary, 'estimated_energy_kcal_hr', 0):>6.0f} kcal/hr"
    ])
    
    # Biomechanics
    if bio_engine:
        try:
            bd = bio_engine.summary_dict()
            if bd:
                lines.extend([
                    "",
                    "BIOMECHANICS  (Butterworth 6 Hz)",
                    "-" * W,
                    f"  L Knee flexion  : {bd.get('left_knee_flexion_mean', 0):>6.1f}° avg  {bd.get('left_knee_flexion_std', 0):.1f}° sd",
                    f"  R Knee flexion  : {bd.get('right_knee_flexion_mean', 0):>6.1f}° avg  {bd.get('right_knee_flexion_std', 0):.1f}° sd",
                    f"  L Hip flexion   : {bd.get('left_hip_flexion_mean', 0):>6.1f}°",
                    f"  R Hip flexion   : {bd.get('right_hip_flexion_mean', 0):>6.1f}°",
                    f"  L Ankle dorsi   : {bd.get('left_ankle_dorsiflexion_mean', 0):>6.1f}°",
                    f"  R Ankle dorsi   : {bd.get('right_ankle_dorsiflexion_mean', 0):>6.1f}°",
                    f"  Trunk lat lean  : {bd.get('trunk_lateral_lean_mean', 0):>6.1f}°",
                    f"  Pelvis obliquity: {bd.get('pelvis_obliquity_mean', 0):>6.1f}°",
                    f"  Arm swing asym  : {bd.get('arm_swing_asymmetry_mean', 0):>6.1f}°",
                    f"  Double support  : {bd.get('double_support_pct', 0):>6.1f}%",
                    f"  Heel strikes L/R: {bd.get('lhs_count', 0)} / {bd.get('rhs_count', 0)}"
                ])
                
                lvc = bd.get('left_valgus_clinical_mean', 0)
                rvc = bd.get('right_valgus_clinical_mean', 0)
                lines.append(f"  L Valgus (clin) : {lvc:>+6.1f}°{'  ⚠ VALGUS' if abs(lvc) > 10 else ''}")
                lines.append(f"  R Valgus (clin) : {rvc:>+6.1f}°{'  ⚠ VALGUS' if abs(rvc) > 10 else ''}")
        except Exception as e:
            lines.append(f"  Biomechanics data error: {e}")
    
    # Risk Indicators
    lines.extend([
        "",
        "RISK INDICATORS",
        "-" * W,
        f"  Peak Risk Score : {getattr(summary, 'peak_risk_score', 0):>6.0f} / 100",
        f"  Gait Symmetry   : {getattr(summary, 'gait_symmetry_pct', 0):>6.1f} %",
        f"  Acute Inj. Risk : {getattr(summary, 'injury_risk_label', 'Unknown')}",
        f"  Body Stress     : {getattr(summary, 'body_stress_label', 'Unknown')}",
        f"  Fatigue Level   : {getattr(summary, 'fatigue_label', 'Unknown')}",
        f"  Risk Detail     : {getattr(summary, 'injury_risk_detail', 'None')}",
        "",
        "=" * W
    ])
    
    return "\n".join(lines)


def save_report(report_text: str, output_path: str) -> bool:
    """Save report to file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"Report saved: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving report: {e}")
        return False