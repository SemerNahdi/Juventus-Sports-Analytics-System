from .models import FrameMetrics
from .math_utils import clamp01

class RiskScorer:
    """Encapsulates biomechanical risk and stress calculations."""
    def __init__(self, model: dict):
        self.rm = model
        # Cache model params as typed floats
        self.knee_cap        = max(1.0, float(model.get("knee_angle_cap_deg", 165)))
        self.lean_stress_sc  = max(1.0, float(model.get("trunk_lean_stress_scale_deg", 35)))
        self.spd_base        = float(model.get("speed_baseline_mps", 4.0))
        self.spd_sc          = max(0.1, float(model.get("speed_scale_mps", 4.0)))
        self.knee_asym_sc    = max(1.0, float(model.get("knee_asym_stress_scale_deg", 25)))
        self.js_knee_w       = float(model.get("joint_stress_knee_w", 0.4))
        self.js_lean_w       = float(model.get("joint_stress_lean_w", 0.3))
        self.js_asym_w       = float(model.get("joint_stress_asym_w", 0.3))
        
        self.valgus_sc       = max(1.0, float(model.get("valgus_scale_deg", 15)))
        self.knee_asym_i_sc  = max(1.0, float(model.get("knee_asym_injury_scale_deg", 20)))
        self.accel_sc        = max(0.1, float(model.get("accel_scale", 10)))
        self.i_valgus_w      = float(model.get("injury_valgus_w", 0.5))
        self.i_asym_w        = float(model.get("injury_knee_asym_w", 0.3))
        self.i_accel_w       = float(model.get("injury_accel_w", 0.2))
        
        self.trunk_cum_sc    = max(1.0, float(model.get("trunk_lean_cumulative_scale_deg", 30)))
        self.cum_js_w        = float(model.get("cumulative_joint_stress_w", 0.4))
        self.cum_trunk_w     = float(model.get("cumulative_trunk_w", 0.3))
        self.cum_fatigue_w   = float(model.get("cumulative_fatigue_w", 0.3))
        self.f_injury_w      = float(model.get("final_injury_w", 0.6))
        self.f_cum_w         = float(model.get("final_cumulative_w", 0.4))

    def joint_stress(self, fm: FrameMetrics) -> float:
        ks = sum((self.knee_cap - ang) / self.knee_cap for ang in
                 [fm.left_knee_angle, fm.right_knee_angle] if ang < self.knee_cap)
        base = min(1., ks / 2)
        
        ls = clamp01(abs(fm.trunk_lateral_lean) / self.lean_stress_sc) * (
            max(0, fm.speed - self.spd_base) / self.spd_sc
        )
        asym = clamp01(abs(fm.left_knee_angle - fm.right_knee_angle) / self.knee_asym_sc)
        
        return clamp01(
            base * self.js_knee_w +
            ls * self.js_lean_w +
            asym * self.js_asym_w
        )

    def injury_risk(self, fm: FrameMetrics) -> float:
        valgus_deg  = (abs(fm.l_valgus_clinical) + abs(fm.r_valgus_clinical)) / 2.0
        p_valgus    = clamp01(valgus_deg / self.valgus_sc)
        p_knee_asym = clamp01(abs(fm.left_knee_angle - fm.right_knee_angle) / self.knee_asym_i_sc)
        p_accel     = clamp01(abs(fm.acceleration) / self.accel_sc)
        
        return (
            self.i_valgus_w * p_valgus +
            self.i_asym_w * p_knee_asym +
            self.i_accel_w * p_accel
        )

    def final_score(self, fm: FrameMetrics, perspective_conf: float) -> float:
        p_trunk = clamp01(abs(fm.trunk_lateral_lean) / self.trunk_cum_sc)
        cumulative = (
            self.cum_js_w * fm.joint_stress +
            self.cum_trunk_w * p_trunk +
            self.cum_fatigue_w * fm.fatigue_index
        )
        return (
            self.f_injury_w * fm.injury_risk +
            self.f_cum_w * cumulative
        ) * perspective_conf
