"""Risk scoring module for biomechanical analysis."""

from typing import Dict, Any, List, Optional
from .models import FrameMetrics
from .math_utils import clamp01


class RiskScorer:
    """Encapsulates biomechanical risk and stress calculations."""
    
    def __init__(self, model: Dict[str, Any], debug: bool = False) -> None:
        """Initialize RiskScorer with model parameters.
        
        Args:
            model: Dictionary of risk model parameters
            debug: Enable debug output
        """
        if not model:
            raise ValueError("Risk model cannot be empty")
        
        self.rm = model
        self.debug = debug
        
        # Cache model params as typed floats with safe defaults
        self.knee_cap = max(1.0, float(model.get("knee_angle_cap_deg", 165)))
        self.lean_stress_sc = max(1.0, float(model.get("trunk_lean_stress_scale_deg", 35)))
        self.spd_base = float(model.get("speed_baseline_mps", 4.0))
        self.spd_sc = max(0.1, float(model.get("speed_scale_mps", 4.0)))
        self.knee_asym_sc = max(1.0, float(model.get("knee_asym_stress_scale_deg", 25)))
        self.js_knee_w = float(model.get("joint_stress_knee_w", 0.4))
        self.js_lean_w = float(model.get("joint_stress_lean_w", 0.3))
        self.js_asym_w = float(model.get("joint_stress_asym_w", 0.3))
        
        self.valgus_sc = max(1.0, float(model.get("valgus_scale_deg", 15)))
        self.knee_asym_i_sc = max(1.0, float(model.get("knee_asym_injury_scale_deg", 20)))
        self.accel_sc = max(0.1, float(model.get("accel_scale", 10)))
        self.i_valgus_w = float(model.get("injury_valgus_w", 0.5))
        self.i_asym_w = float(model.get("injury_knee_asym_w", 0.3))
        self.i_accel_w = float(model.get("injury_accel_w", 0.2))
        
        self.trunk_cum_sc = max(1.0, float(model.get("trunk_lean_cumulative_scale_deg", 30)))
        self.cum_js_w = float(model.get("cumulative_joint_stress_w", 0.4))
        self.cum_trunk_w = float(model.get("cumulative_trunk_w", 0.3))
        self.cum_fatigue_w = float(model.get("cumulative_fatigue_w", 0.3))
        self.f_injury_w = float(model.get("final_injury_w", 0.6))
        self.f_cum_w = float(model.get("final_cumulative_w", 0.4))
        
        if self.debug:
            self._print_debug_info()
    
    def _print_debug_info(self) -> None:
        """Print initialization parameters for debugging."""
        print("[RiskScorer] Initialized with params:")
        for key, value in self.__dict__.items():
            if not key.startswith('_') and not callable(value) and key != 'rm':
                print(f"  {key}: {value}")
    
    def joint_stress(self, fm: FrameMetrics) -> float:
        """Calculate joint stress score (0-1)."""
        if fm is None:
            return 0.0
        
        # Knee stress
        knee_angles = [fm.left_knee_angle, fm.right_knee_angle]
        ks = sum(max(0.0, (self.knee_cap - ang) / self.knee_cap) for ang in knee_angles if ang < self.knee_cap)
        base = min(1.0, ks / 2.0) if knee_angles else 0.0
        
        # Trunk lean stress with speed factor
        speed_factor = max(0.0, (max(0.0, fm.speed) - self.spd_base) / self.spd_sc) if self.spd_sc > 0 else 0.0
        ls = clamp01(abs(fm.trunk_lateral_lean) / self.lean_stress_sc) * speed_factor
        
        # Asymmetry stress
        asym = clamp01(abs(fm.left_knee_angle - fm.right_knee_angle) / self.knee_asym_sc) if self.knee_asym_sc > 0 else 0.0
        
        result = clamp01(
            base * self.js_knee_w +
            ls * self.js_lean_w +
            asym * self.js_asym_w
        )
        
        if self.debug:
            print(f"[DEBUG] Joint stress - base:{base:.3f}, ls:{ls:.3f}, asym:{asym:.3f} -> {result:.3f}")
        
        return result
    
    def injury_risk(self, fm: FrameMetrics) -> float:
        """Calculate injury risk score (0-1)."""
        if fm is None:
            return 0.0
        
        # Valgus risk
        valgus_deg = (abs(fm.l_valgus_clinical) + abs(fm.r_valgus_clinical)) / 2.0
        p_valgus = clamp01(valgus_deg / self.valgus_sc) if self.valgus_sc > 0 else 0.0
        
        # Knee asymmetry risk
        knee_diff = abs(fm.left_knee_angle - fm.right_knee_angle)
        p_knee_asym = clamp01(knee_diff / self.knee_asym_i_sc) if self.knee_asym_i_sc > 0 else 0.0
        
        # Acceleration risk
        p_accel = clamp01(abs(fm.acceleration) / self.accel_sc) if self.accel_sc > 0 else 0.0
        
        result = (
            self.i_valgus_w * p_valgus +
            self.i_asym_w * p_knee_asym +
            self.i_accel_w * p_accel
        )
        
        if self.debug:
            print(f"[DEBUG] Injury risk - valgus:{p_valgus:.3f}, asym:{p_knee_asym:.3f}, accel:{p_accel:.3f} -> {result:.3f}")
        
        return result
    
    def final_score(self, fm: FrameMetrics, perspective_conf: float) -> float:
        """Calculate final risk score (0-100)."""
        if fm is None:
            return 0.0
        
        # Clamp inputs
        perspective_conf = clamp01(perspective_conf)
        
        # Cumulative components
        p_trunk = clamp01(abs(fm.trunk_lateral_lean) / self.trunk_cum_sc) if self.trunk_cum_sc > 0 else 0.0
        
        cumulative = (
            self.cum_js_w * getattr(fm, 'joint_stress', 0.0) +
            self.cum_trunk_w * p_trunk +
            self.cum_fatigue_w * getattr(fm, 'fatigue_index', 0.0)
        )
        
        # Final calculation
        raw_score = (
            self.f_injury_w * getattr(fm, 'injury_risk', 0.0) +
            self.f_cum_w * cumulative
        ) * perspective_conf
        
        result = clamp01(raw_score) * 100.0
        
        if self.debug:
            print(f"[DEBUG] Final score - cumulative:{cumulative:.3f}, raw:{raw_score:.3f} -> {result:.1f}")
        
        return result
    
    def batch_joint_stress(self, frames: List[FrameMetrics]) -> List[float]:
        """Calculate joint stress for multiple frames."""
        if not frames:
            return []
        return [self.joint_stress(fm) for fm in frames]
    
    def batch_injury_risk(self, frames: List[FrameMetrics]) -> List[float]:
        """Calculate injury risk for multiple frames."""
        if not frames:
            return []
        return [self.injury_risk(fm) for fm in frames]
    
    def get_summary(self, frames: List[FrameMetrics]) -> Dict[str, float]:
        """Get summary statistics for a sequence."""
        if not frames:
            return {}
        
        joint_stresses = self.batch_joint_stress(frames)
        injury_risks = self.batch_injury_risk(frames)
        
        return {
            "avg_joint_stress": sum(joint_stresses) / len(joint_stresses),
            "max_joint_stress": max(joint_stresses),
            "avg_injury_risk": sum(injury_risks) / len(injury_risks),
            "max_injury_risk": max(injury_risks),
            "high_risk_frames": sum(1 for r in injury_risks if r > 0.7),
        }