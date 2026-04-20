"""OpenSim/TRC/MOT output writing utilities."""

from .core import *  # noqa: F401,F403

class OpenSimFileWriter:
    """
    Generates valid OpenSim input files from tracked pose data.

    TRC format:
        Standard OpenSim Marker Trajectory (marker positions in metres, 3-D).
        We set Z=0 for all markers (monocular video → 2-D plane).
        Coordinate system: X = horizontal (right), Y = vertical (up, image Y inverted),
        Z = depth (out of plane, zero). This matches the standard OpenSim convention
        used by Sports2D / Pose2Sim.

    MOT format:
        OpenSim Motion file (tab-separated, header block).
        Stores joint angles in degrees, same convention as Sports2D.
    """

    # Subset of our joint names that map to standard OpenSim marker labels
    OPENSIM_MARKERS = [
        "head", "neck",
        "left_shoulder", "right_shoulder",
        "left_elbow",    "right_elbow",
        "left_wrist",    "right_wrist",
        "left_hip",      "right_hip",
        "left_knee",     "right_knee",
        "left_ankle",    "right_ankle",
        "left_foot",     "right_foot",
        "hip_center",    "shoulder_center",
    ]

    # Canonical OpenSim marker label mapping
    _LABEL_MAP = {
        "head":             "Head",
        "neck":             "Neck",
        "left_shoulder":    "L_Shoulder",
        "right_shoulder":   "R_Shoulder",
        "left_elbow":       "L_Elbow",
        "right_elbow":      "R_Elbow",
        "left_wrist":       "L_Wrist",
        "right_wrist":      "R_Wrist",
        "left_hip":         "L_Hip",
        "right_hip":        "R_Hip",
        "left_knee":        "L_Knee",
        "right_knee":       "R_Knee",
        "left_ankle":       "L_Ankle",
        "right_ankle":      "R_Ankle",
        "left_foot":        "L_Foot",
        "right_foot":       "R_Foot",
        "hip_center":       "Hip_Center",
        "shoulder_center":  "Shoulder_Center",
    }

    def write_trc(self, pose_frames: List[PoseFrame], path: str,
                  fps: float, pix_to_m: float, frame_height_px: int) -> bool:
        """
        Write a .trc file with 3-D marker trajectories.

        Coordinate conversion from image (px) to OpenSim (m):
            X_osim =  x_px * pix_to_m          (right is positive)
            Y_osim =  (H - y_px) * pix_to_m    (Y flipped: up is positive)
            Z_osim =  0.0                       (monocular — no depth)
        """
        n_frames  = len(pose_frames)
        n_markers = len(self.OPENSIM_MARKERS)
        H         = frame_height_px

        if n_frames == 0:
            print("[TRC] No pose frames — skipping TRC export.")
            return False

        try:
            with open(path, "w", newline="\r\n") as f:
                # ── Header ────────────────────────────────────────────────────
                # Line 0: file-type header
                f.write(f"PathFileType\t4\t(X/Y/Z)\t{os.path.basename(path)}\n")
                # Line 1: field names
                f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\t"
                        "Units\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
                # Line 2: values
                f.write(f"{fps:.6f}\t{fps:.6f}\t{n_frames}\t{n_markers}\t"
                        f"m\t{fps:.6f}\t1\t{n_frames}\n")
                # Line 3: marker labels — Frame# Time M1 '' '' M2 '' '' ...
                labels_row = "Frame#\tTime"
                for nm in self.OPENSIM_MARKERS:
                    lbl = self._LABEL_MAP[nm]
                    labels_row += f"\t{lbl}\t\t"  # label + 2 empty for Y Z
                f.write(labels_row + "\n")
                # Line 4: X/Y/Z sub-headers
                xyz_row = "\t"
                for _ in self.OPENSIM_MARKERS:
                    xyz_row += "\tX\tY\tZ"
                f.write(xyz_row + "\n")
                # Line 5: blank separator (OpenSim expects this)
                f.write("\n")

                # ── Data rows ─────────────────────────────────────────────────
                for pf in pose_frames:
                    row = f"{pf.frame_idx + 1}\t{pf.timestamp:.6f}"
                    for nm in self.OPENSIM_MARKERS:
                        px, py = getattr(pf.kp, nm)
                        x =  px * pix_to_m
                        y = (H - py) * pix_to_m  # flip Y: image Y↓ → OpenSim Y↑
                        z = 0.0
                        row += f"\t{x:.6f}\t{y:.6f}\t{z:.6f}"
                    f.write(row + "\n")
            print(f"[TRC] Written: {path}  ({n_frames} frames, {n_markers} markers)")
            return True
        except Exception as e:
            print(f"[TRC] Failed to write {path}: {e}")
            return False

    def write_mot(self, bio_frames: List[BioFrame], path: str, fps: float) -> bool:
        """
        Write a .mot (OpenSim Motion) file containing joint angles (degrees).

        The column ordering matches the standard Sports2D MOT output so the
        file can be loaded directly in OpenSim's Motion Visualizer or used
        as input to Inverse Kinematics.
        """
        if not bio_frames:
            print("[MOT] No biomechanics frames — skipping MOT export.")
            return False

        # Columns to export (all continuous angle fields from BioFrame)
        angle_fields = [
            "left_knee_flexion",    "right_knee_flexion",
            "left_hip_flexion",     "right_hip_flexion",
            "left_ankle_dorsiflexion", "right_ankle_dorsiflexion",
            "left_elbow_flexion",   "right_elbow_flexion",
            "trunk_lateral_lean",   "trunk_sagittal_lean",
            "pelvis_obliquity",     "pelvis_rotation",
            "left_thigh_angle",     "right_thigh_angle",
            "left_shank_angle",     "right_shank_angle",
            "trunk_segment_angle",
            "left_valgus_clinical", "right_valgus_clinical",
            "left_arm_swing",       "right_arm_swing",
        ]

        n_rows = len(bio_frames)
        n_cols = 1 + len(angle_fields)  # time + angles

        try:
            with open(path, "w", newline="\r\n") as f:
                # ── OpenSim MOT header ────────────────────────────────────────
                f.write(f"{os.path.basename(path)}\n")
                f.write("version=1\n")
                f.write(f"nRows={n_rows}\n")
                f.write(f"nColumns={n_cols}\n")
                f.write("inDegrees=yes\n")
                f.write("endheader\n")

                # ── Column header row ─────────────────────────────────────────
                header = "time\t" + "\t".join(angle_fields)
                f.write(header + "\n")

                # ── Data rows ─────────────────────────────────────────────────
                for bf in bio_frames:
                    row = f"{bf.timestamp:.6f}"
                    for field in angle_fields:
                        row += f"\t{getattr(bf, field):.6f}"
                    f.write(row + "\n")
            print(f"[MOT] Written: {path}  ({n_rows} rows, {len(angle_fields)} angles)")
            return True
        except Exception as e:
            print(f"[MOT] Failed to write {path}: {e}")
            return False


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYTICS PLOTTER  — saves all plots to /results
# ══════════════════════════════════════════════════════════════════════════════
