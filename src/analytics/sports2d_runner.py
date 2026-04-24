"""Sports2D execution and native output discovery."""

from .core import *  # noqa: F401,F403
import multiprocessing as mp


def _sports2d_worker(config: dict, result_queue):
    """Run Sports2D in a child process and report success/failure."""
    try:
        from .core import SPORTS2D_PROCESS  # local import for spawn safety on Windows
        if SPORTS2D_PROCESS is None:
            raise RuntimeError(
                "Sports2D Python API not available (could not resolve process()). "
                "Install/repair with: pip install sports2d pose2sim"
            )
        SPORTS2D_PROCESS(config)
        result_queue.put({"ok": True})
    except Exception as e:
        import traceback
        result_queue.put({
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })

class Sports2DRunner:
    JOINT_ANGLES = [
        "Right ankle", "Left ankle",
        "Right knee",  "Left knee",
        "Right hip",   "Left hip",
        "Right shoulder", "Left shoulder",
        "Right elbow", "Left elbow",
    ]
    SEGMENT_ANGLES = [
        "Right foot",    "Left foot",
        "Right shank",   "Left shank",
        "Right thigh",   "Left thigh",
        "Pelvis", "Trunk", "Shoulders",
        "Right arm",     "Left arm",
        "Right forearm", "Left forearm",
    ]

    def __init__(self, video_path: str, result_dir: str,
                 player_height_m: float = 1.75,
                 participant_mass_kg: float = 75.0,
                 mode: str = "balanced",
                 show_realtime: bool = False,
                 person_ordering: str = "greatest_displacement",
                 do_ik: bool = False,
                 use_augmentation: bool = False,
                 visible_side: str = "auto front"):
        self.video_path          = video_path
        self.result_dir          = result_dir
        self.player_height_m     = player_height_m
        self.participant_mass_kg = participant_mass_kg
        self.mode                = mode
        self.show_realtime       = show_realtime
        self.person_ordering     = person_ordering
        self.do_ik               = do_ik
        self.use_augmentation    = use_augmentation
        self.visible_side        = visible_side
        self.outputs: dict       = {}

    def run(self) -> dict:
        if not HAS_SPORTS2D:
            print("[S2D] Sports2D not installed — skipping.\n"
                  "      Run: pip install sports2d pose2sim")
            return {}

        os.makedirs(self.result_dir, exist_ok=True)

        # Absolute path so Sports2D can locate the video regardless of cwd
        video_abs = str(os.path.abspath(self.video_path))
        result_abs = str(os.path.abspath(self.result_dir))

        # Sports2D CLI accepts `--visible_side auto front` (two tokens). In the
        # Python config this should be a list like ["auto", "front"], not
        # ["auto front"] (which can crash inside Sports2D).
        vs = self.visible_side
        if isinstance(vs, str):
            vs_tokens = [t for t in vs.strip().split() if t]
        else:
            vs_tokens = list(vs)  # type: ignore[arg-type]
        visible_side_cfg = vs_tokens if vs_tokens else ["auto", "front"]

        config = {
            # ── base: I/O, display, and what to save ──────────────────────────
            "base": {
                "video_input":            video_abs,
                "video_dir":              "",          # video_abs already absolute
                "result_dir":             result_abs,
                "nb_persons_to_detect":   1,
                "person_ordering_method": self.person_ordering,
                "first_person_height":    self.player_height_m,
                "visible_side":           visible_side_cfg,
                "load_trc_px":            "",
                "compare":                False,
                "time_range":             [],
                "webcam_id":              0,
                "input_size":             [1280, 720],
                "show_realtime_results":  self.show_realtime,
                "save_vid":               True,
                "save_img":               False,
                "save_pose":              True,
                "save_angles":            True,
            },
            # ── pose: model and detection parameters ──────────────────────────
            "pose": {
                "pose_model":    "Body_with_feet",
                "mode":          self.mode,
                "det_frequency": 4,
                "slowmo_factor": 1,
                "backend":       "auto",
                "device":        "auto",
                "tracking_mode": "sports2d",
                "keypoint_likelihood_threshold": 0.3,
                "average_likelihood_threshold":  0.5,
                "keypoint_number_threshold":     0.3,
            },
            # ── px_to_meters_conversion: separate section (NOT in base) ───────
            "px_to_meters_conversion": {
                "to_meters":         True,
                "make_c3d":          True,
                "save_calib":        True,
                "floor_angle":       "auto",
                "xy_origin":         ["auto"],
                "perspective_value": 10,
                "perspective_unit":  "distance_m",
                "distortions":       [0.0, 0.0, 0.0, 0.0, 0.0],
                "calib_file":        "",
            },
            # ── angles: which angles to compute and display ───────────────────
            "angles": {
                "calculate_angles":   True,          # ← correct location
                "joint_angles":   self.JOINT_ANGLES,
                "segment_angles": self.SEGMENT_ANGLES,
                "correct_segment_angles_with_floor_angle": True,
                "display_angle_values_on": ["body", "list"],
                "fontSize": 0.3,
            },
            # ── post-processing: filtering and graph saving ───────────────────
            # IMPORTANT: show_graphs and save_graphs live HERE, not in base
            "post-processing": {
                "interpolate":             True,
                "interp_gap_smaller_than": 100,
                "fill_large_gaps_with":    "last_value",
                "sections_to_keep":        "all",
                "min_chunk_size":          10,
                "reject_outliers":         True,
                "filter":                  True,
                "show_graphs":             self.show_realtime,  # mirrors realtime flag
                "save_graphs":             True,
                "filter_type":             "butterworth",
                "butterworth": {
                    "cut_off_frequency": 6,
                    "order": 4,
                },
            },
            # ── kinematics: OpenSim IK (requires full OpenSim install) ────────
            "kinematics": {
                "do_ik":               self.do_ik,
                "use_augmentation":    self.use_augmentation,
                "feet_on_floor":       False,
                "use_simple_model":    False,
                "participant_mass":    [self.participant_mass_kg],
                "right_left_symmetry": True,
                "default_height":      self.player_height_m,
                "fastest_frames_to_remove_percent": 0.1,
                "slowest_frames_to_remove_percent": 0.2,
                "large_hip_knee_angles":            45,
                "trimmed_extrema_percent":          0.5,
                "remove_individual_scaling_setup":  True,
                "remove_individual_ik_setup":       True,
            },
            "logging": {
                "use_custom_logging": False,
            },
        }

        # Run Sports2D out-of-process so heavy execution does not block the
        # current Python execution thread and remains killable via timeout.
        timeout_s = int(os.getenv("SPORTS2D_TIMEOUT_SEC", "1200"))
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_sports2d_worker, args=(config, q), daemon=True)
        p.start()
        p.join(timeout=timeout_s)

        if p.is_alive():
            p.terminate()
            p.join(10)
            print(f"[S2D] Sports2D timed out after {timeout_s}s and was terminated.")
            return {}

        worker_result = None
        try:
            if not q.empty():
                worker_result = q.get_nowait()
        except Exception:
            worker_result = None
        finally:
            try:
                q.close()
            except Exception:
                pass

        if p.exitcode not in (0, None):
            print(f"[S2D] Sports2D subprocess exited with code {p.exitcode}.")
            if worker_result and not worker_result.get("ok", False):
                print(f"[S2D] Sports2D.process() failed: {worker_result.get('error')}")
                tb = worker_result.get("traceback")
                if tb:
                    print(tb)
            return {}

        if worker_result and not worker_result.get("ok", False):
            print(f"[S2D] Sports2D.process() failed: {worker_result.get('error')}")
            tb = worker_result.get("traceback")
            if tb:
                print(tb)
            return {}

        self.outputs = self._collect_outputs()
        return self.outputs

    def _collect_outputs(self) -> dict:
        import glob
        rd = self.result_dir
        out = {
            "annotated_video": [],
            "angle_plots_png": [],
            "trc_pose_px":     [],
            "trc_pose_m":      [],
            "mot_angles":      [],
            "calib_toml":      [],
            "c3d":             [],
            "osim_model":      [],
            "osim_mot":        [],
            "osim_setup":      [],
            "all":             [],
        }
        for f in glob.glob(os.path.join(rd, "**", "*"), recursive=True):
            if not os.path.isfile(f):
                continue
            out["all"].append(f)
            fl   = f.lower()
            name = os.path.basename(fl)
            if fl.endswith(".mp4") or fl.endswith(".avi"):
                if "_h264" not in name:
                    out["annotated_video"].append(f)
            elif fl.endswith(".png"):
                out["angle_plots_png"].append(f)
            elif fl.endswith(".trc"):
                if "_px" in name or "pixel" in name:
                    out["trc_pose_px"].append(f)
                else:
                    out["trc_pose_m"].append(f)
            elif fl.endswith(".mot") and "ik" not in name:
                out["mot_angles"].append(f)
            elif fl.endswith(".toml"):
                out["calib_toml"].append(f)
            elif fl.endswith(".c3d"):
                out["c3d"].append(f)
            elif fl.endswith(".osim"):
                out["osim_model"].append(f)
            elif fl.endswith(".mot") and "ik" in name:
                out["osim_mot"].append(f)
            elif fl.endswith(".xml"):
                out["osim_setup"].append(f)
        return out

    def get_seed_from_trc(self) -> Optional[dict]:
        """
        Read the first few frames of Sports2D's pixel-space TRC file and
        return a seed dict compatible with TargetLock / pick_player_interactive.

        The TRC pixel file contains marker X/Y positions in image pixels.
        We use Hip_Center (or the mean of all markers) to locate the player
        in frame 0, build a rough bounding box, and sample a colour histogram
        from that region.
        """
        trcs_px = self.outputs.get("trc_pose_px", [])
        if not trcs_px:
            return None

        path = trcs_px[0]
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            # Find data start (first numeric row after the 5-line header)
            data_start = None
            for i, line in enumerate(lines):
                stripped = line.strip()
                if i >= 4 and stripped and stripped[0].isdigit():
                    data_start = i
                    break
            if data_start is None:
                return None

            # Header row is 2 lines above data
            header_idx = max(0, data_start - 2)
            df = pd.read_csv(path, sep="	", skiprows=header_idx,
                             encoding="utf-8", on_bad_lines="skip")
            df.columns = [c.strip() for c in df.columns]
            df = df.dropna(axis=1, how="all")

            if df.empty or len(df) < 2:
                return None

            # Robust TRC parsing: infer marker base names from "<Marker>.<X|Y|Z>".
            # We avoid relying on numeric column ordering, because missing markers
            # can shift indices and break modulo assumptions.
            cols = [c for c in df.columns if isinstance(c, str)]
            bases = {}
            for c in cols:
                if c.endswith(".X") or c.endswith(".Y") or c.endswith(".Z"):
                    base, axis = c.rsplit(".", 1)
                    bases.setdefault(base, {})[axis] = c

            if not bases:
                return None

            # Use first valid frame row; coerce to numeric where possible.
            row = df.iloc[0]
            def _val(col):
                try:
                    return float(pd.to_numeric(row.get(col), errors="coerce"))
                except Exception:
                    return float("nan")

            xs: List[float] = []
            ys: List[float] = []

            def _add_marker(base: str):
                ax = bases.get(base, {})
                xcol = ax.get("X")
                ycol = ax.get("Y")
                if not xcol or not ycol:
                    return
                x = _val(xcol)
                y = _val(ycol)
                if not np.isnan(x) and not np.isnan(y) and x > 0 and y > 0:
                    xs.append(x)
                    ys.append(y)

            # Use ALL available markers to find true min/max bounds
            for b in bases.keys():
                _add_marker(b)

            if not xs or not ys:
                return None

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # Base width and height
            w = max(40., max_x - min_x)
            h = max(100., max_y - min_y)
            
            # Add moderate percentage padding
            pad_x = w * 0.15
            pad_y = h * 0.10
            
            bx = int(min_x - pad_x)
            by = int(min_y - pad_y)
            bw = int(w + 2 * pad_x)
            bh = int(h + 2 * pad_y)

            seed_bbox = (bx, by, bw, bh)

            # Try to sample a histogram from the source video at frame 0
            hist = None
            try:
                cap = cv2.VideoCapture(self.video_path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        hist = crop_hist(frame, seed_bbox)
                cap.release()
            except Exception:
                pass

            return {
                "seed_bbox":  seed_bbox,
                "seed_frame": 0,
                "hist":       hist,
            }
        except Exception as e:
            print(f"[S2D] get_seed_from_trc failed: {e}")
            return None

    def load_mot_angles(self) -> Optional[pd.DataFrame]:
        mots = self.outputs.get("mot_angles", [])
        if not mots:
            return None
        path = mots[0]
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            header_idx = next(
                (i for i, l in enumerate(lines) if l.strip().lower().startswith("time")),
                None,
            )
            if header_idx is None:
                return None
            df = pd.read_csv(path, sep="\t", skiprows=header_idx,
                             encoding="utf-8", on_bad_lines="skip")
            df.columns = [c.strip() for c in df.columns]
            return df.dropna(axis=1, how="all")
        except Exception as e:
            print(f"[S2D] Failed to load MOT file {path}: {e}")
            return None

    def load_trc_pose(self, metres: bool = True) -> Optional[pd.DataFrame]:
        key  = "trc_pose_m" if metres else "trc_pose_px"
        trcs = self.outputs.get(key, [])
        if not trcs:
            return None
        path = trcs[0]
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            data_start = None
            for i, line in enumerate(lines):
                stripped = line.strip()
                if i >= 3 and stripped and (stripped[0].isdigit() or
                        stripped.lower().startswith("frame")):
                    data_start = i
                    break
            if data_start is None:
                data_start = 5
            header_line = data_start - 2
            df = pd.read_csv(path, sep="\t", skiprows=header_line,
                             encoding="utf-8", on_bad_lines="skip")
            df.columns = [c.strip() for c in df.columns]
            return df.dropna(axis=1, how="all")
        except Exception as e:
            print(f"[S2D] Failed to load TRC file {path}: {e}")
            return None


# ══════════════════════════════════════════════════════════════════════════════
#  TRC / MOT WRITER  — native OpenSim-compatible file generation
# ══════════════════════════════════════════════════════════════════════════════
