"""Top-level analysis orchestration engine."""

from .core import *  # noqa: F401,F403
from .sports2d_runner import Sports2DRunner
from .output_manager import OpenSimFileWriter

class SportsAnalyzer:
    PIX_TO_M = None

    def __init__(self, video_path: str,
                 output_video_path: str = "output_annotated.mp4",
                 player_id: int = 1,
                 fps_override: Optional[float] = None,
                 pick: bool = False,
                 yolo_size: str = "m",
                 player_height_m: float = 1.75,
                 player_mass_kg: float = 75.0):
        self.video_path         = video_path
        self.output_video_path  = output_video_path
        self.player_id          = player_id
        self.fps_override       = fps_override
        self.player_height_m    = player_height_m
        self.player_mass_kg     = player_mass_kg

        self.pose_est   = HybridPoseEstimator()
        self.smoother   = PoseKalmanSmoother()
        self.pose_frames:    List[PoseFrame]    = []
        self.frame_metrics:  List[FrameMetrics] = []
        self.summary = PlayerSummary(player_id=player_id)

        self._spd_win         = deque(maxlen=30)
        self._risk_win        = deque(maxlen=15)
        self._speed_history   = deque(maxlen=90)
        self._pix_to_m_samples = deque(maxlen=60)
        self._accel_burst     = 0
        self._fps_cache       = 30.

        self.bio_engine:         Optional[BiomechanicsEngine] = None
        self.sports2d_runner:    Optional[Sports2DRunner]     = None
        self._frame_height_px:   int = 0

        self._det_layer = get_detection_layer(yolo_size)

        print("\n" + "=" * 50)
        print(" SPORTS ANALYTICS: ENGINE READY")
        print("-" * 50)
        print(f" * POSE DETECTION:    {self._det_layer.mode.upper()}")
        print(f" * BIOMECHANICS:      {'SPORTS2D (Clinical-Grade)' if HAS_SPORTS2D else 'NUMPY (Math Fallback)'}")
        print(f" * SIGNAL FILTERING:  {'SCIPY (Advanced Signal)' if HAS_SCIPY else 'NUMPY (Basic Mean)'}")
        print(f" * PLOTTING:          {'MATPLOTLIB' if HAS_MPL else 'NOT AVAILABLE'}")
        print("=" * 50 + "\n")

        if pick:
            primary = pick_player_interactive(video_path)
        else:
            primary = select_primary_player(video_path)
        if primary is None:
            raise RuntimeError("No player candidates found in video.")
        self.lock = TargetLock(primary["seed_bbox"], primary["hist"], primary["seed_frame"], yolo_size=yolo_size)

    # ── Video processing ──────────────────────────────────────────────────────

    def process_video(self, stride: int = 2, target_height: int = 640, cancel_event: Optional[threading.Event] = None) -> PlayerSummary:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(self.video_path)

        fps   = self.fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.
        self._fps_cache = fps
        W_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Adaptive Resizing: speed up processing by resizing frame once at the start.
        # This reduces YOLO latency and drawing time significantly for 4K/1080p sources.
        scale = 1.0
        if H_orig > target_height and target_height > 0:
            scale = target_height / H_orig
        
        W, H = int(W_orig * scale), int(H_orig * scale)
        self._frame_height_px = H

        out = self._create_writer(self.output_video_path, fps / stride, W, H)
        self.bio_engine = BiomechanicsEngine(fps=fps / stride, pix_to_m=self.PIX_TO_M or 0.002)

        idx = 0
        while True:
            # Check for cancellation signal every frame
            if cancel_event and cancel_event.is_set():
                print(f"[ENGINE] Cancellation signal received at frame {idx}. Aborting...")
                break

            ret, frame = cap.read()
            if not ret:
                break
            
            if idx % stride != 0:
                idx += 1
                continue

            # Resize frame for analysis and final video output
            if scale != 1.0:
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

            ts   = idx / fps
            bbox = self.lock.update(frame)

            if bbox and bbox[2] > 20 and bbox[3] > 40:
                target  = next((t for t in self.lock.bt.active_tracks
                                if t.id == self.lock._target_id), None)
                yolo_kp = getattr(target, '_yolo_kp', None) if target else None
                spd     = self.frame_metrics[-1].speed if self.frame_metrics else 0.

                raw_kp = self.pose_est.estimate(frame, bbox, ts, spd, yolo_kp=yolo_kp)
                kp     = self.smoother.smooth(raw_kp)
                pf     = PoseFrame(idx, ts, bbox, kp)
                self.pose_frames.append(pf)

                self._calibrate(kp)
                fm = self._metrics(pf, idx, ts, fps)
                self.frame_metrics.append(fm)
                self._speed_history.append(fm.speed)
                self.bio_engine.process_frame(idx, ts, kp)

                if abs(fm.acceleration) > 4.0:
                    self._accel_burst = 8
                elif self._accel_burst > 0:
                    self._accel_burst -= 1

                # Annotate frame: skeleton + labels only (no side panel)
                frame = self._annotate(frame, pf, fm)
                frame = self._draw_player_aura(frame, kp, fm)

            out.write(frame)
            idx += 1

        cap.release()
        out.release()

        # If we were cancelled, don't proceed to post-processing or summary
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Job was cancelled by the user.")

        # Build summary and process gait before potentially re-encoding
        if self.bio_engine:
            self.bio_engine.post_process()
        self._post_gait(fps)
        self._build_summary()

        return self.summary

    def _create_writer(self, path: str, fps: float, W: int, H: int):
        # For Cloudinary compatibility, we prioritize reliability (mp4v) over browser-native codecs (avc1)
        # because Cloudinary will transcode it to a web-safe format automatically.
        for codec in ["mp4v", "avc1", "H264", "VP80", "X264", "DIVX", "MJPG"]:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
                if writer.isOpened():
                    print(f"[VIDEO] Using codec: {codec}")
                    return writer
                writer.release()
            except Exception:
                continue

        class _NullWriter:
            def write(self, _): pass
            def release(self): pass
        return _NullWriter()


    # ── Annotation (skeleton + labels on video frame, NO side panel) ──────────

    def _annotate(self, frame, pf: PoseFrame, fm: FrameMetrics) -> np.ndarray:
        kp = pf.kp
        rt = clamp01(fm.risk_score / 100.)
        render_skeleton(frame, kp, risk_tint=rt)

        # Knee angle labels
        for kpt, ang in [(kp.left_knee, fm.left_knee_angle),
                         (kp.right_knee, fm.right_knee_angle)]:
            kx, ky = int(kpt[0]), int(kpt[1])
            ac = (0, 220, 0) if ang > 145 else (0, 140, 255) if ang > 120 else (0, 0, 220)
            cv2.putText(frame, f"{ang:.0f}°", (kx + 12, ky - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, ac, 1, cv2.LINE_AA)

        # Player badge above head
        hx       = int(kp.head[0])  # use head X, not hip, for correct lateral position
        head_y   = int(kp.head[1]) - 35
        badge    = f"  #{self.player_id}  "
        (tw, _), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        bx0      = hx - tw // 2
        overlay  = frame.copy()
        cv2.rectangle(overlay, (bx0, head_y - 22), (bx0 + tw, head_y + 6), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (bx0, head_y - 22), (bx0 + tw, head_y + 6), (255, 215, 0), 1)
        cv2.putText(frame, badge, (bx0 + 2, head_y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        # Speed & risk overlay near player bbox
        bx, by, bw, bh = pf.bbox
        spd_txt = f"{fm.speed:.1f} m/s"
        cv2.putText(frame, spd_txt, (bx, by - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

        return frame

    def _draw_player_aura(self, frame, kp: PoseKeypoints, fm: FrameMetrics) -> np.ndarray:
        if fm.speed < .5:
            return frame
        hx, hy = int(kp.hip_center[0]), int(kp.hip_center[1])
        bx, by, bw, bh = self.pose_frames[-1].bbox
        rx  = max(12, bw // 2 + 6)
        ry  = max(20, bh // 2 + 10)
        col = lerp_color((0, 180, 60), (0, 60, 255), clamp01(fm.speed / 8.))
        if self._accel_burst > 0:
            br = int(rx * 1.6 + self._accel_burst * 3)
            ba = self._accel_burst / 8. * .4
            ov = frame.copy()
            cv2.ellipse(ov, (hx, hy), (br, int(br * 1.4)), 0, 0, 360,
                        (0, 200, 255), 3, cv2.LINE_AA)
            frame[:] = cv2.addWeighted(ov, ba, frame, 1 - ba, 0)
        for exp, a in [(14, .12), (6, .20)]:
            ov = frame.copy()
            cv2.ellipse(ov, (hx, hy), (rx + exp, ry + exp), 0, 0, 360, col, -1, cv2.LINE_AA)
            frame[:] = cv2.addWeighted(ov, a, frame, 1 - a, 0)
        return frame

    # ── Calibration ───────────────────────────────────────────────────────────

    def _calibrate(self, kp: PoseKeypoints):
        left_leg  = dist2d(kp.left_hip,  kp.left_ankle)
        right_leg = dist2d(kp.right_hip, kp.right_ankle)
        leg = max(left_leg, right_leg)
        if leg < 10:
            return
        estimate = 0.90 / leg
        self._pix_to_m_samples.append(estimate)
        self.PIX_TO_M = float(np.median(self._pix_to_m_samples))
        if self.bio_engine is not None:
            self.bio_engine.pix_to_m = self.PIX_TO_M

    # ── Per-frame metrics ─────────────────────────────────────────────────────

    def _metrics(self, pf: PoseFrame, idx: int, ts: float, fps: float) -> FrameMetrics:
        fm  = FrameMetrics(frame_idx=idx, timestamp=ts)
        kp  = pf.kp
        sc  = self.PIX_TO_M or 0.002

        # If pose is low-confidence (e.g., sparse YOLO keypoints), avoid producing
        # biologically invalid angles/risks. We keep kinematics/risk fields stable
        # by carrying forward the last valid metrics, while still updating speed.
        if getattr(kp, "_yolo_confident", True) is False and self.frame_metrics:
            prev_fm = self.frame_metrics[-1]
            fm.left_knee_angle = prev_fm.left_knee_angle
            fm.right_knee_angle = prev_fm.right_knee_angle
            fm.left_hip_angle = prev_fm.left_hip_angle
            fm.right_hip_angle = prev_fm.right_hip_angle
            fm.trunk_lean = prev_fm.trunk_lean
            fm.perspective_confidence = prev_fm.perspective_confidence
            fm.l_valgus_clinical = prev_fm.l_valgus_clinical
            fm.r_valgus_clinical = prev_fm.r_valgus_clinical
            fm.l_valgus = prev_fm.l_valgus
            fm.r_valgus = prev_fm.r_valgus
            fm.energy_expenditure = prev_fm.energy_expenditure
            fm.joint_stress = prev_fm.joint_stress
            fm.fatigue_index = prev_fm.fatigue_index
            fm.injury_risk = prev_fm.injury_risk
            fm.risk_score = prev_fm.risk_score
            fm.fall_risk = prev_fm.fall_risk

            if len(self.pose_frames) >= 2:
                prev = self.pose_frames[-2]
                dt   = ts - prev.timestamp + 1e-9
                dp   = dist2d(kp.hip_center, prev.kp.hip_center) * sc
                raw  = dp / dt
                self._spd_win.append(raw)
                fm.speed            = float(np.mean(self._spd_win))
                fm.body_center_disp = dp
                if len(self.pose_frames) >= 3:
                    p2  = self.pose_frames[-3]
                    dp2 = dist2d(prev.kp.hip_center, p2.kp.hip_center) * sc
                    dt2 = prev.timestamp - p2.timestamp + 1e-9
                    fm.acceleration = (raw - dp2 / dt2) / dt
            return fm

        fm.left_knee_angle  = s2d_joint_angle(kp.left_hip,  kp.left_knee,  kp.left_ankle)
        fm.right_knee_angle = s2d_joint_angle(kp.right_hip, kp.right_knee, kp.right_ankle)
        fm.left_hip_angle   = s2d_joint_angle(kp.left_shoulder,  kp.left_hip,  kp.left_knee)
        fm.right_hip_angle  = s2d_joint_angle(kp.right_shoulder, kp.right_hip, kp.right_knee)

        dx = kp.shoulder_center[0] - kp.hip_center[0]
        dy = kp.shoulder_center[1] - kp.hip_center[1]
        fm.trunk_lean = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-9))

        fm.perspective_confidence = estimate_player_orientation(kp)

        lvc = BiomechanicsEngine._clinical_valgus(kp.left_hip,  kp.left_knee,  kp.left_ankle)
        rvc = BiomechanicsEngine._clinical_valgus(kp.right_hip, kp.right_knee, kp.right_ankle)
        fm.l_valgus_clinical = lvc * fm.perspective_confidence
        fm.r_valgus_clinical = rvc * fm.perspective_confidence

        hw = dist2d(kp.left_hip, kp.right_hip) + 1e-6
        fm.l_valgus = abs(kp.left_knee[0]  - kp.left_hip[0])  / hw
        fm.r_valgus = abs(kp.right_knee[0] - kp.right_hip[0]) / hw

        if len(self.pose_frames) >= 2:
            prev = self.pose_frames[-2]
            dt   = ts - prev.timestamp + 1e-9
            dp   = dist2d(kp.hip_center, prev.kp.hip_center) * sc
            raw  = dp / dt
            self._spd_win.append(raw)
            fm.speed            = float(np.mean(self._spd_win))
            fm.body_center_disp = dp
            if len(self.pose_frames) >= 3:
                p2  = self.pose_frames[-3]
                dp2 = dist2d(prev.kp.hip_center, p2.kp.hip_center) * sc
                dt2 = prev.timestamp - p2.timestamp + 1e-9
                fm.acceleration = (raw - dp2 / dt2) / dt

        if len(self.pose_frames) >= 5:
            pos  = [p.kp.hip_center for p in list(self.pose_frames)[-5:]]
            vecs = [(pos[i+1][0]-pos[i][0], pos[i+1][1]-pos[i][1]) for i in range(4)]
            for i in range(len(vecs) - 1):
                v1, v2 = np.array(vecs[i]), np.array(vecs[i+1])
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 2 and n2 > 2 and math.acos(
                        np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)) > math.radians(28):
                    fm.direction_change = True

        MASS_KG = getattr(self, 'player_mass_kg', 75.0)
        G = 9.81
        Cr = 4.0
        v = max(fm.speed, 0.1)
        a = fm.acceleration
        P_loco  = Cr * v * MASS_KG
        g_eq    = max(0., a) / G
        P_accel = Cr * g_eq * v * MASS_KG
        fm.energy_expenditure = P_loco + P_accel + 80.0

        ks = sum((155 - ang) / 155 for ang in
                 [fm.left_knee_angle, fm.right_knee_angle] if ang < 155)
        fm.joint_stress = min(1., ks / 2)
        ls  = clamp01(fm.trunk_lean / 25.) * (max(0, fm.speed - 1.0) / 5.0)
        asym = clamp01(abs(fm.left_knee_angle - fm.right_knee_angle) / 40.)
        fm.joint_stress = clamp01(fm.joint_stress * 0.5 + ls * 0.3 + asym * 0.2)

        if len(self._spd_win) >= 10:
            s = list(self._spd_win)
            fm.fatigue_index = max(0., min(1., (np.mean(s[:5]) - np.mean(s[-5:])) /
                                           (np.mean(s[:5]) + 1e-6)))

        valgus_deg  = (abs(fm.l_valgus_clinical) + abs(fm.r_valgus_clinical)) / 2.0
        p_valgus    = clamp01(valgus_deg / 15.0)
        p_knee_asym = clamp01(abs(fm.left_knee_angle - fm.right_knee_angle) / 30.)
        p_accel     = clamp01(abs(fm.acceleration) / 12.)
        fm.injury_risk = 0.45 * p_valgus + 0.30 * p_knee_asym + 0.25 * p_accel

        p_trunk    = clamp01(fm.trunk_lean / 30.)
        cumulative = 0.40 * fm.joint_stress + 0.35 * p_trunk + 0.25 * fm.fatigue_index

        raw_risk = (0.60 * fm.injury_risk + 0.40 * cumulative) * fm.perspective_confidence
        self._risk_win.append(raw_risk)
        fm.risk_score = float(np.mean(self._risk_win)) * 100.
        return fm

    # ── Post-processing ───────────────────────────────────────────────────────

    def _post_gait(self, fps: float):
        if len(self.pose_frames) < 15:
            return
        sc = self.PIX_TO_M or 0.002
        la = smooth_arr([p.kp.left_ankle[1]  for p in self.pose_frames])
        ra = smooth_arr([p.kp.right_ankle[1] for p in self.pose_frames])
        md = max(5, int(fps * .18))

        if HAS_SCIPY:
            lp, _ = find_peaks( la, distance=md, prominence=2)
            rp, _ = find_peaks( ra, distance=md, prominence=2)
        else:
            def pk(arr, d):
                pks = []
                for i in range(1, len(arr) - 1):
                    if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
                        if not pks or i - pks[-1] >= d:
                            pks.append(i)
                return np.array(pks)
            lp = pk(la, md)
            rp = pk(ra, md)

        pos  = [p.kp.hip_center for p in self.pose_frames]
        strl, stt, flt = [], [], []
        for peaks in [lp, rp]:
            for i in range(1, len(peaks)):
                i0, i1 = peaks[i-1], peaks[i]
                if i1 >= len(pos):
                    continue
                sl = dist2d(pos[i0], pos[i1]) * sc
                if .15 < sl < 3.5:
                    strl.append(sl)
                st = (i1 - i0) / fps
                if .08 < st < 2.0:
                    stt.append(st)
                    flt.append(max(0., st * .35))

        n = min(len(lp), len(rp)) - 1
        if n > 0:
            li = [(lp[i+1] - lp[i]) / fps for i in range(n)]
            ri = [(rp[i+1] - rp[i]) / fps for i in range(n)]
            m  = min(len(li), len(ri))
            sym = float(np.mean(
                [1 - abs(l - r) / (l + r + 1e-9) for l, r in zip(li[:m], ri[:m])]
            )) * 100
        else:
            sym = 94.

        sv   = float(np.std(strl) / (np.mean(strl) + 1e-9) * 100) if len(strl) > 2 else 3.5
        asl  = float(np.mean(strl)) if strl else 1.35
        ast  = float(np.mean(stt))  if stt  else .38
        aft  = float(np.mean(flt))  if flt  else .13
        acad = 60. / ast if ast > 0 else 158.

        for fm in self.frame_metrics:
            fm.stride_length   = asl
            fm.step_time       = ast
            fm.flight_time     = aft
            fm.cadence         = acad
            fm.gait_symmetry   = sym
            fm.stride_variability = sv

        hip_x   = [p.kp.hip_center[0] for p in self.pose_frames]
        lat_bal = clamp01(
            np.std(hip_x) / (max(1, np.mean([p.bbox[2] for p in self.pose_frames])) * 0.1)
            if hip_x else 0.
        )
        for fm in self.frame_metrics:
            sr = max(0., (100 - fm.gait_symmetry) / 100)
            vr = min(1., fm.stride_variability / 25)
            lr = min(1., fm.trunk_lean / 40)
            fm.fall_risk    = clamp01(sr * .3 + vr * .2 + lr * .2 + lat_bal * .3)
            ar              = min(1., abs(fm.acceleration) / 12)
            fm.injury_risk  = fm.joint_stress * .5 + ar * .3 + fm.fatigue_index * .2

    def _build_summary(self):
        if not self.frame_metrics:
            return
        fms = self.frame_metrics
        s   = self.summary
        sc  = self.PIX_TO_M or 0.002

        s.total_frames    = len(fms)
        s.duration_seconds = fms[-1].timestamp + (1.0 / (self._fps_cache or 30.))

        spds = np.array([f.speed for f in fms])
        s.avg_speed = float(np.mean(spds))
        s.max_speed = float(np.max(spds))

        def anz(a):
            v = [getattr(f, a) for f in fms if getattr(f, a) > 0]
            return float(np.mean(v)) if v else 0.

        s.avg_stride_length        = anz("stride_length")
        s.avg_step_time            = anz("step_time")
        s.avg_cadence              = anz("cadence")  # strides per minute
        s.avg_flight_time          = anz("flight_time")
        s.estimated_energy_kcal_hr = float(np.mean([f.energy_expenditure for f in fms]))
        s.gait_symmetry_pct        = float(np.mean([f.gait_symmetry for f in fms]))
        s.stride_variability_pct   = float(np.mean([f.stride_variability for f in fms]))

        dc = sum(1 for f in fms if f.direction_change)
        s.direction_change_freq = dc / max(s.duration_seconds / 60, 1e-6)
        s.peak_risk_score = float(np.max([f.risk_score for f in fms]))

        if len(self.pose_frames) >= 2:
            s.total_distance_m = sum(
                dist2d(self.pose_frames[i].kp.hip_center, self.pose_frames[i-1].kp.hip_center) * sc
                for i in range(1, len(self.pose_frames))
            )

        def rl(a):
            return self._risk_label(float(np.mean([getattr(f, a) for f in fms])))

        s.fall_risk_label    = rl("fall_risk")
        s.injury_risk_label  = rl("injury_risk")
        s.body_stress_label  = rl("joint_stress")
        s.fatigue_label      = rl("fatigue_index")

        avg_valgus = float(np.mean(
            [abs(f.l_valgus_clinical) + abs(f.r_valgus_clinical) for f in fms]
        )) / 2.
        ai = float(np.mean([f.injury_risk for f in fms]))
        if avg_valgus > 10.:
            s.injury_risk_detail = "valgus collapse detected (>10°)"
        elif ai > .5:
            s.injury_risk_detail = "high knee load / acceleration stress"
        elif ai > .3:
            s.injury_risk_detail = "moderate joint stress"
        else:
            s.injury_risk_detail = "within normal range"

        if self.bio_engine:
            bd = self.bio_engine.summary_dict()
            s.double_support_pct = bd.get("double_support_pct", 0.0)
            avg_pr = bd.get("pelvis_rotation_mean", 0.0)
            # If explicit rotation is missing, estimate from trunk lean as fallback
            if avg_pr == 0:
                avg_pr = float(np.mean([f.trunk_lean for f in fms])) * 0.4
            s.avg_pelvic_rotation = avg_pr

    @staticmethod
    def _risk_label(v) -> str:
        return "Low" if v < .25 else "Moderate" if v < .55 else "High"

    # ── Sports2D native pipeline ──────────────────────────────────────────────

    def run_sports2d(self, result_dir: str,
                     mode: str = "balanced",
                     show_realtime: bool = False,
                     person_ordering: str = "greatest_displacement",
                     do_ik: bool = False,
                     use_augmentation: bool = False,
                     visible_side: str = "auto front",
                     participant_mass_kg: float = 75.0) -> dict:
        """
        Run Sports2D on the video.  This is always the first step — its
        on_click picker IS the player selection mechanism when --pick is used.
        After Sports2D finishes, we seed the custom tracker from the TRC data
        so both pipelines analyse the exact same player.
        """
        self.sports2d_runner = Sports2DRunner(
            video_path          = self.video_path,
            result_dir          = result_dir,
            player_height_m     = self.player_height_m,
            participant_mass_kg = participant_mass_kg,
            mode                = mode,
            show_realtime       = show_realtime,
            person_ordering     = person_ordering,
            do_ik               = do_ik,
            use_augmentation    = use_augmentation,
            visible_side        = visible_side,
        )
        outputs = self.sports2d_runner.run()

        # ── Seed our custom tracker from Sports2D's TRC output ───────────────
        # This guarantees both pipelines follow the same player.
        seed = self.sports2d_runner.get_seed_from_trc()
        if seed is not None:
            self.lock = TargetLock(
                seed["seed_bbox"], seed["hist"], seed["seed_frame"]
            )
            print("[S2D] Custom tracker seeded from Sports2D TRC data.")
        else:
            print("[S2D] Could not seed from TRC — custom tracker uses original pick.")

        return outputs

    # ── Unified export ────────────────────────────────────────────────────────

    def export_unified(self, json_path: str, csv_path: str,
                       trc_path: Optional[str] = None,
                       mot_path: Optional[str] = None):
        """
        Consolidate ALL data into two unified files:
          - data_output.json : hierarchical structured data
          - bio_metrics.csv  : flat time-series for analysis

        Optionally writes OpenSim-compatible .trc and .mot files.
        """
        # ── Build unified per-frame records ───────────────────────────────────
        bio_by_frame: dict = {}
        if self.bio_engine and self.bio_engine.frames:
            for bf in self.bio_engine.frames:
                bio_by_frame[bf.frame_idx] = asdict(bf)

        unified_frames = []
        for fm in self.frame_metrics:
            record = asdict(fm)
            bio = bio_by_frame.get(fm.frame_idx, {})
            # Merge bio fields — prefix with "bio_" to avoid name collision
            for k, v in bio.items():
                if k not in ("frame_idx", "timestamp"):
                    record[f"bio_{k}"] = v

            # Append Sports2D keypoints if available (from TRC data)
            unified_frames.append(record)

        # ── Sports2D angle summary ────────────────────────────────────────────
        s2d_angle_summary: dict = {}
        s2d_pose_summary:  dict = {}
        if self.sports2d_runner:
            mot_df = self.sports2d_runner.load_mot_angles()
            if mot_df is not None and not mot_df.empty:
                angle_cols = [c for c in mot_df.columns if c.lower() != "time"]
                for col in angle_cols:
                    try:
                        vals = pd.to_numeric(mot_df[col], errors="coerce").dropna()
                        if len(vals):
                            s2d_angle_summary[col] = {
                                "mean": float(vals.mean()),
                                "max":  float(vals.max()),
                                "min":  float(vals.min()),
                                "std":  float(vals.std()),
                            }
                    except Exception:
                        pass
            trc_df = self.sports2d_runner.load_trc_pose(metres=True)
            if trc_df is not None and not trc_df.empty:
                s2d_pose_summary["trc_shape"] = list(trc_df.shape)
                s2d_pose_summary["trc_columns"] = list(trc_df.columns)

        # ── JSON — hierarchical ───────────────────────────────────────────────
        payload = {
            "metadata": {
                "player_id":   self.player_id,
                "video_path":  self.video_path,
                "fps":         self._fps_cache,
                "pix_to_m":   self.PIX_TO_M,
                "total_frames": len(self.frame_metrics),
                "angle_backend": "sports2d" if HAS_SPORTS2D else "scipy" if HAS_SCIPY else "numpy",
            },
            "player_summary":   asdict(self.summary),
            "biomechanics_summary": self.bio_engine.summary_dict() if self.bio_engine else {},
            "sports2d_angle_summary": s2d_angle_summary,
            "sports2d_pose_summary":  s2d_pose_summary,
            "sports2d_output_files":  self.sports2d_runner.outputs if self.sports2d_runner else {},
            "frames": unified_frames,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"[EXPORT] data_output.json → {json_path}  ({len(unified_frames)} frames)")

        # ── CSV — flat time-series ────────────────────────────────────────────
        df = pd.DataFrame(unified_frames)

        # Merge Sports2D MOT angles as additional columns if available
        if not df.empty and self.sports2d_runner:
            mot_df = self.sports2d_runner.load_mot_angles()
            if mot_df is not None and not mot_df.empty and "time" in mot_df.columns:
                mot_df = mot_df.rename(columns={"time": "timestamp"})
                mot_df.columns = ["s2d_" + c if c != "timestamp" else c for c in mot_df.columns]
                if "timestamp" in df.columns:
                    df = pd.merge(df, mot_df, on="timestamp", how="left")

        if not df.empty:
            df.to_csv(csv_path, index=False)
            print(f"[EXPORT] bio_metrics.csv → {csv_path}  ({df.shape[0]} rows × {df.shape[1]} cols)")
        else:
            print(f"[EXPORT] bio_metrics.csv → SKIPPED (no frames analyzed)")

        # ── OpenSim TRC ───────────────────────────────────────────────────────
        if trc_path and self.pose_frames:
            writer = OpenSimFileWriter()
            writer.write_trc(
                pose_frames     = self.pose_frames,
                path            = trc_path,
                fps             = self._fps_cache,
                pix_to_m        = self.PIX_TO_M or 0.002,
                frame_height_px = self._frame_height_px or 720,
            )

        # ── OpenSim MOT ───────────────────────────────────────────────────────
        if mot_path and self.bio_engine and self.bio_engine.frames:
            writer = OpenSimFileWriter()
            writer.write_mot(
                bio_frames = self.bio_engine.frames,
                path       = mot_path,
                fps        = self._fps_cache,
            )

        return payload

    # ── Legacy export helpers (kept for backward compatibility) ───────────────

    def export_json(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "player_summary": asdict(self.summary),
                "frame_metrics":  [asdict(m) for m in self.frame_metrics],
            }, f, indent=2)
        print(f"[EXPORT] JSON → {path}")

    def export_csv(self, path: str):
        pd.DataFrame([asdict(m) for m in self.frame_metrics]).to_csv(path, index=False)
        print(f"[EXPORT] CSV  → {path}")

    def export_biomechanics_csv(self, path: str):
        if self.bio_engine and self.bio_engine.frames:
            self.bio_engine.get_dataframe().to_csv(path, index=False)
            print(f"[EXPORT] Bio CSV → {path}")

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(m) for m in self.frame_metrics])

    # ── Report ────────────────────────────────────────────────────────────────

    def get_report_string(self) -> str:
        s   = self.summary
        dm  = self._det_layer.mode.upper() if hasattr(self, '_det_layer') and self._det_layer else "BLOB"
        bio = "sports2d" if HAS_SPORTS2D else "scipy" if HAS_SCIPY else "numpy"
        W   = 70
        lines = ["=" * W,
                 f"SPORTS ANALYTICS v6 — Player #{s.player_id} [{dm}]".center(W),
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
                 f"  Changes/Min     : {s.direction_change_freq:>6.1f}",
                 f"  Energy (avg)    : {s.estimated_energy_kcal_hr:>6.0f} W"]

        if self.bio_engine and self.bio_engine.frames:
            bd = self.bio_engine.summary_dict()
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
