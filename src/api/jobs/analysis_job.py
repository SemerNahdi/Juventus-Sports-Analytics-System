import os
import shutil
import tempfile
import threading
import gc
from typing import Optional, List, Dict, cast, Any
from dataclasses import asdict

from ..database import safe_supabase_update
from .job_manager import register_active_job, unregister_active_job
from .job_helpers import check_cancel, log_step
from ..storage.cloudinary_client import upload_video_to_cloudinary
from ..storage.supabase_storage import upload_file_to_supabase, upload_directory_to_supabase
from ..utils.email_utils import send_analysis_email
from ..config import DEFAULT_STRIDE, DEFAULT_TARGET_HEIGHT
import datetime

def run_full_analysis_job(
    job_id: str,
    temp_input_path: str,
    player_id: int,
    yolo_size: str,
    player_height: float,
    mass_kg: float,
    session_tags: str,
    run_sports2d: bool,
    original_filename: str,
    email: Optional[str] = None,
    stride: int = DEFAULT_STRIDE,
    target_height: int = DEFAULT_TARGET_HEIGHT,
    seed_bbox: Optional[List[int]] = None,
    seed_frame_idx: int = 0
) -> None:
    """Heavy-lifting background task: AI analysis + results upload."""
    job_logs: List[str] = []
    cancel_event = threading.Event()
    register_active_job(job_id, cancel_event)

    try:
        # Helper that uses your log_step function
        def step(msg: str) -> None:
            log_step(job_id, msg, job_logs, cancel_event)

        step("Initializing AI environment...")
        from src.analytics.sports_analytics import SportsAnalyzer, AnalyticsPlotter, HAS_SPORTS2D

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input_" + original_filename)
            output_video_name = "output_annotated.mp4"
            output_video_path = os.path.join(temp_dir, output_video_name)
            results_dir = os.path.join(temp_dir, "results")
            data_dir = os.path.join(temp_dir, "data")

            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(data_dir, exist_ok=True)

            try:
                shutil.move(temp_input_path, input_path)
                step(f"Video file validated. Size: {os.path.getsize(input_path)//1024}KB")
            except Exception as e:
                step(f"CRITICAL ERROR: File move failed: {e}")
                safe_supabase_update(job_id, {"status": "failed", "error": str(e)})
                return

            try:
                step(f"Loading Neural Network ({yolo_size})...")
                gc.collect()

                seed_bbox_tuple: Optional[tuple[int, int, int, int]] = None
                if seed_bbox is not None and len(seed_bbox) == 4:
                    seed_bbox_tuple = cast(tuple[int, int, int, int], tuple(seed_bbox))

                analyzer = SportsAnalyzer(
                    video_path=input_path,
                    output_video_path=output_video_path,
                    player_id=player_id,
                    yolo_size=yolo_size,
                    player_height_m=player_height,
                    pick=False,
                    seed_bbox=seed_bbox_tuple,
                    seed_frame_idx=seed_frame_idx
                )

                s2d_dir = None
                if run_sports2d:
                    step("Invoking deep clinical pipeline (Sports2D)...")
                    s2d_dir = os.path.join(temp_dir, "Sports2D")
                    analyzer.run_sports2d(
                        result_dir=s2d_dir,
                        mode="balanced",
                        participant_mass_kg=mass_kg
                    )
                    step("Clinical data extracted.")
                    gc.collect()

                step("Commencing Pose Estimation & Tracking...")
                summary = analyzer.process_video(stride=stride, target_height=target_height, cancel_event=cancel_event)
                step(f"Tracking concluded. {len(analyzer.frame_metrics)} frames analyzed.")
                gc.collect()

                if check_cancel(cancel_event):
                    raise InterruptedError("Job cancelled by user.")

                step("Synchronizing biomechanical datasets...")
                json_out = os.path.join(data_dir, "analytics_unified.json")
                csv_out = os.path.join(data_dir, "bio_metrics.csv")
                trc_out = os.path.join(data_dir, "trajectories.trc")
                mot_out = os.path.join(data_dir, "motions.mot")
                report_out = os.path.join(data_dir, "report.txt")

                unified_data = analyzer.export_unified(
                    json_path=json_out,
                    csv_path=csv_out,
                    trc_path=trc_out,
                    mot_path=mot_out,
                )
                unified_payload = cast(Dict[str, object], unified_data)
                unified_frames = unified_payload.get("frames", [])

                if check_cancel(cancel_event):
                    raise InterruptedError("Job cancelled by user.")

                with open(report_out, "w", encoding="utf-8") as f:
                    f.write(analyzer.get_report_string())

                step("Synthesizing graphical metrics...")
                plotter = AnalyticsPlotter(results_dir=results_dir, player_id=player_id)
                plotter.generate_all(
                    frame_metrics=analyzer.frame_metrics,
                    bio_engine=analyzer.bio_engine
                )

                step("Uploading finalized assets to cloud storage...")
                asset_prefix = f"jobs/{job_id}"

                video_url = upload_video_to_cloudinary(output_video_path, f"mitus_ai_analytics_{job_id}", cancel_event=cancel_event)
                if not video_url:
                    step("Cloudinary bypass: using direct Supabase storage.")
                    video_url = upload_file_to_supabase(output_video_path, f"{asset_prefix}/{output_video_name}", cancel_event=cancel_event)

                if check_cancel(cancel_event):
                    raise InterruptedError("Job cancelled by user.")

                data_urls = upload_directory_to_supabase(data_dir, f"{asset_prefix}/data", cancel_event=cancel_event)
                plot_urls = upload_directory_to_supabase(results_dir, f"{asset_prefix}/plots", cancel_event=cancel_event)
                sports2d_urls = upload_directory_to_supabase(s2d_dir, f"{asset_prefix}/Sports2D", cancel_event=cancel_event) if s2d_dir else {}

                if check_cancel(cancel_event):
                    raise InterruptedError("Job cancelled by user.")

                step("Job finalized successfully.")
                full_summary = {
                    "player_summary": asdict(summary),
                    "biomechanics_summary": analyzer.bio_engine.summary_dict() if analyzer.bio_engine else {},
                    "frame_metrics": unified_frames,
                    "sports2d_output_files": sports2d_urls,
                }

                safe_supabase_update(job_id, {
                    "status": "success",
                    "video_url": video_url,
                    "summary": full_summary,
                    "data_urls": data_urls,
                    "plot_urls": plot_urls,
                    "player_height": player_height,
                    "mass_kg": mass_kg,
                    "yolo_size": yolo_size,
                    "run_sports2d": run_sports2d
                })

                if email:
                    print(f"[JOB {job_id[:8]}] Queuing email notification...")
                    safe_supabase_update(job_id, {
                        "logs": job_logs + [f"[{datetime.datetime.now().isoformat()}] - Dispatching report email to {email}..."]
                    })
                    try:
                        send_analysis_email(email, job_id, player_id, video_url)
                        job_logs.append(f"[{datetime.datetime.now().isoformat()}] - Email report delivered successfully.")
                    except Exception as e:
                        job_logs.append(f"[{datetime.datetime.now().isoformat()}] - Email Error: {str(e)}")
                    safe_supabase_update(job_id, {"logs": job_logs})

                print(f"[JOB {job_id[:8]}] Successfully completed.")

            except Exception as e:
                import traceback
                traceback.print_exc()

                final_status = "cancelled" if isinstance(e, InterruptedError) else "failed"
                error_msg = "Job cancelled by user." if isinstance(e, InterruptedError) else str(e)

                try:
                    log_step(job_id, f"FATAL ERROR: {error_msg}", job_logs, cancel_event)
                except InterruptedError:
                    pass
                
                safe_supabase_update(job_id, {
                    "status": final_status,
                    "error": error_msg
                })
                print(f"[JOB {job_id[:8]}] {final_status.capitalize()} with error: {error_msg}")

            finally:
                if os.path.exists(temp_input_path):
                    try:
                        os.remove(temp_input_path)
                    except Exception:
                        pass
                unregister_active_job(job_id)

    except Exception as e:
        print(f"[JOB {job_id[:8]}] Outer exception: {e}")
        unregister_active_job(job_id)