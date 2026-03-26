#!/usr/bin/env python3
"""
Juventus Sports Analytics System — Entry Point
===============================================
QUICK START:
  pip install ultralytics opencv-python numpy pandas scipy
  pip install sports2d pose2sim        # optional: Sports2D angle engine

  # Basic run:
  python run_analysis.py --video match.mp4

  # Interactive player pick + skeleton video output:
  python run_analysis.py --video match.mp4 --pick --skeleton

  # Full output with biomechanics CSV:
  l

  # Model sizes (auto-downloaded):
  --yolo-size n  fastest, CPU-ok
  --yolo-size m  default, best balance    <-- recommended
  --yolo-size l  most accurate, needs GPU
"""

import argparse
import os
import sys
from sports_analytics import SportsAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Juventus Sports Analytics — single-player biomechanical analysis")

    parser.add_argument("--video",      type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--output",     type=str, default="output_annotated.mp4",
                        help="Annotated output video filename")
    parser.add_argument("--player",     type=int, default=1,
                        help="Player jersey/ID label")
    parser.add_argument("--fps",        type=float, default=None,
                        help="Override video FPS")
    parser.add_argument("--json",       type=str, default="metrics.json",
                        help="JSON export filename")
    parser.add_argument("--csv",        type=str, default="metrics.csv",
                        help="Core metrics CSV filename")
    parser.add_argument("--bio-csv",    type=str, default=None,
                        help="Biomechanics CSV (joint angles, gait events, angular velocity)")
    parser.add_argument("--skeleton",   action="store_true",
                        help="Also produce a skeleton-only video (black background)")
    parser.add_argument("--skel-out",   type=str, default="skeleton.mp4",
                        help="Skeleton video filename (default: skeleton.mp4)")
    parser.add_argument("--pick",       action="store_true",
                        help="Interactive player selection (click to choose)")
    parser.add_argument("--yolo-size",  type=str, default="m",
                        help="YOLO model size: n / s / m / l / x  (default: m)")

    args = parser.parse_args()

    output_dir = "Output"
    os.makedirs(output_dir, exist_ok=True)
    def op(name): return os.path.join(output_dir, os.path.basename(name))

    video_out  = op(args.output)
    json_out   = op(args.json)
    csv_out    = op(args.csv)
    report_out = op("report.txt")
    skel_out   = op(args.skel_out) if args.skeleton else None

    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)

    analyzer = SportsAnalyzer(
        video_path          = args.video,
        output_video_path   = video_out,
        player_id           = args.player,
        fps_override        = args.fps,
        pick                = args.pick,
        yolo_size           = args.yolo_size,
        skeleton_video_path = skel_out,
    )

    summary = analyzer.process_video()

    # Report
    report_str = analyzer.get_report_string()
    print("\n" + report_str + "\n")
    with open(report_out, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"[EXPORT] Report   -> {report_out}")

    # Core exports
    analyzer.export_json(json_out)
    analyzer.export_csv(csv_out)

    # Biomechanics CSV
    if args.bio_csv:
        bio_out = op(args.bio_csv)
        if analyzer.bio_engine and analyzer.bio_engine.frames:
            analyzer.bio_engine.get_dataframe().to_csv(bio_out, index=False)
            print(f"[EXPORT] Bio CSV  -> {bio_out}")
        else:
            print("[WARN] BiomechanicsEngine has no frames — bio CSV skipped.")

    # DF/BIO preview
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY PREVIEW".center(70))
    print("="*70)
    
    df = analyzer.get_dataframe()
    if not df.empty:
        print(f"\n[CORE] Data: {df.shape[0]} frames x {df.shape[1]} columns")
        cols = ["frame_idx","timestamp","speed","cadence","stride_length",
                "left_knee_angle","right_knee_angle","risk_score","gait_symmetry"]
        # Print first few and last few rows to see progress
        print(df[cols].head(3).to_string(index=False))
        print("...")
        print(df[cols].tail(3).to_string(index=False, header=False))

    if analyzer.bio_engine and analyzer.bio_engine.frames:
        bio = analyzer.bio_engine.summary_dict()
        bdf = analyzer.bio_engine.get_dataframe()
        print(f"\n[BIO] Biomechanics Data: {bdf.shape[0]} frames x {bdf.shape[1]} columns")
        print(f"  Knee Flexion (avg) : L {bio.get('left_knee_flexion_mean',0):.1f}° | R {bio.get('right_knee_flexion_mean',0):.1f}°")
        print(f"  Valgus Clinical    : L {bio.get('left_valgus_clinical_mean',0):.2f}° | R {bio.get('right_valgus_clinical_mean',0):.2f}°")
        print(f"  Stride/Gait        : Sym {bio.get('gait_symmetry_pct',0):.1f}% | DS {bio.get('double_support_pct',0):.1f}%")
        print(f"  Event Counts       : LHS={bio.get('lhs_count',0)} | RHS={bio.get('rhs_count',0)}")

    print(f"\n[DONE] Outputs saved to: {output_dir}/")
    print(f"  Annotated video -> {video_out}")
    if skel_out:
        print(f"  Skeleton video  -> {skel_out}")


if __name__ == "__main__":
    main()
