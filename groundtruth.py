"""
compare_angles.py
=================
Compares IMU axial angle data (from imu_main.py) against a visual tool's
angle log, matched by closest epoch timestamp.

File formats
------------
Visual tool (file 1):
    EPOCH, TIME_FROM_START, LOWER_ANGLE, UPPER_ANGLE, SEGMENT_ANGLE

IMU log (file 2):
    CH, EPOCH, TIME_FROM_START, Z_ABS, Y_ABS, X_ABS, Z_REL, Y_REL, X_REL

Comparisons
-----------
  ch2 X_ABS  ←→  LOWER_ANGLE
  ch3 X_ABS  ←→  SEGMENT_ANGLE

Metrics per pair
----------------
  RMSE, MAE, Mean Error (bias), Standard Deviation of differences

Usage
-----
  python compare_angles.py <visual_file> <imu_file> [--max-dt 0.05]
                           [--intervals T0_start T0_end T1_start T1_end ...]

  --max-dt     Maximum allowed epoch gap (seconds) for a match to be accepted.
               Defaults to 0.05 s (half a 50 Hz frame).
  --intervals  Even number of time values (seconds, relative to plot start=0)
               defining one or more analysis intervals.
               Example: --intervals 10 30 50 80
               → interval 1: [10 s, 30 s], interval 2: [50 s, 80 s]
               Metrics are computed per interval for each comparison.
"""

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────

def load_visual(path: str) -> np.ndarray:
    """
    Returns array of shape (N, 5):
      col 0  epoch
      col 1  time_from_start
      col 2  lower_angle
      col 3  upper_angle
      col 4  segment_angle
    """
    rows = []
    with open(path, newline="") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            try:
                rows.append([float(p) for p in parts[:5]])
            except ValueError:
                continue
    if not rows:
        sys.exit(f"[ERROR] No valid rows found in visual file: {path}")
    return np.array(rows, dtype=float)


def load_imu(path: str) -> dict[int, np.ndarray]:
    """
    Returns dict keyed by channel int.
    Each value is array of shape (N, 8):
      col 0  epoch
      col 1  time_from_start
      col 2  z_abs
      col 3  y_abs
      col 4  x_abs
      col 5  z_rel
      col 6  y_rel
      col 7  x_rel
    """
    channels: dict[int, list] = {}
    with open(path, newline="") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 9:
                continue
            try:
                ch   = int(parts[0])
                vals = [float(p) for p in parts[1:9]]
            except ValueError:
                continue
            channels.setdefault(ch, []).append(vals)

    result = {}
    for ch, rows in channels.items():
        result[ch] = np.array(rows, dtype=float)

    if not result:
        sys.exit(f"[ERROR] No valid rows found in IMU file: {path}")
    return result


# ─────────────────────────────────────────────
# Matching
# ─────────────────────────────────────────────

def match_by_epoch(
    epochs_a: np.ndarray,
    epochs_b: np.ndarray,
    max_dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each epoch in epochs_a, find the index in epochs_b with the
    closest epoch. Returns (idx_a, idx_b) filtered to pairs
    where |epoch_a - epoch_b| <= max_dt.
    """
    idx_a, idx_b = [], []
    for i, ea in enumerate(epochs_a):
        j = int(np.argmin(np.abs(epochs_b - ea)))
        gap = abs(ea - epochs_b[j])
        if gap <= max_dt:
            idx_a.append(i)
            idx_b.append(j)

    return np.array(idx_a, dtype=int), np.array(idx_b, dtype=int)


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_metrics(ref: np.ndarray, meas: np.ndarray) -> dict:
    """
    ref   : reference signal (visual tool)
    meas  : measured signal (IMU)
    diff  : meas - ref
    """
    diff = meas - ref
    return {
        "n":          len(diff),
        "rmse":       float(np.sqrt(np.mean(diff ** 2))),
        "mae":        float(np.mean(np.abs(diff))),
        "mean_error": float(np.mean(diff)),       # signed bias
        "std":        float(np.std(diff)),
        "diff":       diff,
    }


def filter_finite_pairs(ref: np.ndarray, meas: np.ndarray, time_axis: np.ndarray):
    mask = np.isfinite(ref) & np.isfinite(meas) & np.isfinite(time_axis)
    return ref[mask], meas[mask], time_axis[mask], int((~mask).sum())


def print_metrics(label: str, m: dict) -> None:
    print(f"\n  ┌─ {label}")
    print(f"  │  Matched samples : {m['n']}")
    print(f"  │  RMSE            : {m['rmse']:.4f}°")
    print(f"  │  MAE             : {m['mae']:.4f}°")
    print(f"  │  Mean error      : {m['mean_error']:+.4f}°  (+ = IMU reads higher)")
    print(f"  └  Std of diff     : {m['std']:.4f}°")


def print_interval_metrics(label: str, intervals: list[tuple[float, float]],
                           time: np.ndarray, ref: np.ndarray, meas: np.ndarray) -> list[dict]:
    """Compute and print metrics for each time interval. Returns list of metric dicts."""
    results = []
    for i, (t_start, t_end) in enumerate(intervals):
        mask = (time >= t_start) & (time <= t_end)
        n = mask.sum()
        if n == 0:
            print(f"\n  ┌─ {label}  |  Interval {i+1}: [{t_start:.2f} s – {t_end:.2f} s]")
            print(f"  └  (no samples in this interval)")
            continue
        m = compute_metrics(ref[mask], meas[mask])
        results.append(m)
        print(f"\n  ┌─ {label}  |  Interval {i+1}: [{t_start:.2f} s – {t_end:.2f} s]")
        print(f"  │  Matched samples : {m['n']}")
        print(f"  │  RMSE            : {m['rmse']:.4f}°")
        print(f"  │  MAE             : {m['mae']:.4f}°")
        print(f"  │  Mean error      : {m['mean_error']:+.4f}°  (+ = IMU reads higher)")
        print(f"  └  Std of diff     : {m['std']:.4f}°")
    return results


def print_interval_means(label: str, metrics_list: list[dict]) -> None:
    """Print the mean of each metric across all valid intervals."""
    if not metrics_list:
        return
    keys = ["rmse", "mae", "mean_error", "std"]
    means = {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}
    print(f"\n  ┌─ {label}  |  MEAN across {len(metrics_list)} interval(s)")
    print(f"  │  RMSE            : {means['rmse']:.4f}°")
    print(f"  │  MAE             : {means['mae']:.4f}°")
    print(f"  │  Mean error      : {means['mean_error']:+.4f}°")
    print(f"  └  Std of diff     : {means['std']:.4f}°")


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

COLORS = {
    "visual":        "#3A86FF",
    "imu":           "#FF6B6B",
    "diff":          "#8338EC",
    "zero":          "#AAAAAA",
    "interval_start": "#1D7FDE",   # blue  — interval start
    "interval_end":   "#E03131",   # red   — interval end
}


def plot_comparison(
    time_a: np.ndarray,
    ref: np.ndarray,
    meas: np.ndarray,
    metrics: dict,
    title: str,
    ref_label: str,
    meas_label: str,
    ax_signal,
    ax_diff,
    intervals: list[tuple[float, float]] | None = None,
) -> None:
    # Signal plot
    ax_signal.plot(time_a, ref,  color=COLORS["visual"], lw=1.5, label=ref_label)
    ax_signal.plot(time_a, meas, color=COLORS["imu"],    lw=1.5, label=meas_label, alpha=0.85)
    ax_signal.set_title(title, fontsize=11, fontweight="bold")
    ax_signal.set_ylabel("Angle (°)")
    ax_signal.legend(fontsize=8, loc="upper right")
    ax_signal.grid(True, alpha=0.3)

    # Difference plot
    diff = metrics["diff"]
    ax_diff.plot(time_a, diff, color=COLORS["diff"], lw=1.0, alpha=0.8, label="IMU − Visual")
    ax_diff.axhline(0,                        color=COLORS["zero"], lw=0.8, ls="--")
    ax_diff.axhline(metrics["mean_error"],    color=COLORS["imu"],  lw=1.2, ls=":",
                    label=f"Mean error {metrics['mean_error']:+.3f}°")
    ax_diff.fill_between(
        time_a,
        metrics["mean_error"] - metrics["std"],
        metrics["mean_error"] + metrics["std"],
        alpha=0.15, color=COLORS["diff"], label=f"±1 SD ({metrics['std']:.3f}°)"
    )
    ax_diff.set_ylabel("Difference (°)")
    ax_diff.set_xlabel("Time from start (s)")
    ax_diff.legend(fontsize=7, loc="upper right")
    ax_diff.grid(True, alpha=0.3)

    # Annotation box
    txt = (f"RMSE {metrics['rmse']:.3f}°\n"
           f"MAE  {metrics['mae']:.3f}°\n"
           f"SD   {metrics['std']:.3f}°")
    ax_signal.text(
        0.01, 0.97, txt,
        transform=ax_signal.transAxes,
        fontsize=7.5, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="#cccccc"),
    )

    # Interval vertical lines
    if intervals:
        for i, (t_start, t_end) in enumerate(intervals):
            label_start = f"Start {i+1}" if i == 0 else f"Start {i+1}"
            label_end   = f"End {i+1}"   if i == 0 else f"End {i+1}"
            for ax in [ax_signal, ax_diff]:
                ax.axvline(t_start, color=COLORS["interval_start"], lw=1.4, ls="--",
                           label=label_start if ax is ax_signal else None)
                ax.axvline(t_end,   color=COLORS["interval_end"],   lw=1.4, ls="--",
                           label=label_end   if ax is ax_signal else None)
        # Re-draw legend on signal axes to include interval lines
        ax_signal.legend(fontsize=7.5, loc="upper right")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare visual tool angles vs IMU angles.")
    parser.add_argument("visual_file", help="Path to visual tool log (CSV)")
    parser.add_argument("imu_file",    help="Path to IMU axial angle log (TXT)")
    parser.add_argument(
        "--max-dt", type=float, default=0.05,
        help="Max epoch gap (s) to accept a matched pair (default 0.05)"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save plot to this file path instead of showing interactively"
    )
    parser.add_argument(
        "--intervals", type=float, nargs="+", default=None,
        metavar="T",
        help=(
            "Even number of time values (seconds, relative to plot start=0) "
            "defining analysis intervals. Example: --intervals 10 30 50 80 "
            "gives [10–30 s] and [50–80 s]."
        ),
    )
    args = parser.parse_args()

    # Validate intervals argument
    intervals: list[tuple[float, float]] | None = None
    if args.intervals is not None:
        if len(args.intervals) % 2 != 0:
            sys.exit("[ERROR] --intervals requires an even number of values "
                     "(start/end pairs). Got: " + str(args.intervals))
        intervals = [
            (args.intervals[i], args.intervals[i + 1])
            for i in range(0, len(args.intervals), 2)
        ]
        for k, (t0, t1) in enumerate(intervals):
            if t0 >= t1:
                sys.exit(f"[ERROR] Interval {k+1}: start ({t0}) must be < end ({t1}).")
        print(f"[INFO] Analysis intervals (relative to t=0): "
              + ", ".join(f"[{t0}–{t1} s]" for t0, t1 in intervals))

    print("\n[INFO] Loading files...")
    visual = load_visual(args.visual_file)
    imu    = load_imu(args.imu_file)

    for ch in [2, 3]:
        if ch not in imu:
            sys.exit(f"[ERROR] Channel {ch} not found in IMU file.")

    print(f"       Visual rows : {len(visual)}")
    for ch in sorted(imu):
        print(f"       IMU ch{ch} rows : {len(imu[ch])}")

    # ── Match ch2/ch3 against visual ──────────────────────────────────
    print(f"\n[INFO] Matching by epoch (max_dt={args.max_dt} s)...")

    vis_epochs = visual[:, 0]

    idx_vis2, idx_ch2 = match_by_epoch(vis_epochs, imu[2][:, 0], args.max_dt)
    idx_vis3, idx_ch3 = match_by_epoch(vis_epochs, imu[3][:, 0], args.max_dt)

    if len(idx_vis2) == 0:
        sys.exit("[ERROR] No matched pairs found for ch2. Check epoch ranges or increase --max-dt.")
    if len(idx_vis3) == 0:
        sys.exit("[ERROR] No matched pairs found for ch3. Check epoch ranges or increase --max-dt.")

    print(f"       ch2 ↔ visual : {len(idx_vis2)} matched pairs")
    print(f"       ch3 ↔ visual : {len(idx_vis3)} matched pairs")

    # ── Extract signals ───────────────────────────────────────────────
    # ch2 X_ABS (col 4) ↔ LOWER_ANGLE (col 2)
    lower_ref   = visual[idx_vis2, 2]
    ch2_x_abs   = imu[2][idx_ch2,   4]
    time_lower  = visual[idx_vis2, 1]

    # ch3 X_ABS (col 4) ↔ SEGMENT_ANGLE (col 4)
    segment_ref = visual[idx_vis3, 4]
    ch3_x_abs   = imu[3][idx_ch3,   4]
    time_seg    = visual[idx_vis3, 1]

    lower_ref, ch2_x_abs, time_lower, dropped_lower = filter_finite_pairs(
        lower_ref, ch2_x_abs, time_lower
    )
    segment_ref, ch3_x_abs, time_seg, dropped_segment = filter_finite_pairs(
        segment_ref, ch3_x_abs, time_seg
    )

    if len(lower_ref) == 0:
        sys.exit("[ERROR] No finite matched pairs remain for ch2 after filtering NaN/Inf values.")
    if len(segment_ref) == 0:
        sys.exit("[ERROR] No finite matched pairs remain for ch3 after filtering NaN/Inf values.")

    # ── Normalise time axes to start at 0 ────────────────────────────
    t0_lower = time_lower.min()
    t0_seg   = time_seg.min()
    time_lower = time_lower - t0_lower
    time_seg   = time_seg   - t0_seg

    # ── Global metrics ────────────────────────────────────────────────
    m_lower   = compute_metrics(lower_ref,   ch2_x_abs)
    m_segment = compute_metrics(segment_ref, ch3_x_abs)

    print("\n" + "═" * 52)
    print("  ANGLE COMPARISON RESULTS  (full recording)")
    print("═" * 52)
    if dropped_lower or dropped_segment:
        print(f"  Dropped non-finite pairs : ch2={dropped_lower}, ch3={dropped_segment}")
    print_metrics("ch2 X_abs  vs  Lower Angle",   m_lower)
    print_metrics("ch3 X_abs  vs  Segment Angle", m_segment)
    print("═" * 52)

    # ── Per-interval metrics ──────────────────────────────────────────
    if intervals:
        print("\n" + "═" * 52)
        print("  ANGLE COMPARISON RESULTS  (per interval)")
        print("═" * 52)
        print_interval_metrics(
            "ch2 X_abs  vs  Lower Angle",
            intervals, time_lower, lower_ref, ch2_x_abs,
        )
        print_interval_metrics(
            "ch3 X_abs  vs  Segment Angle",
            intervals, time_seg, segment_ref, ch3_x_abs,
        )
        print("═" * 52)

    # ── Plot ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#F7F7F8")
    fig.suptitle(
        "IMU vs Visual Tool — Angle Comparison",
        fontsize=14, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        hspace=0.55, wspace=0.35,
        top=0.93, bottom=0.07, left=0.07, right=0.97,
        height_ratios=[2, 1],
    )

    ax_lower_sig  = fig.add_subplot(gs[0, 0])
    ax_lower_diff = fig.add_subplot(gs[1, 0], sharex=ax_lower_sig)
    ax_seg_sig    = fig.add_subplot(gs[0, 1])
    ax_seg_diff   = fig.add_subplot(gs[1, 1], sharex=ax_seg_sig)

    for ax in [ax_lower_sig, ax_lower_diff, ax_seg_sig, ax_seg_diff]:
        ax.set_facecolor("#FFFFFF")
        for spine in ax.spines.values():
            spine.set_edgecolor("#DDDDDD")
        ax.xaxis.set_major_locator(plt.MultipleLocator(10))

    plot_comparison(
        time_lower, lower_ref, ch2_x_abs, m_lower,
        title="Lower Angle  (ch2 X_abs  vs  Visual lower)",
        ref_label="Visual — Lower",
        meas_label="IMU ch2 X_abs",
        ax_signal=ax_lower_sig,
        ax_diff=ax_lower_diff,
        intervals=intervals,
    )

    plot_comparison(
        time_seg, segment_ref, ch3_x_abs, m_segment,
        title="Segment Angle  (ch3 X_abs  vs  Visual segment)",
        ref_label="Visual — Segment",
        meas_label="IMU ch3 X_abs",
        ax_signal=ax_seg_sig,
        ax_diff=ax_seg_diff,
        intervals=intervals,
    )

    ax_lower_sig.set_xlabel("Time from start (s)")
    ax_seg_sig.set_xlabel("Time from start (s)")

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"\n[INFO] Plot saved to: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()