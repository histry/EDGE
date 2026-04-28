import argparse
import csv
import json
from pathlib import Path


METRICS = [
    ("keyframe_mpjpe_m_mean", "Keyframe MPJPE mean (cm)", 100.0, 2),
    ("keyframe_mpjpe_m_max", "Keyframe MPJPE max (cm)", 100.0, 2),
    ("keyframe_rot_err_deg_mean", "Keyframe rot err mean (deg)", 1.0, 2),
    ("trajectory_ade_m", "Trajectory ADE (cm)", 100.0, 2),
    ("trajectory_rmse_m", "Trajectory RMSE (cm)", 100.0, 2),
    ("trajectory_max_error_m", "Trajectory max err (cm)", 100.0, 2),
    ("trajectory_final_error_m", "Trajectory final err (cm)", 100.0, 2),
    ("foot_slide_rate", "Foot slide rate (%)", 100.0, 2),
    ("foot_contact_speed_p95_mps", "Foot contact speed P95 (m/s)", 1.0, 3),
    ("beatalign_symmetric", "BeatAlign symmetric", 1.0, 3),
    ("beatalign_motion_to_audio", "BeatAlign motion->audio", 1.0, 3),
    ("beatalign_audio_to_motion", "BeatAlign audio->motion", 1.0, 3),
]


def load_metrics(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "metrics" in data:
        return data["metrics"]
    return data


def label_from_path(path):
    stem = Path(path).stem
    for token in ("resmoothed_", "stage2B_train5_4key_midmove_filter_20s_"):
        if token in stem:
            return stem.split(token)[-1]
    return stem


def fmt_value(value, scale, ndigits):
    if value is None:
        return ""
    try:
        value = float(value) * scale
    except Exception:
        return str(value)
    return f"{value:.{ndigits}f}"


def write_csv(path, labels, table_rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric"] + labels)
        for row in table_rows:
            writer.writerow(row)


def write_markdown(path, labels, table_rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Quantitative Evaluation Comparison\n\n")
        f.write("| Metric | " + " | ".join(labels) + " |\n")
        f.write("| --- | " + " | ".join(["---"] * len(labels)) + " |\n")
        for row in table_rows:
            f.write("| " + " | ".join(row) + " |\n")


def main():
    parser = argparse.ArgumentParser(description="Build comparison table from eval_quantitative JSON outputs.")
    parser.add_argument("json_files", nargs="+")
    parser.add_argument("--labels", default="", help="Comma separated labels matching json_files")
    parser.add_argument("--out_csv", default="output/v5_v6_eval_comparison.csv")
    parser.add_argument("--out_md", default="output/v5_v6_eval_comparison.md")
    args = parser.parse_args()

    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    if labels and len(labels) != len(args.json_files):
        raise ValueError("--labels count must match json_files count")
    if not labels:
        labels = [label_from_path(path) for path in args.json_files]

    metrics_by_file = [load_metrics(path) for path in args.json_files]
    table_rows = []
    for key, title, scale, ndigits in METRICS:
        row = [title]
        for metrics in metrics_by_file:
            row.append(fmt_value(metrics.get(key), scale, ndigits))
        table_rows.append(row)

    write_csv(args.out_csv, labels, table_rows)
    write_markdown(args.out_md, labels, table_rows)
    print(f"Saved CSV: {args.out_csv}")
    print(f"Saved Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
