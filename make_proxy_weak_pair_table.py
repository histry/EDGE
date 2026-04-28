import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def fmt(value, ndigits=3):
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{ndigits}f}"


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, summary, threshold):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Proxy music",
        "BPM",
        "Beat density",
        "Onset density",
        "All hits",
        f"Hits >= {threshold}",
        "High-conf windows",
        "High-conf share",
        "Mean score",
        "Best score",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Proxy Music Weak Pair Statistics\n\n")
        f.write("These are BPM/beat based weak candidates, not real music-motion labels.\n\n")
        f.write(f"- total motion windows: {summary['total_motion_windows']}\n")
        f.write(f"- total top-k candidate rows: {summary['total_candidate_rows']}\n")
        f.write(f"- high-confidence threshold: {threshold}\n")
        f.write(f"- high-confidence candidate rows: {summary['high_conf_candidate_rows']}\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            values = [
                row["proxy_music"],
                fmt(row["bpm"], 2),
                fmt(row["beat_density"], 3),
                fmt(row["onset_density"], 3),
                str(row["all_hit_count"]),
                str(row["high_conf_hit_count"]),
                str(row["high_conf_unique_windows"]),
                fmt(row["high_conf_share"], 3),
                fmt(row["mean_score"], 3),
                fmt(row["best_score"], 3),
            ]
            f.write("| " + " | ".join(values) + " |\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize proxy weak pair distribution.")
    parser.add_argument("--weak_pairs", default="data/proxy_weak_pairs/weak_pairs.csv")
    parser.add_argument("--proxy_features", default="data/proxy_weak_pairs/proxy_music_features.csv")
    parser.add_argument("--score_threshold", type=float, default=0.8)
    parser.add_argument("--out_csv", default="data/proxy_weak_pairs/proxy_weak_pair_stats.csv")
    parser.add_argument("--out_md", default="data/proxy_weak_pairs/proxy_weak_pair_stats.md")
    args = parser.parse_args()

    pairs = read_csv(args.weak_pairs)
    features = {row["audio_path"]: row for row in read_csv(args.proxy_features)}
    total_windows = len({row["window_id"] for row in pairs})
    high_rows = [row for row in pairs if to_float(row["score"]) >= args.score_threshold]

    grouped = defaultdict(list)
    for row in pairs:
        grouped[row["audio_path"]].append(row)

    rows = []
    for audio_path, feature in sorted(features.items()):
        group = grouped.get(audio_path, [])
        high_group = [row for row in group if to_float(row["score"]) >= args.score_threshold]
        scores = [to_float(row["score"]) for row in group]
        high_share = len(high_group) / max(len(high_rows), 1)
        rows.append(
            {
                "proxy_music": Path(audio_path).name,
                "audio_path": audio_path,
                "bpm": to_float(feature.get("bpm")),
                "beat_density": to_float(feature.get("beat_density")),
                "onset_density": to_float(feature.get("onset_density")),
                "all_hit_count": len(group),
                "high_conf_hit_count": len(high_group),
                "high_conf_unique_windows": len({row["window_id"] for row in high_group}),
                "high_conf_share": high_share,
                "mean_score": sum(scores) / len(scores) if scores else 0.0,
                "best_score": max(scores) if scores else 0.0,
            }
        )

    rows = sorted(rows, key=lambda row: row["high_conf_hit_count"], reverse=True)
    summary = {
        "total_motion_windows": total_windows,
        "total_candidate_rows": len(pairs),
        "high_conf_candidate_rows": len(high_rows),
    }
    write_csv(args.out_csv, rows)
    write_markdown(args.out_md, rows, summary, args.score_threshold)
    print(f"Saved CSV: {args.out_csv}")
    print(f"Saved Markdown: {args.out_md}")
    print(f"Total windows={total_windows}, candidates={len(pairs)}, high_conf={len(high_rows)}")


if __name__ == "__main__":
    main()
