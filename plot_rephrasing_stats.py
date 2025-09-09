#!/usr/bin/env python3
"""
Scan subfolders of /fsx/joel_niklaus/logs/finephrase/experiments/rephrasing,
read each stats.json, extract the mean of stats.edu_score_difference.mean,
and generate a Plotly bar chart comparing all folders.

Output: plots/rephrasing_edu_score_difference_means.png
"""

import json
import os
from typing import Dict, List, Tuple

import plotly.graph_objects as go

import kaleido
kaleido.get_chrome_sync()


LOGS_ROOT = "/fsx/joel_niklaus/logs/finephrase/experiments/rephrasing"
OUTPUT_DIR = "/fsx/joel_niklaus/projects/finephrase/plots"
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "rephrasing_edu_score_difference_means.png")

# Image export settings
# Approximate DPI control via pixel dimensions (width/height in pixels)
# For ~300 DPI on 8x4.5 inches canvas
PNG_WIDTH_PX = 2400
PNG_HEIGHT_PX = 1350


def find_stats_files(root_dir: str) -> List[Tuple[str, str]]:
    """Return list of (folder_name, stats_json_path) for subfolders containing stats.json."""
    results: List[Tuple[str, str]] = []
    if not os.path.isdir(root_dir):
        return results
    for entry in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, entry)
        if not os.path.isdir(folder_path):
            continue
        stats_path = os.path.join(folder_path, "stats.json")
        if os.path.isfile(stats_path):
            results.append((entry, stats_path))
    return results


def extract_mean_edu_score_difference(stats_json_path: str) -> float:
    """Parse stats.json and return the mean of stats.edu_score_difference.mean.

    The file is expected to be a JSON array of sections; we search for the
    one whose name contains "Education Score Stats Logger" and then read
    ["stats"]["edu_score_difference"]["mean"].
    """
    with open(stats_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # data is a list of blocks; we find the block with the education score stats
    for block in data:
        if isinstance(block, dict) and "name" in block and isinstance(block["name"], str):
            if "Education Score Stats Logger" in block["name"]:
                stats = block.get("stats", {})
                edu = stats.get("edu_score_difference", {})
                mean = edu.get("mean")
                if isinstance(mean, (int, float)):
                    return float(mean)
    raise ValueError(f"edu_score_difference.mean not found in {stats_json_path}")


def collect_means(root_dir: str) -> Dict[str, float]:
    """Map folder name -> mean edu_score_difference."""
    results: Dict[str, float] = {}
    for folder_name, stats_path in find_stats_files(root_dir):
        try:
            mean_val = extract_mean_edu_score_difference(stats_path)
            results[folder_name] = mean_val
        except Exception as exc:
            # Skip folders without the expected structure
            print(f"Warning: skipping {folder_name}: {exc}")
    return results


def parse_model_size_b(folder_name: str) -> float:
    """Extract model size in billions from folder name like '...-0.6B' or '...-14B'.

    Returns float('inf') if no recognizable size is found.
    """
    try:
        # Find the last '-' segment and look for trailing 'B'
        last = folder_name.rsplit('-', 1)[-1]
        if last.endswith('B'):
            num = last[:-1]
            return float(num)
    except Exception:
        pass
    return float('inf')


def make_bar_chart(values_by_folder: Dict[str, float]) -> go.Figure:
    if not values_by_folder:
        raise RuntimeError("No data found to plot.")
    # Sort: 'rewire' group first, then by parsed model size ascending, then name
    def sort_key(name: str):
        is_rewire = 0 if ('rewire' in name.lower()) else 1
        size_b = parse_model_size_b(name)
        return (is_rewire, size_b, name)

    folders = sorted(values_by_folder.keys(), key=sort_key)
    means = [values_by_folder[f] for f in folders]

    fig = go.Figure(
        data=[
            go.Bar(
                x=folders,
                y=means,
                texttemplate="%{y:.3f}",
                textposition="inside",
                insidetextanchor="end",
                textfont=dict(size=42),
                cliponaxis=False,
                marker=dict(color="rgba(31, 119, 180, 0.8)"),
                hovertemplate="%{x}: %{y:.4f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Mean edu_score_difference by rephrasing run",
        xaxis_title="Run folder",
        yaxis_title="Mean edu_score_difference",
        template="plotly_white",
        margin=dict(l=60, r=40, t=90, b=160),
        xaxis_tickangle=-30,
        font=dict(size=36),
        uniformtext_minsize=36,
    )
    return fig


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    values = collect_means(LOGS_ROOT)
    fig = make_bar_chart(values)
    # Requires the 'kaleido' package: pip install -U kaleido
    fig.write_image(OUTPUT_PNG, width=PNG_WIDTH_PX, height=PNG_HEIGHT_PX, scale=1)
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()


