import subprocess
import re
from typing import List, Tuple


def detect_scenes(video_path: str, threshold: float = 0.3, min_scene_len: float = 2.0) -> List[Tuple[float, float]]:
    """
    Hybrid GPU-accelerated scene detection using ffmpeg (CUDA).
    1. Run ffmpeg with scene detection filter to find candidate cuts.
    2. Merge close cuts into longer segments.

    Args:
        video_path: path to video
        threshold: ffmpeg scene change threshold (0.3 ~ good default)
        min_scene_len: minimum scene length in seconds after merging

    Returns:
        List of (start, end) scene ranges in seconds
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "info",
        "-hwaccel", "cuda",
        "-i", video_path,
        "-filter:v", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null",
        "-"
    ]

    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True)

    scene_times = []
    pattern = re.compile(r"pts_time:(\d+\.\d+)")
    for line in proc.stderr:
        match = pattern.search(line)
        if match:
            t = float(match.group(1))
            if not scene_times or t - scene_times[-1] > 0.5:  # avoid duplicate near cuts
                scene_times.append(t)
    proc.wait()

    # Build segments from cuts
    if not scene_times:
        return [(0.0, None)]  # whole video if no cuts

    scenes = []
    prev_time = 0.0
    for t in scene_times:
        if t - prev_time >= min_scene_len:
            scenes.append((prev_time, t))
            prev_time = t

    # Last scene continues until end of video
    scenes.append((prev_time, None))
    return scenes
