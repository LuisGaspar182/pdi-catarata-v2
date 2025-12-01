#!/usr/bin/env python3
"""
preprocessing.py
- Extrai frames de todos os vídeos em data/raw_videos/
- Cria pasta data/preprocessed/<video_name>/ com frames nomeados frame_000001.png
- Usa ffmpeg via subprocess (recomendado para performance/compatibilidade)
"""

from pathlib import Path
import subprocess
import shlex
import argparse

RAW_DIR = Path("data/videos")
OUT_DIR = Path("data/preprocessed")
FPS_DEFAULT = 30

def extract_frames(video_path: Path, fps: int = FPS_DEFAULT) -> None:
    video_name = video_path.stem
    output_folder = OUT_DIR / video_name
    output_folder.mkdir(parents=True, exist_ok=True)

    out_pattern = str(output_folder / "frame_%06d.png")

    cmd = f'ffmpeg -y -i {shlex.quote(str(video_path))} -vf fps={fps} {shlex.quote(out_pattern)}'
    print(f"Processing {video_path} -> {output_folder} (fps={fps})")
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0:
        print(f"ffmpeg failed for {video_path} (return code {proc.returncode})")

def process_all(fps: int = FPS_DEFAULT) -> None:
    mp4s = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        mp4s.extend(RAW_DIR.glob(ext))
    if not mp4s:
        print(f"No videos found in {RAW_DIR.resolve()}")
        return
    for v in mp4s:
        if v.is_file():
            extract_frames(v, fps=fps)
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Extract frames for all videos in data/raw_videos/")
    parser.add_argument("--fps", type=int, default=FPS_DEFAULT, help="frames per second to extract")
    args = parser.parse_args()
    process_all(fps=args.fps)

if __name__ == "__main__":
    main()