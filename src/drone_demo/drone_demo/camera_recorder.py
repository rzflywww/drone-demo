#!/usr/bin/env python3
"""
Record video/snapshots from Gazebo camera via gz topic pipe.

Usage:
  # Single PNG snapshot
  gz topic -e -t /ground_camera -n 1 | python3 camera_recorder.py -o snap.png

  # 5-second MP4 video
  timeout 8 gz topic -e -t /ground_camera | python3 camera_recorder.py -o flight.mp4 -d 5

  # N-frame MP4 video
  timeout 15 gz topic -e -t /ground_camera | python3 camera_recorder.py -o flight.mp4 -n 300
"""

import argparse
import os
import subprocess
import sys
import time


def extract_frame(input_stream):
    """
    State-machine parser for gz topic -e protobuf text.
    Handles very large data fields efficiently.
    Yields (width, height, raw_bytes).
    """
    width = height = 0
    in_data_block = False
    data_parts = []

    # We process the stream in chunks to handle the massive data field
    chunk = input_stream.read(8192)
    leftover = ""

    while chunk:
        text = leftover + chunk
        leftover = ""

        lines = text.split("\n")

        # The last line might be incomplete; save it for next iteration
        for line in lines[:-1]:
            stripped = line.lstrip()

            if "width:" in stripped:
                try:
                    width = int(stripped.split(":", 1)[1].strip())
                except ValueError:
                    pass

            elif "height:" in stripped:
                try:
                    height = int(stripped.split(":", 1)[1].strip())
                except ValueError:
                    pass

            elif stripped.startswith('data: "'):
                # Start of data - extract everything between quotes
                idx = stripped.index('"')
                content = stripped[idx + 1:]
                in_data_block = True
                # Check if this line also ends the data
                if content.endswith('"'):
                    escaped = content[:-1]
                    if width and height:
                        yield (width, height, unescape(escaped))
                        width = height = 0
                    in_data_block = False
                    data_parts = []
                else:
                    data_parts = [content]

            elif in_data_block:
                if stripped.endswith('"'):
                    # End of data
                    data_parts.append(stripped[:-1])
                    escaped = "".join(data_parts)
                    if width and height:
                        yield (width, height, unescape(escaped))
                        width = height = 0
                    in_data_block = False
                    data_parts = []
                else:
                    data_parts.append(stripped)

        leftover = lines[-1] if lines else ""

        chunk = input_stream.read(8192)

    # Process any remaining leftover
    if leftover and in_data_block:
        stripped = leftover.lstrip()
        if stripped.endswith('"'):
            data_parts.append(stripped[:-1])
            escaped = "".join(data_parts)
            if width and height:
                yield (width, height, unescape(escaped))


def unescape(s):
    """Decode C-style octal-escaped string to raw bytes (uses C codec for speed)."""
    import codecs
    # codecs.escape_decode handles \ooo, \xhh, \\, \n, \t, etc.
    raw, _ = codecs.escape_decode(s.encode("latin-1"))
    return raw


def main():
    parser = argparse.ArgumentParser(
        description="Record Gazebo camera via gz topic pipe")
    parser.add_argument("-o", "--output", required=True,
                        help="Output file (.mp4 or .png)")
    parser.add_argument("-n", "--frames", type=int, default=None,
                        help="Max frames")
    parser.add_argument("-d", "--duration", type=float, default=None,
                        help="Max duration (seconds)")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    frames_raw = []
    width = height = 0
    start_time = time.time()

    sys.stderr.write("Recording... (pipe from gz topic -e stdin)\n")

    for w, h, data in extract_frame(sys.stdin):
        width, height = w, h
        expected = width * height * 3
        actual = len(data)

        if actual == expected:
            frames_raw.append(data)
            if len(frames_raw) == 1:
                sys.stderr.write(f"Resolution: {width}x{height}\n")
        else:
            sys.stderr.write(
                f"[WARN] Frame {len(frames_raw)+1}: "
                f"size mismatch (got {actual}, expected {expected})\n"
            )

        elapsed = time.time() - start_time
        sys.stderr.write(f"\rFrames: {len(frames_raw)} ({elapsed:.1f}s)")
        sys.stderr.flush()

        if args.frames and len(frames_raw) >= args.frames:
            break
        if args.duration and elapsed >= args.duration:
            break

    sys.stderr.write("\n")

    if not frames_raw:
        sys.stderr.write("ERROR: No frames captured!\n")
        return 1

    sys.stderr.write(
        f"Captured {len(frames_raw)} frames ({width}x{height})\n"
    )
    sys.stderr.write("Encoding with ffmpeg...\n")

    is_image = args.output.endswith((".png", ".jpg", ".jpeg"))
    is_single = len(frames_raw) <= 1

    if is_image or is_single:
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", "1",
            "-i", "-", "-frames:v", "1",
        ]
        if args.output.endswith(".png"):
            cmd += ["-c:v", "png"]
        elif args.output.endswith((".jpg", ".jpeg")):
            cmd += ["-c:v", "mjpeg", "-q:v", "2"]
        else:
            cmd += ["-c:v", "png"]
        cmd.append(args.output)
    else:
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(args.fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "fast",
            "-crf", "23", "-pix_fmt", "yuv420p",
            args.output,
        ]

    raw_data = b"".join(frames_raw)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    _, ffmpeg_stderr = proc.communicate(input=raw_data)

    if proc.returncode == 0:
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        sys.stderr.write(f"Done: {args.output} ({size_mb:.1f} MB)\n")
    else:
        sys.stderr.write(
            f"ffmpeg error (code {proc.returncode}):\n"
            f"{ffmpeg_stderr.decode('utf-8', errors='replace')[:800]}\n"
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
