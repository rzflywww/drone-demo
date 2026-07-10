#!/usr/bin/env python3
"""Wrapper script to record video/snapshots from Gazebo ground camera."""

import argparse
import os
import subprocess
import sys
import time

# Default service and topic names — override with --service / --topic
DEFAULT_SERVICE = "/ground_camera/record_video"
DEFAULT_TOPIC = "/ground_camera"


def _gz_service(service, reqtype, reptype, req, timeout=5000):
    """Call a gz service with better error reporting."""
    cmd = [
        "gz", "service", "-s", service,
        "--reqtype", reqtype,
        "--reptype", reptype,
        "--timeout", str(timeout),
        "--req", req,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(
            f"ERROR: gz service call failed (rc={result.returncode})\n"
            f"  service: {service}\n"
            f"  stderr: {result.stderr.strip()}\n"
        )
    return result


def _discover_service(suffix):
    """Try to discover the actual service path via gz service --list."""
    try:
        result = subprocess.run(
            ["gz", "service", "--list"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if line.endswith(suffix):
                    return line
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _discover_topic(suffix):
    """Try to discover the actual topic path via gz topic --list."""
    try:
        result = subprocess.run(
            ["gz", "topic", "--list"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if line.endswith(suffix):
                    return line
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def record_video(output, duration, service=None):
    """Record video using CameraVideoRecorder service."""
    if not output.endswith(".mp4"):
        output += ".mp4"

    output = os.path.abspath(output)
    svc = service or DEFAULT_SERVICE

    sys.stderr.write(f"Recording {duration}s to {output}...\n")

    # Start recording
    result = _gz_service(
        svc,
        "gz.msgs.VideoRecord",
        "gz.msgs.Boolean",
        f'start: true, format: "mp4", save_filename: "{output}"',
    )

    if "data: true" not in result.stdout:
        # Auto-discover the correct service path
        sys.stderr.write(
            f"Service [{svc}] not responding, attempting discovery...\n"
        )
        discovered = _discover_service("record_video")
        if discovered and discovered != svc:
            sys.stderr.write(f"Found service at: {discovered}\n")
            svc = discovered
            result = _gz_service(
                svc,
                "gz.msgs.VideoRecord",
                "gz.msgs.Boolean",
                f'start: true, format: "mp4", save_filename: "{output}"',
            )

        if "data: true" not in result.stdout:
            sys.stderr.write(
                f"ERROR: Failed to start recording\n"
                f"  stdout: {result.stdout.strip()}\n"
                f"  stderr: {result.stderr.strip()}\n"
                f"  Hint: run 'gz service --list' to check available services\n"
            )
            return 1

    # Wait
    sys.stderr.write(f"Recording... ({duration}s)\n")
    for remaining in range(duration, 0, -1):
        sys.stderr.write(f"\r  {remaining}s remaining...")
        sys.stderr.flush()
        time.sleep(1)
    sys.stderr.write("\rDone waiting.        \n")

    # Stop recording
    result = _gz_service(
        svc,
        "gz.msgs.VideoRecord",
        "gz.msgs.Boolean",
        "stop: true",
    )

    if "data: true" in result.stdout:
        # Give Gazebo time to flush & move the temp file
        for _ in range(10):
            time.sleep(0.5)
            if os.path.exists(output):
                break
        if os.path.exists(output):
            size_mb = os.path.getsize(output) / (1024 * 1024)
            sys.stderr.write(f"Saved: {output} ({size_mb:.1f} MB)\n")
        else:
            sys.stderr.write(f"WARNING: File not found: {output}\n")
    else:
        sys.stderr.write(
            f"ERROR: Failed to stop recording\n"
            f"  stdout: {result.stdout.strip()}\n"
            f"  stderr: {result.stderr.strip()}\n"
        )

    return 0


def snapshot(output, topic=None):
    """Take a single PNG snapshot via gz topic pipe."""
    if not output.endswith(".png"):
        output += ".png"

    output = os.path.abspath(output)
    cam_topic = topic or DEFAULT_TOPIC
    sys.stderr.write(f"Capturing snapshot to {output}...\n")

    # Use the camera_recorder.py pipeline
    script_dir = os.path.dirname(os.path.abspath(__file__))
    recorder = os.path.join(script_dir, "camera_recorder.py")

    # Step 1: Capture frame to temp file (ensure file handle is closed)
    tmp_file = "/tmp/gz_frame.txt"
    with open(tmp_file, "w") as fh:
        subprocess.run(
            ["timeout", "15", "gz", "topic", "-e", "-t", cam_topic, "-n", "1"],
            stdout=fh,
            stderr=subprocess.DEVNULL,
        )

    # If the default topic didn't produce output, try auto-discovery
    if not os.path.exists(tmp_file) or os.path.getsize(tmp_file) == 0:
        sys.stderr.write(
            f"Topic [{cam_topic}] produced no data, attempting discovery...\n"
        )
        discovered = _discover_topic("ground_camera")
        if discovered and discovered != cam_topic:
            sys.stderr.write(f"Found topic at: {discovered}\n")
            cam_topic = discovered
            with open(tmp_file, "w") as fh:
                subprocess.run(
                    ["timeout", "15", "gz", "topic", "-e", "-t", cam_topic, "-n", "1"],
                    stdout=fh,
                    stderr=subprocess.DEVNULL,
                )

    # Step 2: Parse and convert
    with open(tmp_file, "r") as f:
        result = subprocess.run(
            ["python3", recorder, "-o", output],
            stdin=f,
        )

    if os.path.exists(output) and os.path.getsize(output) > 0:
        sys.stderr.write(f"Saved: {output}\n")
    else:
        sys.stderr.write(
            "ERROR: Snapshot failed\n"
            f"  Hint: run 'gz topic --list' to check available topics\n"
        )
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Record video/snapshots from Gazebo ground camera"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # video subcommand
    vid = sub.add_parser("video", help="Record MP4 video")
    vid.add_argument("-o", "--output", default="drone_flight.mp4",
                     help="Output MP4 file")
    vid.add_argument("-d", "--duration", type=int, default=5,
                     help="Recording duration (seconds)")
    vid.add_argument("-s", "--service", default=None,
                     help="Gazebo service path (default: auto-detect)")

    # snap subcommand
    snap = sub.add_parser("snap", help="Take a PNG snapshot")
    snap.add_argument("-o", "--output", default="snapshot.png",
                      help="Output PNG file")
    snap.add_argument("-t", "--topic", default=None,
                      help="Gazebo camera topic (default: auto-detect)")

    args = parser.parse_args()

    if args.command == "video":
        return record_video(args.output, args.duration, args.service)
    elif args.command == "snap":
        return snapshot(args.output, args.topic)


if __name__ == "__main__":
    sys.exit(main())
