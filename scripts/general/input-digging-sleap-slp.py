#!/usr/bin/env python3
"""Fill a SLEAP track over a frame range.

This is useful when an animal disappears into substrate and you want to
programmatically add a stable manual annotation instead of clicking every frame.

Run with the SLEAP environment, for example:

    conda run -n sleap python input-digging-sleap-slp.py \
        /path/to/file.tracks.slp --track track_0 \
        --start-frame 2749 --end-frame 3599 \
        --source-frame 2749

Or set exact coordinates:

    conda run -n sleap python input-digging-sleap-slp.py \
        /path/to/file.tracks.slp --track track_0 \
        --start-frame 2749 --end-frame 3599 \
        --point head:805,364 --point body:783,348 --point tail:767,356
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import sleap
from sleap.instance import Instance, LabeledFrame, Track


DEFAULT_SLP = (
    "/Users/cochral/Desktop/SLAEP/h-h/SI-S-S/"
    "2026-06-22_12-05-22_td10.tracks.slp"
)


def parse_point(text: str) -> Tuple[str, Tuple[float, float]]:
    """Parse a point like 'body:123.4,567.8'."""
    if ":" not in text:
        raise argparse.ArgumentTypeError(
            f"Point must look like node:x,y; got {text!r}"
        )

    node, xy = text.split(":", 1)
    node = node.strip()
    if not node:
        raise argparse.ArgumentTypeError(f"Point has an empty node name: {text!r}")

    if "," not in xy:
        raise argparse.ArgumentTypeError(
            f"Point coordinate must look like x,y; got {text!r}"
        )

    x_text, y_text = xy.split(",", 1)
    try:
        x = float(x_text)
        y = float(y_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Could not parse x,y coordinates from {text!r}"
        ) from exc

    return node, (x, y)


def edited_path_for(slp_path: Path) -> Path:
    """Return '/path/name_edited.slp' for '/path/name.slp'."""
    return slp_path.with_name(f"{slp_path.stem}_edited{slp_path.suffix}")


def get_track(labels: sleap.Labels, track_name: str) -> Track:
    for track in labels.tracks:
        if track.name == track_name:
            return track

    available = ", ".join(track.name for track in labels.tracks[:20])
    if len(labels.tracks) > 20:
        available += ", ..."
    raise ValueError(f"Could not find track {track_name!r}. Available: {available}")


def get_video(labels: sleap.Labels, video_index: int):
    if video_index < 0 or video_index >= len(labels.videos):
        raise ValueError(
            f"Video index {video_index} is out of range. "
            f"This file has {len(labels.videos)} video(s)."
        )
    return labels.videos[video_index]


def get_skeleton(labels: sleap.Labels, skeleton_index: int):
    if skeleton_index < 0 or skeleton_index >= len(labels.skeletons):
        raise ValueError(
            f"Skeleton index {skeleton_index} is out of range. "
            f"This file has {len(labels.skeletons)} skeleton(s)."
        )
    return labels.skeletons[skeleton_index]


def get_or_create_frame(labels: sleap.Labels, video, frame_idx: int) -> LabeledFrame:
    matches = labels.find(video=video, frame_idx=frame_idx)
    if matches:
        return matches[0]

    labeled_frame = LabeledFrame(video=video, frame_idx=frame_idx)
    labels.append(labeled_frame)
    return labeled_frame


def remove_track_instances(
    labels: sleap.Labels, labeled_frame: LabeledFrame, track: Track
) -> int:
    matches = [inst for inst in labeled_frame.instances if inst.track == track]
    for inst in matches:
        labels.remove_instance(labeled_frame, inst)
    return len(matches)


def points_from_source_frame(
    labels: sleap.Labels, video, track: Track, source_frame: int
) -> np.ndarray:
    source_matches = labels.find(video=video, frame_idx=source_frame)
    if not source_matches:
        raise ValueError(f"Frame {source_frame} was not found in this SLEAP file.")

    source_instances = [
        inst for inst in source_matches[0].instances if inst.track == track
    ]
    if not source_instances:
        raise ValueError(
            f"Track {track.name!r} was not found on source frame {source_frame}."
        )

    manual_source_instances = [
        inst for inst in source_instances if type(inst) == Instance
    ]
    source_instance = (
        manual_source_instances[0]
        if manual_source_instances
        else source_instances[0]
    )

    if len(source_instances) > 1:
        print(
            f"Warning: found {len(source_instances)} instances for {track.name!r} "
            f"on frame {source_frame}; using "
            f"{'a manual instance' if manual_source_instances else 'the first one'}."
        )

    return source_instance.numpy().astype(float)


def points_from_cli(
    skeleton, point_args: Iterable[Tuple[str, Tuple[float, float]]]
) -> np.ndarray:
    node_names = list(skeleton.node_names)
    points_by_node: Dict[str, Tuple[float, float]] = dict(point_args)

    unknown_nodes = sorted(set(points_by_node) - set(node_names))
    if unknown_nodes:
        raise ValueError(
            f"Unknown node(s): {', '.join(unknown_nodes)}. "
            f"Available nodes: {', '.join(node_names)}"
        )

    points = np.full((len(node_names), 2), np.nan, dtype=float)
    for node_idx, node_name in enumerate(node_names):
        if node_name in points_by_node:
            points[node_idx] = points_by_node[node_name]

    if np.isnan(points).all():
        raise ValueError("No usable points were supplied.")

    return points


def count_visible_points(points: np.ndarray) -> int:
    return int(np.sum(~np.isnan(points).any(axis=1)))


def list_file_contents(labels: sleap.Labels) -> None:
    print(f"Videos: {len(labels.videos)}")
    print(f"Skeletons: {len(labels.skeletons)}")
    for i, skeleton in enumerate(labels.skeletons):
        print(f"  skeleton {i}: {', '.join(skeleton.node_names)}")

    print(f"Tracks: {len(labels.tracks)}")
    print("  " + ", ".join(track.name for track in labels.tracks))

    if labels.labeled_frames:
        frame_idxs = [lf.frame_idx for lf in labels.labeled_frames]
        print(f"Frames: {min(frame_idxs)} to {max(frame_idxs)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Add or replace one SLEAP track over an inclusive frame range, "
            "saving a new *_edited.slp file by default."
        )
    )
    parser.add_argument(
        "slp_path",
        nargs="?",
        default=DEFAULT_SLP,
        help=f"Input .slp file. Default: {DEFAULT_SLP}",
    )
    parser.add_argument("--track", help="Track name to edit, e.g. track_0.")
    parser.add_argument(
        "--start-frame",
        type=int,
        help="First frame to edit. Inclusive.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        help="Last frame to edit. Inclusive.",
    )
    parser.add_argument(
        "--source-frame",
        type=int,
        help="Copy this track's pose from one existing frame into the frame range.",
    )
    parser.add_argument(
        "--point",
        action="append",
        type=parse_point,
        default=[],
        help=(
            "Exact node coordinate to write, e.g. --point body:123,456. "
            "Repeat for head/body/tail. Mutually exclusive with --source-frame."
        ),
    )
    parser.add_argument(
        "--output",
        help="Output .slp path. Default is the input name with _edited before .slp.",
    )
    parser.add_argument(
        "--video-index",
        type=int,
        default=0,
        help="Video index inside the SLEAP file. Default: 0.",
    )
    parser.add_argument(
        "--skeleton-index",
        type=int,
        default=0,
        help="Skeleton index inside the SLEAP file. Default: 0.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help=(
            "Do not overwrite frames where the chosen track already exists. "
            "Default is to replace that track in the selected frame range."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting the output .slp file if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed, but do not save a file.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List videos, skeleton nodes, tracks, and frame range, then exit.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.list:
        return

    missing = [
        name
        for name in ("track", "start_frame", "end_frame")
        if getattr(args, name) is None
    ]
    if missing:
        formatted = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise ValueError(f"Missing required argument(s): {formatted}")

    if args.start_frame < 0 or args.end_frame < 0:
        raise ValueError("Frame numbers must be zero or higher.")
    if args.end_frame < args.start_frame:
        raise ValueError("--end-frame must be greater than or equal to --start-frame.")

    has_source = args.source_frame is not None
    has_points = len(args.point) > 0
    if has_source == has_points:
        raise ValueError("Use exactly one of --source-frame or one/more --point values.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_args(args)

        slp_path = Path(args.slp_path).expanduser().resolve()
        if not slp_path.exists():
            raise FileNotFoundError(slp_path)

        labels = sleap.load_file(str(slp_path))
        if args.list:
            list_file_contents(labels)
            return

        video = get_video(labels, args.video_index)
        skeleton = get_skeleton(labels, args.skeleton_index)
        track = get_track(labels, args.track)

        if args.source_frame is not None:
            points = points_from_source_frame(labels, video, track, args.source_frame)
            source_text = f"source frame {args.source_frame}"
        else:
            points = points_from_cli(skeleton, args.point)
            source_text = "command-line coordinates"

        visible_points = count_visible_points(points)
        if visible_points == 0:
            raise ValueError("The chosen pose has no visible/non-NaN points.")

        output_path = (
            Path(args.output).expanduser().resolve()
            if args.output
            else edited_path_for(slp_path)
        )
        if output_path.exists() and not args.force and not args.dry_run:
            raise FileExistsError(
                f"{output_path} already exists. Use --force or choose --output."
            )

        edited = 0
        skipped = 0
        removed = 0
        created_frames = 0

        for frame_idx in range(args.start_frame, args.end_frame + 1):
            existing_matches = labels.find(video=video, frame_idx=frame_idx)
            if existing_matches:
                labeled_frame = existing_matches[0]
            else:
                labeled_frame = get_or_create_frame(labels, video, frame_idx)
                created_frames += 1

            current_track_instances = [
                inst for inst in labeled_frame.instances if inst.track == track
            ]
            if args.keep_existing and current_track_instances:
                skipped += 1
                continue

            removed += remove_track_instances(labels, labeled_frame, track)

            new_instance = Instance.from_numpy(points, skeleton=skeleton, track=track)
            labels.add_instance(labeled_frame, new_instance)
            edited += 1

        print(
            f"Prepared {edited} frame(s) for {track.name!r} using {source_text}. "
            f"Visible nodes per frame: {visible_points}."
        )
        if removed:
            print(f"Replaced {removed} existing instance(s) for that track.")
        if skipped:
            print(f"Skipped {skipped} frame(s) because --keep-existing was used.")
        if created_frames:
            print(f"Created {created_frames} missing labeled frame(s).")

        if args.dry_run:
            print("Dry run only: no file saved.")
            return

        os.makedirs(output_path.parent, exist_ok=True)
        labels.save(str(output_path))
        print(f"Saved edited SLEAP file to: {output_path}")

    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    main()
