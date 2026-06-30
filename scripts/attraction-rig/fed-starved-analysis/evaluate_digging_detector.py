from pathlib import Path
import importlib.util
import sys
import types

import numpy as np
import pandas as pd


DATA_ROOT = Path(
    "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/"
    "social-isolation/head-head/2/agarose-plates"
)

KNOWN_DIGGING = {
    DATA_ROOT / "socially-isolated/starved-starved/2026-06-22_12-05-22_td10.tracks.feather": {
        0: 1000,
        1: 2721,
    },
    DATA_ROOT / "socially-isolated/fed-starved/2026-06-22_11-59-31_td5.tracks.feather": {
        1: 2340,
    },
    DATA_ROOT / "group-housed/starved-starved/2026-06-23_15-58-45_td12.tracks.feather": {
        1: 1410,
    },
}


def load_analysis_class():
    install_import_stubs()
    analysis_path = Path(__file__).with_name("fed-starved_analysis.py")
    spec = importlib.util.spec_from_file_location("fed_starved_analysis", analysis_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FedStarvedAnalysis


def install_import_stubs():
    def noop(*args, **kwargs):
        return None

    def stub_module(name, attrs=None):
        if name in sys.modules:
            return sys.modules[name]

        module = types.ModuleType(name)
        for attr, value in (attrs or {}).items():
            setattr(module, attr, value)
        sys.modules[name] = module
        return module

    class Dummy:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return None

    optional_modules = {
        "shapely": {},
        "shapely.geometry": {"Polygon": Dummy, "Point": Dummy},
        "shapely.affinity": {"scale": noop},
        "shapely.wkt": {"dumps": noop, "loads": noop},
        "scipy": {},
        "scipy.spatial": {"ConvexHull": Dummy},
        "scipy.spatial.distance": {"cdist": noop, "pdist": noop},
        "scipy.ndimage": {"label": noop, "find_objects": noop},
        "scipy.stats": {"gaussian_kde": Dummy},
        "seaborn": {},
        "matplotlib": {},
        "matplotlib.pyplot": {},
        "cv2": {},
        "joblib": {"Parallel": Dummy, "delayed": lambda func: func},
    }

    for name, attrs in optional_modules.items():
        try:
            __import__(name)
        except ImportError:
            stub_module(name, attrs)


def contiguous_segments(frames):
    frames = np.asarray(frames, dtype=int)
    if len(frames) == 0:
        return []

    breaks = np.where(np.diff(frames) > 1)[0] + 1
    runs = np.split(frames, breaks)
    return [(int(run[0]), int(run[-1]), int(len(run))) for run in runs]


def rel_path(path):
    return str(path.relative_to(DATA_ROOT))


def main():
    analysis_cls = load_analysis_class()
    analysis = analysis_cls.__new__(analysis_cls)

    files = sorted(DATA_ROOT.glob("*/*/*.tracks.feather"))
    print(f"Checking {len(files)} track files under {DATA_ROOT}")
    print("Assumption: only KNOWN_DIGGING entries are verified digging.")
    print()

    tp = fp = fn = tn = 0
    onset_rows = []
    ambiguous_segments = []

    for path in files:
        df = pd.read_feather(path)
        result = analysis.compute_digging(df.copy())

        for track_id, group in result.groupby("track_id"):
            track_id = int(track_id)
            frames = group["frame"].to_numpy()
            predicted = group["digging_status"].to_numpy(dtype=bool)
            truth = np.zeros(len(group), dtype=bool)

            onset = KNOWN_DIGGING.get(path, {}).get(track_id)
            if onset is not None:
                truth = frames >= onset

            tp += int((predicted & truth).sum())
            fp += int((predicted & ~truth).sum())
            fn += int((~predicted & truth).sum())
            tn += int((~predicted & ~truth).sum())

            if onset is not None:
                candidate = np.where(predicted & (frames >= onset - 150))[0]
                first = int(frames[candidate[0]]) if len(candidate) else None
                onset_rows.append(
                    {
                        "file": rel_path(path),
                        "track_id": track_id,
                        "labelled_onset": onset,
                        "first_detected": first,
                        "offset": None if first is None else first - onset,
                        "missed_frames_after_onset": int((~predicted & truth).sum()),
                    }
                )
            elif predicted.any():
                for start, end, length in contiguous_segments(frames[predicted]):
                    ambiguous_segments.append(
                        {
                            "file": rel_path(path),
                            "track_id": track_id,
                            "start": start,
                            "end": end,
                            "frames": length,
                        }
                    )

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0

    print("Known digging onsets")
    print(pd.DataFrame(onset_rows).to_string(index=False))
    print()
    print("Frame-level summary")
    print(f"true_positive_frames: {tp}")
    print(f"false_positive_frames_under_assumption: {fp}")
    print(f"false_negative_frames: {fn}")
    print(f"true_negative_frames: {tn}")
    print(f"precision_under_assumption: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"specificity_under_assumption: {specificity:.4f}")
    print()

    print("Segments detected outside the verified labels")
    if ambiguous_segments:
        print(pd.DataFrame(ambiguous_segments).to_string(index=False))
        print()
        print(
            "These are the segments to visually verify: they look track-wise like "
            "long confined/compact states, so they may be unlabelled digging or "
            "tracking/edge cases rather than ordinary crawling."
        )
    else:
        print("None")


if __name__ == "__main__":
    main()
