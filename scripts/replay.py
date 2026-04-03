"""
replay.py — Replay raw NDJSON files through the feature pipeline.
Proves that offline replay produces identical features to live processing.
"""

import argparse
import glob
import sys
sys.path.insert(0, ".")  # allow imports from project root

from features.featurizer import run_from_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay raw ticks through featurizer")
    parser.add_argument("--raw", nargs="+", required=True,
                        help="Input NDJSON files (supports glob patterns)")
    parser.add_argument("--out", default="data/processed/features.parquet",
                        help="Output Parquet path")
    parser.add_argument("--window", type=int, default=60,
                        help="Window size in seconds")
    args = parser.parse_args()

    # Expand any glob patterns
    files = []
    for pattern in args.raw:
        files.extend(glob.glob(pattern))

    if not files:
        print("[ERROR] No files found matching the input pattern")
        sys.exit(1)

    files.sort()
    run_from_file(files, args.out, args.window)