#!/usr/bin/env python3
"""
BMLrub Dataset Deduplicator
============================
Strategy:
  - BMLrub has ~111 subjects (rub001-rub115), each doing the same ~32 activities.
  - We want motion diversity (different activity types), NOT subject diversity.
  - Keep ONE subject per activity type (the one with the most frames = richest motion).
  - Also de-prioritise low-value activities (sitting, motorcycle, knocking) and
    keep high-value ones (walking, jogging, jumping, kicking, ROM, treadmill).

Output:
  - Prints the curated list of NPZ files to KEEP
  - Copies them into ~/Downloads/BMLrub_curated/ (preserving filenames)
  - Reports: total before, total after, breakdown by activity
"""

import os
import re
import shutil
from collections import defaultdict

BMLRUB_DIR = os.path.expanduser("~/Downloads/BMLrub/")
OUTPUT_DIR  = os.path.expanduser("~/Downloads/BMLrub_curated/")

# ---- Activity priority tiers ----
# Tier 1: Full-body dynamic leg motion (highest value for our 3D lifter)
TIER1 = [
    "treadmill_norm", "treadmill_fast", "treadmill_slow", "treadmill_jog",
    "normal_walk", "normal_jog",
    "jumping", "circle_walk", "scamper",
    "kicking", "rom",                # ROM = Range of Motion = great for joint angles
]

# Tier 2: Good upper-body + some leg involvement
TIER2 = [
    "throwing", "catching_and_throwing", "lifting_heavy", "lifting_light",
]

# Tier 3: Mostly static / low leg movement — keep only 1-2 examples total
TIER3 = [
    "sitting", "knocking", "motorcycle",
]

# How many subjects to keep PER activity per tier
MAX_SUBJECTS_TIER1 = 5   # e.g. 5 different people walking (body shape variety)
MAX_SUBJECTS_TIER2 = 3
MAX_SUBJECTS_TIER3 = 1   # just 1 example of sitting/knocking


def get_activity_key(filename):
    """Strip subject number and sequence index → canonical activity name."""
    # filename like: 0005_normal_walk3_poses.npz
    name = os.path.splitext(filename)[0]  # strip .npz
    # Remove leading index (e.g. 0005_)
    name = re.sub(r'^\d+_', '', name)
    # Remove trailing _poses
    name = re.sub(r'_poses$', '', name)
    # Remove trailing sequence number (walk1 → walk, jog3 → jog)
    name = re.sub(r'\d+$', '', name)
    # Normalise trailing underscore
    name = name.rstrip('_')
    return name


def tier_of(activity_key):
    for t in TIER1:
        if t in activity_key:
            return 1
    for t in TIER2:
        if t in activity_key:
            return 2
    for t in TIER3:
        if t in activity_key:
            return 3
    return 2  # default: treat unknown as tier 2


def main():
    # Collect all npz files: {activity_key: [(num_frames, path), ...]}
    activity_map = defaultdict(list)

    for root, dirs, files in os.walk(BMLRUB_DIR):
        for fname in sorted(files):
            if not fname.endswith('.npz'):
                continue
            fpath = os.path.join(root, fname)
            act   = get_activity_key(fname)
            # Use file size as a cheap proxy for num_frames (bigger = more frames)
            size  = os.path.getsize(fpath)
            activity_map[act].append((size, fpath))

    print(f"Found {sum(len(v) for v in activity_map.values())} total NPZ files")
    print(f"Found {len(activity_map)} unique activity types\n")

    # Sort each activity bucket by size descending (largest = most frames first)
    for act in activity_map:
        activity_map[act].sort(reverse=True)

    # Select files to keep
    to_keep = []
    tier_counts = defaultdict(lambda: defaultdict(int))  # tier → activity → count

    for act, entries in sorted(activity_map.items()):
        tier = tier_of(act)
        limit = {1: MAX_SUBJECTS_TIER1, 2: MAX_SUBJECTS_TIER2, 3: MAX_SUBJECTS_TIER3}[tier]

        kept = 0
        for size, fpath in entries:
            if kept >= limit:
                break
            to_keep.append((tier, act, fpath))
            tier_counts[tier][act] += 1
            kept += 1

    print(f"Curated selection: {len(to_keep)} files (from {sum(len(v) for v in activity_map.values())})")
    print()

    # Print breakdown by tier
    for tier in [1, 2, 3]:
        tier_name = {1: "Tier 1 (Full-body dynamic)", 2: "Tier 2 (Upper body + legs)", 3: "Tier 3 (Static / low motion)"}[tier]
        acts = tier_counts[tier]
        if acts:
            total = sum(acts.values())
            print(f"  {tier_name}: {total} files across {len(acts)} activity types")
            for act, cnt in sorted(acts.items()):
                print(f"    {act:40s}  {cnt} file(s)")
            print()

    # Copy to output directory
    print(f"\nCopying {len(to_keep)} files to {OUTPUT_DIR} ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    copied = 0
    for tier, act, fpath in to_keep:
        fname = os.path.basename(fpath)
        # Include parent subject folder in name to keep unique filenames
        subject = os.path.basename(os.path.dirname(fpath))
        dest_name = f"{subject}_{fname}"
        dest = os.path.join(OUTPUT_DIR, dest_name)
        shutil.copy2(fpath, dest)
        copied += 1
        if copied % 20 == 0:
            print(f"  {copied}/{len(to_keep)} copied...")

    print(f"\nDone! {copied} files in {OUTPUT_DIR}")
    print(f"Reduction: {sum(len(v) for v in activity_map.values())} → {copied} files "
          f"({100*(1 - copied/sum(len(v) for v in activity_map.values())):.0f}% smaller)")


if __name__ == "__main__":
    main()
