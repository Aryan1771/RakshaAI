"""Stable group-level splits; related incident/source records never cross splits."""
from collections import defaultdict
import hashlib


def group_keys(row):
    keys = []
    for name in ("incident_id", "source_video_id", "source_recording_id", "original_recording_id"):
        value = row.get(name)
        if value is not None and str(value).strip():
            keys.append(f"{name}:{value}")
    return keys


def grouped_split(rows, seed=42, fractions=(0.8, 0.1, 0.1)):
    """Assign rows by connected incident/source groups with deterministic balancing.

    Rows lacking both incident and source identity are returned as unassigned;
    they are never independently randomized into a split.
    """
    if len(fractions) != 3 or any(f < 0 for f in fractions) or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("fractions must be three nonnegative values summing to one")
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    owner = {}
    unassigned = []
    for i, row in enumerate(rows):
        keys = group_keys(row)
        if not keys:
            unassigned.append(i)
            continue
        for key in keys:
            if key in owner:
                union(i, owner[key])
            else:
                owner[key] = i
    groups = defaultdict(list)
    for i in range(len(rows)):
        if i not in unassigned:
            groups[find(i)].append(i)

    names = ("train", "validation", "test")
    target = [f * max(1, len(rows) - len(unassigned)) for f in fractions]
    active_splits = [i for i, fraction in enumerate(fractions) if fraction > 0]
    counts = [0, 0, 0]
    assignments = {}
    ordered = sorted(
        groups.values(),
        key=lambda group: (-len(group), hashlib.sha256(f"{seed}:{min(group)}".encode()).hexdigest()),
    )
    for group in ordered:
        # Choose the split with the largest remaining proportional deficit.
        split = max(active_splits, key=lambda j: ((target[j] - counts[j]) / target[j] if target[j] else -1, -j))
        for i in group:
            assignments[i] = names[split]
        counts[split] += len(group)
    return {
        "assignments": assignments,
        "unassigned": unassigned,
        "group_count": len(groups),
        "split_counts": {name: counts[i] for i, name in enumerate(names)},
    }

