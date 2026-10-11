"""Undefined-safe clip metrics, validation threshold sweeps and event matching."""
from collections import defaultdict
import numpy as np


def classification_metrics(y_true, probabilities, threshold=0.5):
    from sklearn.metrics import average_precision_score, roc_auc_score

    truth = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(probabilities, dtype=np.float64)
    if truth.shape != prob.shape or truth.ndim != 1 or not len(truth):
        raise ValueError("y_true and probabilities must be nonempty one-dimensional arrays of equal length")
    if not np.isin(truth, [0, 1]).all() or not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise ValueError("Labels must be binary and probabilities finite in [0,1]")
    pred = (prob >= threshold).astype(np.int64)
    tn = int(((truth == 0) & (pred == 0)).sum())
    fp = int(((truth == 0) & (pred == 1)).sum())
    fn = int(((truth == 1) & (pred == 0)).sum())
    tp = int(((truth == 1) & (pred == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None
    result = {
        "threshold": float(threshold), "support": {"negative": int((truth == 0).sum()), "positive": int((truth == 1).sum()), "total": len(truth)},
        "confusion_matrix_labels_0_1": [[tn, fp], [fn, tp]], "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": (tp + tn) / len(truth), "precision": precision, "recall_sensitivity": recall, "f1": f1,
        "average_precision_pr_auc": float(average_precision_score(truth, prob)) if truth.sum() else None,
        "roc_auc": float(roc_auc_score(truth, prob)) if len(np.unique(truth)) == 2 else None,
        "undefined": {"precision": bool(tp + fp == 0), "recall": bool(tp + fn == 0),
                      "f1": bool(precision is None or recall is None or precision + recall == 0),
                      "roc_auc": bool(len(np.unique(truth)) != 2), "average_precision": bool(truth.sum() == 0)},
    }
    return result


def threshold_sweep(y_true, probabilities, thresholds, negative_camera_hours=None):
    rows = []
    for threshold in thresholds:
        row = classification_metrics(y_true, probabilities, threshold)
        row["false_alerts_per_camera_hour"] = (
            row["fp"] / negative_camera_hours if negative_camera_hours and negative_camera_hours > 0 else None
        )
        rows.append(row)
    return rows


def choose_operating_point(rows, *, recall_target=0.98, false_alert_budget_per_camera_hour=0.1):
    feasible = [r for r in rows if r["recall_sensitivity"] is not None
                and r["false_alerts_per_camera_hour"] is not None
                and r["recall_sensitivity"] >= recall_target
                and r["false_alerts_per_camera_hour"] <= false_alert_budget_per_camera_hour]
    if feasible:
        return {"status": "target_and_budget_met", "selected": max(feasible, key=lambda r: (r["precision"] or 0, r["threshold"]))}
    return {"status": "no_threshold_meets_both_constraints", "selected": None,
            "recall_best": max(rows, key=lambda r: r["recall_sensitivity"] if r["recall_sensitivity"] is not None else -1, default=None),
            "false_alert_best": min((r for r in rows if r["false_alerts_per_camera_hour"] is not None),
                                     key=lambda r: r["false_alerts_per_camera_hour"], default=None)}


def merge_predictions(predictions, merge_gap_seconds=2.0):
    """Merge adjacent thresholded predictions independently per camera."""
    groups = defaultdict(list)
    for row in predictions:
        if row.get("positive"):
            groups[str(row.get("camera_id") or "unknown")].append(row)
    merged = []
    for camera_id, rows in groups.items():
        rows.sort(key=lambda r: float(r["decision_timestamp_seconds"]))
        current = None
        for row in rows:
            at = float(row["decision_timestamp_seconds"])
            start = float(row.get("window_start_seconds", at))
            if current is None or at - current["last_positive_timestamp_seconds"] > merge_gap_seconds:
                if current:
                    merged.append(current)
                current = {"camera_id": camera_id, "start_seconds": start, "alert_timestamp_seconds": at,
                           "last_positive_timestamp_seconds": at, "end_seconds": at,
                           "score": float(row["score"]), "prediction_count": 1}
            else:
                current["last_positive_timestamp_seconds"] = at
                current["end_seconds"] = at
                current["score"] = max(current["score"], float(row["score"]))
                current["prediction_count"] += 1
        if current:
            merged.append(current)
    return sorted(merged, key=lambda r: r["alert_timestamp_seconds"])


def event_evaluation(events, candidates, camera_hours, deadlines=(1, 3, 5, 10), early_tolerance=0.0):
    """One-to-one causal event matching; unknown camera IDs do not match silently."""
    matchable = [e for e in events if e.get("event_start_seconds") is not None and e.get("camera_id")]
    result = []
    for deadline in deadlines:
        used = set()
        matched, delays = [], []
        for event in sorted(matchable, key=lambda e: e["event_start_seconds"]):
            onset = float(event["event_start_seconds"])
            choices = [(float(c["alert_timestamp_seconds"]) - onset, i) for i, c in enumerate(candidates)
                       if i not in used and c.get("camera_id") == event.get("camera_id")
                       and onset - early_tolerance <= float(c["alert_timestamp_seconds"]) <= onset + deadline]
            if choices:
                delay, index = min(choices)
                used.add(index)
                matched.append(event.get("incident_id"))
                delays.append(max(0.0, delay))
        duplicate = 0
        false_alerts = 0
        for i, candidate in enumerate(candidates):
            if i in used:
                continue
            nearby = any(e.get("camera_id") == candidate.get("camera_id")
                         and e.get("event_start_seconds") is not None
                         and float(e["event_start_seconds"]) - early_tolerance <= float(candidate["alert_timestamp_seconds"])
                         <= float(e.get("event_end_seconds") or e["event_start_seconds"]) + deadline
                         for e in matchable)
            duplicate += int(nearby)
            false_alerts += int(not nearby)
        hours_known = bool(camera_hours) and sum(camera_hours.values()) > 0
        denominator = len(matchable)
        result.append({
            "deadline_seconds": float(deadline), "matched_incidents": len(matched),
            "event_support_with_known_onset_and_camera": denominator,
            "event_recall": len(matched) / denominator if denominator else None,
            "missed_incidents": denominator - len(matched), "false_alert_candidates": false_alerts,
            "duplicate_alerts": duplicate,
            "false_alerts_per_camera_hour": false_alerts / sum(camera_hours.values()) if hours_known else None,
            "median_detection_delay_seconds": float(np.median(delays)) if delays else None,
            "p95_detection_delay_seconds": float(np.percentile(delays, 95)) if delays else None,
            "events_missing_camera_or_onset": len(events) - denominator,
            "camera_hours_denominator": float(sum(camera_hours.values())) if hours_known else None,
        })
    return result


def grouped_recall_bootstrap(event_outcomes, iterations=2000, seed=42):
    """Bootstrap incident-level hit/miss outcomes, never repeated windows."""
    values = np.asarray(event_outcomes, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        return {"support": 0, "recall": None, "ci95": None}
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return {"support": len(values), "recall": float(values.mean()),
            "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "unit": "independent incident"}
