"""TridentNet ensemble: merge candidates from Brains A (detector), B
(segmenter, optional) and C (anomaly autoencoder) into one candidate list.

Rules, in order:

1. A Brain-C anomaly blob that overlaps a Brain-A box (IoU above
   ``corroborate_iou``) *corroborates* it: the A detection keeps its class and
   box but gains provenance ``"AC"`` and a small score lift (two independent
   brains agreeing is real signal; the lift is capped so calibration stays
   honest).
2. Standalone Brain-C blobs survive as open-set ``unknown_anomaly``
   candidates — the things no supervised class covers.
3. Brain-B masks, when present, refine the matching A/C boxes (tight bbox of
   the mask) and add provenance ``"B"``; standalone masks become ghost_net
   candidates (nets/ropes are what the segmenter is trained on).
"""

from __future__ import annotations

from dataclasses import replace

from tridentnet.detector import Detection, box_iou

CORROBORATE_IOU = 0.30
CORROBORATION_LIFT = 0.05  # score bump when two brains agree (capped at 0.99)


def merge_brains(
    detections_a: list[Detection],
    anomalies_c: list[Detection] | None = None,
    segments_b: list[Detection] | None = None,
    corroborate_iou: float = CORROBORATE_IOU,
) -> list[Detection]:
    """Fuse per-brain candidate lists; every output keeps its provenance."""
    merged: list[Detection] = []
    used_c: set[int] = set()

    for det in detections_a:
        corroborated = False
        for idx, blob in enumerate(anomalies_c or []):
            if blob.side == det.side and box_iou(det, blob) >= corroborate_iou:
                used_c.add(idx)
                corroborated = True
        if corroborated:
            merged.append(
                replace(
                    det,
                    brain="AC",
                    score=min(det.score + CORROBORATION_LIFT, 0.99),
                )
            )
        else:
            merged.append(det)

    for idx, blob in enumerate(anomalies_c or []):
        if idx not in used_c:
            merged.append(blob)

    for seg in segments_b or []:
        overlaps = [
            i
            for i, det in enumerate(merged)
            if det.side == seg.side and box_iou(det, seg) >= corroborate_iou
        ]
        if overlaps:
            for i in overlaps:
                det = merged[i]
                merged[i] = replace(
                    det,
                    brain="".join(sorted(set(det.brain + "B"))),
                    score=min(det.score + CORROBORATION_LIFT, 0.99),
                )
        else:
            merged.append(seg)

    merged.sort(key=lambda d: d.score, reverse=True)
    return merged
