"""Spike event clustering — groups overlapping/consecutive positive segments into events."""

from collections import defaultdict

REQUIRED_PHASES = {"jump", "swing"}


def cluster_spike_events(segments, db, max_gap_frames=20):
    """
    Group consecutive spike-positive segments (per track) into discrete spike events,
    then filter to only events that have both a jump and swing phase recorded.

    Args:
        segments: list of Segment ORM objects already filtered to positive labels.
        db: SQLAlchemy Session (used to fetch phase records).
        max_gap_frames: max gap between segment end and next segment start to still merge
                        (default 20 = 2 strides; handles gaps from single missed window).

    Returns:
        List of event dicts, each with keys:
            track_id, start_frame, end_frame, segment_ids,
            phases_present, confidence, segment_count
    """
    if not segments:
        return []

    # Import here to avoid circular imports
    from spike_platform.models.db_models import SegmentPhase

    by_track = defaultdict(list)
    for seg in segments:
        by_track[seg.track_id].append(seg)

    # Prefetch all phases for the involved segments in one query
    seg_ids = [s.id for s in segments]
    all_phases = (
        db.query(SegmentPhase)
        .filter(SegmentPhase.segment_id.in_(seg_ids))
        .all()
    )
    phases_by_seg = defaultdict(list)
    for p in all_phases:
        phases_by_seg[p.segment_id].append(p)

    result = []
    for track_id, segs in by_track.items():
        segs.sort(key=lambda s: s.start_frame)
        groups = _group_consecutive(segs, max_gap_frames)
        for group in groups:
            event = _make_event(track_id, group, phases_by_seg)
            if event is not None:  # None = phase gate not met
                result.append(event)

    result.sort(key=lambda e: (e["track_id"], e["start_frame"]))
    return result


def _group_consecutive(segs, max_gap_frames):
    """Split a sorted list of segments into groups of consecutive windows."""
    groups = []
    group = [segs[0]]
    for seg in segs[1:]:
        if seg.start_frame - group[-1].end_frame <= max_gap_frames:
            group.append(seg)
        else:
            groups.append(group)
            group = [seg]
    groups.append(group)
    return groups


def _make_event(track_id, group, phases_by_seg):
    """
    Build an event dict from a group of consecutive segments.
    Returns None if the group doesn't contain both jump and swing phases.
    """
    event_start = group[0].start_frame
    event_end = group[-1].end_frame

    # Collect all phase records that overlap this event window
    phases_in_event = defaultdict(list)
    for seg in group:
        for p in phases_by_seg.get(seg.id, []):
            if p.start_frame <= event_end and p.end_frame >= event_start:
                phases_in_event[p.phase].append(p)

    # Phase gate: require both jump and swing
    if not REQUIRED_PHASES.issubset(phases_in_event.keys()):
        return None

    # Refine event bounds to phase boundaries (more precise than segment edges)
    if "approach" in phases_in_event:
        refined_start = min(p.start_frame for p in phases_in_event["approach"])
    else:
        refined_start = min(p.start_frame for p in phases_in_event["jump"])

    if "land" in phases_in_event:
        refined_end = max(p.end_frame for p in phases_in_event["land"])
    else:
        refined_end = max(p.end_frame for p in phases_in_event["swing"])

    confs = [s.confidence for s in group if s.confidence is not None]
    return {
        "track_id": track_id,
        "start_frame": refined_start,
        "end_frame": refined_end,
        "segment_ids": [s.id for s in group],
        "phases_present": sorted(phases_in_event.keys()),
        "confidence": float(sum(confs) / len(confs)) if confs else None,
        "segment_count": len(group),
    }
