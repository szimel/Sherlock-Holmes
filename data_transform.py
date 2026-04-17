from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json

# Could and should use this... but I don't understand the conversion that has to happen 
# that turns this into a json object. So not gonna do it. TODO in future? 
@dataclass
class Item: 
	name: str = ''
	count: float = 0
	seconds: float = 0
	pct: float = 0

	def __post_init__(self):
		if self.name and self.name not in NOTE_NAMES:
			raise ValueError("Name not in list of notes")

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

CHORD_INTERVALS = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "7":    [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
}

def initialize_map(type: bool = False) -> dict:
	map = {}
	for note in NOTE_NAMES: 
		map[note] = {"name": note, "count": 0, "seconds": 0, "pct":0}
		if type: map[note + 'm'] = {"name": note + 'm', "count": 0, "seconds": 0, "pct":0}
	
	return map

def set_values(map: dict, segment: dict, note: str):
	if note == 'N': return
	seconds = segment["end"] - segment["start"]
	map[note]["count"] += 1
	map[note]["seconds"] += seconds

# handles input of chord and note segments. Returns parsed dictionary for front end
def process_segments(segments: list) -> dict:
	isChord = isinstance(segments[0]["label"], str)
	map = initialize_map(isChord)
	count = 0

	# find values for data
	for segment in segments: 
		#check typing of label (list for notes, string for chords)
		if isChord:
			if segment["label"] == 'N': continue
			# splits string at ':'
			name, chord_type = segment["label"].split(":", 1)
			if chord_type == 'min': name = name + 'm'
			set_values(map, segment, name)
			count += 1
		else:
			for note in segment["label"]: 
				set_values(map, segment, note)
				count += 1

	# set values for pct after counts have been set
	for item in map: map[item]["pct"] = map[item]["count"] / count

	return map

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_INDEX = {n: i for i, n in enumerate(NOTE_NAMES)}

CHORD_INTERVALS = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "7":    [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
}


def normalize_quality(q: Optional[str]) -> str:
    if not q:
        return "maj"
    q = q.strip().lower()
    if q in ("m", "minor"):
        return "min"
    if q == "major":
        return "maj"
    return q


def quality_family(q: Optional[str]) -> str:
    qn = normalize_quality(q)
    if qn == "min" or qn.startswith("min") or qn == "dim":
        return "min"
    return "maj"


def parse_chord_label(label: str) -> Tuple[Optional[str], Optional[str], bool]:
    t = (label or "").strip()
    if not t or t == "N" or ("no" in t.lower() and "chord" in t.lower()):
        return None, None, True
    if ":" not in t:
        return None, None, True

    root, quality = [p.strip() for p in t.split(":", 1)]
    if root not in NOTE_INDEX:
        return None, None, True

    return root, normalize_quality(quality), False


def chord_key_from_label(label: str) -> Optional[str]:
    root, quality, no_chord = parse_chord_label(label)
    if no_chord or root is None:
        return None
    return f"{root}m" if quality_family(quality) == "min" else root


def is_note_in_chord(note: str, chord_label: str) -> bool:
    if chord_label == 'N':
        return False

    root, quality, no_chord = parse_chord_label(chord_label)
    if no_chord or root is None:
        return False

    root_idx = NOTE_INDEX[root]
    intervals = CHORD_INTERVALS.get(normalize_quality(quality), [0, 4, 7])
    chord_notes = [NOTE_NAMES[(root_idx + step) % 12] for step in intervals]
    return note in chord_notes


def _build_fingerprint_pairs(n_segs: List[Dict[str, Any]], c_segs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Recreates the old frontend-needed pair rows:
    {note, chord, start, end, related}
    """
    if not n_segs:
        return []

    # Sort chords by start once for deterministic behavior
    chords = sorted(c_segs, key=lambda x: float(x.get("start", 0.0))) if c_segs else []
    fallback_chord = chords[0]["label"] if chords else "N"

    pairs: List[Dict[str, Any]] = []

    for n_seg in n_segs:
        n_start = float(n_seg.get("start", 0.0))
        n_end = float(n_seg.get("end", n_start))
        labels = n_seg.get("label", [])
        notes = labels if isinstance(labels, list) else []

        # IMPORTANT: reset per note segment (old code had this outside loop)
        matching: List[str] = []

        # Chords that START inside [n_start, n_end]
        for c_seg in chords:
            c_start = float(c_seg.get("start", 0.0))
            c_label = c_seg.get("label", "N")

            if n_start <= c_start <= n_end:
                matching.append(c_label)
            elif not matching and c_start >= n_end:
                # fallback to next chord if none started in this note window
                matching.append(c_label)
                break
            elif matching and c_start > n_end:
                # once we passed note window and already have matches, stop
                break

        if not matching:
            matching = [fallback_chord]

        for note in notes:
            if note not in NOTE_INDEX:
                continue
            for chord in matching:
                pairs.append({
                    "note": note,
                    "chord": chord,
                    "start": n_start,
                    "end": n_end,
                    "related": is_note_in_chord(note, chord),
                })

    return pairs


def _collect_fingerprint_summary(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    note_pair_count = [0] * 12
    note_related_count = [0] * 12
    first_start_by_note: List[Optional[float]] = [None] * 12  # JSON-friendly; frontend can treat None as missing

    chord_pair_count: Dict[str, int] = {}
    chord_related_count: Dict[str, int] = {}
    first_start_by_chord: Dict[str, float] = {}

    related_count = 0
    max_pair_duration = 0.05

    for row in pairs:
        note = row.get("note", "")
        chord = row.get("chord", "N")
        start = float(row.get("start", 0.0))
        end = float(row.get("end", start))
        related = bool(row.get("related", False))
        dur = max(0.01, end - start)
        max_pair_duration = max(max_pair_duration, dur)

        note_idx = NOTE_INDEX.get(note)
        if note_idx is not None:
            note_pair_count[note_idx] += 1
            first_note_start = first_start_by_note[note_idx]
            if first_note_start is None or start < first_note_start:
                first_start_by_note[note_idx] = start
            if related:
                note_related_count[note_idx] += 1

        ck = chord_key_from_label(chord)
        if ck:
            chord_pair_count[ck] = chord_pair_count.get(ck, 0) + 1
            if ck not in first_start_by_chord or start < first_start_by_chord[ck]:
                first_start_by_chord[ck] = start
            if related:
                chord_related_count[ck] = chord_related_count.get(ck, 0) + 1

        if related:
            related_count += 1

    fingerprint_count = max(1, len(pairs))
    related_ratio = related_count / fingerprint_count

    note_related_ratio = []
    for i in range(12):
        c = note_pair_count[i]
        note_related_ratio.append((note_related_count[i] / c) if c > 0 else related_ratio)

    chord_related_ratio = {}
    for ck, c in chord_pair_count.items():
        chord_related_ratio[ck] = (chord_related_count.get(ck, 0) / c) if c > 0 else related_ratio

    return {
        "fingerprintCount": fingerprint_count,
        "relatedRatio": related_ratio,
        "notePairCount": note_pair_count,
        "noteRelatedRatio": note_related_ratio,
        "chordPairCount": chord_pair_count,
        "chordRelatedRatio": chord_related_ratio,
        "firstStartByChord": first_start_by_chord,
        "firstStartByNote": first_start_by_note,
        "maxPairDuration": max_pair_duration,
    }


def _sample_fingerprint_pairs(pairs: List[Dict[str, Any]], max_samples: int = 360) -> List[Dict[str, Any]]:
    if max_samples <= 0:
        return []
    if len(pairs) <= max_samples:
        return pairs

    # Deterministic stride sampling (same behavior style as your frontend)
    sampled: List[Dict[str, Any]] = []
    step = len(pairs) / max_samples
    for i in range(max_samples):
        idx = int(i * step)
        sampled.append(pairs[idx])
    return sampled


def create_fingerprint(
    n_segs: List[Dict[str, Any]],
    c_segs: List[Dict[str, Any]],
    max_samples: int = 360
) -> Dict[str, Any]:
    """
    Compact replacement for huge raw fingerprint payload.

    Returns:
    {
      "summary": {...},
      "sampled": [{note, chord, start, end, related}, ... up to max_samples]
    }
    """
    pairs = _build_fingerprint_pairs(n_segs, c_segs)
    summary = _collect_fingerprint_summary(pairs)
    sampled = _sample_fingerprint_pairs(pairs, max_samples=max_samples)

    return {
        "summary": summary,
        "sampled": sampled,
    }