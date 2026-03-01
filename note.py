from __future__ import annotations
import numpy as np


# Standard pitch class names corresponding to the 12 chroma bins
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def get_active_notes(chroma, fps, max_notes=3, global_thresh=0.15, relative_thresh=0.6, min_duration=0.1):
    """
    Analyzes a chromagram to extract active notes, filtering out harmonics and rapid jitter.
    
    Parameters:
    - chroma: The (12, N) chromagram array.
    - fps: Frames per second (Sample Rate / Hop Length).
    - max_notes: Maximum number of notes to return per segment.
    - global_thresh: Absolute minimum magnitude required to not be considered silence.
    - relative_thresh: The "Solo Note Logic". A note must be at least this % as strong 
      as the loudest note in the frame to be counted.
    - min_duration: Minimum duration (in seconds) a note combination must hold to be recorded.
    """
    num_frames = chroma.shape[1]
    active_notes_per_frame = []

    # --- STEP 1: Frame-by-Frame Analysis ---
    for i in range(num_frames):
        frame = chroma[:, i]
        max_val = np.max(frame)

        # 1. Global Threshold: If the loudest note is basically silent, record no notes.
        if max_val < global_thresh:
            active_notes_per_frame.append(tuple())
            continue

        # 2. Relative Threshold (Solo Note Logic):
        # We calculate a dynamic threshold based on the loudest note right now.
        # If the loudest note is 1.0, and relative_thresh is 0.6, all other notes MUST be >= 0.6.
        # This naturally ignores quiet harmonics (which might be 0.2 or 0.3) during a solo piano C.
        dynamic_thresh = max_val * relative_thresh

        # Get indices of the strongest bins, sorted descending by magnitude
        sorted_indices = np.argsort(frame)[::-1]

        current_frame_notes = []
        for idx in sorted_indices[:max_notes]:
            if frame[idx] >= dynamic_thresh:
                current_frame_notes.append(NOTE_NAMES[idx])

        # Sort the note names alphabetically so combinations like ('C', 'E') and ('E', 'C') 
        # are treated as the exact same state for our grouping logic below.
        current_frame_notes.sort()
        active_notes_per_frame.append(tuple(current_frame_notes))

    # --- STEP 2: Run-Length Encoding (Preventing the "Huge Array") ---
    # We only create a new object when the detected notes ACTUALLY change.
    segments = []
    if not active_notes_per_frame:
        return segments

    current_notes = active_notes_per_frame[0]
    start_frame = 0

    for i in range(1, num_frames):
        if active_notes_per_frame[i] != current_notes:
            end_frame = i
            start_time = start_frame / fps
            end_time = end_frame / fps

            # Only save the segment if it's not silence AND it lasted longer than min_duration
            if current_notes and (end_time - start_time) >= min_duration:
                segments.append({
                    "start": round(start_time, 3),
                    "end": round(end_time, 3),
                    "label": list(current_notes)
                })

            current_notes = active_notes_per_frame[i]
            start_frame = i

    # Handle the very last segment in the file
    end_frame = num_frames
    start_time = start_frame / fps
    end_time = end_frame / fps
    if current_notes and (end_time - start_time) >= min_duration:
        segments.append({
            "start": round(start_time, 3),
            "end": round(end_time, 3),
            "label": list(current_notes)
        })

    return segments