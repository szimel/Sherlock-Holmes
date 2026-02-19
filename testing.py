import numpy as np

NOTE_NAMES = np.array(["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"])

CHORD_INTERVALS = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "7":    [0, 4, 7, 10],   # dominant 7
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
}

def _harmonic_pc_offsets(num_harmonics: int) -> np.ndarray:
    """Pitch-class offsets for harmonics 1..H: round(12*log2(i)) mod 12."""
    i = np.arange(1, num_harmonics + 1, dtype=float)
    offsets = np.mod(np.rint(12.0 * np.log2(i)).astype(int), 12)
    return offsets

def build_templates(
    chord_types=("maj","min"),
    num_harmonics=4,
    s=0.6,
    eps=1e-16,
    include_no_chord=True,
):
    """
    Returns:
      P: (12, K) templates, L1-normalized
      labels: list[str] length K (e.g., "C:maj")
    """
    offsets = _harmonic_pc_offsets(num_harmonics)
    weights = (s ** np.arange(num_harmonics)).astype(float)  # i=1 -> s^0 = 1

    templates = []
    labels = []

    for chord_type in chord_types:
        intervals = CHORD_INTERVALS[chord_type]

        for root in range(12):
            p = np.zeros(12, dtype=float)

            if num_harmonics == 1:
                # Binary mask model: chord tones only
                for iv in intervals:
                    p[(root + iv) % 12] = 1.0
            else:
                # Harmonic-dependent model: add harmonic contributions
                for iv in intervals:
                    note_pc = (root + iv) % 12
                    for h_idx, off in enumerate(offsets):
                        pc = (note_pc + off) % 12
                        p[pc] += weights[h_idx]

            # Avoid zeros (paper uses tiny epsilon to avoid log/div issues)
            p = p + eps

            # Paper normalizes templates so sum = 1
            p = p / np.sum(p)

            templates.append(p)
            labels.append(f"{NOTE_NAMES[root]}:{chord_type}")

    if include_no_chord:
        # Simple "no chord": uniform template (also L1 = 1)
        pN = np.ones(12, dtype=float)
        pN = pN / np.sum(pN)
        templates.append(pN)
        labels.append("N")

    P = np.stack(templates, axis=1)  # (12, K)
    return P, labels


def chord_criteria(C, P, measure="KL2", eps=1e-16):
    """
    C: (12, N) chroma (nonnegative)
    P: (12, K) templates (L1 normalized, strictly positive)
    Returns D: (K, N) criteria (lower is better)
    """
    C = np.asarray(C, dtype=float)
    P = np.asarray(P, dtype=float)

    # enforce positivity for divisions/logs
    C = np.maximum(C, eps)
    P = np.maximum(P, eps)

    M, N = C.shape
    _, K = P.shape

    if measure == "EUC":
        # d_{k,n} = sqrt( sum p^2 - (sum c p)^2 / sum c^2 )
        dot = P.T @ C  # (K,N)  sum_m p_m,k * c_m,n
        c2 = np.sum(C**2, axis=0, keepdims=True)  # (1,N)
        p2 = np.sum(P**2, axis=0, keepdims=True).T  # (K,1)
        D = np.sqrt(np.maximum(p2 - (dot**2) / np.maximum(c2, eps), 0.0))
        return D

    if measure == "IS1":
        # d = M log( (1/M) sum (c/p) ) - sum log(c/p)
        ratio = C[:, None, :] / P[:, :, None]  # (12,K,N)
        s1 = np.sum(ratio, axis=0)             # (K,N)
        slog = np.sum(np.log(ratio), axis=0)   # (K,N)
        D = M * np.log(np.maximum(s1 / M, eps)) - slog
        return D

    if measure == "IS2":
        # d = M log( (1/M) sum (p/c) ) - sum log(p/c)
        ratio = P[:, :, None] / C[:, None, :]  # (12,K,N)
        s1 = np.sum(ratio, axis=0)             # (K,N)
        slog = np.sum(np.log(ratio), axis=0)   # (K,N)
        D = M * np.log(np.maximum(s1 / M, eps)) - slog
        return D

    # For KL measures, the paper uses c' = c / ||c||_1 in Table I :contentReference[oaicite:8]{index=8}
    csum = np.sum(C, axis=0, keepdims=True)  # (1,N)
    Cn = C / np.maximum(csum, eps)           # (12,N)
    Cn = np.maximum(Cn, eps)

    logP = np.log(P)
    logCn = np.log(Cn)

    if measure == "KL1":
        # d = 1 - exp( - sum_m c'_m * log(c'_m / p_m,k) )
        # compute S = sum c' * (log c' - log p)
        S = np.sum(Cn[:, None, :] * (logCn[:, None, :] - logP[:, :, None]), axis=0)  # (K,N)
        D = 1.0 - np.exp(-S)
        return D

    if measure == "KL2":
        # d = sum_m [ p_m,k log(p_m,k / c'_m) - p_m,k + c'_m ]
        # (the -p + c' sums to 0, but we keep it explicit like the table)
        term = P[:, :, None] * (logP[:, :, None] - logCn[:, None, :])  # (12,K,N)
        D = np.sum(term, axis=0) + (np.sum(Cn, axis=0, keepdims=True) - np.sum(P, axis=0, keepdims=True).T)
        return D

    raise ValueError(f"Unknown measure: {measure}")



def smooth_criteria(D, L=15, mode="median"):
    """
    D: (K,N)
    L: odd window length
    mode: "median" or "mean"
    """
    D = np.asarray(D, dtype=float)
    K, N = D.shape

    if L <= 1:
        return D
    if L % 2 == 0:
        L += 1  # enforce odd like the paper

    pad = L // 2
    Dp = np.pad(D, ((0, 0), (pad, pad)), mode="edge")

    if mode == "mean":
        kernel = np.ones(L, dtype=float) / L
        out = np.empty((K, N), dtype=float)
        for k in range(K):
            out[k] = np.convolve(Dp[k], kernel, mode="valid")
        return out

    if mode == "median":
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(Dp, window_shape=L, axis=1)  # (K,N,L)
            return np.median(windows, axis=-1)
        except Exception:
            # fallback: slower but safe
            out = np.empty((K, N), dtype=float)
            for k in range(K):
                for n in range(N):
                    out[k, n] = np.median(Dp[k, n:n+L])
            return out

    raise ValueError(f"Unknown smoothing mode: {mode}")



def decode_chords(C, fps, chord_types=("maj","min"), num_harmonics=4,
                  measure="KL2", smooth="median", seconds=2.0,
                  include_no_chord=True, eps=1e-16):
    """
    Returns:
      frame_labels: list[str] length N
      segments: list[dict] with start/end seconds and label
      debug: dict with templates/criteria if you want it
    """
    P, labels = build_templates(
        chord_types=chord_types,
        num_harmonics=num_harmonics,
        s=0.6,          # paper uses s=0.6 :contentReference[oaicite:10]{index=10}
        eps=eps,
        include_no_chord=include_no_chord
    )

    D = chord_criteria(C, P, measure=measure, eps=eps)

    # Paper notes optimal L ~ 13..19 in their setting (~2 seconds). :contentReference[oaicite:11]{index=11}
    L = int(round(seconds * fps))
    if L % 2 == 0:
        L += 1
    L = max(L, 1)

    Df = smooth_criteria(D, L=L, mode=("mean" if smooth == "mean" else "median"))

    idx = np.argmin(Df, axis=0)
    frame_labels = [labels[i] for i in idx]

    # Merge to segments
    dt = 1.0 / fps
    segments = []
    if len(frame_labels) > 0:
        cur = frame_labels[0]
        start = 0
        for n in range(1, len(frame_labels)):
            if frame_labels[n] != cur:
                segments.append({
                    "start": start * dt,
                    "end": n * dt,
                    "label": cur
                })
                cur = frame_labels[n]
                start = n
        segments.append({"start": start * dt, "end": len(frame_labels) * dt, "label": cur})

    debug = {"criteria_raw": D, "criteria_smooth": Df, "templates": P, "labels": labels, "L": L}
    return frame_labels, segments, debug

def majority_filter_int(arr: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    Simple majority filter over integers (helps remove 1-frame spikes).
    Window size = 2*radius+1. Keeps center value on ties.
    """
    n = arr.shape[0]
    out = arr.copy()
    for i in range(n):
        a = max(0, i - radius)
        b = min(n, i + radius + 1)
        window = arr[a:b]
        # bincount requires non-negative ints
        counts = np.bincount(window)
        top = np.max(counts) if counts.size else 0
        winners = np.where(counts == top)[0] if top > 0 else np.array([], dtype=int)

        if winners.size == 1:
            out[i] = int(winners[0])
        else:
            # tie: keep original center value
            out[i] = int(arr[i])
    return out