import logging
import os
import gc
import asyncio
import tempfile

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Header, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

import testing

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a global lock (1-vCPU queue safety)
analysis_lock = asyncio.Lock()

# Configuration from environment (Railway sets env vars)
API_KEY = os.getenv("API_KEY")  # if set, endpoint requires header 'x-api-key'
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS")  # comma-separated list or '*' for all

if ALLOWED_ORIGINS:
    if ALLOWED_ORIGINS.strip() == "*":
        origins = ["*"]
    else:
        origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
else:
    origins = ["http://localhost:3000"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Simple endpoint to verify the server is alive."""
    return {"status": "healthy"}


def require_api_key(x_api_key: str = Header(None)):
    """Dependency to enforce API key when `API_KEY` is set in env.

    - In production set `API_KEY` on Railway and clients must send header `x-api-key: <API_KEY>`.
    - When `API_KEY` is not set, the endpoint is open (convenient for local development).
    """
    if API_KEY:
        if x_api_key != API_KEY:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return True


@app.post("/analyze", dependencies=[Depends(require_api_key)])
async def analyze(file: UploadFile = File(...)):
    # Wait for our turn in the 1-vCPU queue
    async with analysis_lock:
        tmp_path = None
        try:
            gc.collect()

            # Save upload to disk
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            info = sf.info(tmp_path)
            total_duration = float(info.duration)
            sr_original = int(info.samplerate)
            logger.info(f"Original sample rate: {sr_original}")

            # Analysis params
            T_SR = 11025 * 2  # 22050
            HL = 512

            # Accumulators (chunked)
            all_chroma = []
            all_cqt = []
            all_rms = []
            all_f0 = []
            all_voiced = []
            all_voiced_prob = []

            # Stream through file w/ soundfile (10s blocks to avoid huge RAM)
            block_samples = sr_original * 10

            with sf.SoundFile(tmp_path) as f:
                for block in f.blocks(blocksize=block_samples, dtype="float32", always_2d=True):
                    # mono
                    y_chunk = np.mean(block, axis=1)

                    # resample
                    y_resampled = librosa.resample(y_chunk, orig_sr=sr_original, target_sr=T_SR)

                    # harmonic emphasis (helps pitch/chroma stability)
                    y_harm = librosa.effects.harmonic(y_resampled, margin=3.0)

                    # --- chroma (12 x T_chunk) ---
                    chroma_chunk = librosa.feature.chroma_cqt(
                        y=y_harm,
                        sr=T_SR,
                        hop_length=HL,
                        n_octaves=7,
                        n_chroma=12,
                        bins_per_octave=36,
                        fmin=librosa.note_to_hz("C1"),
                    )

                    # --- CQT magnitude (F x T_chunk) for spectrogram UI ---
                    cqt_chunk = np.abs(librosa.cqt(y=y_harm, sr=T_SR, hop_length=HL, n_bins=84))

                    # --- RMS aligned to hop_length (1 x T_chunk) ---
                    rms_chunk = librosa.feature.rms(
                        y=y_harm,
                        frame_length=2048,
                        hop_length=HL,
                        center=True,
                    )

                    # --- pYIN (F0 + voiced flags/prob) aligned to hop_length (T_chunk,) ---
                    # Pick a pitch range that covers most melodic content.
                    # Tighten these if your content is known (reduces octave errors + compute).
                    fmin = librosa.note_to_hz("C2")
                    fmax = librosa.note_to_hz("C7")

                    f0, voiced_flag, voiced_prob = librosa.pyin(
                        y_harm,
                        fmin=fmin,
                        fmax=fmax,
                        sr=T_SR,
                        hop_length=HL,
                        frame_length=2048,
                        center=True,
                    )

                    # --- Align all time axes for this chunk ---
                    # chroma: (12, Tc), cqt: (84, Tc2), rms: (1, Tr), pyin: (Tp,)
                    m = min(
                        chroma_chunk.shape[1],
                        cqt_chunk.shape[1],
                        rms_chunk.shape[1],
                        f0.shape[0],
                        voiced_flag.shape[0],
                        voiced_prob.shape[0],
                    )

                    if m <= 0:
                        continue

                    chroma_chunk = chroma_chunk[:, :m]
                    cqt_chunk = cqt_chunk[:, :m]
                    rms_chunk = rms_chunk[:, :m]

                    f0 = f0[:m]
                    voiced_flag = voiced_flag[:m]
                    voiced_prob = voiced_prob[:m]

                    all_chroma.append(chroma_chunk)
                    all_cqt.append(cqt_chunk)
                    all_rms.append(rms_chunk.squeeze(0))  # (m,)
                    all_f0.append(f0)
                    all_voiced.append(voiced_flag)
                    all_voiced_prob.append(voiced_prob)

            # Concatenate across chunks
            if not all_chroma:
                raise HTTPException(status_code=400, detail="No audio frames extracted.")

            full_chroma = np.concatenate(all_chroma, axis=1)  # (12, T)
            full_cqt = np.concatenate(all_cqt, axis=1)        # (84, T)
            full_rms = np.concatenate(all_rms, axis=0)        # (T,)
            full_f0 = np.concatenate(all_f0, axis=0)          # (T,)
            full_voiced = np.concatenate(all_voiced, axis=0)  # (T,)
            full_vprob = np.concatenate(all_voiced_prob, axis=0)  # (T,)

            # Delete temp file early
            os.unlink(tmp_path)
            tmp_path = None

            # --- Spectrogram payload (uint8 for frontend) ---
            cqt_db = librosa.amplitude_to_db(full_cqt, ref=np.max)  # (F, T)
            cqt_db = np.clip(cqt_db, -80, 0)
            spec_u8 = np.round((cqt_db + 80) * (255 / 80)).astype(np.uint8)  # (F, T)
            spectrogram_payload = spec_u8.T.tolist()  # (T, F)

            # --- RMS in dB (0..-80) aligned to chroma frames ---
            full_rms_safe = np.maximum(full_rms.astype(float), 1e-12)
            rms_db = librosa.amplitude_to_db(full_rms_safe, ref=np.max)
            rms_db = np.clip(rms_db, -80, 0)

            # --- Pitch class per frame (-1 = unvoiced), from pYIN f0 ---
            pitch_class = np.full(full_f0.shape, -1, dtype=int)
            idx = np.where(full_voiced & np.isfinite(full_f0))[0]
            if idx.size > 0:
                midi = librosa.hz_to_midi(full_f0[idx])
                midi_round = np.rint(midi).astype(int)
                pitch_class[idx] = midi_round % 12

            # --- Chord decoding (expects (12, T)) ---
            C = np.maximum(full_chroma.astype(float), 1e-16)

            _, chord_segments, _ = testing.decode_chords(
                C,
                fps=T_SR / HL,
                chord_types=("maj", "min"),
                num_harmonics=4,
                measure="KL2",
                smooth="median",
                seconds=2,
                include_no_chord=True,
            )

            # JSON cannot contain NaN; convert f0 NaNs to None
            f0_list = [float(x) if np.isfinite(x) else None for x in full_f0]
            voiced_prob_list = [float(x) if np.isfinite(x) else 0.0 for x in full_vprob]
            voiced_list = full_voiced.astype(bool).tolist()

            gc.collect()

            return {
                "file_name": file.filename,
                "duration": total_duration,
                "fps": float(T_SR / HL),
                "sr": int(T_SR),
                "hop_length": int(HL),

                # Chroma (time-major for frontend): (T, 12)
                "tonal_profile": np.mean(full_chroma, axis=1).tolist(),
                "time_series_notes": full_chroma.T.tolist(),

                # Loudness gate aligned to frames: (T,)
                "time_series_rms_db": rms_db.tolist(),

                # pYIN monophonic pitch track aligned to frames: (T,)
                "time_series_f0_hz": f0_list,                      # None when unvoiced
                "time_series_voiced": voiced_list,                 # bool
                "time_series_voiced_prob": voiced_prob_list,       # 0..1-ish
                "time_series_pitch_class": pitch_class.tolist(),   # -1..11

                # Spectrogram (time-major): (T, F)
                "spectrogram_data": spectrogram_payload,

                # Chord segments: list of {"start": float, "end": float, "chord": str}
                "c_chord_segments": chord_segments,

                "status": "Success",
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal Server Error")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            gc.collect()
