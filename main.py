import logging
import librosa
import numpy as np
import os
from fastapi import FastAPI, UploadFile, File, Header, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import soundfile as sf
import tempfile
import chord, note
import asyncio
import gc

# --- added for m4a support (system ffmpeg required) ---
from pathlib import Path
import subprocess

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a global lock
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


def _decode_to_wav_ffmpeg(in_path: str) -> str:
    """
    Decode any input audio file to a temporary WAV using the system 'ffmpeg' binary.
    This enables reading formats that soundfile/libsndfile can't handle (commonly .m4a).
    """
    out_fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(out_fd)

    cmd = [
        "ffmpeg", "-nostdin", "-y",
        "-i", in_path,
        "-vn",
        "-ac", "1",  # mono (you were averaging to mono anyway)
        out_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        # clean up the wav we created
        try:
            if os.path.exists(out_path):
                os.unlink(out_path)
        except Exception:
            pass
        raise RuntimeError(
            "ffmpeg failed to decode audio (is ffmpeg installed on the server?). "
            + proc.stderr.decode("utf-8", errors="ignore")
        )
    return out_path


@app.post("/analyze", dependencies=[Depends(require_api_key)])
async def analyze(file: UploadFile = File(...)):
    # Wait for our turn in the 1-vCPU queue
    async with analysis_lock:
        tmp_path = None
        decoded_tmp_path = None
        decoded_path = None
        try:
            gc.collect()

            # Save to disk (IMPORTANT: preserve the uploaded extension; don't force .mp3)
            orig_ext = Path(file.filename or "").suffix.lower()
            # allow common audio suffixes; if unknown, keep it extensionless
            if orig_ext not in {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac", ".mp4"}:
                orig_ext = ""

            with tempfile.NamedTemporaryFile(delete=False, suffix=orig_ext) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            # Always decode through ffmpeg so every codec (mp3, m4a, etc.)
            # takes the exact same path: ffmpeg → mono WAV → soundfile.
            # This eliminates inconsistencies between libsndfile's decoder
            # and ffmpeg's decoder (different PCM output, different mono
            # downmixing algorithms).
            decoded_tmp_path = _decode_to_wav_ffmpeg(tmp_path)
            decoded_path = decoded_tmp_path
            info = sf.info(decoded_path)

            total_duration = info.duration
            sr_original = info.samplerate
            T_SR = 11024 * 2.5
            HL = 1024

            all_chroma = []
            all_cqt = []
            block_samples = sr_original * 15 # scared of that 500mb

            # Stream through file w/ soundfile
            with sf.SoundFile(decoded_path) as f:
                for block in f.blocks(blocksize=block_samples, dtype='float32', always_2d=True):
                    y_chunk = np.mean(block, axis=1)
                    y_resampled = librosa.resample(y_chunk, orig_sr=sr_original, target_sr=T_SR) # 11025 scales down data

                    # Only grab harmonics
                    y_harm = librosa.effects.harmonic(y_resampled, margin=3.0)

                    # cens seems to do better for note detection
                    chroma_chunk = librosa.feature.chroma_cens(
                        y=y_harm,
                        sr=T_SR,
                        hop_length=HL,
                        n_octaves=7,
                        n_chroma=12,
                        bins_per_octave=84 # scared of mb increase cost of this
                    )

                    # CQT for chord detection - seems to do better than chroma for chords, especially with more harmonics
                    cqt_chunk = librosa.feature.chroma_cqt(
                        y=y_harm,
                        sr=T_SR,
                        hop_length=HL,
                        n_octaves=7,
                        n_chroma=12,
                        bins_per_octave=84
                    )
                    all_cqt.append(cqt_chunk)
                    # for note detection
                    all_chroma.append(chroma_chunk)

            # cleanup
            full_chroma = np.concatenate(all_chroma, axis=1)
            full_cqt = np.concatenate(all_cqt, axis=1)

            # log scaling for spectrogram visualization
            cqt_db = librosa.amplitude_to_db(full_cqt, ref=np.max)
            db = cqt_db  # (F,N)
            db = np.clip(db, -80, 0)
            spec_u8 = np.round((db + 80) * (255/80)).astype(np.uint8)  # (F,N)
            payload = spec_u8.T.tolist()

            # Get chord segments from chord.py
            C = full_chroma.astype(float)
            C = np.maximum(C, 1e-16)
            _, chord_segments, _ = chord.decode_chords(
                C,
                fps=T_SR/HL,
                chord_types=("maj", "min"),
                num_harmonics=4,
                measure="KL2",
                smooth="median",
                seconds=2,
                include_no_chord=True
            )

            # Get note segments from note.py
            note_segments = note.get_active_notes(
                chroma=full_chroma.astype(float),
                fps=T_SR / HL,
                max_notes=3,
                global_thresh=0.45,     # Tweak this to ignore general background noise
                relative_thresh=0.5,    # Tweak this higher (e.g., 0.7) to be stricter about what counts as a note vs a harmonic
                min_duration=0.10       # Ignores blips shorter than 100ms
            )

            gc.collect() # Final cleanup before returning data
            return {
                "file_name": file.filename,
                "duration": float(total_duration),
                "fps": T_SR / HL,
                "spectrogram_data": payload,
                "chord_segments": chord_segments,
                "note_segments": note_segments,
                "status": "Success"
            }

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal Server Error")

        finally:
            # Crucial: Clean up the file and RAM even if it fails
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass
            try:
                if decoded_tmp_path and os.path.exists(decoded_tmp_path):
                    os.unlink(decoded_tmp_path)
            except Exception:
                pass
            gc.collect()