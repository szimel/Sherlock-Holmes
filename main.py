import logging
import librosa
import numpy as np
import os
from fastapi import FastAPI, UploadFile, File, Header, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import soundfile as sf
import tempfile
import testing
import asyncio
import gc

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

@app.post("/analyze", dependencies=[Depends(require_api_key)])
async def analyze(file: UploadFile = File(...)):
    # Wait for our turn in the 1-vCPU queue
    async with analysis_lock:
        tmp_path = None
        try:
            gc.collect()
      
            # Save to disk
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            info = sf.info(tmp_path)
            total_duration = info.duration
            sr_original = info.samplerate
            T_SR = 11025
            HL = 512

            all_chroma = []
            all_1D_data = []
            all_cqt = []
            block_samples = sr_original * 10 # scared of that 500mb

            # Stream through file w/ soundfile
            with sf.SoundFile(tmp_path) as f:
                for block in f.blocks(blocksize=block_samples, dtype='float32', always_2d=True):
                    y_chunk = np.mean(block, axis=1)
                    y_resampled = librosa.resample(y_chunk, orig_sr=sr_original, target_sr=T_SR) # 11025 scales down data
                    
                    # creates 3D line animation data
                    y_visual = librosa.resample(y_resampled, orig_sr=11025, target_sr=1000)
                    all_1D_data.append(y_visual)
                    
                    # Only grab harmonics
                    y_harm = librosa.effects.harmonic(y_resampled, margin=3.0)
                    
                    # decomposes & classifies underlying notes
                    chroma_chunk = librosa.feature.chroma_cens( # TODO: try .chroma_cens() - more accurate?
                        y=y_harm, 
                        sr=T_SR, 
                        hop_length=HL,
                        n_octaves=7,
                        n_chroma=12,
                        bins_per_octave=36
                    )
                    all_chroma.append(chroma_chunk)
                    cqt_chunk = np.abs(librosa.cqt(y=y_harm, sr=T_SR, hop_length=HL, n_bins=84))
                    all_cqt.append(cqt_chunk)

            # cleanup
            full_chroma = np.concatenate(all_chroma, axis=1)
            full_cqt = np.concatenate(all_cqt, axis=1)
            full_raw = np.concatenate(all_1D_data)
            os.unlink(tmp_path) # Delete the temp file from disk
            # log scaling for display
            cqt_db = librosa.amplitude_to_db(full_cqt, ref=np.max)    

            # CHAT's
            C = full_chroma.astype(float)
            C = np.maximum(C, 1e-16)
            

            frame_labels, chord_segments, _debug = testing.decode_chords(
                C,
                fps=T_SR/HL,
                chord_types=("maj", "min"), 
                num_harmonics=4,               
                measure="KL2",               
                smooth="median",               
                seconds=2,
                include_no_chord=True
            )

            gc.collect() # Final cleanup before returning data
            return {
                "file_name": file.filename,
                "duration": float(total_duration),
                "fps": T_SR / HL,
                "tonal_profile": np.mean(full_chroma, axis=1).tolist(),
                "time_series_notes": full_chroma.T.tolist(),
                "spectrogram_data": cqt_db.T.tolist(), # The 84-bin 3D terrain
                "raw_data": full_raw.tolist(),
                "c_frame_labels": frame_labels, # len 12 array of which chords 
                "c_chord_segments": chord_segments,
                "status": "Success"
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal Server Error")
            
        finally:
            # Crucial: Clean up the file and RAM even if it fails
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            gc.collect()