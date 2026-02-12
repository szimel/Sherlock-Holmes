import librosa
# import scipy
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
# import psutil
import os
import soundfile as sf
import tempfile
import testing

# def log_memory(step_name):
#     process = psutil.Process(os.getpid())
#     # Convert bytes to Megabytes
#     mem_mb = process.memory_info().rss / (1024 * 1024)
#     print(f"[{step_name}] Memory Usage: {mem_mb:.2f} MB")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # gc.collect()
    
    # Save to disk. Failsafe for large files
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
    block_samples = sr_original * 15 # 15 second blocks for processing

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
            # E. Clear "Pool C" (Workspace)
            # del y_chunk, y_resampled, y_harm, chroma_chunk
            # gc.collect()

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
				chord_types=("maj", "min"),     # start simple
				num_harmonics=4,               # paper’s strong setting :contentReference[oaicite:12]{index=12}
				measure="KL2",                 # paper’s best overall in their table :contentReference[oaicite:13]{index=13}
				smooth="median",               # their best smoothing :contentReference[oaicite:14]{index=14}
				seconds=2,
				include_no_chord=True
		)

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