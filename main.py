import librosa
import numpy as np
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import psutil
import os
import gc
import soundfile as sf
import tempfile

def log_memory(step_name):
    process = psutil.Process(os.getpid())
    # Convert bytes to Megabytes
    mem_mb = process.memory_info().rss / (1024 * 1024)
    print(f"[{step_name}] Memory Usage: {mem_mb:.2f} MB")
    
# Settings w/ 500MB RAM / 1 vCPU in mind:
SR = 11025 # gets rid of really high frequencies via downsampling
HOP_LENGTH = 512  # data samples per second, 512 = 21.5 data fps
CHUNK_SEC = 5    # Process 30 seconds at a time

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
    
    log_memory("1 ")

    # 1. Save to disk to keep RAM free
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # 2. Get Duration and SR from the header
    info = sf.info(tmp_path)
    total_duration = info.duration
    sr_original = info.samplerate

    all_chroma = []
    # 30-second blocks at the original sample rate
    block_samples = sr_original * 15 
    log_memory("3 ")

    # 3. Stream through the file
    with sf.SoundFile(tmp_path) as f:
        # always_2d ensures we get (samples, channels) even for mono
        for block in f.blocks(blocksize=block_samples, dtype='float32', always_2d=True):
            log_memory("4 ")
            
            # A. Downmix to Mono (Average the channels)
            y_chunk = np.mean(block, axis=1)
            log_memory("5 ")
            
            # B. Resample chunk to our analysis rate (11025)
            y_resampled = librosa.resample(y_chunk, orig_sr=sr_original, target_sr=11025)
            log_memory("6 ")
            
            # C. HPSS on the chunk
            y_harm, _ = librosa.effects.hpss(y_resampled, margin=3.0)
            log_memory("7 ")
            
            # D. Chroma CQT
            chroma_chunk = librosa.feature.chroma_cqt(
                y=y_harm, 
                sr=11025, 
                hop_length=512,
            )
            log_memory("8 ")
            all_chroma.append(chroma_chunk)
            log_memory("9 ")

            # E. Clear "Pool C" (Workspace)
            # del y_chunk, y_resampled, y_harm, chroma_chunk
            # gc.collect()
            log_memory("10")

    # 4. Cleanup
    log_memory("11")
    full_chroma = np.concatenate(all_chroma, axis=1)
    os.unlink(tmp_path) # Delete the temp file from disk
    log_memory("12")
    
    tonal_profile = np.mean(full_chroma, axis=1).tolist()
    log_memory("13")
    time_series_notes = full_chroma.T.tolist()
    log_memory("14")

    return {
        "file_name": file.filename,
        "duration": float(total_duration),
        "fps": 11025 / 512,
        "tonal_profile": tonal_profile,
        "time_series_notes": time_series_notes,
        "status": "Success"
    }
# async def analyze(file: UploadFile = File(...)):
#     gc.collect()
    
#     # 1. Save to a temporary file on DISK (not RAM)
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
#         content = await file.read()
#         tmp.write(content)
#         tmp_path = tmp.name

#     all_chroma = []

#     # 2. Open a Stream Pointer
#     # We use sf.blocks to read the file in chunks without loading the whole thing
#     with sf.SoundFile(tmp_path) as f:
#         sr_original = f.samplerate
#         # We process in blocks of 30 seconds
#         block_size = sr_original * 30 
        
#         for block in f.blocks(blocksize=block_size, dtype='float32', always_2d=True):
#             # Downmix to mono manually (average the channels)
#             y_chunk = np.mean(block, axis=1)
            
#             # Resample the chunk to 11025Hz
#             y_resampled = librosa.resample(y_chunk, orig_sr=sr_original, target_sr=11025)
            
#             # --- Perform your HPSS and Chroma math here on y_resampled ---
# 						# A. Separate Harmonic (Notes) from Percussive (Drums)
# 						# Margin 3.0 keeps it strict.
#             y_harm, _ = librosa.effects.hpss(y_resampled, margin=3.0)
						
# 						# log_memory("@@__Chunk Middle")
						
# 						# B. Run CQT on the slice
# 						# n_bins=48 maps 4 octaves (The musical "sweet spot")
# 						# fmin=librosa.note_to_hz('C2') starts the map at a low C
#             chroma_chunk = librosa.feature.chroma_cqt(
# 								y=y_resampled, 
# 								sr=SR, 
# 								hop_length=HOP_LENGTH,
# 								fmin=librosa.note_to_hz('C2'), 
# 								n_octaves=4,
# 								n_chroma=12,
# 								bins_per_octave=12
# 						)
            
#             all_chroma.append(chroma_chunk)
#             del y_chunk, y_resampled
#             gc.collect()

#     # 3. Clean up the temp file from disk
#     os.unlink(tmp_path)
# 		# # make sure we start w/ ram at a minimum
#     # gc.collect()
    
#     # # log_memory("Start")
#     # # 1. Read file into memory
#     # log_memory('psycho 1 ')
#     # content = await file.read()
#     # log_memory('psycho 2 ')
#     # audio_file = io.BytesIO(content)
#     # log_memory('psycho 3 ')

#     # # 2. Load the 1D Waveform
#     # # by default, this will downsample the file. Meaning, smaller size but data is lost.
#     # # to turn this off, do: (audio_file, sr=None, mono=False)
#     # log_memory('psycho 4 ')
#     # y, sr = librosa.load(audio_file, sr=SR) 
#     # log_memory('psycho 5 ')
#     # # log_memory("After Load")
    
#     # log_memory('psycho 6 ')
#     # chunk_size = SR * CHUNK_SEC
#     # log_memory('psycho 7 ')
#     # all_chroma = []
#     # log_memory('psycho 8 ')
    
# 		# # 3. The Chunking Loop
#     # for i in range(0, len(y), chunk_size):
#     #     # log_memory("$$__Chunking Start__$$")
#     #     y_chunk = y[i : i + chunk_size]
#     #     log_memory('psycho 9 ')
        
#         # # A. Separate Harmonic (Notes) from Percussive (Drums)
#         # # Margin 3.0 keeps it strict.
#         # y_harm, _ = librosa.effects.hpss(y_chunk, margin=3.0)
#         # log_memory('psycho 10')
        
        
#         # # log_memory("@@__Chunk Middle")
        
#         # # B. Run CQT on the slice
#         # # n_bins=48 maps 4 octaves (The musical "sweet spot")
#         # # fmin=librosa.note_to_hz('C2') starts the map at a low C
#         # chroma_chunk = librosa.feature.chroma_cqt(
#         #     y=y_harm, 
#         #     sr=sr, 
#         #     hop_length=HOP_LENGTH,
#         #     fmin=librosa.note_to_hz('C2'), 
#         #     n_octaves=4,
#         #     n_chroma=12,
#         #     bins_per_octave=12
#         # )
#     #     log_memory('psycho 11')
        
#     #     all_chroma.append(chroma_chunk)
#     #     log_memory('psycho 12')
#     #     # log_memory("##__Chunking End__##")
        
#     #     # # # C. MANUALLY PURGE RAM (The "Safety Valve")
#     #     # log_memory("BEFORE")
#     #     # del y_harm
#     #     # del chroma_chunk
#     #     # gc.collect() # Forces the "Workspace" to empty before next loop
#     #     # log_memory("AFTER")
        
# 		# 4. Final Assembly
#     # Concatenate all chunks along the time axis (axis 1)
#     full_chroma = np.concatenate(all_chroma, axis=1)
#     log_memory('psycho 13')
    
#     # Transpose to get [Time][12 Notes] structure for JSON
#     time_series_notes = full_chroma.T.tolist()
#     log_memory('psycho 14')
    
#     # Global DNA profile
#     tonal_profile = np.mean(full_chroma, axis=1).tolist()
#     log_memory('psycho 15')
#     # log_memory("End")

#     # # 3. Harmonic-Percussive Source Separation (HPSS)
#     # # Since sounds like drums or highly distorted electric guitars can't be categorized, 
#     # # we run this, which filters out those sounds from the categorizable ones. 
#     # y_harmonic, y_percussive = librosa.effects.hpss(y, margin=3.0) # margin 3 makes it strict
#     # log_memory("After HPSS")

#     # # 4. takes filtered y_harmonic and categorizes it into notes. 
#     # chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=HOP_LENGTH, bins_per_octave=12)
#     # log_memory("After Chroma")
    
# 		# # 5. DATA FOR FRONTEND
#     # # 'chroma' is a 12 x T matrix. 
#     # # To send it over JSON, we transpose it so it's T x 12 (Time-steps of 12 notes)
#     # time_series_notes = chroma.T.tolist()

#     # # 5. Calculate Global Tonal Weight (The "DNA" of the song)
#     # tonal_profile = np.mean(chroma, axis=1).tolist()
#     # log_memory("End")

#     return {
#         "filename": file.filename,
#         "duration": float(librosa.get_duration(y=y, sr=sr)),
#         "tonal_profile": tonal_profile, # 12 values representing C through B
#         "time_series_notes": time_series_notes,
#         "status": "Success"
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)