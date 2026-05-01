# D&D Session Transcription — Setup & Run Guide

Real-time speech-to-text for hybrid D&D sessions. Captures the EMEET conference mic (local
players) and Discord audio (remote players) via OBS, transcribes with Whisper v3 Turbo on
the RTX 5060, and writes a timestamped text file.

---

## What You Need (all already installed)

- OBS Studio 32+ with scene `DnDGroup_luna_transcribe`
- VB-Audio Virtual Cable
- Python 3.14 at `C:\Python314`
- This project folder: `C:\Users\<USERNAME>\claude_code\dndtranscribe\`

---

## One-Time Setup (already done — skip to "Every Session" below)

### 1. Create the virtual environment

```
cd C:\Users\<USERNAME>\claude_code\dndtranscribe
C:\Python314\python.exe -m venv .venv
```

### 2. Install dependencies

```
.venv\Scripts\pip.exe install -r requirements.txt
```

This installs faster-whisper, sounddevice, numpy, and the NVIDIA CUDA 12 runtime pip packages
(`nvidia-cuda-runtime-cu12`, `nvidia-cublas-cu12`). These pip packages supply the `cublas64_12.dll`
and `cudart64_12.dll` libraries that ctranslate2 needs to run on the GPU — no separate CUDA
Toolkit installation is required. The script automatically adds their location to the DLL search
path on startup.

### 3. Download and cache the Whisper model

The model downloads automatically on first run (~1.6 GB from Hugging Face). It is cached at
`C:\Users\<USERNAME>\.cache\huggingface\hub\` and will not re-download on subsequent runs.

To pre-download it before a session:

```
.venv\Scripts\python.exe -c "from faster_whisper import WhisperModel; WhisperModel('deepdml/faster-whisper-large-v3-turbo-ct2', device='cuda', compute_type='float16'); print('Model cached.')"
```

### 4. Configure OBS (one-time, settings persist across reboots)

**a. Add your audio sources**

In OBS, with the scene `DnDGroup_luna_transcribe` selected:

1. Click **+** in the Sources panel → **Audio Input Capture** → name it `EMEET Mic` → OK
   - Device: `Microphone (EMEET OfficeCore Luna Plus)`

2. Click **+** → **Application Audio Capture** → name it `Discord Audio` → OK
   - Application: select `Discord.exe`
   - *(If Application Audio Capture is not listed, use Audio Output Capture → Desktop Audio)*

**b. Set both sources to Monitor Only**

In the Audio Mixer at the bottom of OBS:

1. Click the **gear icon (⚙)** on `EMEET Mic` → **Advanced Audio Properties**
2. In the `Audio Monitoring` column, set `EMEET Mic` → **Monitor Only (mute output)**
3. Set `Discord Audio` → **Monitor Only (mute output)**

**c. Set the monitoring device**

**File → Settings → Audio → Advanced section**
- **Monitoring Device**: `CABLE Input (VB-Audio Virtual Cable)`
- Click **Apply → OK**

> **What this does:** OBS silently blends both audio streams and routes them into the virtual
> cable. Your Python script reads the other end of that cable.

---

## Every Session Startup (after reboot)

Follow these steps in order each time.

### Step 1 — Start Discord and join your voice channel

Log into Discord and connect to the game voice channel before starting OBS. This ensures
Discord is a running process that OBS can capture.

### Step 2 — Open OBS and load the scene

Open OBS Studio. In the Scenes panel, select **DnDGroup_luna_transcribe**.

Verify audio is flowing:
- Speak into the EMEET mic — the `EMEET Mic` level meter in the Audio Mixer should bounce.
- Someone in Discord should talk — the `Discord Audio` meter should bounce.

> If the meters are flat, check that Audio Monitoring is still set to "Monitor Only" for both
> sources (gear icon → Advanced Audio Properties).

### Step 3 — Check the virtual cable has audio

Open Windows **Volume Mixer** (right-click the speaker icon in the taskbar → Open Volume Mixer).
You should see `CABLE Input` receiving audio when you speak or when Discord plays audio.

Alternatively, run the device check:

```
cd C:\Users\<USERNAME>\claude_code\dndtranscribe
.venv\Scripts\python.exe list_devices.py
```

You should see `CABLE Output (VB-Audio Virtual` highlighted with `>>>`.

### Step 4 — Start transcription

```
cd C:\Users\<USERNAME>\claude_code\dndtranscribe
.venv\Scripts\activate
python transcribe.py
```

Or without activating:

```
cd C:\Users\<USERNAME>\claude_code\dndtranscribe
.venv\Scripts\python.exe transcribe.py
```

**First run after a fresh install:** The model loads in ~5–10 seconds from the local cache.
You will see:

```
Loading model into VRAM...
Model loaded.

Listening on: 5
Transcript:   transcripts\session_2026-04-30_19-30-00.txt
Press Ctrl+C to stop.
```

After that, speech will appear in the console within a few seconds of silence, and every line
is immediately written to the transcript file.

### Step 5 — End the session

Press **Ctrl+C** in the terminal. The script will flush any remaining audio in the buffer and
write a final duration line to the transcript file.

Transcripts are saved to:
```
C:\Users\<USERNAME>\claude_code\dndtranscribe\transcripts\
```

---

## Troubleshooting

### No audio detected / transcript is empty

1. Open OBS and confirm the level meters for `EMEET Mic` and `Discord Audio` are moving.
2. Confirm Monitoring Device is set to `CABLE Input` (File → Settings → Audio).
3. Confirm both sources are set to `Monitor Only` (gear icon → Advanced Audio Properties).
4. Run the 3-second capture test:
   ```
   .venv\Scripts\python.exe -c "
   import sounddevice as sd, numpy as np
   a = sd.rec(3*16000, samplerate=16000, channels=1, dtype='float32', device=5)
   sd.wait()
   print(f'Peak: {np.max(np.abs(a)):.4f}')
   "
   ```
   Peak should be above `0.01` if audio is flowing.

### "Invalid sample rate" error

The WASAPI device doesn't support 16kHz directly. Run with the explicit MME device index:

```
.venv\Scripts\python.exe transcribe.py --device 5
```

### VRAM pressure / stuttering in browser

Switch to a smaller memory footprint:

```
.venv\Scripts\python.exe transcribe.py --compute-type int8_float16
```

This drops the model from ~1.6 GB to ~1.0 GB VRAM with negligible accuracy loss.

### `cublas64_12.dll is not found` error

This means the CUDA 12 runtime DLLs aren't on the PATH. They are supplied by the pip packages
and the script adds their location automatically — but only from inside the venv. Make sure
you're running with `.venv\Scripts\python.exe`, not the system Python:

```
cd C:\Users\<USERNAME>\claude_code\dndtranscribe
.venv\Scripts\python.exe transcribe.py
```

If the error persists, re-install the packages:
```
.venv\Scripts\pip.exe install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 --force-reinstall
```

### Other CUDA errors / driver issues

Fall back to CPU (slower but works):

```
.venv\Scripts\python.exe transcribe.py --cpu
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Start transcription | `.venv\Scripts\python.exe transcribe.py` |
| List audio devices | `.venv\Scripts\python.exe list_devices.py` |
| Use less VRAM | `... transcribe.py --compute-type int8_float16` |
| Explicit device index | `... transcribe.py --device 5` |
| CPU fallback | `... transcribe.py --cpu` |

All commands run from `C:\Users\<USERNAME>\claude_code\dndtranscribe\`.

---

## Future Work

### Speaker Diarization

The current pipeline produces raw text without identifying who is speaking. The next step is
to add speaker diarization — labeling each segment with a speaker identity (e.g., "Speaker 1",
"Speaker 2", or ideally by name after a training/enrollment step).

Candidates to research:
- **pyannote-audio** — leading open-source diarization library, GPU-accelerated
- **whisperX** — wraps faster-whisper + pyannote for aligned, speaker-attributed transcription
- **NeMo MSDD** — NVIDIA's multi-scale diarization decoder

Key constraint: 8 GB VRAM on the RTX 5060 must fit both the whisper model (~1.6 GB float16)
and the diarization model simultaneously alongside Discord + OBS + browser.
