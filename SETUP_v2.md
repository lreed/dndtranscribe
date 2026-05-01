# v2 Setup — Whisper large-v3 + Speaker Diarization

This is the setup for `transcribe_v2.py` (`start_v2.bat`). v2 produces speaker-labeled
transcripts. v1 setup ([SETUP.md](SETUP.md)) covers the OBS audio routing and is
shared by both versions — read that first if you haven't already.

---

## Prerequisites (one-time)

1. **OBS audio routing** — same as v1. EMEET mic + Discord audio routed into VB-Audio
   Virtual Cable via "Monitor Only" sources. See [SETUP.md](SETUP.md) §"One-Time Setup".

2. **Python 3.12** at `C:\Users\<USERNAME>\AppData\Local\Programs\Python\Python312\python.exe`
   (installed via `winget install Python.Python.3.12`). v1 stays on Python 3.14;
   v2 needs 3.12 because pyannote 3.3.2 / speechbrain 0.5.x have wheel and API
   compatibility issues with newer Python.

3. **NVIDIA RTX 5060 (Blackwell, sm_120)** with current drivers. Confirmed working
   with PyTorch 2.7.x + CUDA 12.8 wheels. Earlier CUDA versions (12.6 etc.) compile
   and load but their kernels are not built for sm_120 — runtime falls back to CPU
   or errors.

4. **Hugging Face account + token** — free. The token is only used to authenticate
   one-time downloads of the gated pyannote models. After download, runtime is
   fully offline. See "HF token" below.

---

## Install (one-time)

```
cd C:\Users\<USERNAME>\claude_code\dndtranscribe
py -3.12 -m venv .venv_v2
.venv_v2\Scripts\python.exe -m pip install torch==2.7.* torchaudio==2.7.* --index-url https://download.pytorch.org/whl/cu128
.venv_v2\Scripts\python.exe -m pip install -r requirements_v2.txt
```

The `cu128` index is critical — it has wheels built for Blackwell. The default
PyPI torch is CPU-only, and the `cu126` and earlier indexes lack sm_120 kernels.

---

## HF token + EULAs (one-time)

1. **Generate a token**: https://huggingface.co/settings/tokens → "New token" →
   role "Read" → save it. You only need this for the initial model downloads.

2. **Accept the gated-model EULAs in browser** (one click each):
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-3.1

3. **Save the token to local credential storage** (not env vars):

   ```
   .venv_v2\Scripts\python.exe -c "from huggingface_hub import login; login(token='YOUR_TOKEN', add_to_git_credential=False)"
   ```

   This writes `C:\Users\<USERNAME>\.cache\huggingface\token`. The huggingface_hub
   library auto-discovers it. After model download you can revoke the token —
   weights cache locally and don't need re-authentication.

---

## First run

```
start_v2.bat
```

On first launch, three downloads happen automatically (~3.5 GB total, one time):

| Cache | Size | Path |
|---|---|---|
| Whisper large-v3 (CT2 format) | ~3.0 GB | `C:\Users\<USERNAME>\.cache\huggingface\hub\models--Systran--faster-whisper-large-v3` |
| pyannote diarization 3.1 + segmentation 3.0 + wespeaker embedding | ~500 MB | `C:\Users\<USERNAME>\.cache\torch\pyannote` |
| SpeechBrain ECAPA-TDNN | ~50 MB | `C:\Users\<USERNAME>\.cache\huggingface\hub\models--speechbrain--spkrec-ecapa-voxceleb` and `pretrained_models/EncoderClassifier-*/` |

After download, every subsequent launch loads from cache in ~5–10 s.

You'll see:

```
Loading Whisper large-v3 into VRAM...
Whisper loaded.
Loading pyannote diarization pipeline...
Diarization loaded.
Loading ECAPA-TDNN speaker encoder...
Encoder loaded.

All models loaded.

Listening on: 5
Transcript:   transcripts\session_v2_2026-04-30_23-39-55.txt
Embeddings:   transcripts\session_v2_2026-04-30_23-39-55_embeddings.jsonl
Press Ctrl+C to stop.
```

---

## Speaker enrollment (optional, anytime)

Without enrollment, transcripts use generic `Speaker_01:` etc. labels. With enrollment,
players are identified by name. Quick start:

```
.venv_v2\Scripts\python.exe enroll.py Frodo
```

See [ENROLLMENT.md](ENROLLMENT.md) for the full guide — three recording modes,
mic-selection guidance, the channel-mismatch workaround for Discord vs local audio,
verification, re-enrollment, and troubleshooting.

---

## Tuning

### Compute type (VRAM trade-off)

```
start_v2.bat --compute-type float16       # ~3.5 GB Whisper VRAM, slightly higher accuracy
start_v2.bat --compute-type int8_float16  # default, ~2.0 GB Whisper VRAM
```

### Speaker-match threshold

Cosine-similarity cutoff for matching a chunk's speaker cluster to an enrolled
voiceprint or a previously-seen anonymous speaker. **Lower = more permissive merging.**

```
start_v2.bat --threshold 0.55   # default — strict, conservative matching
start_v2.bat --threshold 0.40   # more permissive — short utterances of the same
                                #   speaker are more likely to be linked together
                                #   (good starting point if you see many spurious
                                #   Speaker_NN splits in your transcripts)
start_v2.bat --threshold 0.30   # very permissive — risk of merging different people
```

The threshold is logged in the session header so each transcript records what value
was used. Typical range to experiment in: **0.35–0.60**. Re-enrolling from the same
audio path used at session time (see [ENROLLMENT.md](ENROLLMENT.md)) is usually a
bigger lever than threshold tuning — try that first.

### Whisper-only mode (skip diarization)

```
start_v2.bat --no-diarize
```

Useful for debugging or to compare ASR quality against v1 without the diarization layer.

### CPU fallback

```
start_v2.bat --cpu
```

Diarization on CPU is slow — expect chunks to fall behind real-time after a few minutes.

---

## Output

Per session:

- `transcripts/session_v2_<timestamp>.txt` — line-by-line, flushed per segment.
  Survives crashes.
- `transcripts/session_v2_<timestamp>_embeddings.jsonl` — append-per-chunk JSONL,
  one record per `(chunk, speaker_cluster)` pair:

  ```json
  {"offset": 123.4, "label": "Speaker_01", "embedding": [0.1, -0.05, ...]}
  ```

  Used for retroactive speaker naming — see [ROADMAP.md](ROADMAP.md).

---

## Troubleshooting

### `Speaker_??:` for many lines

A Whisper segment overlapped a pyannote turn shorter than the 0.5 s cutoff used
for embedding extraction. Common with single-word utterances. See ARCHITECTURE.md
§"Diarization quirks".

### Same speaker getting different IDs across chunks

Short utterances produce noisy ECAPA embeddings, and the running-mean cosine match
in `SpeakerRegistry` fires below threshold. Mitigations:

1. Enroll the regulars — anchored matching against a 30 s reference is way more
   stable than running-mean clustering.
2. Lower `SPEAKER_MATCH_THRESHOLD` from 0.55 toward 0.45 (in `transcribe_v2.py`).
3. Long-term: dual-stream OBS split (Discord/EMEET separate), see ROADMAP.md.

### `WinError 1314: A required privilege is not held by the client`

Symlink permission. The scripts already include a fallback that copies instead.
If it slips through somehow: enable Windows Developer Mode (Settings → System →
For developers → Developer Mode = On), or run terminal as administrator once for
the initial download.

### `Weights only load failed` from torch.load

PyTorch 2.6+ default. The scripts monkey-patch `torch.load` to force
`weights_only=False`. If you see this, you ran a Python file that doesn't include
the shim — check the file's import block.

### `module 'torchaudio' has no attribute 'list_audio_backends'`

You installed torch >2.7. Pin to `torch==2.7.* torchaudio==2.7.*` from the cu128
index. Newer torchaudio removed the API that pyannote 3.3.2 calls.

### Other CUDA errors

The transcribe_v2.py script adds CUDA DLL directories to PATH at startup the same
way v1 does (see SETUP.md §"cublas64_12.dll is not found"). If you see "DLL not
found": make sure you're running with `.venv_v2\Scripts\python.exe`, not the system
Python.
