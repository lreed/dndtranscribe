# Architecture & Design Notes

This document captures the *why* — design decisions, version pins, runtime compat
shims, and known limitations. It exists so future-us (or future-Claude picking up
the project) doesn't have to rediscover this from scratch.

## Audio pipeline (shared by v1 and v2)

```
EMEET conference mic ─┐
                      ├─ OBS scene "DnDGroup_luna_transcribe"
Discord remote audio ─┘                │
                                       │ (Audio Monitoring = Monitor Only)
                                       ▼
                          OBS Monitoring Device
                                       │
                                       ▼
                       VB-Audio Virtual Cable (Input)
                                       │
                                       ▼
                      VB-Audio Virtual Cable (Output)
                                       │
                                       ▼
                  Python sounddevice InputStream (16 kHz mono)
                                       │
                                       ▼
                              buffer_worker thread
                          (30 s buffer, flush on
                           1.5 s silence, min 5 s)
                                       │
                                       ▼
                               transcription_queue
                                       │
                                       ▼
                                main inference loop
                                  (per chunk)
```

The buffer_worker pattern is what makes this near-real-time without dropping audio:
the `sounddevice` callback is non-blocking and just enqueues, the buffer thread
detects silence and hands chunks off, and the main thread does the heavy ML work.

Both v1 and v2 use exactly this audio path — diarization in v2 plugs in as an
extra step in the per-chunk inference loop.

## v1: faster-whisper large-v3-turbo

Single model, single inference per chunk. Buffered + VAD-filtered transcription.

- Model: `deepdml/faster-whisper-large-v3-turbo-ct2` (CT2 quant of OpenAI's turbo)
- Compute: float16 default, int8_float16 fallback for VRAM pressure
- Library: `faster-whisper` (built on ctranslate2)
- VRAM: ~1.6 GB model + ~1.5 GB buffers/cuDNN ≈ 3 GB total

CUDA DLL handling: faster-whisper's ctranslate2 backend needs `cublas64_12.dll`
and `cudart64_12.dll` on PATH at runtime. We get those from the pip-installed
`nvidia-cuda-runtime-cu12` and `nvidia-cublas-cu12` packages. The script prepends
their `bin` dirs to `PATH` at startup — see the `_extra` block at the top of both
`transcribe.py` and `transcribe_v2.py`.

## v2: Whisper large-v3 + pyannote + ECAPA + registry

Three models loaded simultaneously, sequential inference per chunk:

```
chunk arrives ─┬─▶ faster-whisper large-v3  ─▶ Whisper segments (text + timing)
               │
               └─▶ pyannote 3.1 diarization ─▶ pyannote turns (start/end/cluster_id)
                                              │
                                              └▶ for each cluster_id with ≥0.5s of speech:
                                                  ECAPA-TDNN encode → 192-dim embedding
                                                  │
                                                  └▶ SpeakerRegistry.assign() ─▶ stable global label
                                                                                  │
                                                                                  └▶ append to JSONL sidecar

merge: for each Whisper segment, find pyannote turn with greatest overlap → use that
       cluster's global label
```

### Key components in `transcribe_v2.py`

- **`SpeakerRegistry`** — maintains running-mean ECAPA embeddings per global speaker
  label. New chunk's clusters are cosine-matched against (a) enrolled voiceprints
  and (b) previously-seen running-mean embeddings. If no match exceeds
  `SPEAKER_MATCH_THRESHOLD` (default 0.55, overridable at runtime via `--threshold`),
  a new `Speaker_NN` label is created.
- **`compute_cluster_embeddings`** — averages ECAPA embeddings across each
  pyannote cluster's turns within the chunk. Skips turns under 0.5 s as too noisy.
- **`speaker_for_segment`** — for a Whisper segment's `[start, end]` range, returns
  the pyannote cluster with the most temporal overlap.
- **`append_embedding`** — opens, writes one JSONL line, closes (per-write durability
  for crash resilience over 4+ hour sessions).

### VRAM budget on RTX 5060 (8 GB)

| Component | VRAM |
|---|---|
| faster-whisper large-v3 (int8_float16) | ~2.0 GB |
| pyannote 3.1 segmentation + embedding | ~2.5 GB peak |
| SpeechBrain ECAPA-TDNN | ~50 MB |
| Buffers / cuDNN workspace | ~0.5 GB |
| **Total typical** | **~5 GB** |

Leaves ~3 GB for OBS, Discord, and a browser. Tested OK during real sessions.

## Why the dependency tree is so painful

The "obvious" stack — newest pyannote, newest torch, newest Python — does not work.
We hit a chain of regressions and removed APIs. Here's what we found and what we
pinned, with the reasoning.

### Python 3.12 (not 3.14) for the v2 venv

- pyannote 3.3.2 + torchaudio 2.11 (only torch with cu128 on Python 3.14) hit
  `torchaudio.list_audio_backends` and `torchaudio.AudioMetaData` removals.
- speechbrain 1.x has a lazy-import bug interacting with `inspect.stack()` calls
  inside pytorch_lightning that fires unrelated `k2` import errors.
- Wheels for several deps lag Python 3.14 release.
- Python 3.12 has stable wheels for everything and the older torchaudio 2.7 still
  has the APIs pyannote calls.

### torch / torchaudio 2.7.x with cu128

- **Need cu128**: RTX 5060 is Blackwell sm_120. Earlier CUDA versions (12.6 etc.)
  build but lack sm_120 kernels — runtime warns and falls back.
- **Cap at 2.7.x**: torchaudio 2.8+ deprecated and 2.9+ removed `list_audio_backends`
  and `AudioMetaData`, both of which pyannote 3.3.2 calls at module-import time.

### pyannote.audio == 3.3.2 (avoid 4.x)

- 4.0.x has a 6× VRAM regression (issue #1963). 3.3.2 fits in our budget; 4.x does not.

### speechbrain < 1.0

- pyannote 3.3.2 *declares* `speechbrain>=1.0` but speechbrain 1.x has a lazy-import
  bug where `inspect.stack()` (called by pytorch_lightning during checkpoint load)
  triggers an unrelated `k2_fsa` lazy module, which then tries to `import k2` (not
  installed) and raises ImportError up through the load.
- We override the pin to `<1.0` (0.5.16) — the `EncoderClassifier` API is the same
  for our usage, and the lazy module isn't there.
- Side effect: import path is `from speechbrain.pretrained import EncoderClassifier`
  in 0.5.x, vs `from speechbrain.inference.speaker import EncoderClassifier` in 1.x.

### huggingface_hub < 1.0

- 1.0+ removed the `use_auth_token` kwarg. pyannote 3.3.2 still passes it. 0.36
  works.

### matplotlib

- pyannote 3.3.2 imports `matplotlib.pyplot` at module load even though we don't
  use any plotting. Has to be installed.

## Runtime compat shims

Three monkey-patches at the top of `transcribe_v2.py` and `enroll.py`. Each is
documented inline. Removing any of them re-introduces a known failure mode.

### 1. `os.symlink` and `Path.symlink_to` → fallback to copy

Windows requires admin or Developer Mode for `os.symlink`. Both huggingface_hub
(model weights) and speechbrain (ECAPA fetch) symlink-from-cache during model
download. We catch `OSError` and `shutil.copy2` instead.

Long-term cleanup: enable Windows Developer Mode and remove the shim. Saves a few
GB of duplicated cache files. Cosmetic.

### 2. `torch.load` → force `weights_only=False`

PyTorch 2.6 changed the default of `weights_only` from `False` to `True`. pyannote
3.3.2 checkpoints contain `torch.torch_version.TorchVersion` instances that aren't
in the safe-globals allowlist, so loading fails under the new default.

`weights_only=False` allows arbitrary code execution from a malicious checkpoint.
Acceptable here because we only load weights from the official pyannote HF repos.

### 3. (CUDA DLL PATH) — same as v1

ctranslate2's lazy-loaded `cublas64_12.dll` isn't covered by `os.add_dll_directory`.
We prepend the pip-installed `nvidia-*-cu12` package paths to `PATH`.

## Known limitations & quirks

### Channel mismatch — same person, Discord vs EMEET

ECAPA produces meaningfully different embeddings for the same speaker depending
on the input channel (codec, mic, noise floor). Cosine similarity between a player's
EMEET-mic voice and their Discord voice can be ~0.3 — well below any reasonable
threshold for "same speaker." This is fundamental to off-the-shelf voice biometrics
and won't be fixed by tuning.

The architectural fix is to split EMEET and Discord into two audio streams (in OBS),
diarize each independently, and merge transcripts at the end. See
[ROADMAP.md](ROADMAP.md).

### Short utterances → noisy embeddings

A 1 s "yeah" produces an ECAPA embedding far from the same speaker's 30 s reference
embedding. The 0.5 s minimum-turn cutoff in `compute_cluster_embeddings` catches
some of this, but Whisper still transcribes those clips and they get tagged
`Speaker_??` (cluster present but no embedding computed). Not a bug per se — short
isolated speech is genuinely hard to attribute.

### TF32 disabled

pyannote disables TF32 globally for reproducibility. Costs ~2-3× on diarization
inference but is intentional. Can be re-enabled with
`torch.backends.cuda.matmul.allow_tf32 = True` if real-time falls behind.

### Speaker_?? handling

When a pyannote cluster has only short turns (all <0.5 s), no embedding is computed
and the cluster has no entry in `cluster_to_label`. Whisper segments overlapping
that cluster get `Speaker_??`. Could be fixed by lowering the cutoff or falling back
to "most recent named speaker" — left as future work.

## File layout (runtime)

```
C:\Users\<USERNAME>\
├── claude_code\dndtranscribe\        ← repo
│   ├── .venv\                        ← v1, Python 3.14
│   ├── .venv_v2\                     ← v2, Python 3.12
│   ├── transcripts\                  ← session output (gitignored)
│   ├── voiceprints\                  ← enrolled .npy files (gitignored)
│   └── pretrained_models\            ← speechbrain ECAPA cache (gitignored)
└── .cache\
    ├── huggingface\
    │   ├── token                     ← HF credential (read by huggingface_hub)
    │   └── hub\                      ← model weight cache (Whisper, ECAPA)
    └── torch\pyannote\               ← pyannote model cache
```

Nothing inside `.cache\` should ever be committed — it's regenerable from HF.
