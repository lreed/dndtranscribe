# Roadmap & Open Work

Tracks planned work, deferred decisions, and the v3 vision. Items roughly ordered
by impact-per-effort.

## Near-term (low effort, high value)

### Tune the speaker-match threshold based on real session data

The `--threshold` CLI flag is in (default 0.55). Open question: what value works
best in real D&D sessions? In a single-speaker test, 0.55 over-segmented short
utterances (same Bilbo voice → 14 different `Speaker_NN` labels). Worth measuring
accuracy at 0.45 / 0.40 / 0.35 across a real session — both for false splits (same
person → multiple labels, what we have today) and false merges (different people
→ same label, the risk of going too low).

If 0.45–0.40 turns out reliably better, change the default in `transcribe_v2.py`.

### Speaker enrollment for the regulars

Run `enroll.py` for each consistent player (Frodo, Bilbo, Samwise, etc.). Anchored
matching against a clean 30 s reference is much more stable than the running-mean
clustering approach, especially for short utterances.

### Retroactive speaker naming script

Use the `_embeddings.jsonl` sidecar to relabel `Speaker_NN` → real names after a
session. Workflow:

1. Listen to a section of the transcript at a specific timestamp.
2. Identify "Speaker_03 was Bilbo" by ear.
3. Run a tool that reads the JSONL, finds all `Speaker_03` rows, and:
   - Updates the transcript text in place
   - Optionally saves the average embedding as a new voiceprint for next time

Not built yet. The data is being written; we just need the script.

### Fix the `Speaker_??` fallback

When a pyannote cluster has only short turns (<0.5 s threshold in
`compute_cluster_embeddings`), no embedding is computed and overlapping Whisper
segments get `Speaker_??`. Two options, in order of simplicity:

- Drop the 0.5 s cutoff entirely — accept noisier embeddings for short clusters.
- Inherit the most-recently-named speaker if none is available — naive but might
  work in practice given conversational continuity.

## Medium-term (channel split — solves the structural diarization problem)

### Dual-stream OBS routing

The single biggest accuracy improvement for diarization: route EMEET (local) and
Discord (remote) into **two separate** virtual audio cables instead of mixing them.

Why it matters: ECAPA embeddings from the same human's voice differ by codec/mic
enough that Discord-Bilbo and EMEET-Bilbo cosine-match at ~0.3. There's no
threshold tuning that fixes this — only physical separation.

Architecture:

- OBS: add a second VB-Audio cable (CABLE-B). Route EMEET to cable A, Discord to
  cable B (separate Monitor sources).
- Python: open two `sounddevice` streams in parallel. Run independent
  Whisper+pyannote pipelines per stream. Merge transcripts at output by chunk
  timestamp.
- Result: Discord-side speakers are diarized only against each other; same for
  EMEET-side. Two label namespaces (`Discord_01`, `Local_01`).
- Voiceprint enrollment becomes per-channel — Bilbo has a Discord voiceprint AND
  an EMEET voiceprint.

VRAM: roughly doubles (~9 GB) — likely won't fit on the 5060. Mitigations:

- Run with `--compute-type int8` for Whisper (saves ~1 GB)
- Share one pyannote pipeline serially across the two streams (alternate per chunk)
- Or run Discord-side on CPU (accept higher latency on remote audio)

### Smarter speaker assignment in cross-chunk continuity

Current `SpeakerRegistry` uses running-mean cosine similarity, which drifts with
noise. A better approach: keep all per-chunk embeddings, periodically re-cluster
globally (every N chunks), and remap labels. Trades some real-time-ness for
stability.

## v3: Move to NVIDIA Parakeet on WSL2

This is the long-term ambition: replace the entire Whisper+pyannote stack with
NVIDIA's purpose-built ASR + diarization, which is currently top of the Open ASR
Leaderboard and ~50× faster.

### Why

- **Parakeet-TDT-0.6B-v2** beats Whisper large-v3 by ~1.3 absolute WER points
  (6.05% vs 7.4%) on the Open ASR Leaderboard. Beats large-v3-turbo by ~1.5
  points.
- **Streaming Sortformer** (released Aug 2025) is purpose-built for live, chunked
  speaker diarization with consistent labels across chunks — solves the
  cross-chunk consistency problem we have today, by design.
- **Lower VRAM** — Parakeet is ~2.1 GB, Sortformer ~1.5 GB. Total ~3.5 GB vs our
  current ~5 GB.

### Why not yet

- **Speaker cap of 4** in Streaming Sortformer 4spk-v2.1 (NeMo issue #14546).
  Hard architectural cap — we have 6 D&D players, so this is disqualifying as-is.
  Watch for a higher-speaker-count variant; alternatively combine Parakeet with
  pyannote (as we do for Whisper today).
- **Windows-native NeMo is not officially supported** — requires WSL2 + Linux for
  the toolchain. The user has WSL2 Ubuntu 25.10 already running.
- **Audio routing into WSL2** — VB-Audio Cable lives in Windows host. WSL2 doesn't
  natively access Windows audio devices.

### Plan

1. **Audio routing — pick one**:
   - **Option A: PulseAudio bridge.** Run PulseAudio in WSL2, expose a TCP socket
     back to Windows; Windows forwards VB-Audio Cable Output into the socket.
     There are guides for this; non-trivial.
   - **Option B: Windows side capture, network pipe.** Keep `sounddevice` capture
     in a small Windows-side script, pipe raw 16 kHz mono PCM over a TCP socket
     to the WSL2 inference service. Cleanest separation; minimal Windows-side code.
   - Recommend Option B.

2. **WSL2-side stack**:
   ```
   conda create -n parakeet python=3.11
   conda activate parakeet
   # NeMo install: nvidia-pyindex, nemo_toolkit[asr]
   # Parakeet-TDT-0.6B-v2 from HF
   # diarization: see decision below
   ```

3. **Diarization decision** — not Sortformer yet (4-speaker cap). Two options:
   - **pyannote on Linux** — much smoother install than Windows; same model.
   - **Wait for Sortformer multi-speaker variant** — track NeMo releases.

4. **Output back to Windows**:
   - Write transcripts to a path mapped via `\\wsl$\Ubuntu-25.10\...` or share a
     workspace dir.

### What's already in place for v3

- WSL2 Ubuntu 25.10 is running on the user's machine (verified by user).
- The current Windows-side audio capture code (`sounddevice` 16 kHz mono in
  `transcribe.py` / `transcribe_v2.py`) is small and reusable as the network
  source.

## Deferred decisions / things considered and rejected

- **WhisperX** (large-v3 + bundled pyannote) — at 8 GB VRAM, OOM risk too high
  with concurrent OBS/Discord/browser. Rejected in favor of separately-loaded
  faster-whisper + pyannote with sequential per-chunk inference.
- **CrisperWhisper** — better word-level timestamps than large-v3, but same
  ASR accuracy and same VRAM cost; not worth the swap. Reconsider if word-level
  alignment becomes a feature requirement.
- **Reverb (Rev.com)** — best-in-class on noisy/spontaneous English. Worth a
  bake-off vs Parakeet on actual D&D session audio in v3.
- **Sortformer for v2** — rejected because 4-speaker hard cap doesn't fit a 6-player
  D&D group.
- **Cloud APIs (AssemblyAI, Deepgram, etc.)** — explicit non-goal; user wants
  fully local, no audio leaving the laptop.

## Bug-tracker-style open items

- [ ] `Speaker_??` lines in real sessions — investigate frequency, decide whether
      to drop short-turn cutoff or inherit-most-recent. (See ARCHITECTURE.md
      §"Speaker_?? handling".)
- [x] Add `--threshold` CLI flag. (Done. Default 0.55. See SETUP_v2.md.)
- [ ] Build the retroactive-rename script for the JSONL sidecar.
- [ ] Test: does `start_v2.bat --compute-type float16` actually fit alongside
      pyannote on 8 GB? Was tested only with int8_float16.
- [ ] Decide on Windows Developer Mode vs the symlink monkey-patches long-term.
