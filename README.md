# dndtranscribe

Real-time speech-to-text for hybrid D&D sessions. Captures the EMEET conference mic
(local players) and Discord audio (remote players) via OBS, transcribes locally on
an RTX 5060, and writes timestamped text to `transcripts/`.

All inference runs on-device. Audio never leaves the laptop.

## Two versions

| | v1 (`transcribe.py`) | v2 (`transcribe_v2.py`) |
|---|---|---|
| ASR | Whisper large-v3-turbo | Whisper large-v3 (full) |
| Diarization | none | pyannote.audio 3.3.2 |
| Speaker labels | none | `Speaker_NN:` / enrolled name |
| Speaker enrollment | n/a | `enroll.py` (live mic, 30 s) |
| Embeddings sidecar | n/a | JSONL append-per-chunk |
| Python | 3.14 (`.venv`) | 3.12 (`.venv_v2`) |
| VRAM (idle / typical) | ~3.0 GB | ~4.5 GB |

Both share the same audio pipeline: OBS routes EMEET + Discord into VB-Audio Virtual
Cable, and the script reads the cable's output.

## Quick start

**v1:** double-click [start.bat](start.bat)
**v2:** double-click [start_v2.bat](start_v2.bat)

Both pass through any extra args to the underlying Python script. See:

- [SETUP.md](SETUP.md) — v1 setup (OBS audio routing, virtual cable, etc.)
- [SETUP_v2.md](SETUP_v2.md) — v2 setup (Python 3.12, pyannote, HF token, EULAs)
- [ENROLLMENT.md](ENROLLMENT.md) — speaker enrollment guide for v2 (turns `Speaker_NN:` into real names)

## Why two versions?

v1 is the proven, low-friction baseline. v2 trades higher VRAM and a more complex
dependency tree for higher ASR accuracy and per-speaker labeling. We kept v1 untouched
so you can A/B compare against the same audio pipeline.

The v2 dependency tree took some untangling — see [ARCHITECTURE.md](ARCHITECTURE.md)
for why each version is pinned where it is, and the runtime compat shims baked into
the Python files.

## Where we are headed

Future work — see [ROADMAP.md](ROADMAP.md):

- **v3:** WSL2 + NVIDIA Parakeet ASR (top of Open ASR Leaderboard, ~50× faster than
  Whisper) and Streaming Sortformer diarization, if we can solve audio routing into
  WSL2 from the Windows-side VB-Audio Cable.
- **Channel split:** route EMEET and Discord into separate streams in OBS, run two
  diarizers — fixes the channel-mismatch problem where Discord and local audio of
  the same person look like different speakers.
- **Speaker enrollment workflow:** retroactive labeling using the JSONL embeddings
  sidecar (listen back, identify "Speaker_03 was Bilbo," patch transcript).

## Repository layout

```
transcribe.py          v1 main script (Whisper turbo)
transcribe_v2.py       v2 main script (Whisper large-v3 + pyannote)
enroll.py              v2 voiceprint enrollment helper
list_devices.py        prints audio devices to find the cable index
start.bat              v1 launcher
start_v2.bat           v2 launcher
requirements.txt       v1 dependencies
requirements_v2.txt    v2 dependencies (with pin reasoning)
SETUP.md               v1 setup guide (OBS, cable)
SETUP_v2.md            v2 setup guide
ENROLLMENT.md          v2 speaker enrollment guide
ARCHITECTURE.md        design notes, compat shims, version pin reasoning
ROADMAP.md             planned work, including v3 WSL2/Parakeet plan
transcripts/           output (gitignored)
voiceprints/           enrolled voice samples (gitignored)
.venv/                 v1 venv, Python 3.14 (gitignored)
.venv_v2/              v2 venv, Python 3.12 (gitignored)
```
