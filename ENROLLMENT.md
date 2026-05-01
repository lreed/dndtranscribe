# Speaker Enrollment Guide (v2)

Enrollment turns the generic `Speaker_01:`, `Speaker_02:` labels in v2 transcripts
into real names — `Bilbo:`, `Samwise:`, etc.

It's optional. v2 works fine without it; you'll just see anonymous speaker IDs.
Enroll the regulars once and every future session auto-labels them.

---

## How it works

`enroll.py` records (or reads) a clean voice sample for one person, computes a
192-dim ECAPA-TDNN embedding ("voiceprint"), and saves it to `voiceprints/<name>.npy`.

At session time, `transcribe_v2.py`:

1. Loads every `.npy` file in `voiceprints/` at startup.
2. For each diarized speaker cluster in a chunk, computes the cluster's mean ECAPA
   embedding.
3. Cosine-matches it against the enrolled voiceprints.
4. If the best match is above the threshold (default 0.55), uses the matched name;
   otherwise falls back to `Speaker_NN`.

So enrolling improves accuracy in two ways:
- Stable names across an entire session (vs running-mean clustering that can drift
  on short utterances)
- Stable names across *different* sessions

---

## Quick reference

```
enroll.py NAME                            live record 30 s from default mic
enroll.py NAME --seconds 45               custom record length
enroll.py NAME --device 5                 specific input device index
enroll.py NAME --wav clip.wav             use a pre-recorded WAV file
enroll.py NAME --cpu                      run encoder on CPU (slower, no GPU needed)
enroll.py NAME --overwrite                replace an existing voiceprint
```

All examples below assume you're in the project directory:

```
cd C:\Users\<USERNAME>\claude_code\dndtranscribe
```

---

## Mode 1: Live recording from default mic (typical)

```
.venv_v2\Scripts\python.exe enroll.py Frodo
```

The script prints a 3-second countdown, then records for 30 seconds, then computes
and saves the voiceprint:

```
>>> Recording will start in 3 seconds. Speak naturally for ~30s.
>>> Read a paragraph aloud, or have a normal conversation. Avoid background voices.
   3...
   2...
   1...
>>> RECORDING NOW. Speak.

>>> Done recording.

Audio energy: 0.0432  (warn if <0.005)
Loading ECAPA-TDNN encoder on cuda...

Voiceprint saved: voiceprints\Frodo.npy
Future v2 sessions will label this speaker as 'Frodo'.
```

The `Audio energy` line is a sanity check — if it's below 0.005, your mic was
muted or you weren't speaking. Re-run.

### What to say

- Read a paragraph from a book, or just talk naturally for 30 seconds about
  anything (your day, the weather, etc.).
- 30 seconds of clean speech is far better than 60 seconds of speech mixed with
  silence/background.
- Use the same kind of voice you use during sessions (don't whisper for enrollment
  if you normally speak at conversational volume).

---

## Mode 2: Specific input device

If you have multiple mics (EMEET, Discord, headset, laptop builtin) and want to
enroll using a non-default one, find the device index first:

```
.venv_v2\Scripts\python.exe list_devices.py
```

You'll see something like:

```
   0  Microsoft Sound Mapper - Input  (MME)        [in: 2  out: 0]
   1  Microphone (EMEET OfficeCore Luna Plus)      [in: 1  out: 0]
   5  CABLE Output (VB-Audio Virtual Cable)        [in: 1  out: 0]   <<< default for sessions
```

Then pass the index with `--device`:

```
.venv_v2\Scripts\python.exe enroll.py Frodo --device 1
```

### Important: which mic should you enroll from?

The enrollment mic should match the mic that captures the player at session time:

- **Local players** (talking into the EMEET mic in the room) → enroll from device
  index of the EMEET mic, not the virtual cable.
- **Remote players** (joining via Discord) → ask them to enroll on their end
  using their own mic and send you the `.npy` file. Or capture their voice through
  Discord on your machine and enroll from that audio (see Mode 3).

This matters because of *channel mismatch*: a person's voice through their Discord
mic + codec sounds quite different to ECAPA than the same person through the EMEET
mic in your room. A voiceprint enrolled from one channel will match poorly against
the other. See "Channel mismatch workaround" below.

---

## Mode 3: From a pre-recorded WAV file

If you have a recording from a past session, a Discord clip, or anything else:

```
.venv_v2\Scripts\python.exe enroll.py Bilbo --wav recordings\lance_solo_30sec.wav
```

The WAV can be any sample rate — the script will resample to 16 kHz. Stereo files
are downmixed to mono. The clip should be 10 s or longer; 30 s is ideal.

### Capturing a WAV from a remote player

If a remote player can't run enroll.py themselves, you can capture their audio:

1. Get them on Discord, alone in the voice channel with you.
2. Use OBS or Audacity to record just their stream for ~30 s while they talk.
3. Save as WAV.
4. `enroll.py PlayerName --wav their_clip.wav`

---

## Channel mismatch workaround

ECAPA voiceprints are channel-sensitive. The same human's voice produces
embeddings that differ by codec (Discord Opus, EMEET, etc.), mic, and noise floor.
Cosine similarity between EMEET-Bilbo and Discord-Bilbo can be ~0.3 — well below
any reasonable match threshold.

**Workaround until v3:** enroll the same person twice with channel-suffixed names:

```
.venv_v2\Scripts\python.exe enroll.py Lance_emeet   --device <emeet-index>
.venv_v2\Scripts\python.exe enroll.py Lance_discord --wav lance_discord_clip.wav
```

Both voiceprints load. Bilbo's local audio gets labeled `Lance_emeet:`; Bilbo's
Discord audio gets `Lance_discord:`. Slightly ugly in the transcript, but stable.

The architectural fix (one label, two channels handled correctly) is the dual-stream
OBS routing planned for v3 — see [ROADMAP.md](ROADMAP.md).

---

## Verifying enrollment

After running `enroll.py`, check the file exists:

```
dir voiceprints\
```

Should list `Frodo.npy`, etc. (Each file is small — about 800 bytes.)

To confirm the embedding loads cleanly:

```
.venv_v2\Scripts\python.exe -c "import numpy as np; e = np.load('voiceprints/Frodo.npy'); print(e.shape, e.dtype, 'norm:', np.linalg.norm(e))"
```

Expect: `(192,) float32 norm: 1.0` (the file is pre-normalized).

To confirm v2 picks it up, start a session and look at the header:

```
.\start_v2.bat
```

```
=== D&D Session Transcript (v2) ===
Started: 2026-05-01 ...
ASR: Systran/faster-whisper-large-v3 (int8_float16 on cuda)
Diarization: pyannote/speaker-diarization-3.1
Enrolled speakers: Frodo, Bilbo, Samwise          <<< should list yours
========================================
```

---

## Re-enrolling

To replace an existing voiceprint (e.g., you've changed mics, or the original
recording was poor):

```
.venv_v2\Scripts\python.exe enroll.py Frodo --overwrite
```

Without `--overwrite`, the script refuses to clobber an existing file.

---

## Removing a voiceprint

Just delete the file — no other state to clean up:

```
del voiceprints\Frodo.npy
```

Future sessions will fall back to `Speaker_NN:` for that person.

---

## Troubleshooting

### `Audio energy: 0.0001  (warn if <0.005)`

You weren't speaking, or the wrong device was selected, or the mic is muted in
Windows. Re-run with `--device <index>` after checking `list_devices.py`.

### Speaker still labels as `Speaker_NN` even after enrollment

Cosine similarity didn't exceed the 0.55 threshold. Causes, in order of likelihood:
1. **Channel mismatch** — see workaround above.
2. **Bad enrollment recording** — too short, too quiet, or too much background noise.
   Re-enroll with `--overwrite` and a cleaner sample.
3. **Threshold too strict** — edit `SPEAKER_MATCH_THRESHOLD` in `transcribe_v2.py`,
   try 0.45. Risk: distinct people getting merged.

### `OSError: [WinError 1314] A required privilege is not held by the client`

First-run-only symlink permission issue when downloading the ECAPA model. The
script has a copy-fallback, but if it slips through: enable Windows Developer Mode
once (Settings → System → For developers → Developer Mode = On), then re-run.

### `cublas64_12.dll is not found`

Make sure you're using `.venv_v2\Scripts\python.exe` (not the system Python).
The venv adds CUDA DLL paths at startup.

---

## Best-practice checklist

- [ ] Quiet room, no other people talking
- [ ] Mic at normal session distance (don't lean in)
- [ ] 30 seconds of natural speech (read a paragraph, not single words)
- [ ] Same mic the player will use during sessions
- [ ] Energy reading above 0.01 in the script output
- [ ] One enrollment per channel (EMEET vs Discord) until v3 dual-stream lands
