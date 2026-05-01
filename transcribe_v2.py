"""
v2 transcription pipeline: faster-whisper large-v3 + pyannote diarization + speaker enrollment.

Differences from v1 (transcribe.py):
  - Uses Whisper large-v3 (full, not turbo) for higher accuracy
  - Adds speaker diarization via pyannote-audio 3.1
  - Tags each segment with a global speaker ID consistent across chunks
  - If voiceprints/<name>.npy files exist, resolves speaker IDs to names via ECAPA-TDNN cosine match
  - Saves per-segment ECAPA embeddings to a sidecar .npz file for retroactive enrollment

Output format per line:
    [HH:MM:SS] Bilbo: text...
    [HH:MM:SS] Speaker_03: text...   (when no enrollment match)
"""

import argparse
import json
import os
import queue
import signal
import site
import threading
import time
from datetime import datetime
from pathlib import Path

# Add CUDA DLL directories to PATH for ctranslate2 (same as v1).
_extra = []
for _sp in site.getsitepackages():
    for _pkg in ("cuda_runtime", "cublas"):
        _dll_dir = os.path.join(_sp, "nvidia", _pkg, "bin")
        if os.path.isdir(_dll_dir):
            _extra.append(_dll_dir)
if _extra:
    os.environ["PATH"] = os.pathsep.join(_extra) + os.pathsep + os.environ.get("PATH", "")

# Windows compat: huggingface_hub and speechbrain create symlinks during model fetch,
# which require admin or Developer Mode. Fall back to copy on permission errors.
import os as _os
from pathlib import Path
import shutil as _shutil

_orig_os_symlink = _os.symlink
def _os_symlink_or_copy(src, dst, target_is_directory=False, *, dir_fd=None):
    try:
        _orig_os_symlink(src, dst, target_is_directory, dir_fd=dir_fd)
    except OSError:
        # huggingface_hub passes a relative `src` resolved against the dst's directory.
        src_path = src if _os.path.isabs(src) else _os.path.join(_os.path.dirname(dst), src)
        _shutil.copy2(src_path, dst)
_os.symlink = _os_symlink_or_copy

_orig_symlink_to = Path.symlink_to
def _symlink_to_or_copy(self, target, target_is_directory=False):
    try:
        _orig_symlink_to(self, target, target_is_directory)
    except OSError:
        _shutil.copy2(target, self)
Path.symlink_to = _symlink_to_or_copy

import numpy as np
import sounddevice as sd
import torch

# torch >=2.6 made `weights_only=True` the default in torch.load; pyannote 3.3.2
# checkpoints contain types not in the safe-globals allowlist. Force the old behavior.
_orig_torch_load = torch.load
def _torch_load_compat(*a, **kw):
    kw["weights_only"] = False
    return _orig_torch_load(*a, **kw)
torch.load = _torch_load_compat

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline as DiarizationPipeline
from speechbrain.pretrained import EncoderClassifier

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 0.5
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)
DEFAULT_BUFFER_SECONDS = 30
SILENCE_THRESHOLD_SECONDS = 1.5
MIN_FLUSH_SECONDS = 5.0
SILENCE_ENERGY = 0.005

WHISPER_MODEL_ID = "Systran/faster-whisper-large-v3"
DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"
ECAPA_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"

TRANSCRIPT_DIR = "transcripts"
VOICEPRINT_DIR = "voiceprints"

# Cosine similarity threshold for cross-chunk speaker matching and voiceprint identification.
# 0.55-0.70 is the common ECAPA range; lower = more permissive, may merge distinct speakers.
SPEAKER_MATCH_THRESHOLD = 0.55

running = threading.Event()
running.set()

audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
transcription_queue: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(maxsize=5)


def find_device(name_or_index: str) -> int:
    try:
        return int(name_or_index)
    except ValueError:
        pass
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    mme_idx = next((i for i, h in enumerate(hostapis) if h["name"].lower() == "mme"), None)
    matches = [
        (i, d)
        for i, d in enumerate(devices)
        if name_or_index.lower() in d["name"].lower() and d["max_input_channels"] > 0
    ]
    if not matches:
        raise ValueError(f"No input device found matching '{name_or_index}'. Run list_devices.py.")
    if len(matches) == 1:
        return matches[0][0]
    if mme_idx is not None:
        for i, d in matches:
            if d["hostapi"] == mme_idx:
                print(f"[INFO] Multiple matches for '{name_or_index}', using MME: [{i}] {d['name']}", flush=True)
                return i
    i, d = matches[0]
    print(f"[INFO] Multiple matches for '{name_or_index}', using first: [{i}] {d['name']}", flush=True)
    return i


def write_line(path: str, line: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def append_embedding(path: str, chunk_offset: float, label: str, embedding: np.ndarray):
    """Append a single embedding row to a JSONL file. Open/close per write so a crash
    after this returns cannot lose the row."""
    record = {
        "offset": float(chunk_offset),
        "label": label,
        "embedding": embedding.astype(np.float32).tolist(),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def format_elapsed(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[AUDIO] {status}", flush=True)
    try:
        audio_queue.put_nowait(indata[:, 0].copy())
    except queue.Full:
        pass


def buffer_worker(buffer_max_samples: int):
    buffer = np.zeros(buffer_max_samples, dtype=np.float32)
    write_pos = 0
    silence_samples = 0
    total_samples = 0

    while running.is_set():
        try:
            chunk = audio_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        chunk_len = len(chunk)
        energy = np.mean(np.abs(chunk))
        if energy < SILENCE_ENERGY:
            silence_samples += chunk_len
        else:
            silence_samples = 0

        space = buffer_max_samples - write_pos
        if chunk_len <= space:
            buffer[write_pos:write_pos + chunk_len] = chunk
            write_pos += chunk_len
        else:
            elapsed = total_samples / SAMPLE_RATE
            try:
                transcription_queue.put_nowait((buffer[:write_pos].copy(), elapsed))
            except queue.Full:
                print("[WARNING] Transcription falling behind, dropping audio chunk", flush=True)
            total_samples += write_pos
            buffer = np.zeros(buffer_max_samples, dtype=np.float32)
            buffer[:chunk_len] = chunk
            write_pos = chunk_len
            silence_samples = 0
            continue

        silence_seconds = silence_samples / SAMPLE_RATE
        buffered_seconds = write_pos / SAMPLE_RATE
        if silence_seconds >= SILENCE_THRESHOLD_SECONDS and buffered_seconds >= MIN_FLUSH_SECONDS:
            elapsed = total_samples / SAMPLE_RATE
            try:
                transcription_queue.put_nowait((buffer[:write_pos].copy(), elapsed))
            except queue.Full:
                print("[WARNING] Transcription falling behind, dropping audio chunk", flush=True)
            total_samples += write_pos
            buffer = np.zeros(buffer_max_samples, dtype=np.float32)
            write_pos = 0
            silence_samples = 0

    if write_pos > 0:
        elapsed = total_samples / SAMPLE_RATE
        try:
            transcription_queue.put((buffer[:write_pos].copy(), elapsed), timeout=5.0)
        except queue.Full:
            pass


def load_voiceprints(directory: str) -> dict[str, np.ndarray]:
    """Load enrolled voiceprints. Each is a 192-dim ECAPA embedding stored as <name>.npy."""
    out: dict[str, np.ndarray] = {}
    p = Path(directory)
    if not p.is_dir():
        return out
    for f in p.glob("*.npy"):
        try:
            emb = np.load(f).astype(np.float32).flatten()
            emb /= np.linalg.norm(emb) + 1e-9
            out[f.stem] = emb
        except Exception as e:
            print(f"[WARN] Failed to load voiceprint {f}: {e}", flush=True)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class SpeakerRegistry:
    """
    Tracks global speaker identities across chunks.

    Each call to assign() takes a list of (cluster_id, embedding) pairs from one chunk
    and returns a mapping from cluster_id to a stable global label (name from voiceprints
    if matched, else "Speaker_NN").
    """

    def __init__(self, voiceprints: dict[str, np.ndarray], threshold: float = SPEAKER_MATCH_THRESHOLD):
        self.voiceprints = voiceprints
        self.threshold = threshold
        self.global_embeddings: dict[str, np.ndarray] = {}  # label -> running mean embedding
        self.counts: dict[str, int] = {}
        self.next_anon = 1

    def _match(self, emb: np.ndarray) -> tuple[str | None, float]:
        emb_n = emb / (np.linalg.norm(emb) + 1e-9)
        best_label, best_score = None, -1.0
        for name, ref in self.voiceprints.items():
            s = cosine(emb_n, ref)
            if s > best_score:
                best_label, best_score = name, s
        for label, ref in self.global_embeddings.items():
            s = cosine(emb_n, ref)
            if s > best_score:
                best_label, best_score = label, s
        return (best_label, best_score) if best_score >= self.threshold else (None, best_score)

    def assign(self, embeddings_by_cluster: dict[str, np.ndarray]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for cluster_id, emb in embeddings_by_cluster.items():
            label, score = self._match(emb)
            if label is None:
                label = f"Speaker_{self.next_anon:02d}"
                self.next_anon += 1
                self.global_embeddings[label] = emb / (np.linalg.norm(emb) + 1e-9)
                self.counts[label] = 1
            else:
                if label in self.global_embeddings:
                    n = self.counts[label]
                    new_mean = (self.global_embeddings[label] * n + emb / (np.linalg.norm(emb) + 1e-9)) / (n + 1)
                    self.global_embeddings[label] = new_mean / (np.linalg.norm(new_mean) + 1e-9)
                    self.counts[label] = n + 1
            mapping[cluster_id] = label
        return mapping


def diarize_chunk(
    diar_pipeline: DiarizationPipeline,
    audio: np.ndarray,
    sample_rate: int,
) -> list[tuple[float, float, str]]:
    """Run pyannote on a numpy chunk; returns list of (start, end, speaker_id) in seconds."""
    waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)
    annotation = diar_pipeline({"waveform": waveform, "sample_rate": sample_rate})
    turns = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append((turn.start, turn.end, speaker))
    return turns


def speaker_for_segment(
    seg_start: float,
    seg_end: float,
    turns: list[tuple[float, float, str]],
) -> str | None:
    """Pick the speaker whose turn overlaps most with [seg_start, seg_end]."""
    best, best_overlap = None, 0.0
    for t_start, t_end, spk in turns:
        overlap = max(0.0, min(seg_end, t_end) - max(seg_start, t_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best = spk
    return best


def compute_cluster_embeddings(
    encoder: EncoderClassifier,
    audio: np.ndarray,
    sample_rate: int,
    turns: list[tuple[float, float, str]],
) -> dict[str, np.ndarray]:
    """For each pyannote cluster id, compute the mean ECAPA embedding across its turns."""
    by_cluster: dict[str, list[np.ndarray]] = {}
    for t_start, t_end, spk in turns:
        s = int(t_start * sample_rate)
        e = int(t_end * sample_rate)
        if e - s < int(0.5 * sample_rate):  # skip turns under 0.5s
            continue
        segment = audio[s:e]
        wav = torch.from_numpy(segment).float().unsqueeze(0)
        with torch.no_grad():
            emb = encoder.encode_batch(wav).squeeze().cpu().numpy().astype(np.float32)
        by_cluster.setdefault(spk, []).append(emb)
    return {k: np.mean(v, axis=0) for k, v in by_cluster.items()}


def main():
    parser = argparse.ArgumentParser(description="D&D session transcription v2 (Whisper + diarization)")
    parser.add_argument("--device", default="CABLE Output (VB-Audio Virtual", help="Audio input device name or index")
    parser.add_argument("--compute-type", default="int8_float16",
                        choices=["int8_float16", "float16", "int8"],
                        help="Whisper compute type (default: int8_float16, lower VRAM)")
    parser.add_argument("--buffer-seconds", type=int, default=DEFAULT_BUFFER_SECONDS, help="Max audio buffer before flush")
    parser.add_argument("--cpu", action="store_true", help="Run inference on CPU")
    parser.add_argument("--no-diarize", action="store_true", help="Disable diarization (Whisper-only mode for debugging)")
    args = parser.parse_args()

    audio_device = find_device(args.device)
    buffer_max_samples = SAMPLE_RATE * args.buffer_seconds
    compute_device = "cpu" if args.cpu else "cuda"
    torch_device = torch.device(compute_device)

    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    os.makedirs(VOICEPRINT_DIR, exist_ok=True)

    session_start = datetime.now()
    stamp = session_start.strftime('%Y-%m-%d_%H-%M-%S')
    transcript_path = os.path.join(TRANSCRIPT_DIR, f"session_v2_{stamp}.txt")
    embeddings_path = os.path.join(TRANSCRIPT_DIR, f"session_v2_{stamp}_embeddings.jsonl")

    voiceprints = load_voiceprints(VOICEPRINT_DIR)
    enrolled_summary = ", ".join(voiceprints.keys()) if voiceprints else "(none — speakers will be Speaker_NN)"

    header = (
        f"=== D&D Session Transcript (v2) ===\n"
        f"Started: {session_start.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"ASR: {WHISPER_MODEL_ID} ({args.compute_type} on {compute_device})\n"
        f"Diarization: {'disabled' if args.no_diarize else DIARIZATION_MODEL_ID}\n"
        f"Enrolled speakers: {enrolled_summary}\n"
        f"{'=' * 40}\n"
    )
    write_line(transcript_path, header)
    print(header)

    print("Loading Whisper large-v3 into VRAM...", flush=True)
    whisper = WhisperModel(WHISPER_MODEL_ID, device=compute_device, compute_type=args.compute_type)
    print("Whisper loaded.", flush=True)

    diar_pipeline = None
    encoder = None
    registry = None
    if not args.no_diarize:
        print("Loading pyannote diarization pipeline...", flush=True)
        diar_pipeline = DiarizationPipeline.from_pretrained(DIARIZATION_MODEL_ID)
        diar_pipeline.to(torch_device)
        print("Diarization loaded.", flush=True)

        print("Loading ECAPA-TDNN speaker encoder...", flush=True)
        encoder = EncoderClassifier.from_hparams(
            source=ECAPA_MODEL_ID,
            run_opts={"device": compute_device},
            savedir=os.path.join(os.path.expanduser("~"), ".cache", "speechbrain", "ecapa"),
        )
        print("Encoder loaded.", flush=True)

        registry = SpeakerRegistry(voiceprints)

    print("\nAll models loaded.\n", flush=True)

    def signal_handler(sig, frame):
        print("\n[INFO] Shutting down... flushing remaining audio", flush=True)
        running.clear()

    signal.signal(signal.SIGINT, signal_handler)

    buf_thread = threading.Thread(target=buffer_worker, args=(buffer_max_samples,), daemon=True)
    buf_thread.start()

    print(f"Listening on: {audio_device}", flush=True)
    print(f"Transcript:   {transcript_path}", flush=True)
    if not args.no_diarize:
        print(f"Embeddings:   {embeddings_path}", flush=True)
    print("Press Ctrl+C to stop.\n", flush=True)

    stream = sd.InputStream(
        device=audio_device,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=audio_callback,
    )
    stream.start()

    embedding_count = 0

    try:
        while running.is_set() or not transcription_queue.empty():
            try:
                audio_data, chunk_offset = transcription_queue.get(timeout=2.0)
            except queue.Empty:
                if not running.is_set():
                    break
                continue

            segments_iter, _info = whisper.transcribe(
                audio_data,
                language="en",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200,
                    threshold=0.35,
                ),
                word_timestamps=False,
                condition_on_previous_text=True,
                no_speech_threshold=0.6,
            )
            segments = list(segments_iter)

            cluster_to_label: dict[str, str] = {}
            turns: list[tuple[float, float, str]] = []
            if not args.no_diarize:
                turns = diarize_chunk(diar_pipeline, audio_data, SAMPLE_RATE)
                if turns:
                    cluster_embs = compute_cluster_embeddings(encoder, audio_data, SAMPLE_RATE, turns)
                    cluster_to_label = registry.assign(cluster_embs)
                    for cid, emb in cluster_embs.items():
                        append_embedding(embeddings_path, chunk_offset, cluster_to_label[cid], emb)
                        embedding_count += 1

            for segment in segments:
                wall_seconds = chunk_offset + segment.start
                ts = format_elapsed(wall_seconds)
                text = segment.text.strip()
                if not text:
                    continue
                if turns:
                    cid = speaker_for_segment(segment.start, segment.end, turns)
                    label = cluster_to_label.get(cid, "Speaker_??") if cid else "Speaker_??"
                else:
                    label = "Speaker_??" if not args.no_diarize else "?"
                line = f"[{ts}] {label}: {text}"
                print(line, flush=True)
                write_line(transcript_path, line)

    finally:
        stream.stop()
        stream.close()
        buf_thread.join(timeout=5.0)

        if embedding_count and not args.no_diarize:
            print(f"Wrote {embedding_count} speaker embeddings to {embeddings_path}", flush=True)

        elapsed = format_elapsed(time.time() - session_start.timestamp())
        end_msg = f"\n[INFO] Session ended. Duration: {elapsed}"
        print(end_msg, flush=True)
        write_line(transcript_path, end_msg)
        print(f"Transcript saved to: {transcript_path}")


if __name__ == "__main__":
    main()
