import argparse
import os
import queue
import signal
import site
import threading
import time
from datetime import datetime

# Add CUDA DLL directories to PATH so ctranslate2's C++ runtime can find
# cublas64_12.dll at inference time (lazy-loaded, not covered by os.add_dll_directory).
_extra = []
for _sp in site.getsitepackages():
    for _pkg in ("cuda_runtime", "cublas"):
        _dll_dir = os.path.join(_sp, "nvidia", _pkg, "bin")
        if os.path.isdir(_dll_dir):
            _extra.append(_dll_dir)
if _extra:
    os.environ["PATH"] = os.pathsep.join(_extra) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 0.5
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)
DEFAULT_BUFFER_SECONDS = 30
SILENCE_THRESHOLD_SECONDS = 1.5
MIN_FLUSH_SECONDS = 5.0
SILENCE_ENERGY = 0.005
MODEL_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"
TRANSCRIPT_DIR = "transcripts"

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
        raise ValueError(f"No input device found matching '{name_or_index}'. Run list_devices.py to see available devices.")
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
            transcription_queue.put(
                (buffer[:write_pos].copy(), elapsed), timeout=5.0
            )
        except queue.Full:
            pass


def main():
    parser = argparse.ArgumentParser(description="Real-time D&D session transcription")
    parser.add_argument("--device", default="CABLE Output (VB-Audio Virtual", help="Audio input device name or index")
    parser.add_argument("--compute-type", default="float16", help="Model compute type (float16, int8_float16, int8)")
    parser.add_argument("--buffer-seconds", type=int, default=DEFAULT_BUFFER_SECONDS, help="Max audio buffer before flush")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU")
    args = parser.parse_args()

    device = find_device(args.device)
    buffer_max_samples = SAMPLE_RATE * args.buffer_seconds
    compute_device = "cpu" if args.cpu else "cuda"

    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    session_start = datetime.now()
    transcript_path = os.path.join(
        TRANSCRIPT_DIR,
        f"session_{session_start.strftime('%Y-%m-%d_%H-%M-%S')}.txt",
    )

    header = (
        f"=== D&D Session Transcript ===\n"
        f"Started: {session_start.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Model: {MODEL_ID}\n"
        f"Compute: {args.compute_type} on {compute_device}\n"
        f"{'=' * 40}\n"
    )
    write_line(transcript_path, header)
    print(header)

    print("Loading model into VRAM...", flush=True)
    model = WhisperModel(MODEL_ID, device=compute_device, compute_type=args.compute_type)
    print("Model loaded.\n", flush=True)

    def signal_handler(sig, frame):
        print("\n[INFO] Shutting down... flushing remaining audio", flush=True)
        running.clear()

    signal.signal(signal.SIGINT, signal_handler)

    buf_thread = threading.Thread(target=buffer_worker, args=(buffer_max_samples,), daemon=True)
    buf_thread.start()

    print(f"Listening on: {device}", flush=True)
    print(f"Transcript:   {transcript_path}", flush=True)
    print("Press Ctrl+C to stop.\n", flush=True)

    stream = sd.InputStream(
        device=device,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=audio_callback,
    )
    stream.start()

    last_status = time.time()
    status_interval = 60

    try:
        while running.is_set() or not transcription_queue.empty():
            try:
                audio_data, chunk_offset = transcription_queue.get(timeout=2.0)
            except queue.Empty:
                if not running.is_set():
                    break
                now = time.time()
                if now - last_status >= status_interval:
                    elapsed = format_elapsed(now - session_start.timestamp())
                    msg = f"[STATUS] Queues: audio={audio_queue.qsize()}, transcription={transcription_queue.qsize()} (elapsed: {elapsed})"
                    print(msg, flush=True)
                    write_line(transcript_path, msg)
                    last_status = now
                continue

            segments, info = model.transcribe(
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

            for segment in segments:
                wall_seconds = chunk_offset + segment.start
                ts = format_elapsed(wall_seconds)
                text = segment.text.strip()
                if text:
                    line = f"[{ts}] {text}"
                    print(line, flush=True)
                    write_line(transcript_path, line)

            now = time.time()
            if now - last_status >= status_interval:
                elapsed = format_elapsed(now - session_start.timestamp())
                msg = f"[STATUS] Queues: audio={audio_queue.qsize()}, transcription={transcription_queue.qsize()} (elapsed: {elapsed})"
                print(msg, flush=True)
                write_line(transcript_path, msg)
                last_status = now

    finally:
        stream.stop()
        stream.close()
        buf_thread.join(timeout=5.0)
        elapsed = format_elapsed(time.time() - session_start.timestamp())
        end_msg = f"\n[INFO] Session ended. Duration: {elapsed}"
        print(end_msg, flush=True)
        write_line(transcript_path, end_msg)
        print(f"Transcript saved to: {transcript_path}")


if __name__ == "__main__":
    main()
