"""
Enroll a player's voice for v2 speaker labeling.

Usage:
    enroll.py <name>                            # live record 30s from default mic
    enroll.py <name> --seconds 45               # custom record length
    enroll.py <name> --device 5                 # explicit input device index
    enroll.py <name> --wav path/to/clip.wav     # use a pre-recorded WAV file

Saves a 192-dim ECAPA-TDNN embedding to voiceprints/<name>.npy.
At session time, transcribe_v2.py auto-loads any voiceprints in that folder.
"""

import argparse
import os
import site
import sys
import time
from pathlib import Path

# CUDA DLL path setup (same as transcribe_v2.py)
_extra = []
for _sp in site.getsitepackages():
    for _pkg in ("cuda_runtime", "cublas"):
        _dll_dir = os.path.join(_sp, "nvidia", _pkg, "bin")
        if os.path.isdir(_dll_dir):
            _extra.append(_dll_dir)
if _extra:
    os.environ["PATH"] = os.pathsep.join(_extra) + os.pathsep + os.environ.get("PATH", "")

# Windows compat: huggingface_hub and speechbrain symlink during fetch. Fall back to copy.
import os as _os
from pathlib import Path
import shutil as _shutil

_orig_os_symlink = _os.symlink
def _os_symlink_or_copy(src, dst, target_is_directory=False, *, dir_fd=None):
    try:
        _orig_os_symlink(src, dst, target_is_directory, dir_fd=dir_fd)
    except OSError:
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
import soundfile as sf
import torch

# torch >=2.6 weights_only default broke pyannote checkpoint loading; force the old default.
_orig_torch_load = torch.load
def _torch_load_compat(*a, **kw):
    kw["weights_only"] = False
    return _orig_torch_load(*a, **kw)
torch.load = _torch_load_compat

from speechbrain.pretrained import EncoderClassifier

SAMPLE_RATE = 16000
VOICEPRINT_DIR = "voiceprints"
ECAPA_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"


def record_live(seconds: int, device: int | None) -> np.ndarray:
    print(f"\n>>> Recording will start in 3 seconds. Speak naturally for ~{seconds}s.")
    print(">>> Read a paragraph aloud, or have a normal conversation. Avoid background voices.")
    for n in (3, 2, 1):
        print(f"   {n}...", flush=True)
        time.sleep(1)
    print(">>> RECORDING NOW. Speak.\n", flush=True)
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    print(">>> Done recording.\n", flush=True)
    return audio[:, 0]


def load_wav(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        # Quick-and-dirty resample via numpy linspace; for serious use, use librosa.
        idx = np.linspace(0, len(audio) - 1, int(len(audio) * SAMPLE_RATE / sr)).astype(np.int64)
        audio = audio[idx]
    return audio.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Enroll a player voiceprint for v2")
    parser.add_argument("name", help="Player name (used as the transcript label and filename)")
    parser.add_argument("--seconds", type=int, default=30, help="Live recording length in seconds")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index (run list_devices.py to see options)")
    parser.add_argument("--wav", help="Use this WAV file instead of recording live")
    parser.add_argument("--cpu", action="store_true", help="Run encoder on CPU instead of GPU")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing voiceprint")
    args = parser.parse_args()

    safe_name = args.name.strip()
    if not safe_name or any(c in safe_name for c in "/\\:*?\"<>|"):
        sys.exit(f"Invalid name: {args.name!r}")

    os.makedirs(VOICEPRINT_DIR, exist_ok=True)
    out_path = Path(VOICEPRINT_DIR) / f"{safe_name}.npy"
    if out_path.exists() and not args.overwrite:
        sys.exit(f"{out_path} already exists. Pass --overwrite to replace.")

    if args.wav:
        print(f"Loading {args.wav}...")
        audio = load_wav(args.wav)
        print(f"Loaded {len(audio) / SAMPLE_RATE:.1f}s of audio.")
    else:
        audio = record_live(args.seconds, args.device)

    energy = float(np.mean(np.abs(audio)))
    print(f"Audio energy: {energy:.4f}  (warn if <0.005)")
    if energy < 0.005:
        print("[WARN] Audio looks very quiet — voiceprint may be unreliable.")

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading ECAPA-TDNN encoder on {device}...")
    encoder = EncoderClassifier.from_hparams(
        source=ECAPA_MODEL_ID,
        run_opts={"device": device},
        savedir=os.path.join(os.path.expanduser("~"), ".cache", "speechbrain", "ecapa"),
    )

    wav = torch.from_numpy(audio).float().unsqueeze(0)
    with torch.no_grad():
        emb = encoder.encode_batch(wav).squeeze().cpu().numpy().astype(np.float32)
    emb /= np.linalg.norm(emb) + 1e-9

    np.save(out_path, emb)
    print(f"\nVoiceprint saved: {out_path}")
    print(f"Future v2 sessions will label this speaker as '{safe_name}'.")


if __name__ == "__main__":
    main()
