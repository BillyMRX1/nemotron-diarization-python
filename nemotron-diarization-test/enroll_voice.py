"""Save a local voice profile from a clean WAV containing ONLY the named person."""
import argparse
import json
from pathlib import Path
import re
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="WAV with only the person to enroll; 10-30 seconds recommended")
    parser.add_argument("--name", required=True, help="Display name, e.g. Person")
    parser.add_argument("--output", type=Path, help="Destination ending .voice.json; default profiles/<name>.voice.json")
    args = parser.parse_args()
    name = args.name.strip()
    if not name or not name.isprintable():
        parser.error("Name must be nonempty printable text.")
    if not args.audio.is_file() or args.audio.suffix.lower() != ".wav":
        parser.error("Supply an existing WAV file containing only the named speaker.")
    slug = re.sub(r"[^a-z0-9_-]+", "_", name.lower()).strip("_") or "speaker"
    output = args.output or Path(__file__).resolve().parent / "profiles" / f"{slug}.voice.json"
    if output.suffixes[-2:] != [".voice", ".json"]:
        parser.error("Output must end with .voice.json so Git excludes the voice profile.")
    if output.exists():
        parser.error(f"Profile already exists: {output}. Choose a different --output; profiles are not overwritten.")

    import numpy as np
    import soundfile as sf
    import soxr
    import torch
    from voice_matching import MODEL_ID, SAMPLE_RATE, VoiceEncoder, select_clips

    audio, rate = sf.read(args.audio, dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Enrollment audio must be nonempty and finite.")
    print(f"Enrollment input: {rate} Hz, {audio.shape[1]} channel(s), {len(audio)/rate:.2f}s")
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        audio = soxr.resample(audio, rate, SAMPLE_RATE).astype(np.float32)
    clips = select_clips(audio, [(0, len(audio) / SAMPLE_RATE)], max_clips=6)
    used_seconds = sum(len(c) for c in clips) / SAMPLE_RATE
    if used_seconds < 5:
        raise ValueError("Need at least 5 seconds of usable audio; record 10-30 seconds of only your voice.")
    print("Enrollment assumes a single speaker. It cannot verify who the speaker is.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: CUDA unavailable; using CPU.", file=sys.stderr)
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    encoder = VoiceEncoder(device)
    embedding = encoder.encode(clips)
    profile = {
        "schema_version": 1,
        "name": name,
        "model_id": MODEL_ID,
        "model_revision": getattr(encoder.model.config, "_commit_hash", None),
        "sample_rate": SAMPLE_RATE,
        "enrollment_seconds": used_seconds,
        "embedding": embedding.tolist(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Saved local profile for {name}: {output}")
    print(f"Used {used_seconds:.1f}s in {len(clips)} clip(s). No recording is stored in the profile.")
    print("Verify on a different recording; same-audio matching is not an accuracy test.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nVoice enrollment failed; no automatic package changes were made.", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
