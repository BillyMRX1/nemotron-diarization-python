"""Minimal offline inference using NVIDIA's official Transformers example."""
import argparse
import sys
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Path to a WAV recording")
    parser.add_argument("--voice-profile", type=Path, help="Optional local .voice.json profile created by enroll_voice.py")
    parser.add_argument("--match-threshold", type=float, default=0.86, help="Cosine threshold for voice matching (default 0.86; needs calibration)")
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"Audio file does not exist: {args.audio}")
    if args.audio.suffix.lower() != ".wav":
        parser.error("This experiment accepts WAV files.")
    if not -1 <= args.match_threshold <= 1:
        parser.error("--match-threshold must be between -1 and 1.")

    profile = None
    if args.voice_profile:
        from voice_matching import load_profile
        profile = load_profile(args.voice_profile)

    import numpy as np
    import soundfile as sf
    import soxr
    import torch
    import transformers
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"CUDA availability: {cuda}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0) if cuda else 'None'}")
    print(f"Device: {device}")
    if not cuda:
        print("WARNING: CUDA is unavailable; falling back to CPU (may be slow).", file=sys.stderr)

    audio, sample_rate = sf.read(args.audio, dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio must be nonempty and contain only finite samples.")
    duration = len(audio) / sample_rate
    print(f"Input: {sample_rate} Hz, {audio.shape[1]} channel(s)")
    print(f"Audio duration: {duration:.2f} sec")
    audio = audio.mean(axis=1)

    model_id = "nvidia/Nemotron-3-Diarization"
    print(f"Loading {model_id} (first run downloads to the Hugging Face cache)...", flush=True)
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id)
    # Explicit placement avoids an otherwise unnecessary Accelerate dependency.
    model = AutoModelForAudioFrameClassification.from_pretrained(model_id).to(device).eval()
    if cuda:
        torch.cuda.synchronize()
    load_time = time.perf_counter() - started
    expected_rate = processor.feature_extractor.sampling_rate
    if sample_rate != expected_rate:
        audio = soxr.resample(audio, sample_rate, expected_rate).astype(np.float32)
    print(f"Model input: {expected_rate} Hz, mono")
    inputs = processor(audio, sampling_rate=expected_rate).to(model.device, dtype=model.dtype)
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(**inputs).logits
    if cuda:
        torch.cuda.synchronize()
    inference_time = time.perf_counter() - started
    segments = processor.extract_speaker_dict(logits, inputs.attention_mask)[0]
    labels = {}
    if profile is not None:
        from voice_matching import match_speakers
        match_started = time.perf_counter()
        labels = match_speakers(audio, expected_rate, segments, profile, device, args.match_threshold)
        if cuda:
            torch.cuda.synchronize()
        print(f"Voice matching time: {time.perf_counter() - match_started:.2f} sec (includes voice model loading)")
    print("\nDiarization:")
    for segment in segments:
        speaker = segment['Speaker']
        label = f"{labels[speaker]} (speaker_{speaker})" if speaker in labels else f"speaker_{speaker}"
        print(f"{label}: {segment['Start']:.2f}s -> {segment['End']:.2f}s")
    if not segments:
        print("(No speaker segments detected.)")
    print(f"Number of detected speakers: {len({s['Speaker'] for s in segments})}")
    print(f"Model load time: {load_time:.2f} sec (includes download on first run)")
    print(f"Inference time: {inference_time:.2f} sec (forward pass only)")
    print(f"Model parameter device: {next(model.parameters()).device}")
    print(f"Output tensor device: {logits.device}")
    if cuda:
        print(f"Peak PyTorch allocated VRAM: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")
        print(f"Peak PyTorch reserved VRAM: {torch.cuda.max_memory_reserved() / 2**20:.1f} MiB")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nDiarization failed. Full error follows; no automatic package changes were made.", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
