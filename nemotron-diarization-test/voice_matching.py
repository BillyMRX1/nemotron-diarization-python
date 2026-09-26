"""Optional local voice matching using Microsoft's official WavLM XVector API."""
import json
from pathlib import Path
import re

import numpy as np

MODEL_ID = "microsoft/wavlm-base-plus-sv"
SAMPLE_RATE = 16000
DEFAULT_THRESHOLD = 0.86  # Model-card example; must be calibrated for your audio.
MATCH_MARGIN = 0.05  # Sample policy: abstain if two diarized speakers score similarly.
MIN_CLIP_SECONDS = 2.0
MAX_CLIP_SECONDS = 10.0


def normalized(vector):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("Voice embedding must be a finite one-dimensional vector.")
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Voice embedding has zero or invalid norm.")
    return vector / norm


def load_profile(path):
    path = Path(path)
    if path.suffixes[-2:] != [".voice", ".json"]:
        raise ValueError("Voice profiles must use the .voice.json suffix (excluded from Git).")
    profile = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict) or profile.get("schema_version") != 1:
        raise ValueError("Unsupported voice profile format; enroll again.")
    if profile.get("model_id") != MODEL_ID or profile.get("sample_rate") != SAMPLE_RATE:
        raise ValueError("Voice profile uses an incompatible model or sample rate; enroll again.")
    revision = profile.get("model_revision")
    if revision is not None and (not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)):
        raise ValueError("Invalid model revision in voice profile; enroll again.")
    name = profile.get("name")
    if not isinstance(name, str) or not name.strip() or not name.isprintable():
        raise ValueError("Voice profile name must be nonempty printable text.")
    vector = normalized(profile.get("embedding"))
    if vector.shape != (512,):
        raise ValueError("Expected a 512-dimensional WavLM voice embedding; enroll again.")
    return {**profile, "embedding": vector}


def select_clips(audio, intervals, max_clips=3):
    """Take bounded, non-silent clips, preferring longer speech intervals."""
    clips = []
    duration = len(audio) / SAMPLE_RATE
    for start, end in sorted(intervals, key=lambda x: x[1] - x[0], reverse=True):
        start, end = max(0.0, start), min(duration, end)
        while end - start >= MIN_CLIP_SECONDS:
            stop = min(end, start + MAX_CLIP_SECONDS)
            clip = audio[round(start * SAMPLE_RATE):round(stop * SAMPLE_RATE)]
            if len(clip) >= int(MIN_CLIP_SECONDS * SAMPLE_RATE):
                rms = float(np.sqrt(np.mean(clip.astype(np.float64) ** 2)))
                if rms >= 1e-4:
                    clips.append(clip)
                    if len(clips) == max_clips:
                        return clips
            start = stop
    return clips


def exclusive_intervals(segments, speaker):
    """Subtract other speakers' activity so overlap never enters a voice profile."""
    result = []
    others = sorted((s["Start"], s["End"]) for s in segments if s["Speaker"] != speaker)
    for segment in segments:
        if segment["Speaker"] != speaker:
            continue
        pieces = [(segment["Start"], segment["End"])]
        for left, right in others:
            remaining = []
            for start, end in pieces:
                if right <= start or left >= end:
                    remaining.append((start, end))
                else:
                    if start < left:
                        remaining.append((start, left))
                    if right < end:
                        remaining.append((right, end))
            pieces = remaining
        result.extend(pieces)
    return result


def choose_match(scores, threshold):
    """One saved identity may label at most one diarization channel per file."""
    if not np.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("Match threshold must be a finite cosine similarity between -1 and 1.")
    if any(not np.isfinite(score) for score in scores.values()):
        raise ValueError("Non-finite voice similarity; cannot choose a match.")
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] < threshold:
        return None, "no speaker reached the threshold"
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < MATCH_MARGIN:
        return None, "ambiguous: top speakers have similar scores"
    return ranked[0][0], "matched"


class VoiceEncoder:
    def __init__(self, device, revision=None):
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

        print(f"Loading voice model: {MODEL_ID} on {device}...", flush=True)
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, revision=revision)
        self.model = WavLMForXVector.from_pretrained(MODEL_ID, revision=revision).to(device).eval()
        self.device = device

    def encode(self, clips):
        import torch

        if not clips:
            raise ValueError("No usable speech clips: need at least 2 seconds of clean, non-silent audio per clip.")
        vectors = []
        with torch.inference_mode():
            for clip in clips:
                inputs = self.extractor(
                    clip, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
                ).to(self.device)
                output = self.model(**inputs).embeddings
                vectors.append(normalized(output[0].float().cpu().numpy()))
        return normalized(np.mean(vectors, axis=0))


def match_speakers(audio, sample_rate, segments, profile, device, threshold):
    import soxr

    if sample_rate != SAMPLE_RATE:
        audio = soxr.resample(audio, sample_rate, SAMPLE_RATE).astype(np.float32)
    candidates = {}
    print(f"\nVoice matching: {profile['name']} (threshold {threshold:.3f})")
    for speaker in sorted({s["Speaker"] for s in segments}):
        clips = select_clips(audio, exclusive_intervals(segments, speaker))
        if sum(len(c) for c in clips) >= 5 * SAMPLE_RATE:
            candidates[speaker] = clips
        else:
            print(f"speaker_{speaker}: insufficient clean speech; keeping anonymous label")
    if not candidates:
        print("No usable speaker clips; keeping anonymous labels.")
        return {}
    encoder = VoiceEncoder(device, revision=profile.get("model_revision"))
    scores = {}
    for speaker, clips in candidates.items():
        embedding = encoder.encode(clips)
        scores[speaker] = float(np.clip(np.dot(embedding, profile["embedding"]), -1, 1))
        print(f"speaker_{speaker}: cosine similarity {scores[speaker]:.3f} "
              f"({sum(len(c) for c in clips) / SAMPLE_RATE:.1f}s compared)")
    winner, reason = choose_match(scores, threshold)
    print(f"Voice match result: {reason}. Scores are similarities, not probabilities.")
    print(f"Voice model parameter device: {next(encoder.model.parameters()).device}")
    return {} if winner is None else {winner: profile["name"]}
