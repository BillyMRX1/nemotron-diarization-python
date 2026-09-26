# Nemotron Diarization Sample

Minimal Python sample code for local speaker diarization with NVIDIA's official
`nvidia/Nemotron-3-Diarization` model: **audio -> speaker timestamps**.

The sample adapts the [offline Transformers example in NVIDIA's model card](https://huggingface.co/nvidia/Nemotron-3-Diarization#-transformers-usage).
It adds WAV input, format conversion, automatic CUDA selection, timing, and GPU
memory reporting. The model, feature extraction, and speaker-segment decoding
use the official Hugging Face implementation. This is an independent sample,
not an NVIDIA-maintained repository.

There is no transcription, translation, summarization, server, or UI.
Optional voice enrollment adds a second model to match a speaker to a saved name.

## Setup on Windows (PowerShell)

Tested on Windows 11 with Python 3.13.15 and an RTX 5060 Ti 16 GB connected by
eGPU to a ROG Ally X. Git must be installed for the Transformers source dependency.

From the repository root:

```powershell
cd nemotron-diarization-test
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Dependencies: PyTorch, Transformers, NumPy, SoundFile, soxr, and librosa, plus
their required dependencies. NVIDIA's model card was checked on September 26,
2026 and instructed installing Transformers from GitHub. `requirements.txt`
pins the inspected Git revision. `installed-versions.txt` records the tested
package versions, including PyTorch 2.11.0+cu128 and Transformers 5.18.0.dev0.
The CUDA PyTorch wheel includes its runtime; no separate CUDA toolkit is needed.

## Run

In the activated environment, from `nemotron-diarization-test`:

```powershell
python test_diarization.py "C:\path\to\meeting.wav"
```

The script accepts actual WAV files, detects their sample rate and channels,
averages channels to mono, and resamples in Python to the processor's expected
rate (16 kHz). SoundFile's Windows wheel bundles its audio decoder; ffmpeg is
not required for WAV input. Renaming an M4A file to WAV does **not** convert it.

For M4A recordings, optionally install ffmpeg and convert first:

```powershell
winget install --id Gyan.FFmpeg --exact --source winget
# Reopen PowerShell after installation to refresh PATH.
ffmpeg -i "C:\path\to\Recording.m4a" -ar 16000 -ac 1 -c:a pcm_s16le meeting_converted.wav
python test_diarization.py meeting_converted.wav
```

Audio files are deliberately excluded from Git. Supply your own recording.

## Optional: enroll a voice and recognize it later

This belongs in the same sample: Nemotron still produces speaker timestamps;
a separate voice-verification step can attach a saved name. It does not retrain
Nemotron or teach it persistent identities.

From `nemotron-diarization-test`, with the virtual environment activated:

```powershell
# billy_only.wav must contain ONLY Billy's voice, ideally 10-30 seconds.
python enroll_voice.py billy_only.wav --name Billy

# Compare speakers in a different recording with that saved profile.
python test_diarization.py meeting_converted.wav --voice-profile profiles/billy.voice.json
```

Enrollment saves `profiles/billy.voice.json` next to the scripts. To choose a
different destination, pass `--output profiles/billy_new.voice.json`. Existing
profiles are never overwritten. Delete a profile locally to forget it, or create
a new one to replace an old enrollment. All `profiles/` directories and
`*.voice.json` files are excluded from Git, along with recordings. Profiles
contain a name, model metadata, and an embedding, not the original audio; treat
them as private voice data. They are plain local JSON, not encrypted storage.

The second model is [`microsoft/wavlm-base-plus-sv`](https://huggingface.co/microsoft/wavlm-base-plus-sv#speaker-verification).
Its official example uses `Wav2Vec2FeatureExtractor`, `WavLMForXVector`,
normalized embeddings, and cosine similarity. It downloads automatically through
the Hugging Face cache. Our existing dependencies suffice; no new Python packages
are needed. It runs on CUDA when available and CPU otherwise. A saved profile
records the model revision, which is reused when matching. Plain diarization
without `--voice-profile` never loads this second model.

This sample supports one saved identity per run. Enrollment assumes single-speaker
speech; it cannot check that the person is actually the name supplied. Do not use
a whole multi-speaker meeting as enrollment. It requires at least 5 seconds of
usable audio, splits input into 2-10 second clips, and uses at most six clips.
Its simple energy check rejects silence, not music or other non-speech sounds.

For matching, the sample subtracts overlapping speakers' intervals, skips clips
shorter than 2 seconds, and compares up to three clips per speaker, requiring at
least 5 seconds total. It averages
normalized clip embeddings and normalizes that average. These clip-selection and
decision rules are sample code around the official encoder, not NVIDIA features.
Only the best matching channel is renamed, and the original channel remains
visible, for example `Billy (speaker_0): 1.66s -> 8.79s`. Short or unmatched
speakers keep their original `speaker_N` labels. Segment times are unchanged.

Cosine scores are printed for inspection; they are **not confidence percentages**.
The default threshold of `0.86` comes from Microsoft's example, which notes it is
dataset-dependent. This sample also abstains when the top two speakers' scores
differ by less than `0.05`. Neither rule guarantees identity accuracy. Test with
separate recordings of the enrolled person and other people before adjusting:

```powershell
python test_diarization.py meeting.wav --voice-profile profiles/billy.voice.json --match-threshold 0.90
```

A higher threshold is stricter; lowering it can incorrectly label other people.
Microphones, noise, voice changes, and diarization mistakes affect results.
This is an experimental naming convenience, not identity authentication.
Voice matching time is reported separately; GPU peak memory includes the optional
voice model when used. No transcript or external voice-identification service is
involved.

Run the local policy tests (no model downloads):

```powershell
python -m unittest test_voice_matching -v
```

Validation on the RTX 5060 Ti: seven policy tests passed, and real GPU enrollment
and matching completed without additional packages. A 7.13-second anonymous
speaker clip was enrolled; testing on a later portion of the same recording
(excluding enrollment audio) gave similarities of `0.973` for that speaker and
`0.773` for the other speaker. The default threshold matched only the former.
Matching added 1.87 seconds on that run, with 902.8 MiB peak allocated VRAM for
the combined pipeline. This checks integration on one recording, not recognition
accuracy across recordings or microphones. No real-person name was assigned in
the test. Plain diarization also passed its original silence smoke test.

WavLM emitted an upstream PyTorch attention-mask type deprecation warning;
inference completed successfully. This is separate from the existing Transformers
docstring diagnostic noted below.

## Output and verification

The script reports PyTorch/Transformers versions, CUDA availability and runtime,
GPU name, selected device, audio duration, speaker segments/count, model load
time, forward-pass time, and peak PyTorch allocated/reserved VRAM.

Example excerpt from a real recording tested on this machine:

```text
GPU: NVIDIA GeForce RTX 5060 Ti
Device: cuda:0
Audio duration: 115.58 sec
Diarization:
speaker_0: 1.66s -> 8.79s
speaker_0: 9.22s -> 14.14s
...
Number of detected speakers: 2
Model load time: 2.14 sec (includes download on first run)
Inference time: 15.23 sec (forward pass only)
Model parameter device: cuda:0
Output tensor device: cuda:0
Peak PyTorch allocated VRAM: 420.1 MiB
Peak PyTorch reserved VRAM: 440.0 MiB
```

These measurements are one run, not a benchmark guarantee. The recording owner
checked the output by listening and reported matching speaker labels/timestamps;
no formal diarization error rate was measured. A separate 3-second dummy-silence
smoke test also completed successfully. `pip check` passed.

GPU execution is evidenced by the model and output tensor devices, rather than
CUDA availability alone. Timing synchronizes CUDA. VRAM figures cover PyTorch
allocations, not the entire GPU or desktop. `speaker_0`, `speaker_1`, etc. are
anonymous labels within one recording. Pauses can split one speaker into many
segments; overlapping speech can produce overlapping segments.

## Model download and cache

`AutoProcessor.from_pretrained` and
`AutoModelForAudioFrameClassification.from_pretrained` automatically download
`nvidia/Nemotron-3-Diarization` through the normal Hugging Face cache, usually
`%USERPROFILE%\.cache\huggingface\hub`. `HF_HOME` and `HF_HUB_CACHE` are honored.
Later runs reuse cached files. The tested model revision was
`f667ed73aee57d40cc39428eb768b4fd87a0a29e`.

No model weights or tokens are included. The tested download needed no login.
If authentication is required, accept any required model terms and use:

```powershell
hf auth login
```

## Implementation and limitations

The official processor handles features and
`processor.extract_speaker_dict(logits, inputs.attention_mask)` handles decoding.
Explicit model placement with `.to(device)` avoids needing Accelerate. CUDA is
selected when available; otherwise the script prints a CPU fallback warning.

- Native Windows inference succeeded in the tested environment. NVIDIA's model
  card lists Linux as its preferred/supported OS; WSL2 was not needed here.
- The inspected Transformers development build prints an `[ERROR]` diagnostic
  about an undocumented `image_like_kwargs` argument during import. This is
  upstream docstring validation and did not prevent successful inference.
- Hugging Face may warn about unavailable Windows cache symlinks. Downloads still
  work but may require extra disk space; Windows Developer Mode enables symlinks.
- This sample uses offline full-recording inference. Long recordings can exhaust
  memory; start with a short clip. The model supports up to eight speaker channels.
- Opposite-phase stereo audio can cancel when averaged; use clean mono audio if
  needed. Very short detected segments may be artifacts: verify by listening.
- Errors preserve the full traceback and exit nonzero. Inspect the actual error
  before changing dependencies; no automatic installs or downgrades occur.

See [NVIDIA's model card](https://huggingface.co/nvidia/Nemotron-3-Diarization)
for model usage, limitations, and model license terms.
