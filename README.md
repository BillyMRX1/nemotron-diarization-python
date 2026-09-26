# Nemotron Diarization Sample

Minimal Python sample code for local speaker diarization with NVIDIA's official
`nvidia/Nemotron-3-Diarization` model: **audio -> speaker timestamps**.

The sample adapts the [offline Transformers example in NVIDIA's model card](https://huggingface.co/nvidia/Nemotron-3-Diarization#-transformers-usage).
It adds WAV input, format conversion, automatic CUDA selection, timing, and GPU
memory reporting. The model, feature extraction, and speaker-segment decoding
use the official Hugging Face implementation. This is an independent sample,
not an NVIDIA-maintained repository.

There is no transcription, translation, summarization, server, or UI.

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
