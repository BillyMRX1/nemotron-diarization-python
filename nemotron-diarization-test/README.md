# Nemotron Diarization Sample

See the [repository README](../README.md) for setup, usage, the official NVIDIA
implementation reference, and native Windows GPU test results.

From this directory, activate the environment and run your own WAV file:

```powershell
.\.venv\Scripts\Activate.ps1
python test_diarization.py "C:\path\to\meeting.wav"
```

Recordings and the virtual environment are excluded from Git.

Optional voice enrollment (use a WAV with only the named person):

```powershell
python enroll_voice.py billy_only.wav --name Billy
python test_diarization.py meeting.wav --voice-profile profiles/billy.voice.json
```

Voice profiles are also excluded from Git. See the repository README for
thresholds, enrollment requirements, and matching limitations.
