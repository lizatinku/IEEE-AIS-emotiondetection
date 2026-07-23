# Vibration recordings

These sessions contain surface-vibration recordings captured with an ESP32,
ADS131M02 ADC, and SM-24 geophone. Samples are stored as NumPy `int32` arrays
in raw ADC counts.

## Collection configuration

- Sample rate: 8,000 Hz
- ADC PGA gain: 32
- ADC offset: 0
- Surface: hardwood floor
- Mounting: geophone taped firmly to the floor
- Playback device: Echo Dot at volume setting 10
- Playback-device distance: 0.05 m from the geophone
- Source audio: one concatenated CREMA-D WAV per actor
- Quiet lead-in and at least five seconds of quiet tail around every actor playback

The same hardware configuration and physical placement were retained for the
background and all six actor sessions.

## Sessions

- `vib_bg/`: 62.880 seconds of quiet background vibration.
- `vib_1016/`: 234.048 seconds containing actor 1016 playback.
- `vib_1020/`: 248.704 seconds containing actor 1020 playback.
- `vib_1034/`: 215.616 seconds containing actor 1034 playback.
- `vib_1050/`: 231.680 seconds containing actor 1050 playback.
- `vib_1067/`: 231.840 seconds containing actor 1067 playback.
- `vib_1087/`: 224.992 seconds containing actor 1087 playback.

Each actor session contains 82 clip events. Across the six actors,
`events.csv` provides 72 `calm`, 252 `distress`, and 168 `exclude` intervals.
The initial three-class experiments map `NEU` to `calm`; `ANG`, `FEA`, and
`SAD` to `distress`; and omit `DIS` and `HAP` using the `exclude` label.

## Timing and validation

The source-audio energy envelope was aligned to the measured vibration
envelope after each recording. The resulting onset and small clock-scale
correction are stored under `playback_alignment` in `session.json`; the times
in `events.csv` already include this correction. Use those event times directly
when extracting individual clips.

All seven sessions passed the following checks:

- one sensor node and one continuous boot per session;
- 8,000 Hz declared and measured sample stream;
- zero missing, lost, duplicate, or old packets;
- zero clipped ADC samples;
- valid NumPy shapes, masks, and SHA-256 checksums;
- complete source playback plus quiet tail;
- all 82 event intervals within each actor recording.

For every actor, all 82 clip intervals had higher 80-2000 Hz RMS than the
matched background recording. Full-recording source/vibration envelope
correlations ranged from 0.515 to 0.615.

## Reading a session

- `session.json`: configuration, environment, playback source, and alignment metadata.
- `manifest.csv`: chunk paths, sample counts, rate, integrity status, and SHA-256.
- `telemetry.csv`: packet sequence, sample index, arrival time, and temperature.
- `events.csv`: aligned clip boundaries, labels, source filenames, and emotions.
- `node-0001/boot-*/vibration_*.npy`: raw ADC sample chunks.
- `node-0001/boot-*/valid_*.npy`: Boolean masks; `true` marks a valid sample.

Example:

```python
import csv
from pathlib import Path

import numpy as np

session = Path("VibrationData/vib_1016")
rows = list(csv.DictReader((session / "manifest.csv").open()))
samples = np.concatenate(
    [np.load(session / row["relative_path"]) for row in rows]
)

events = list(csv.DictReader((session / "events.csv").open()))
first = events[0]
sample_rate = int(rows[0]["sample_rate_hz"])
start = round(float(first["start_s"]) * sample_rate)
end = round(float(first["end_s"]) * sample_rate)
first_clip = samples[start:end]

print(first["source_clip_id"], first["label"], first_clip.shape)
```

Use actor identity as the grouping key when creating train, validation, and
test splits so clips from the same speaker do not cross partitions.
