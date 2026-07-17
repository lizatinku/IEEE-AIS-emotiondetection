# Vibration recordings

These are the first validated surface-vibration recordings captured with the
ESP32 + ADS131M02 + SM-24 geophone USB collector. Samples are stored as NumPy
`int32` arrays in raw ADC counts at a calibrated 6 kHz sample rate.

## Sessions

- `20260717_conference-room-01_conference-table_background/`: 55.33 seconds of
  no-playback background vibration.
- `20260717_conference-room-01_conference-table_crema-d-actor-1016/`: 241.75
  seconds captured during CREMA-D actor 1016 playback. Its `events.csv` maps
  each source clip to `calm`, `distress`, or `exclude`.

## Reading a session

- `session.json`: collection metadata and calibrated rate.
- `manifest.csv`: chunk index, sample count, data-integrity status, and SHA-256.
- `telemetry.csv`: packet-arrival and temperature log.
- `events.csv`: per-clip labels; only the actor playback session has populated
  events.
- `node-0001/boot-*/vibration_*.npy`: raw sample chunks.
- `node-0001/boot-*/valid_*.npy`: Boolean masks; `true` means a valid sample.

Example:

```python
import numpy as np

samples = np.load("node-0001/boot-<id>/vibration_000000.npy")
valid = np.load("node-0001/boot-<id>/valid_000000.npy")
```

Use `events.csv` rather than assigning the whole actor session one label. The
actor timeline maps `NEU` to `calm`; `ANG`, `FEA`, and `SAD` to `distress`; and
`DIS`/`HAP` are excluded from the initial three-class dataset.
