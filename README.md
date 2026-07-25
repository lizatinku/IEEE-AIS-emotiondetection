## Preparing the CREMA-D Test Dataset

Before running inference, generate the test NumPy clips from the recorded vibration data.

### 1. Generate Calm and Distress Clips
```bash
python3 prepare_cremad_test.py
```
This script:
- Reconstructs the continuous vibration recording from the chunked `.npy` files.
- Uses the timestamps in `events.csv` to extract the annotated calm and distress segments.
- Saves the extracted clips 

> **Note:** To generate clips for a different actor, update the Path("VibrationData/vib_1034") to 1016/1020 etc in `prepare_cremad_test.py`.

---

### 2. Generate Background Clips
```bash
python3 prepare_bg_test.py
```
This script:
- Reconstructs the continuous background vibration recording.
- Splits the recording into fixed-length background clips.

After running both scripts, your test dataset should have the following structure:
```text
CREMA_D_Test/
├── background/
├── calm_speech/
└── distress/
```