from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("CREMA_D_Test")
SAMPLE_RATE = 8000

FRAME_SEC = 0.10
FRAME_LEN = int(FRAME_SEC * SAMPLE_RATE)
ACTIVITY_THRESHOLD = 0.20

def frame_rms(signal, frame_len):
    rms_values = []

    for start in range(0, len(signal), frame_len):
        frame = signal[start:start + frame_len]

        if len(frame) == 0:
            continue

        rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
        rms_values.append(rms)

    return np.array(rms_values)


rows = []

for class_name in ["calm_speech", "distress"]:
    folder = DATA_DIR / class_name

    for fpath in sorted(folder.glob("*.npy")):
        signal = np.load(fpath).astype(np.float32).ravel()
        rms = frame_rms(signal, FRAME_LEN)
        if len(rms) == 0 or rms.max() == 0:
            continue

        threshold = ACTIVITY_THRESHOLD * rms.max()
        active = rms >= threshold

        if not active.any():
            continue

        first_active = np.argmax(active)
        last_active = len(active) - 1 - np.argmax(active[::-1])

        leading_dead_sec = first_active * FRAME_SEC
        trailing_dead_sec = (len(active) - 1 - last_active) * FRAME_SEC

        rows.append({
            "filename": fpath.name,
            "class": class_name,
            "duration_sec": len(signal) / SAMPLE_RATE,
            "leading_dead_sec": leading_dead_sec,
            "trailing_dead_sec": trailing_dead_sec,
            "total_dead_sec": leading_dead_sec + trailing_dead_sec,
        })


df = pd.DataFrame(rows)

print(df.head())

print("\nAverage dead space:")
print(df[
    ["leading_dead_sec", "trailing_dead_sec", "total_dead_sec"]
].mean())

print("\nMedian dead space:")
print(df[
    ["leading_dead_sec", "trailing_dead_sec", "total_dead_sec"]
].median())

print("\nLargest dead-space clips:")
print(
    df.sort_values(
        "total_dead_sec",
        ascending=False
    ).head(10)
)