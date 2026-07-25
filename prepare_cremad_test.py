from pathlib import Path

import numpy as np
import pandas as pd


SESSION = Path("VibrationData/vib_1034")
MANIFEST_PATH = SESSION / "manifest.csv"
EVENTS_PATH = SESSION / "events.csv"

OUTPUT_ROOT = Path("CREMA_D_Test")

LABEL_TO_FOLDER = {
    "calm": "calm_speech",
    "distress": "distress",
}

def load_continuous_signal():
    manifest = pd.read_csv(MANIFEST_PATH).sort_values("chunk_index")

    sample_rates = manifest["sample_rate_hz"].dropna().unique()

    if len(sample_rates) != 1:
        raise ValueError(f"Unexpected sample rates: {sample_rates}")

    sample_rate = int(sample_rates[0])
    chunks = []

    for relative_path in manifest["relative_path"]:
        chunk_path = SESSION / relative_path

        if not chunk_path.exists():
            raise FileNotFoundError(f"Missing chunk: {chunk_path}")

        chunk = np.load(chunk_path).reshape(-1)
        chunks.append(chunk)

    signal = np.concatenate(chunks)

    return signal, sample_rate


signal, sample_rate = load_continuous_signal()
events = pd.read_csv(EVENTS_PATH)

print(f"Loaded continuous signal: {len(signal):,} samples")
print(f"Sample rate: {sample_rate} Hz")
print(f"Duration: {len(signal) / sample_rate:.2f} seconds")

for folder in LABEL_TO_FOLDER.values():
    (OUTPUT_ROOT / folder).mkdir(parents=True, exist_ok=True)

saved_counts = {
    "calm_speech": 0,
    "distress": 0,
}

skipped = 0

for _, event in events.iterrows():
    label = str(event["label"])

    if label not in LABEL_TO_FOLDER:
        skipped += 1
        continue

    start_sample = round(float(event["start_s"]) * sample_rate)
    end_sample = round(float(event["end_s"]) * sample_rate)

    if start_sample < 0:
        start_sample = 0

    if end_sample > len(signal):
        print(
            f"Skipping {event['source_clip_id']}: "
            f"event ends beyond recording"
        )
        skipped += 1
        continue

    clip = signal[start_sample:end_sample]

    if len(clip) == 0:
        print(f"Skipping empty clip: {event['source_clip_id']}")
        skipped += 1
        continue

    output_folder = LABEL_TO_FOLDER[label]

    filename = Path(str(event["source_clip_id"])).stem + ".npy"
    output_path = OUTPUT_ROOT / output_folder / filename

    np.save(output_path, clip)

    saved_counts[output_folder] += 1

print("\nSaved CREMA-D test clips:")
print(f"  Calm speech: {saved_counts['calm_speech']}")
print(f"  Distress:    {saved_counts['distress']}")
print(f"  Skipped:     {skipped}")

print(f"\nOutput folder: {OUTPUT_ROOT.resolve()}")