from pathlib import Path
import numpy as np
import pandas as pd


SESSION = Path("VibrationData/vib_bg")
MANIFEST_PATH = SESSION / "manifest.csv"

OUTPUT_FOLDER = Path("CREMA_D_Test/background")

WINDOW_SECONDS = 3.0
HOP_SECONDS = 3.0


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

    return np.concatenate(chunks), sample_rate


signal, sample_rate = load_continuous_signal()

window_samples = int(WINDOW_SECONDS * sample_rate)
hop_samples = int(HOP_SECONDS * sample_rate)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

saved = 0

for start in range(0, len(signal) - window_samples + 1, hop_samples):
    end = start + window_samples
    clip = signal[start:end]

    output_path = OUTPUT_FOLDER / f"background_{saved:03d}.npy"
    np.save(output_path, clip)

    saved += 1

print(f"Background signal duration: {len(signal) / sample_rate:.2f} seconds")
print(f"Saved background clips: {saved}")
print(f"Output folder: {OUTPUT_FOLDER.resolve()}")