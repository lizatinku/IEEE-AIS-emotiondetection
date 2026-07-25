from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch


BACKGROUND_SESSION = Path("VibrationData/20260717_confroom_bg")
ACTOR_SESSION = Path("VibrationData/20260717_confroom_crema_1016")

def load_signal(session: Path):
    manifest = pd.read_csv(session / "manifest.csv")
    manifest = manifest.sort_values("chunk_index")

    sample_rate = int(manifest["sample_rate_hz"].iloc[0])

    chunks = []

    for relative_path in manifest["relative_path"]:
        chunk = np.load(session / relative_path).reshape(-1)
        chunks.append(chunk)

    signal = np.concatenate(chunks).astype(np.float64)

    # Remove DC offset.
    signal -= np.median(signal)

    return signal, sample_rate


background, background_rate = load_signal(BACKGROUND_SESSION)
actor, actor_rate = load_signal(ACTOR_SESSION)

if background_rate != actor_rate:
    raise ValueError("Background and actor sample rates do not match.")

sample_rate = actor_rate

events = pd.read_csv(ACTOR_SESSION / "events.csv")

# Use only the portion containing labeled playback events.
playback_start_s = float(events["start_s"].min())
playback_end_s = float(events["end_s"].max())

start_sample = int(playback_start_s * sample_rate)
end_sample = int(playback_end_s * sample_rate)

actor_playback = actor[start_sample:end_sample]

# Use the same amount of background data when possible.
comparison_samples = min(len(background), len(actor_playback))

background_comparison = background[:comparison_samples]
actor_comparison = actor_playback[:comparison_samples]

# Calculate power spectral density.
background_frequency, background_psd = welch(
    background_comparison,
    fs=sample_rate,
    nperseg=8192
)

actor_frequency, actor_psd = welch(
    actor_comparison,
    fs=sample_rate,
    nperseg=8192
)

# Convert power to decibels.
background_db = 10 * np.log10(background_psd + 1e-12)
actor_db = 10 * np.log10(actor_psd + 1e-12)

plt.figure(figsize=(14, 5))

plt.plot(
    background_frequency,
    background_db,
    label="Background"
)

plt.plot(
    actor_frequency,
    actor_db,
    label="Actor 1016 playback"
)

plt.xlim(0, 2000)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power spectral density (dB)")
plt.title("Background vs Actor 1016 frequency content")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("background_vs_actor1016_frequency.png", dpi=180)
plt.show()

print(f"Sample rate: {sample_rate} Hz")
print(
    f"Actor playback interval: "
    f"{playback_start_s:.2f}–{playback_end_s:.2f} seconds"
)
print("Saved background_vs_actor1016_frequency.png")