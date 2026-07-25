from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BACKGROUND_SESSION = Path("VibrationData/20260717_confroom_bg")
ACTOR_SESSION = Path("VibrationData/20260717_confroom_crema_1016")


def load_signal(session):
    manifest = pd.read_csv(session / "manifest.csv")
    manifest = manifest.sort_values("chunk_index")

    fs = int(manifest["sample_rate_hz"].iloc[0])

    chunks = []

    for rel in manifest["relative_path"]:
        x = np.load(session / rel).reshape(-1)
        chunks.append(x)

    signal = np.concatenate(chunks).astype(np.float64)
    signal -= np.median(signal)

    return signal, fs


background, fs = load_signal(BACKGROUND_SESSION)
actor, _ = load_signal(ACTOR_SESSION)

events = pd.read_csv(ACTOR_SESSION / "events.csv")

results = []

for _, row in events.iterrows():

    start = int(row["start_s"] * fs)
    end = int(row["end_s"] * fs)

    actor_seg = actor[start:end]

    if len(actor_seg) == 0:
        continue

    # Use the same time interval from the background recording
    bg_end = min(end, len(background))
    bg_seg = background[start:bg_end]

    # Make equal lengths
    n = min(len(actor_seg), len(bg_seg))

    if n < 100:
        continue

    actor_seg = actor_seg[:n]
    bg_seg = bg_seg[:n]

    actor_rms = np.sqrt(np.mean(actor_seg**2))
    bg_rms = np.sqrt(np.mean(bg_seg**2))

    results.append({
        "clip": row["source_clip_id"],
        "label": row["label"],
        "actor_rms": actor_rms,
        "background_rms": bg_rms,
        "difference": actor_rms - bg_rms
    })

results = pd.DataFrame(results)

print(results.head())

plt.figure(figsize=(14,5))

colors = {
    "calm":"green",
    "distress":"red",
    "exclude":"gray"
}

for label in ["calm","distress","exclude"]:

    subset = results[results.label==label]

    plt.scatter(
        subset.index,
        subset["difference"],
        color=colors[label],
        label=label,
        alpha=0.8
    )

plt.axhline(0,color="black",linestyle="--")

plt.ylabel("Actor RMS − Background RMS")
plt.xlabel("Playback event")
plt.title("Difference between actor and background for every event")

plt.legend()

plt.tight_layout()
plt.show()

print("\nAverage difference by label\n")
print(results.groupby("label")["difference"].describe())