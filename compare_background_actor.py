from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BACKGROUND_SESSION = Path("VibrationData/20260717_confroom_bg")
ACTOR_SESSION = Path("VibrationData/20260717_confroom_crema_1016")

WINDOW_SECONDS = 0.1
PLOT_SECONDS = 30


def load_signal(session_path: Path):
    manifest_path = session_path / "manifest.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    manifest = manifest.sort_values("chunk_index")

    sample_rates = manifest["sample_rate_hz"].dropna().unique()

    if len(sample_rates) != 1:
        raise ValueError(
            f"{session_path.name} has unexpected sample rates: {sample_rates}"
        )

    sample_rate = int(sample_rates[0])
    chunks = []

    for relative_path in manifest["relative_path"]:
        chunk_path = session_path / relative_path

        if not chunk_path.exists():
            raise FileNotFoundError(f"Missing vibration file: {chunk_path}")

        chunk = np.load(chunk_path).reshape(-1)
        chunks.append(chunk)

    signal = np.concatenate(chunks).astype(np.float64)

    # Remove DC offset independently for each session.
    signal = signal - np.median(signal)

    return signal, sample_rate


def calculate_rms(
    signal: np.ndarray,
    sample_rate: int,
    window_seconds: float
):
    window_samples = int(window_seconds * sample_rate)

    rms_values = []

    for start in range(0, len(signal), window_samples):
        segment = signal[start:start + window_samples]

        if len(segment) == 0:
            continue

        rms = np.sqrt(np.mean(segment ** 2))
        rms_values.append(rms)

    rms_values = np.asarray(rms_values)
    time_s = np.arange(len(rms_values)) * window_seconds

    return time_s, rms_values


# ---------------------------------------------------
# Load background
# ---------------------------------------------------

background_signal, background_rate = load_signal(BACKGROUND_SESSION)

# ---------------------------------------------------
# Load actor 1016
# ---------------------------------------------------

actor_signal, actor_rate = load_signal(ACTOR_SESSION)

if background_rate != actor_rate:
    raise ValueError(
        f"Sample-rate mismatch: background={background_rate}, "
        f"actor={actor_rate}"
    )

sample_rate = actor_rate

# ---------------------------------------------------
# Load actor events
# ---------------------------------------------------

events_path = ACTOR_SESSION / "events.csv"

if not events_path.exists():
    raise FileNotFoundError(f"Missing events.csv: {events_path}")

events = pd.read_csv(events_path)

required_columns = {"start_s", "end_s", "label", "source_clip_id"}

missing_columns = required_columns - set(events.columns)

if missing_columns:
    raise ValueError(
        f"events.csv is missing columns: {sorted(missing_columns)}"
    )

# Keep only events that overlap the displayed time range.
visible_events = events[
    (events["start_s"] < PLOT_SECONDS)
    & (events["end_s"] > 0)
].copy()

# ---------------------------------------------------
# Calculate RMS
# ---------------------------------------------------

background_time, background_rms = calculate_rms(
    background_signal,
    sample_rate,
    WINDOW_SECONDS
)

actor_time, actor_rms = calculate_rms(
    actor_signal,
    sample_rate,
    WINDOW_SECONDS
)

# Only plot first 30 seconds.
background_mask = background_time <= PLOT_SECONDS
actor_mask = actor_time <= PLOT_SECONDS

# ---------------------------------------------------
# Print integrity summary
# ---------------------------------------------------

print("Background")
print(f"  Duration:   {len(background_signal) / sample_rate:.2f} s")
print(f"  Mean RMS:   {np.mean(background_rms):.2f}")
print(f"  Median RMS: {np.median(background_rms):.2f}")

print("\nActor 1016")
print(f"  Duration:   {len(actor_signal) / sample_rate:.2f} s")
print(f"  Mean RMS:   {np.mean(actor_rms):.2f}")
print(f"  Median RMS: {np.median(actor_rms):.2f}")

print(f"\nEvents shown in first {PLOT_SECONDS} seconds:")
print(
    visible_events[
        ["start_s", "end_s", "label", "source_clip_id"]
    ].to_string(index=False)
)

# ---------------------------------------------------
# Plot background and actor RMS
# ---------------------------------------------------

plt.figure(figsize=(15, 6))

plt.plot(
    background_time[background_mask],
    background_rms[background_mask],
    linewidth=1,
    label="Background RMS"
)

plt.plot(
    actor_time[actor_mask],
    actor_rms[actor_mask],
    linewidth=1,
    label="Actor 1016 RMS"
)

# Event label styles.
label_styles = {
    "calm": {
        "alpha": 0.16,
        "text_y": 0.97
    },
    "distress": {
        "alpha": 0.12,
        "text_y": 0.93
    },
    "exclude": {
        "alpha": 0.07,
        "text_y": 0.89
    }
}

used_legend_labels = set()

for _, event in visible_events.iterrows():
    start_s = max(float(event["start_s"]), 0)
    end_s = min(float(event["end_s"]), PLOT_SECONDS)
    label = str(event["label"])

    style = label_styles.get(
        label,
        {"alpha": 0.08, "text_y": 0.85}
    )

    legend_label = f"Event: {label}"

    if legend_label in used_legend_labels:
        legend_label = None
    else:
        used_legend_labels.add(f"Event: {label}")

    plt.axvspan(
        start_s,
        end_s,
        alpha=style["alpha"],
        label=legend_label
    )

    plt.axvline(
        start_s,
        linewidth=0.7,
        alpha=0.45
    )

    plt.text(
        (start_s + end_s) / 2,
        style["text_y"],
        label,
        rotation=90,
        horizontalalignment="center",
        verticalalignment="top",
        transform=plt.gca().get_xaxis_transform(),
        fontsize=7
    )

plt.xlabel("Time (seconds)")
plt.ylabel("RMS vibration")
plt.title(
    "Background vs Actor 1016 RMS with events.csv timing"
)
plt.xlim(0, PLOT_SECONDS)
plt.grid(True)
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig(
    "background_vs_actor1016_events.png",
    dpi=180
)
plt.show()

print("\nSaved background_vs_actor1016_events.png")