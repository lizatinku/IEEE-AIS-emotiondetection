from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAMPLE_RATE = 6000

SESSION_DIR = Path("VibrationData/actor-1016")
EVENTS_PATH = SESSION_DIR / "events.csv"

# Change this only if your continuous file has a different name.
CONTINUOUS_PATH = Path(
    "prepared_recordings/"
    "actor-1016__continuous.npy"
)

OUTPUT_PLOT = Path("actor_1016_alignment.png")


def rolling_rms(signal: np.ndarray, window_samples: int) -> np.ndarray:
    """Calculate rolling RMS without requiring SciPy."""
    squared = signal.astype(np.float64) ** 2
    kernel = np.ones(window_samples, dtype=np.float64) / window_samples
    mean_squared = np.convolve(squared, kernel, mode="same")
    return np.sqrt(mean_squared)


if not CONTINUOUS_PATH.exists():
    raise FileNotFoundError(
        f"Continuous recording not found:\n{CONTINUOUS_PATH.resolve()}\n"
        "Update CONTINUOUS_PATH to match the actual filename."
    )

if not EVENTS_PATH.exists():
    raise FileNotFoundError(
        f"events.csv not found:\n{EVENTS_PATH.resolve()}"
    )

signal = np.load(CONTINUOUS_PATH).squeeze()
events = pd.read_csv(EVENTS_PATH)

if signal.ndim != 1:
    raise ValueError(f"Expected a 1-D signal, but got shape {signal.shape}")

duration_s = len(signal) / SAMPLE_RATE
expected_end_s = float(events["end_s"].max()) + 5.0
extra_s = duration_s - expected_end_s

print(f"Samples:              {len(signal):,}")
print(f"Sample rate:          {SAMPLE_RATE} Hz")
print(f"Recording duration:   {duration_s:.3f} s")
print(f"Final event ends:     {events['end_s'].max():.3f} s")
print(f"Expected end + 5 s:   {expected_end_s:.3f} s")
print(f"Extra recording time: {extra_s:.3f} s")

# Remove the overall DC offset before calculating energy.
centered = signal.astype(np.float64) - np.median(signal)

# Calculate RMS with a 100 ms window.
rms_window_s = 0.100
rms_window_samples = int(rms_window_s * SAMPLE_RATE)
rms = rolling_rms(centered, rms_window_samples)

# Smooth RMS over 500 ms.
smooth_window_s = 0.500
smooth_window_samples = int(smooth_window_s * SAMPLE_RATE)

kernel = np.ones(smooth_window_samples) / smooth_window_samples
smoothed_rms = np.convolve(rms, kernel, mode="same")

time_s = np.arange(len(signal)) / SAMPLE_RATE

# Estimate background energy using the first 3 seconds.
baseline_samples = int(3 * SAMPLE_RATE)
baseline = smoothed_rms[:baseline_samples]

baseline_median = np.median(baseline)
baseline_mad = np.median(np.abs(baseline - baseline_median))

# Initial automatic threshold.
threshold = baseline_median + 8 * baseline_mad

# Require energy to remain above threshold for at least 300 ms.
minimum_active_s = 0.300
minimum_active_samples = int(minimum_active_s * SAMPLE_RATE)

above = smoothed_rms > threshold
active_run = np.convolve(
    above.astype(np.int32),
    np.ones(minimum_active_samples, dtype=np.int32),
    mode="same",
)

candidates = np.where(active_run >= minimum_active_samples)[0]

if len(candidates) > 0:
    detected_activity_s = candidates[0] / SAMPLE_RATE

    # The first speech event begins 5 seconds after playback timeline time 0.
    # Therefore, estimated recording timeline start is:
    estimated_timeline_zero_s = detected_activity_s - 5.0

    print(f"\nBaseline median RMS:   {baseline_median:.3f}")
    print(f"Baseline MAD:          {baseline_mad:.3f}")
    print(f"Detection threshold:   {threshold:.3f}")
    print(f"First activity found:  {detected_activity_s:.3f} s")
    print(
        "Estimated events.csv time 0 in recording: "
        f"{estimated_timeline_zero_s:.3f} s"
    )
else:
    detected_activity_s = None
    estimated_timeline_zero_s = None

    print("\nNo clear activity onset was automatically detected.")
    print("Use the saved plot to inspect the beginning manually.")

# Plot the complete energy envelope.
plt.figure(figsize=(15, 6))
plt.plot(time_s, smoothed_rms, linewidth=0.8, label="Smoothed vibration RMS")
plt.axhline(
    threshold,
    linestyle="--",
    linewidth=1,
    label="Automatic threshold",
)

if detected_activity_s is not None:
    plt.axvline(
        detected_activity_s,
        linestyle="--",
        linewidth=1.5,
        label=f"First detected activity: {detected_activity_s:.2f} s",
    )

    if estimated_timeline_zero_s >= 0:
        plt.axvline(
            estimated_timeline_zero_s,
            linestyle=":",
            linewidth=1.5,
            label=f"Estimated timeline zero: {estimated_timeline_zero_s:.2f} s",
        )

plt.xlabel("Recording time (seconds)")
plt.ylabel("Smoothed RMS")
plt.title("Actor 1016 vibration recording — playback alignment")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=180)
plt.show()

print(f"\nSaved alignment plot to:\n{OUTPUT_PLOT.resolve()}")