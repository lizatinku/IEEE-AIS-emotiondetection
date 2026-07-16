from pathlib import Path

import numpy as np
from scipy.io import wavfile

wav_folder = Path("./AudioWAV")
output_folder = Path("./CombinedActors")

selected_actor_ids = [
    "1016",
    "1020",
    "1034",
    "1050",
    "1067",
    "1087"
]

output_folder.mkdir(exist_ok=True)

for actor_id in selected_actor_ids:

    actor_files = sorted(wav_folder.glob(f"{actor_id}_*.wav"))

    if len(actor_files) == 0:
        print(f"No recordings found for actor {actor_id}. Skipping.")
        continue

    print(f"\nProcessing Actor {actor_id} ({len(actor_files)} recordings)...")

    combined_audio = []
    sample_rate = None

    for file in actor_files:

        sr, audio = wavfile.read(file)

        # Convert stereo -> mono if needed
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        # Save sample rate from first clip
        if sample_rate is None:
            sample_rate = sr

        # Check sample rate consistency
        if sr != sample_rate:
            raise ValueError(
                f"{file.name} has sample rate {sr}, expected {sample_rate}"
            )

        combined_audio.append(audio)

    # Combine all recordings
    combined_audio = np.concatenate(combined_audio)

    output_path = output_folder / f"actor_{actor_id}.wav"

    wavfile.write(
        output_path,
        sample_rate,
        combined_audio
    )

    duration = len(combined_audio) / sample_rate

    print(f"Saved: {output_path}")
    print(f"Duration: {duration:.2f} s ({duration/60:.2f} min)")

print("\nAll actors processed successfully!")