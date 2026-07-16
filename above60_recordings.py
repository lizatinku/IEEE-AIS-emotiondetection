import os
from collections import Counter, defaultdict

import pandas as pd

wav_folder = "./AudioWAV"
demographics_csv = "./VideoDemographics.csv"
output_file = "./above 60 recordings.txt"

df = pd.read_csv(demographics_csv)

# Make sure ActorID is treated as text
df["ActorID"] = df["ActorID"].astype(str)

# Keep actors aged 60+
older_actors = df[df["Age"] >= 60].copy()

# Actor ID -> age
actor_ages = dict(zip(older_actors["ActorID"], older_actors["Age"]))

selected_actor_ids = set(actor_ages.keys())

# Store filenames by actor
files_by_actor = defaultdict(list)

for filename in os.listdir(wav_folder):
    if not filename.lower().endswith(".wav"):
        continue

    parts = filename.split("_")

    # ActorID_Sentence_Emotion_Intensity.wav
    if len(parts) < 4:
        print(f"Skipping unexpected filename: {filename}")
        continue

    actor_id = parts[0]

    if actor_id in selected_actor_ids:
        files_by_actor[actor_id].append(filename)

# Sort filenames for each actor
for actor_id in files_by_actor:
    files_by_actor[actor_id].sort()

total_recordings = sum(len(files) for files in files_by_actor.values())

emotion_names = {
    "ANG": "Anger",
    "DIS": "Disgust",
    "FEA": "Fear",
    "HAP": "Happiness",
    "NEU": "Neutral",
    "SAD": "Sadness",
}

with open(output_file, "w", encoding="utf-8") as file:
    file.write("CREMA-D Recordings for Actors Aged 60 and Above\n")
    file.write("=" * 55 + "\n\n")

    file.write(f"Total actors: {len(files_by_actor)}\n")
    file.write(f"Total recordings: {total_recordings}\n\n")

    for actor_id in sorted(files_by_actor, key=int):
        actor_files = files_by_actor[actor_id]
        age = actor_ages[actor_id]

        emotion_counts = Counter()

        for filename in actor_files:
            parts = filename.split("_")
            emotion_code = parts[2]
            emotion_counts[emotion_code] += 1

        file.write(f"Actor {actor_id}\n")
        file.write(f"Age: {age}\n")
        file.write(f"Total recordings: {len(actor_files)}\n")
        file.write("Recordings by emotion:\n")

        for emotion_code in ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]:
            count = emotion_counts.get(emotion_code, 0)
            emotion_name = emotion_names[emotion_code]

            file.write(
                f"  {emotion_code} ({emotion_name}): {count} recordings\n"
            )

        file.write("\nFilenames:\n")

        for filename in actor_files:
            file.write(f"  {filename}\n")

        file.write("\n" + "-" * 55 + "\n\n")

print(f"Found {total_recordings} recordings from {len(files_by_actor)} actors.")
print(f"Results saved to: {output_file}")