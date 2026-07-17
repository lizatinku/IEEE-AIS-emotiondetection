from __future__ import annotations

import csv
import wave
from pathlib import Path


# ----------------------------
# CONFIGURATION
# ----------------------------

AUDIO_DIR = Path("AudioWAV")
OUTPUT_DIR = Path("CombinedActors")

ACTOR_IDS = [
    "1016",
    "1020",
    "1034",
    "1050",
    "1067",
    "1087",
]

EMOTION_TO_LABEL = {
    "NEU": "calm",
    "ANG": "distress",
    "FEA": "distress",
    "SAD": "distress",
    "DIS": "exclude",
    "HAP": "exclude",
}


# ----------------------------
# HELPER FUNCTIONS
# ----------------------------

def parse_emotion(filename: str) -> str:
    """
    Example filename:
        1016_DFA_ANG_XX.wav

    Parts:
        1016 = actor
        DFA  = sentence
        ANG  = emotion
        XX   = emotion level
    """
    parts = Path(filename).stem.split("_")

    if len(parts) < 4:
        raise ValueError(f"Unexpected CREMA-D filename: {filename}")

    return parts[2]


def get_actor_files(actor_id: str) -> list[Path]:
    """
    Return all WAV files for one actor in alphabetical order.
    This order must be the same order used for concatenation
    and timeline generation.
    """
    files = sorted(AUDIO_DIR.glob(f"{actor_id}_*.wav"))

    if not files:
        raise FileNotFoundError(
            f"No WAV files found for actor {actor_id} in {AUDIO_DIR.resolve()}"
        )

    return files


def build_actor_audio_and_timeline(actor_id: str) -> None:
    actor_files = get_actor_files(actor_id)

    output_wav = OUTPUT_DIR / f"actor_{actor_id}.wav"
    output_csv = OUTPUT_DIR / f"actor_{actor_id}_timeline.csv"

    timeline_rows = []
    current_frame = 0
    expected_params = None

    with wave.open(str(output_wav), "wb") as combined_wav:
        for index, clip_path in enumerate(actor_files):
            with wave.open(str(clip_path), "rb") as clip:
                params = clip.getparams()

                # Audio settings needed for concatenation
                clip_format = (
                    params.nchannels,
                    params.sampwidth,
                    params.framerate,
                    params.comptype,
                )

                if index == 0:
                    expected_params = clip_format

                    combined_wav.setnchannels(params.nchannels)
                    combined_wav.setsampwidth(params.sampwidth)
                    combined_wav.setframerate(params.framerate)
                    combined_wav.setcomptype(params.comptype, params.compname)

                elif clip_format != expected_params:
                    raise ValueError(
                        f"Audio format mismatch in {clip_path.name}.\n"
                        f"Expected: {expected_params}\n"
                        f"Found:    {clip_format}"
                    )

                frame_rate = params.framerate
                number_of_frames = params.nframes

                start_s = current_frame / frame_rate
                end_frame = current_frame + number_of_frames
                end_s = end_frame / frame_rate

                emotion = parse_emotion(clip_path.name)

                if emotion not in EMOTION_TO_LABEL:
                    raise ValueError(
                        f"Unknown emotion code '{emotion}' in {clip_path.name}"
                    )

                label = EMOTION_TO_LABEL[emotion]

                timeline_rows.append(
                    {
                        "start_s": f"{start_s:.6f}",
                        "end_s": f"{end_s:.6f}",
                        "source_clip_id": clip_path.name,
                        "emotion": emotion,
                        "label": label,
                    }
                )

                # Append the original clip audio to the combined actor file
                combined_wav.writeframes(clip.readframes(number_of_frames))

                current_frame = end_frame

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "start_s",
            "end_s",
            "source_clip_id",
            "emotion",
            "label",
        ]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timeline_rows)

    total_duration_s = float(timeline_rows[-1]["end_s"])

    print(f"\nActor {actor_id}")
    print(f"  Clips:    {len(actor_files)}")
    print(f"  Duration: {total_duration_s:.2f} seconds")
    print(f"  WAV:      {output_wav}")
    print(f"  Timeline: {output_csv}")


def main() -> None:
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Could not find the AudioWAV folder at {AUDIO_DIR.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for actor_id in ACTOR_IDS:
        build_actor_audio_and_timeline(actor_id)

    print("\nDone! All combined WAV files and timeline CSV files were created.")


if __name__ == "__main__":
    main()