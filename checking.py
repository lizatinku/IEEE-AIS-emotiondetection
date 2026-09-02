from pathlib import Path
import numpy as np
from scipy.io import wavfile

VIB_DIR = Path("VibrationData/vib_1087/node-0001/boot-d757b473")
OUT_DIR = Path("TA_Samples")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 8000

vibration_files = sorted(VIB_DIR.glob("vibration_*.npy"))
indices = np.linspace(0, len(vibration_files) - 1, 5, dtype=int)

for sample_num, idx in enumerate(indices, start=1):

    npy_path = vibration_files[idx]

    signal = np.load(npy_path).astype(np.float32)

    # Remove DC offset
    signal = signal - np.mean(signal)

    # Normalize
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val

    signal_int16 = (signal * 32767).astype(np.int16)
    wav_path = OUT_DIR / f"1087_sample_{sample_num}.wav"
    wavfile.write(wav_path, SAMPLE_RATE, signal_int16)

    print(
        f"Sample {sample_num}: "
        f"{npy_path.name} → {wav_path}"
    )