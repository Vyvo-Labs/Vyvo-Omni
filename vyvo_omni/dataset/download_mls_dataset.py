import os
import logging
import soundfile as sf
from tqdm import tqdm
from datasets import load_dataset, Audio
from huggingface_hub import hf_hub_download

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset settings
AUDIO_DATASET = "parler-tts/mls_eng_10k"
JSON_DATASET = "Vyvo-Research/Omni-Mls-Json"
DATASET_SPLIT = "train"

# Output settings
DATA_DIR = "../../data"
AUDIO_DIR = "../../data/audio"

# Audio settings
TARGET_SAMPLE_RATE = 16000
MAX_SAMPLES = None

# Download settings
MAX_WORKERS = 16  # Parallel download threads

# ============================================================================
# MAIN SCRIPT
# ============================================================================


def extract_json_files():
    """Extract JSON files from HF Hub cache to local directory."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir_abs = os.path.abspath(os.path.join(script_dir, DATA_DIR))
    os.makedirs(data_dir_abs, exist_ok=True)

    files = ["train.json", "train_eval.json", "train_conversational.json"]

    for filename in files:
        hf_hub_download(
            repo_id=JSON_DATASET,
            filename=filename,
            repo_type="dataset",
            local_dir=data_dir_abs,
        )

    logger.info("✓ JSON extracted")


def extract_audio_files():
    """Extract audio files from cached dataset."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir_abs = os.path.abspath(os.path.join(script_dir, AUDIO_DIR))
    os.makedirs(audio_dir_abs, exist_ok=True)

    # Load from cache
    dataset = load_dataset(AUDIO_DATASET, split=DATASET_SPLIT)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))

    if MAX_SAMPLES:
        dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))

    logger.info(f"Total: {len(dataset):,} samples")

    # Extract audio files
    saved = 0
    for idx, item in enumerate(tqdm(dataset, desc="Audio", ncols=80)):
        try:
            audio_data = item["audio"]
            audio_array = audio_data["array"]
            sample_rate = audio_data["sampling_rate"]

            audio_filename = f"audio_{idx:06d}.wav"
            audio_path = os.path.join(audio_dir_abs, audio_filename)

            sf.write(audio_path, audio_array, sample_rate)
            saved += 1

        except Exception as e:
            logger.error(f"Error {idx}: {e}")
            continue

    logger.info(f"✓ {saved:,} files saved")


def main():
    """Process and extract dataset files from HF cache."""

    extract_json_files()
    extract_audio_files()

    logger.info("Done! Next: cd ../.. && python train.py")


if __name__ == "__main__":
    main()
