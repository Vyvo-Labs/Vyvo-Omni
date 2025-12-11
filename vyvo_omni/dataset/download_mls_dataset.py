#!/usr/bin/env python3
"""
Download MLS-eng dataset (audio + JSON).

Audio: parler-tts/mls_eng_10k
JSON: Vyvo-Research/Omni-Mls-Json

Usage:
    python download_mls_dataset.py
"""

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
DATA_DIR = "./data"
AUDIO_DIR = "./data/audio"

# Audio settings
TARGET_SAMPLE_RATE = 16000
MAX_SAMPLES = None

# ============================================================================
# MAIN SCRIPT
# ============================================================================


def download_json_files():
    """Download JSON files from HF Hub."""

    logger.info("Downloading JSON files...")
    os.makedirs(DATA_DIR, exist_ok=True)

    files = ["train.json", "train_eval.json"]

    for filename in files:
        logger.info(f"  {filename}")
        hf_hub_download(
            repo_id=JSON_DATASET,
            filename=filename,
            repo_type="dataset",
            local_dir=DATA_DIR,
        )

    logger.info("✓ JSON files downloaded")


def download_audio_files():
    """Download audio files from HF dataset."""

    logger.info("Downloading audio files...")
    os.makedirs(AUDIO_DIR, exist_ok=True)

    # Load dataset
    dataset = load_dataset(AUDIO_DATASET, split=DATASET_SPLIT)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))

    if MAX_SAMPLES:
        dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))

    logger.info(f"Total: {len(dataset):,} samples")

    # Save audio files
    saved = 0
    for idx, item in enumerate(tqdm(dataset, desc="Audio", ncols=80)):
        try:
            audio_data = item["audio"]
            audio_array = audio_data["array"]
            sample_rate = audio_data["sampling_rate"]

            audio_filename = f"audio_{idx:06d}.wav"
            audio_path = os.path.join(AUDIO_DIR, audio_filename)

            sf.write(audio_path, audio_array, sample_rate)
            saved += 1

        except Exception as e:
            logger.error(f"Error {idx}: {e}")
            continue

    logger.info(f"✓ Saved {saved:,} audio files")


def main():
    """Download dataset (JSON + audio)."""

    logger.info(f"Audio: {AUDIO_DATASET}")
    logger.info(f"JSON: {JSON_DATASET}")
    logger.info(f"Output: {DATA_DIR}")

    download_json_files()
    download_audio_files()

    logger.info("Done! Start training: python train.py")


if __name__ == "__main__":
    main()
