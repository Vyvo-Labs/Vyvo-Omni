#!/usr/bin/env python3
"""
Dataset preparation script for VyvoOmni.

Converts HuggingFace datasets to VyvoOmni JSON format with relative paths.

Usage:
    python prepare_dataset.py
"""

import os
import json
import logging
import soundfile as sf
from tqdm import tqdm
from datasets import load_dataset, Audio

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset settings
DATASET_NAME = "parler-tts/mls_eng_10k"
DATASET_CONFIG = None
DATASET_SPLIT = "train"

# Column names
AUDIO_COLUMN = "audio"
TEXT_COLUMN = "transcript"

# Output settings
OUTPUT_DIR = "../../data"
OUTPUT_JSON = "train.json"

# Task configuration
TASK_TOKEN = "<|transcribe|>"
INSTRUCTION_TEXT = ""

# Audio settings
TARGET_SAMPLE_RATE = 16000
MAX_SAMPLES = None

# Train/eval split
EVAL_SPLIT_RATIO = 0.1
CREATE_EVAL_SPLIT = True

# ============================================================================
# PROCESSING - Don't edit below unless you know what you're doing
# ============================================================================


def process_dataset():
    """Process HuggingFace dataset and create VyvoOmni JSON format."""

    logger.info(f"Dataset: {DATASET_NAME}")
    logger.info(f"Output: {OUTPUT_DIR}")

    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir_abs = os.path.abspath(os.path.join(script_dir, OUTPUT_DIR))
    audio_dir = os.path.join(output_dir_abs, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    # Load dataset
    logger.info("Loading dataset...")
    if DATASET_CONFIG:
        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    else:
        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    dataset = dataset.cast_column(AUDIO_COLUMN, Audio(sampling_rate=TARGET_SAMPLE_RATE))

    if MAX_SAMPLES:
        dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))

    logger.info(f"Total: {len(dataset):,} samples")

    # Process samples
    samples = []
    for idx, item in enumerate(tqdm(dataset, desc="Processing", ncols=80)):
        try:
            audio_data = item[AUDIO_COLUMN]
            audio_array = audio_data["array"]
            sample_rate = audio_data["sampling_rate"]

            text = item[TEXT_COLUMN]
            if not text or not text.strip():
                continue

            audio_filename = f"audio_{idx:06d}.wav"
            audio_path = os.path.join(audio_dir, audio_filename)
            sf.write(audio_path, audio_array, sample_rate)

            # Use relative path (relative to JSON location)
            rel_audio_path = os.path.join("audio", audio_filename)
            samples.append({
                "audio_path": rel_audio_path,
                "task_token": TASK_TOKEN,
                "instruction": INSTRUCTION_TEXT,
                "response": text.strip(),
            })

        except Exception as e:
            logger.error(f"Error {idx}: {e}")
            continue

    logger.info(f"Processed: {len(samples):,} samples")

    # Save JSONs
    if CREATE_EVAL_SPLIT and EVAL_SPLIT_RATIO > 0:
        split_idx = int(len(samples) * (1 - EVAL_SPLIT_RATIO))
        train_samples = samples[:split_idx]
        eval_samples = samples[split_idx:]

        train_json_path = os.path.join(output_dir_abs, OUTPUT_JSON)
        with open(train_json_path, "w", encoding="utf-8") as f:
            json.dump({"samples": train_samples}, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Train: {len(train_samples):,} samples")

        eval_json_path = os.path.join(output_dir_abs, OUTPUT_JSON.replace(".json", "_eval.json"))
        with open(eval_json_path, "w", encoding="utf-8") as f:
            json.dump({"samples": eval_samples}, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Eval: {len(eval_samples):,} samples")

    else:
        json_path = os.path.join(output_dir_abs, OUTPUT_JSON)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"samples": samples}, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Saved: {len(samples):,} samples")

    logger.info("Done!")


if __name__ == "__main__":
    process_dataset()
