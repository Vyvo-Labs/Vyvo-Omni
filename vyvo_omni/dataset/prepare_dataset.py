#!/usr/bin/env python3
"""
Dataset preparation script for VyvoOmni.

Converts HuggingFace datasets with audio-text columns to VyvoOmni JSON format.

Usage:
    1. Edit the configuration variables below
    2. Run: python prepare_dataset.py
"""

import os
import json
import soundfile as sf
from tqdm import tqdm
from datasets import load_dataset, Audio

# ============================================================================
# CONFIGURATION - Edit these variables
# ============================================================================

# HuggingFace dataset settings
DATASET_NAME = "parler-tts/mls_eng_10k"  # HF dataset name or path
DATASET_CONFIG = None  # Set to None if dataset has no config (only "default")
DATASET_SPLIT = "train"  # Split to use: "train", "test", "validation"

# Column names in the dataset
AUDIO_COLUMN = "audio"  # Column containing audio data
TEXT_COLUMN = "transcript"  # Column containing transcription text

# Output settings
OUTPUT_DIR = "./data"  # Directory to save audio files and JSON
OUTPUT_JSON = "train.json"  # Output JSON filename
SYSTEM_PROMPT = "Transcribe the following audio:"  # Global system prompt

# Audio settings
TARGET_SAMPLE_RATE = 16000  # Whisper expects 16kHz
MAX_SAMPLES = 100000  # Limit number of samples (None = use all)

# Train/eval split
EVAL_SPLIT_RATIO = 0.1  # Fraction for evaluation set (0.1 = 10%)
CREATE_EVAL_SPLIT = True  # Whether to create separate eval JSON

# ============================================================================
# PROCESSING - Don't edit below unless you know what you're doing
# ============================================================================


def process_dataset():
    """Process HuggingFace dataset and create VyvoOmni JSON format."""

    print("=" * 60)
    print("VyvoOmni Dataset Preparation")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Config: {DATASET_CONFIG}")
    print(f"Split: {DATASET_SPLIT}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Create output directory
    audio_dir = os.path.join(OUTPUT_DIR, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    # Load dataset
    print("\nLoading dataset...")
    if DATASET_CONFIG:
        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    else:
        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    # Cast audio column to correct format
    dataset = dataset.cast_column(AUDIO_COLUMN, Audio(sampling_rate=TARGET_SAMPLE_RATE))

    # Limit samples if specified
    if MAX_SAMPLES is not None:
        dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))

    print(f"Total samples: {len(dataset)}")

    # Process samples
    samples = []
    print("\nProcessing samples...")

    for idx, item in enumerate(tqdm(dataset, desc="Processing")):
        try:
            # Get audio data
            audio_data = item[AUDIO_COLUMN]
            audio_array = audio_data["array"]
            sample_rate = audio_data["sampling_rate"]

            # Get text
            text = item[TEXT_COLUMN]
            if not text or not text.strip():
                continue

            # Save audio file
            audio_filename = f"audio_{idx:06d}.wav"
            audio_path = os.path.join(audio_dir, audio_filename)

            sf.write(audio_path, audio_array, sample_rate)

            # Add to samples
            samples.append({
                "audio_path": os.path.abspath(audio_path),
                "text": text.strip(),
            })

        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            continue

    print(f"\nSuccessfully processed {len(samples)} samples")

    # Split into train/eval if requested
    if CREATE_EVAL_SPLIT and EVAL_SPLIT_RATIO > 0:
        split_idx = int(len(samples) * (1 - EVAL_SPLIT_RATIO))
        train_samples = samples[:split_idx]
        eval_samples = samples[split_idx:]

        # Save train JSON
        train_data = {
            "system_prompt": SYSTEM_PROMPT,
            "samples": train_samples,
        }
        train_json_path = os.path.join(OUTPUT_DIR, OUTPUT_JSON)
        with open(train_json_path, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        print(f"Train JSON saved: {train_json_path} ({len(train_samples)} samples)")

        # Save eval JSON
        eval_data = {
            "system_prompt": SYSTEM_PROMPT,
            "samples": eval_samples,
        }
        eval_json_path = os.path.join(OUTPUT_DIR, OUTPUT_JSON.replace(".json", "_eval.json"))
        with open(eval_json_path, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)
        print(f"Eval JSON saved: {eval_json_path} ({len(eval_samples)} samples)")

    else:
        # Save single JSON
        data = {
            "system_prompt": SYSTEM_PROMPT,
            "samples": samples,
        }
        json_path = os.path.join(OUTPUT_DIR, OUTPUT_JSON)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"JSON saved: {json_path} ({len(samples)} samples)")

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    process_dataset()
