#!/usr/bin/env python3
"""
Fast dataset preparation script using snapshot_download for VyvoOmni.

This script uses snapshot_download instead of load_dataset for faster downloads,
and creates relative paths in the JSON output.

Usage:
    python fast_prepare_dataset.py
"""

import os
import json
import soundfile as sf
import numpy as np
from tqdm import tqdm
from huggingface_hub import snapshot_download
from datasets import load_dataset, Audio

# ============================================================================
# CONFIGURATION
# ============================================================================

# HuggingFace dataset settings
DATASET_NAME = "parler-tts/mls_eng_10k"
DATASET_SPLIT = "train"

# Column names
AUDIO_COLUMN = "audio"
TEXT_COLUMN = "transcript"

# Output settings
OUTPUT_DIR = "../../data"  # Relative to script location
OUTPUT_JSON = "train.json"

# Task configuration (following OmniAudio format)
TASK_TOKEN = "<|transcribe|>"  # Task token for transcription
INSTRUCTION_TEXT = ""  # Empty instruction for transcription task

# Audio settings
TARGET_SAMPLE_RATE = 16000
MAX_SAMPLES = None  # None = use all

# Train/eval split
EVAL_SPLIT_RATIO = 0.1
CREATE_EVAL_SPLIT = True

# ============================================================================
# PROCESSING
# ============================================================================


def process_dataset():
    """Process dataset using fast snapshot download."""

    print("=" * 60)
    print("VyvoOmni Fast Dataset Preparation")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Split: {DATASET_SPLIT}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Get absolute path for output dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir_abs = os.path.abspath(os.path.join(script_dir, OUTPUT_DIR))
    audio_dir = os.path.join(output_dir_abs, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    print(f"\nAbsolute output path: {output_dir_abs}")

    # Download dataset snapshot (faster than load_dataset)
    print("\n[1/3] Downloading dataset snapshot...")
    cache_dir = snapshot_download(
        repo_id=DATASET_NAME,
        repo_type="dataset",
        cache_dir=os.path.join(output_dir_abs, ".cache")
    )
    print(f"Dataset cached at: {cache_dir}")

    # Now load just the metadata quickly
    print("\n[2/3] Loading dataset metadata...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    # Cast audio column
    dataset = dataset.cast_column(AUDIO_COLUMN, Audio(sampling_rate=TARGET_SAMPLE_RATE))

    # Limit samples if specified
    if MAX_SAMPLES is not None:
        dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))

    print(f"Total samples: {len(dataset)}")

    # Process samples
    print("\n[3/3] Processing audio files...")
    samples = []

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
            audio_path_abs = os.path.join(audio_dir, audio_filename)

            sf.write(audio_path_abs, audio_array, sample_rate)

            # Create RELATIVE path for JSON (relative to the output JSON file)
            # JSON will be at: data/train.json
            # Audio will be at: data/audio/audio_000000.wav
            # So relative path is: audio/audio_000000.wav
            audio_path_rel = os.path.join("audio", audio_filename)

            # OmniAudio format with task token
            samples.append({
                "audio_path": audio_path_rel,
                "task_token": TASK_TOKEN,
                "instruction": INSTRUCTION_TEXT,
                "response": text.strip(),
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

        # Save train JSON (OmniAudio format - no system_prompt field)
        train_data = {
            "samples": train_samples,
        }
        train_json_path = os.path.join(output_dir_abs, OUTPUT_JSON)
        with open(train_json_path, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        print(f"\nTrain JSON saved: {train_json_path}")
        print(f"  - {len(train_samples)} samples")
        print(f"  - Using OmniAudio format with task token: {TASK_TOKEN}")
        print(f"  - Using relative paths (e.g., 'audio/audio_000000.wav')")

        # Save eval JSON (OmniAudio format - no system_prompt field)
        eval_data = {
            "samples": eval_samples,
        }
        eval_json_path = os.path.join(output_dir_abs, OUTPUT_JSON.replace(".json", "_eval.json"))
        with open(eval_json_path, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)
        print(f"\nEval JSON saved: {eval_json_path}")
        print(f"  - {len(eval_samples)} samples")
        print(f"  - Using OmniAudio format with task token: {TASK_TOKEN}")
        print(f"  - Using relative paths (e.g., 'audio/audio_009000.wav')")

    else:
        # Save single JSON (OmniAudio format - no system_prompt field)
        data = {
            "samples": samples,
        }
        json_path = os.path.join(output_dir_abs, OUTPUT_JSON)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\nJSON saved: {json_path} ({len(samples)} samples)")
        print(f"Using OmniAudio format with task token: {TASK_TOKEN}")

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print(f"\nDirectory structure:")
    print(f"  {output_dir_abs}/")
    print(f"  ├── train.json (relative paths)")
    print(f"  ├── train_eval.json (relative paths)")
    print(f"  └── audio/")
    print(f"      ├── audio_000000.wav")
    print(f"      ├── audio_000001.wav")
    print(f"      └── ...")


if __name__ == "__main__":
    process_dataset()
