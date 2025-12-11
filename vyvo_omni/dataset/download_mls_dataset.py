import os
import logging
import soundfile as sf
from tqdm import tqdm
from datasets import load_dataset, Audio
from huggingface_hub import hf_hub_download, snapshot_download

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

AUDIO_DATASET = "parler-tts/mls_eng_10k"
JSON_DATASET = "Vyvo-Research/Omni-Mls-Json"
DATASET_SPLIT = "train"

DATA_DIR = "../../data"
AUDIO_DIR = "../../data/audio"

TARGET_SAMPLE_RATE = 16000
MAX_SAMPLES = None

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir_abs = os.path.abspath(os.path.join(script_dir, DATA_DIR))
    audio_dir_abs = os.path.abspath(os.path.join(script_dir, AUDIO_DIR))

    os.makedirs(data_dir_abs, exist_ok=True)
    os.makedirs(audio_dir_abs, exist_ok=True)

    logger.info("Downloading audio dataset...")
    snapshot_download(
        repo_id=AUDIO_DATASET,
        repo_type="dataset",
        allow_patterns=["*.parquet", "*.json"],
    )

    logger.info("Extracting JSON files...")
    files = ["train.json", "train_eval.json", "train_conversational.json"]
    for filename in files:
        hf_hub_download(
            repo_id=JSON_DATASET,
            filename=filename,
            repo_type="dataset",
            local_dir=data_dir_abs,
        )

    logger.info("Loading dataset...")
    dataset = load_dataset(AUDIO_DATASET, split=DATASET_SPLIT)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))

    if MAX_SAMPLES:
        dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))

    logger.info(f"Total: {len(dataset):,} samples")

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
    logger.info("Done! Next: cd ../.. && python train.py")
