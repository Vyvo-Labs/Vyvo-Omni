import os

# Enable fast downloads with hf_transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import logging
from huggingface_hub import snapshot_download

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset settings
AUDIO_DATASET = "parler-tts/mls_eng_10k"
JSON_DATASET = "Vyvo-Research/Omni-Mls-Json"

# Download settings
MAX_WORKERS = 16  # Parallel download threads

# ============================================================================
# MAIN SCRIPT
# ============================================================================


def download_snapshots():
    """Download dataset snapshots to HF cache."""

    # Verify hf_transfer
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") != "1":
        logger.error("HF_TRANSFER not enabled! Install: pip install hf-transfer")
        return

    logger.info(f"Workers: {MAX_WORKERS}")

    # Download JSON dataset
    logger.info(f"[1/2] {JSON_DATASET}")
    json_cache = snapshot_download(
        repo_id=JSON_DATASET,
        repo_type="dataset",
        max_workers=MAX_WORKERS,
    )
    logger.info(f"✓ Cached")

    # Download audio dataset
    logger.info(f"[2/2] {AUDIO_DATASET}")
    audio_cache = snapshot_download(
        repo_id=AUDIO_DATASET,
        repo_type="dataset",
        allow_patterns=["*.parquet", "*.json"],
        max_workers=MAX_WORKERS,
    )
    logger.info(f"✓ Cached")

    logger.info("Done! Next: python download_mls_dataset.py")


if __name__ == "__main__":
    download_snapshots()
