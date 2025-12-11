# VyvoOmni Dataset Tools

Dataset preparation and download utilities for VyvoOmni training.

## Scripts

### 1. Download MLS Dataset

```bash
python download_mls_dataset.py
```
Downloads and processes the complete dataset.

**Downloads:**
- Audio: `parler-tts/mls_eng_10k` → `data/audio/`
- JSON: `Vyvo-Research/Omni-Mls-Json` → `data/`

**Output:**
```
data/
├── train.json
├── train_eval.json
└── audio/
    ├── audio_000000.wav
    ├── audio_000001.wav
    └── ...
```

### 2. Prepare Custom Dataset
Convert any HuggingFace dataset to VyvoOmni format:

```bash
python prepare_dataset.py
```

**Configuration:** Edit variables in `prepare_dataset.py`:
- `DATASET_NAME`: HuggingFace dataset ID
- `AUDIO_COLUMN`: Audio column name
- `TEXT_COLUMN`: Transcript column name
- `MAX_SAMPLES`: Limit samples (None = all)

**Features:**
- ✅ Relative paths (portable across servers)
- ✅ 16kHz resampling
- ✅ Train/eval split (90/10)
- ✅ OmniAudio format

### 3. Generate Synthetic Conversations
Create conversational Q&A from transcriptions:

```bash
python generate_synthetic_conversations.py
```

**Requires:** `.env` file with `DEEPSEEK_API_KEY`

**Configuration:**
- `MAX_SAMPLES`: Number of samples to process
- `MAX_WORKERS`: Parallel API workers
- `QA_PAIRS_PER_TRANSCRIPTION`: Q&A pairs per audio

## Quick Start

```bash
cd vyvo_omni/dataset

# Download dataset
python download_mls_dataset.py

# Generate conversational data (optional)
python generate_synthetic_conversations.py

# Train model
cd ../..
python train.py
```

## Notes

- All JSONs use relative paths (`audio/audio_*.wav`)
- Works across different servers/directories
- No path fixing needed
