#!/usr/bin/env python3
"""
Training script for VyvoOmni model.

Single GPU:
    python train.py

Multi-GPU with Accelerate:
    accelerate launch --config_file accelerate_config.yaml train.py

Multi-GPU with torchrun:
    torchrun --nproc_per_node=4 train.py
"""

import os
from accelerate.utils import set_seed

from vyvo_omni import (
    VyvoOmniConfig,
    VyvoOmniModel,
    SpeechTextDataset,
    SpeechTextCollator,
    VyvoOmniTrainer,
)
from vyvo_omni.config import VyvoOmniTrainingConfig

# ============================================================================
# CONFIGURATION - Edit these variables
# ============================================================================

# Model
LLM_MODEL = "Qwen/Qwen3-0.6B"
WHISPER_MODEL = "openai/whisper-large-v3"
USE_FLASH_ATTENTION = True
PROJECTION_LAYERS = 2
PROJECTION_DROPOUT = 0.1

# Data
DATA_PATH = "./data/train.json"
EVAL_DATA_PATH = "./data/train_eval.json"
MAX_AUDIO_LENGTH = 30.0
MAX_SEQ_LENGTH = 2048

# Training
OUTPUT_DIR = "./vyvo_omni_output"
NUM_EPOCHS = 3
BATCH_SIZE = 8  # Per GPU batch size
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
LR_SCHEDULER = "cosine"

# Precision
BF16 = True
FP16 = False

# Logging and Saving
LOGGING_STEPS = 10
SAVE_STEPS = 500
EVAL_STEPS = 500
SAVE_TOTAL_LIMIT = 3

# Logging
REPORT_TO = "wandb"  # "wandb", "tensorboard", or "none"
WANDB_PROJECT = "vyvo-omni"
WANDB_RUN_NAME = None  # Auto-generated if None
WANDB_ENTITY = "narkadir-ai"  # Your wandb username or team name

# Misc
SEED = 42
NUM_WORKERS = 4
GRADIENT_CHECKPOINTING = True
RESUME_FROM_CHECKPOINT = None  # Set to checkpoint path to resume

# ============================================================================
# TRAINING - Don't edit below unless you know what you're doing
# ============================================================================


def main():
    set_seed(SEED)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = local_rank == 0

    if is_main_process:
        print("=" * 60)
        print("VyvoOmni Training")
        print("=" * 60)

    # Model config
    model_config = VyvoOmniConfig(
        llm_model_name=LLM_MODEL,
        whisper_model_name=WHISPER_MODEL,
        llm_use_flash_attention=USE_FLASH_ATTENTION,
        projection_num_layers=PROJECTION_LAYERS,
        projection_dropout=PROJECTION_DROPOUT,
        max_audio_length=MAX_AUDIO_LENGTH,
        freeze_whisper=True,
        freeze_llm=True,
        train_projection_only=True,
    )

    if is_main_process:
        print(f"LLM: {LLM_MODEL}")
        print(f"Whisper: {WHISPER_MODEL}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Precision: {'bf16' if BF16 else 'fp16' if FP16 else 'fp32'}")
        print("\nLoading model...")

    # Create model
    model = VyvoOmniModel(model_config)

    # Training config
    training_config = VyvoOmniTrainingConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
        lr_scheduler_type=LR_SCHEDULER,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=BF16,
        fp16=FP16,
        max_seq_length=MAX_SEQ_LENGTH,
        dataloader_num_workers=NUM_WORKERS,
        seed=SEED,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        report_to=REPORT_TO,
        wandb_project=WANDB_PROJECT,
        wandb_run_name=WANDB_RUN_NAME,
        wandb_entity=WANDB_ENTITY,
    )

    # Datasets
    if is_main_process:
        print("\nPreparing datasets...")

    train_dataset = SpeechTextDataset(
        data_path=DATA_PATH,
        feature_extractor=model.feature_extractor,
        tokenizer=model.tokenizer,
        max_audio_length=MAX_AUDIO_LENGTH,
        audio_start_token=model_config.audio_start_token,
        audio_end_token=model_config.audio_end_token,
    )

    eval_dataset = None
    if EVAL_DATA_PATH and os.path.exists(EVAL_DATA_PATH):
        eval_dataset = SpeechTextDataset(
            data_path=EVAL_DATA_PATH,
            feature_extractor=model.feature_extractor,
            tokenizer=model.tokenizer,
            max_audio_length=MAX_AUDIO_LENGTH,
            audio_start_token=model_config.audio_start_token,
            audio_end_token=model_config.audio_end_token,
        )

    if is_main_process:
        print(f"Train samples: {len(train_dataset)}")
        if eval_dataset:
            print(f"Eval samples: {len(eval_dataset)}")

    # Collator
    collator = SpeechTextCollator(
        tokenizer=model.tokenizer,
        audio_start_token=model_config.audio_start_token,
        audio_end_token=model_config.audio_end_token,
        max_length=MAX_SEQ_LENGTH,
    )

    # Trainer
    if is_main_process:
        print("\nInitializing trainer...")

    trainer = VyvoOmniTrainer(
        model=model,
        training_config=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        collator=collator,
    )

    # Resume if specified
    if RESUME_FROM_CHECKPOINT:
        trainer.resume_from_checkpoint(RESUME_FROM_CHECKPOINT)

    # Train
    trainer.train()

    if is_main_process:
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Model saved to: {OUTPUT_DIR}")
        print("=" * 60)


if __name__ == "__main__":
    main()
