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

# Training Stage
# Stage 1: Train projection layer only (freeze Whisper + LLM)
# Stage 2: Train LLM only (freeze Whisper + projection)
TRAINING_STAGE = 2  # Set to 1 or 2

# Model
LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
WHISPER_MODEL = "openai/whisper-large-v3"
USE_FLASH_ATTENTION = True
PROJECTION_LAYERS = 2
PROJECTION_DROPOUT = 0.1

# Stage 1 checkpoint to load for Stage 2 training
# Set this to the path of your Stage 1 trained projector when starting Stage 2
STAGE1_CHECKPOINT = None  # e.g., "./vyvo_omni_output/best_model/audio_projector.pt"

# Data
DATA_PATH = "./data/train_conversational.json"  # Updated to use conversational data
EVAL_DATA_PATH = "./data/train_eval.json"
MAX_AUDIO_LENGTH = 30.0
MAX_SEQ_LENGTH = 2048

# Training
# Output directory will be stage-specific by default
OUTPUT_DIR = f"./vyvo_omni_output_stage{TRAINING_STAGE}" if TRAINING_STAGE in [1, 2] else "./vyvo_omni_output"
NUM_EPOCHS = 3
BATCH_SIZE = 4  # Per GPU batch size
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2

# Learning rate recommendations:
# Stage 1 (projection layer): 1e-4 to 2e-4
# Stage 2 (LLM fine-tuning): 5e-6 to 2e-5 (lower than Stage 1)
LEARNING_RATE = 2e-5 if TRAINING_STAGE == 1 else 1e-5
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
WANDB_RUN_NAME = f"stage{TRAINING_STAGE}" if TRAINING_STAGE in [1, 2] else None  # Auto-generated if None
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
        training_stage=TRAINING_STAGE,
    )

    if is_main_process:
        print(f"Training Stage: {TRAINING_STAGE}")
        if TRAINING_STAGE == 1:
            print("Stage 1: Training projection layer only")
        elif TRAINING_STAGE == 2:
            print("Stage 2: Training LLM only")
        print(f"LLM: {LLM_MODEL}")
        print(f"Whisper: {WHISPER_MODEL}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Precision: {'bf16' if BF16 else 'fp16' if FP16 else 'fp32'}")
        print("\nLoading model...")

    # Create model
    model = VyvoOmniModel(model_config)

    # For Stage 2: Explicitly unfreeze LLM parameters
    if TRAINING_STAGE == 2:
        if is_main_process:
            print("\nUnfreezing LLM parameters for Stage 2 training...")

        # Ensure all LLM parameters are trainable
        for param in model.llm.parameters():
            param.requires_grad = True

        # Ensure projection layer is frozen
        for param in model.audio_projector.parameters():
            param.requires_grad = False

        if is_main_process:
            llm_trainable = sum(p.numel() for p in model.llm.parameters() if p.requires_grad)
            proj_trainable = sum(p.numel() for p in model.audio_projector.parameters() if p.requires_grad)
            print(f"✓ LLM trainable params: {llm_trainable:,}")
            print(f"✓ Projection trainable params: {proj_trainable:,} (should be 0)")

    # For Stage 2: Load the trained projection layer from Stage 1
    if TRAINING_STAGE == 2 and STAGE1_CHECKPOINT is not None:
        if is_main_process:
            print(f"\nLoading Stage 1 checkpoint from: {STAGE1_CHECKPOINT}")

        if os.path.isdir(STAGE1_CHECKPOINT):
            # Load from directory (checkpoint folder)
            projector_path = os.path.join(STAGE1_CHECKPOINT, "audio_projector.pt")
        else:
            # Direct path to projector file
            projector_path = STAGE1_CHECKPOINT

        if os.path.exists(projector_path):
            import torch
            model.audio_projector.load_state_dict(
                torch.load(projector_path, weights_only=True, map_location="cpu")
            )
            if is_main_process:
                print("✓ Stage 1 projection layer loaded successfully")
        else:
            if is_main_process:
                print(f"Warning: Stage 1 checkpoint not found at {projector_path}")
                print("Starting Stage 2 with random projection layer initialization")
    elif TRAINING_STAGE == 2 and STAGE1_CHECKPOINT is None:
        if is_main_process:
            print("\nWarning: STAGE1_CHECKPOINT is None")
            print("For Stage 2 training, you should load the trained projection layer from Stage 1")
            print("Set STAGE1_CHECKPOINT to the path of your Stage 1 checkpoint")

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
