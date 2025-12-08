from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VyvoOmniConfig:
    """Configuration for VyvoOmni model."""

    # LLM configuration
    llm_model_name: str = "Qwen/Qwen3-0.6B"
    llm_use_flash_attention: bool = True

    # Whisper encoder configuration
    whisper_model_name: str = "openai/whisper-large-v3"

    # Projection layer configuration
    projection_hidden_size: Optional[int] = None  # If None, uses whisper hidden size
    projection_num_layers: int = 2
    projection_dropout: float = 0.1

    # Training configuration
    freeze_whisper: bool = True
    freeze_llm: bool = True
    train_projection_only: bool = True

    # Audio configuration
    audio_sample_rate: int = 16000
    max_audio_length: float = 30.0  # seconds

    # Special tokens
    audio_start_token: str = "<|audio_start|>"
    audio_end_token: str = "<|audio_end|>"

    def __post_init__(self):
        if self.train_projection_only:
            self.freeze_whisper = True
            self.freeze_llm = True


@dataclass
class VyvoOmniTrainingConfig:
    """Training configuration for VyvoOmni."""

    # Basic training args
    output_dir: str = "./vyvo_omni_output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Scheduler
    lr_scheduler_type: str = "cosine"

    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3

    # Mixed precision
    bf16: bool = True
    fp16: bool = False

    # Data
    max_seq_length: int = 2048
    dataloader_num_workers: int = 4

    # Misc
    seed: int = 42
    gradient_checkpointing: bool = True
    report_to: str = "wandb"  # "wandb", "tensorboard", or "none"

    # Wandb
    wandb_project: str = "vyvo-omni"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
