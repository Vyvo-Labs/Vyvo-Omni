import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from accelerate import Accelerator
from accelerate.utils import set_seed
from typing import Optional, Dict, Any, Callable
from tqdm import tqdm
import json
from datetime import datetime

from .model import VyvoOmniModel
from .config import VyvoOmniConfig, VyvoOmniTrainingConfig
from .data import SpeechTextCollator


class VyvoOmniTrainer:
    """
    Trainer for VyvoOmni model with multi-GPU support.

    Uses HuggingFace Accelerate for distributed training.

    Stage 1: Only the audio projector is wrapped in DDP (Whisper and LLM are frozen).
    Stage 2: Only the LLM is wrapped in DDP (Whisper and projector are frozen).
    """

    def __init__(
        self,
        model: VyvoOmniModel,
        training_config: VyvoOmniTrainingConfig,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        collator: Optional[SpeechTextCollator] = None,
        compute_metrics: Optional[Callable] = None,
    ):
        self.config = training_config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics

        # Set seed for reproducibility
        set_seed(training_config.seed)

        # Initialize Accelerator
        mixed_precision = None
        if training_config.bf16:
            mixed_precision = "bf16"
        elif training_config.fp16:
            mixed_precision = "fp16"

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            log_with=training_config.report_to if training_config.report_to != "none" else None,
            project_dir=training_config.output_dir,
        )

        # Store model reference
        self.model = model

        # Set up collator
        self.collator = collator or SpeechTextCollator(
            tokenizer=model.tokenizer,
            audio_start_token=model.config.audio_start_token,
            audio_end_token=model.config.audio_end_token,
            max_length=training_config.max_seq_length,
        )

        # Set up data loaders
        self.train_dataloader = self._create_dataloader(train_dataset, is_train=True)
        self.eval_dataloader = self._create_dataloader(eval_dataset, is_train=False) if eval_dataset else None

        # Store training stage
        self.training_stage = model.config.training_stage

        # Move frozen components to device (NOT wrapped in DDP)
        self.device = self.accelerator.device
        self.model.whisper_encoder.to(self.device)

        # Move appropriate components based on training stage
        if self.training_stage == 1:
            # Stage 1: LLM is frozen, move to device without DDP
            self.model.llm.to(self.device)
            # Projector will be wrapped in DDP later
        elif self.training_stage == 2:
            # Stage 2: Projector is frozen, move to device without DDP
            # Cast projector to same dtype as LLM to avoid dtype mismatches
            llm_dtype = self.model.llm.dtype if hasattr(self.model.llm, 'dtype') else torch.bfloat16
            self.model.audio_projector.to(device=self.device, dtype=llm_dtype)
            # LLM will be wrapped in DDP later

        # Set up optimizer based on training stage
        if self.training_stage == 1:
            # Stage 1: Train projector only
            trainable_params = self.model.audio_projector.parameters()
        elif self.training_stage == 2:
            # Stage 2: Train LLM only
            trainable_params = self.model.llm.parameters()
        else:
            raise ValueError(f"Invalid training stage: {self.training_stage}. Must be 1 or 2.")

        self.optimizer = AdamW(
            trainable_params,
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )

        # Calculate total training steps
        num_update_steps_per_epoch = len(self.train_dataloader) // training_config.gradient_accumulation_steps
        self.total_steps = num_update_steps_per_epoch * training_config.num_train_epochs

        # Set up scheduler
        warmup_steps = int(self.total_steps * training_config.warmup_ratio)
        self.scheduler = get_scheduler(
            training_config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
        )

        # Set up gradient checkpointing if enabled
        if training_config.gradient_checkpointing:
            if self.training_stage == 2 and hasattr(self.model.llm, "gradient_checkpointing_enable"):
                # Only enable gradient checkpointing for LLM in stage 2
                self.model.llm.gradient_checkpointing_enable()

        # Prepare trainable components with Accelerate (wraps in DDP)
        if self.training_stage == 1:
            # Stage 1: Wrap projector in DDP
            self.model.audio_projector, self.optimizer, self.train_dataloader, self.scheduler = self.accelerator.prepare(
                self.model.audio_projector, self.optimizer, self.train_dataloader, self.scheduler
            )
        elif self.training_stage == 2:
            # Stage 2: Wrap LLM in DDP
            # IMPORTANT: Ensure LLM parameters require grad BEFORE wrapping
            for param in self.model.llm.parameters():
                param.requires_grad = True

            self.model.llm, self.optimizer, self.train_dataloader, self.scheduler = self.accelerator.prepare(
                self.model.llm, self.optimizer, self.train_dataloader, self.scheduler
            )

            # Verify after DDP wrapping
            unwrapped_llm = self.accelerator.unwrap_model(self.model.llm)
            for param in unwrapped_llm.parameters():
                if not param.requires_grad:
                    param.requires_grad = True

        if self.eval_dataloader is not None:
            self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float("inf")

        # Logging
        self.log_history = []

        # Create output directory (only on main process)
        if self.accelerator.is_main_process:
            os.makedirs(training_config.output_dir, exist_ok=True)

        # Initialize trackers (wandb/tensorboard)
        if training_config.report_to != "none":
            tracker_config = {
                "llm_model": model.config.llm_model_name,
                "whisper_model": model.config.whisper_model_name,
                "learning_rate": training_config.learning_rate,
                "batch_size": training_config.per_device_train_batch_size,
                "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
                "num_epochs": training_config.num_train_epochs,
                "warmup_ratio": training_config.warmup_ratio,
                "max_seq_length": training_config.max_seq_length,
                "projection_layers": model.config.projection_num_layers,
            }

            init_kwargs = {}
            if training_config.report_to == "wandb":
                init_kwargs["wandb"] = {
                    "name": training_config.wandb_run_name,
                    "entity": training_config.wandb_entity,
                }

            self.accelerator.init_trackers(
                project_name=training_config.wandb_project,
                config=tracker_config,
                init_kwargs=init_kwargs,
            )

        self.accelerator.wait_for_everyone()

    def _create_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        is_train: bool = True,
    ) -> DataLoader:
        """Create data loader for training or evaluation."""
        if dataset is None:
            return None

        batch_size = (
            self.config.per_device_train_batch_size
            if is_train
            else self.config.per_device_eval_batch_size
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            collate_fn=self.collator,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=True,
            drop_last=is_train,
        )

    def _forward_pass(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Custom forward pass that handles frozen and trainable components separately.

        Stage 1: Gradients flow through projector only
        Stage 2: Gradients flow through LLM only

        Returns:
            loss: Scalar loss tensor with gradients through trainable component
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        audio_features = batch["audio_features"]
        audio_positions = batch["audio_positions"]
        labels = batch["labels"]

        batch_size = input_ids.shape[0]

        # Get LLM dtype - handle both wrapped and unwrapped models
        if self.training_stage == 2:
            # LLM is wrapped in DDP
            unwrapped_llm = self.accelerator.unwrap_model(self.model.llm)
            dtype = unwrapped_llm.dtype if hasattr(unwrapped_llm, 'dtype') else torch.bfloat16
        else:
            dtype = self.model.llm.dtype if hasattr(self.model.llm, 'dtype') else torch.bfloat16

        # Step 1: Encode audio with frozen Whisper (always no gradients)
        audio_features = audio_features.to(dtype=dtype)
        with torch.no_grad():
            audio_hidden = self.model.whisper_encoder(audio_features).last_hidden_state

        # Step 2: Project audio to LLM space
        if self.training_stage == 1:
            # Stage 1: Projector is trainable - allow gradients
            audio_embeds = self.model.audio_projector(audio_hidden)
        elif self.training_stage == 2:
            # Stage 2: Projector is frozen - no gradients
            # Projector is already cast to LLM dtype when moved to device
            with torch.no_grad():
                audio_embeds = self.model.audio_projector(audio_hidden)
                # Ensure output is in correct dtype
                audio_embeds = audio_embeds.to(dtype=dtype)

        audio_seq_len = audio_embeds.shape[1]

        # Step 3: Get text embeddings
        if self.training_stage == 1:
            # Stage 1: LLM embeddings are frozen - no gradients
            with torch.no_grad():
                text_embeds = self.model.llm.get_input_embeddings()(input_ids)
        elif self.training_stage == 2:
            # Stage 2: LLM is trainable - allow gradients
            # Access embedding layer through unwrapped model
            unwrapped_llm = self.accelerator.unwrap_model(self.model.llm)
            text_embeds = unwrapped_llm.get_input_embeddings()(input_ids)
            # Explicitly detach audio embeddings to avoid gradient issues
            audio_embeds = audio_embeds.detach()

        # Step 4: Merge audio and text embeddings
        merged_embeds_list = []
        merged_labels_list = []
        merged_attention_list = []

        for i in range(batch_size):
            start_pos, end_pos = audio_positions[i]

            pre_audio = text_embeds[i, :start_pos + 1]
            post_audio = text_embeds[i, end_pos:]

            # Merge embeddings
            # In Stage 2: text parts have gradients, audio part is detached
            merged = torch.cat([pre_audio, audio_embeds[i], post_audio], dim=0)
            merged_embeds_list.append(merged)

            # Handle labels
            if labels is not None:
                pre_labels = labels[i, :start_pos + 1]
                post_labels = labels[i, end_pos:]
                audio_labels = torch.full(
                    (audio_seq_len,), -100, dtype=labels.dtype, device=self.device
                )
                merged_label = torch.cat([pre_labels, audio_labels, post_labels], dim=0)
                merged_labels_list.append(merged_label)

            # Handle attention mask
            pre_attn = attention_mask[i, :start_pos + 1]
            post_attn = attention_mask[i, end_pos:]
            audio_attn = torch.ones(audio_seq_len, dtype=attention_mask.dtype, device=self.device)
            merged_attn = torch.cat([pre_attn, audio_attn, post_attn], dim=0)
            merged_attention_list.append(merged_attn)

        # Pad to same length
        max_len = max(e.shape[0] for e in merged_embeds_list)

        # Create padded tensors - use list comprehension to preserve gradients
        padded_embeds_list = []
        padded_attention_list = []
        padded_labels_list = [] if labels is not None else None

        for i in range(batch_size):
            seq_len = merged_embeds_list[i].shape[0]
            pad_len = max_len - seq_len

            # Pad embeddings (preserve gradients)
            if pad_len > 0:
                pad_tensor = torch.zeros(
                    pad_len, self.model.llm_hidden_size,
                    device=self.device, dtype=dtype
                )
                padded_embed = torch.cat([merged_embeds_list[i], pad_tensor], dim=0)
            else:
                padded_embed = merged_embeds_list[i]
            padded_embeds_list.append(padded_embed)

            # Pad attention mask
            if pad_len > 0:
                pad_attn = torch.zeros(pad_len, device=self.device, dtype=attention_mask.dtype)
                padded_attn = torch.cat([merged_attention_list[i], pad_attn], dim=0)
            else:
                padded_attn = merged_attention_list[i]
            padded_attention_list.append(padded_attn)

            # Pad labels
            if padded_labels_list is not None:
                if pad_len > 0:
                    pad_labels = torch.full((pad_len,), -100, device=self.device, dtype=torch.long)
                    padded_label = torch.cat([merged_labels_list[i], pad_labels], dim=0)
                else:
                    padded_label = merged_labels_list[i]
                padded_labels_list.append(padded_label)

        # Stack into batch tensors (preserves gradients)
        padded_embeds = torch.stack(padded_embeds_list, dim=0)
        padded_attention = torch.stack(padded_attention_list, dim=0)
        padded_labels = torch.stack(padded_labels_list, dim=0) if padded_labels_list is not None else None

        # In Stage 2, ensure padded_embeds requires grad
        if self.training_stage == 2:
            # Verify gradients are enabled
            if not padded_embeds.requires_grad:
                # This shouldn't happen, but just in case
                padded_embeds.requires_grad_(True)

        # Step 5: Forward through LLM
        # Stage 1: LLM is frozen, gradients only through projector
        # Stage 2: LLM is trainable, gradients through LLM parameters and embeddings
        outputs = self.model.llm(
            inputs_embeds=padded_embeds,
            attention_mask=padded_attention,
            labels=padded_labels,
            return_dict=True,
        )

        return outputs.loss

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on the eval dataset."""
        if self.eval_dataloader is None:
            return {}

        # Set trainable component to eval mode
        if self.training_stage == 1:
            self.model.audio_projector.eval()
        elif self.training_stage == 2:
            self.model.llm.eval()

        total_loss = 0.0
        num_batches = 0

        eval_dataloader = self.eval_dataloader
        if self.accelerator.is_main_process:
            eval_dataloader = tqdm(eval_dataloader, desc="Evaluating")

        for batch in eval_dataloader:
            loss = self._forward_pass(batch)
            gathered_loss = self.accelerator.gather(loss)
            total_loss += gathered_loss.mean().item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        metrics = {"eval_loss": avg_loss}

        # Set trainable component back to train mode
        if self.training_stage == 1:
            self.model.audio_projector.train()
        elif self.training_stage == 2:
            self.model.llm.train()

        return metrics

    def _log(self, metrics: Dict[str, Any]):
        """Log metrics (only on main process)."""
        if not self.accelerator.is_main_process:
            return

        metrics["step"] = self.global_step
        metrics["epoch"] = self.current_epoch
        metrics["timestamp"] = datetime.now().isoformat()
        self.log_history.append(metrics)

        # Format floats - use scientific notation for very small values like learning_rate
        def format_value(k, v):
            if isinstance(v, float):
                if k == "learning_rate":
                    return f"{k}: {v:.2e}"
                return f"{k}: {v:.4f}"
            return f"{k}: {v}"

        log_str = " | ".join([format_value(k, v) for k, v in metrics.items()])
        print(log_str)

        # Log to wandb/tensorboard via accelerate
        if self.config.report_to != "none":
            # Filter out non-numeric values for tracker
            tracker_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            self.accelerator.log(tracker_metrics, step=self.global_step)

        # Save to local JSON
        log_path = os.path.join(self.config.output_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(self.log_history, f, indent=2)

    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        self.accelerator.wait_for_everyone()

        if not self.accelerator.is_main_process:
            return

        checkpoint_dir = os.path.join(
            self.config.output_dir,
            f"checkpoint-{self.global_step}",
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Unwrap and save trainable component based on training stage
        if self.training_stage == 1:
            # Stage 1: Save projector
            unwrapped_component = self.accelerator.unwrap_model(self.model.audio_projector)
            component_path = os.path.join(checkpoint_dir, "audio_projector.pt")
            torch.save(unwrapped_component.state_dict(), component_path)
        elif self.training_stage == 2:
            # Stage 2: Save LLM
            unwrapped_component = self.accelerator.unwrap_model(self.model.llm)
            component_path = os.path.join(checkpoint_dir, "llm_model")
            unwrapped_component.save_pretrained(component_path)

        # Save config
        config_path = os.path.join(checkpoint_dir, "vyvo_config.json")
        with open(config_path, "w") as f:
            json.dump(self.model.config.__dict__, f, indent=2)

        # Save tokenizer
        self.model.tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "best_eval_loss": self.best_eval_loss,
            "training_stage": self.training_stage,
        }
        torch.save(
            training_state,
            os.path.join(checkpoint_dir, "training_state.pt"),
        )

        # Save optimizer and scheduler state
        self.accelerator.save_state(os.path.join(checkpoint_dir, "accelerator_state"))

        # Save training config
        training_config_path = os.path.join(checkpoint_dir, "training_config.json")
        with open(training_config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

        print(f"Checkpoint saved to {checkpoint_dir}")

        if is_best:
            best_dir = os.path.join(self.config.output_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)

            if self.training_stage == 1:
                torch.save(unwrapped_component.state_dict(), os.path.join(best_dir, "audio_projector.pt"))
            elif self.training_stage == 2:
                unwrapped_component.save_pretrained(os.path.join(best_dir, "llm_model"))

            with open(os.path.join(best_dir, "vyvo_config.json"), "w") as f:
                json.dump(self.model.config.__dict__, f, indent=2)
            self.model.tokenizer.save_pretrained(best_dir)
            print(f"Best model saved to {best_dir}")

        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Remove old checkpoints keeping only the most recent ones."""
        if not self.accelerator.is_main_process:
            return

        checkpoints = []
        for name in os.listdir(self.config.output_dir):
            if name.startswith("checkpoint-"):
                checkpoint_path = os.path.join(self.config.output_dir, name)
                checkpoints.append((checkpoint_path, os.path.getmtime(checkpoint_path)))

        checkpoints.sort(key=lambda x: x[1], reverse=True)

        for checkpoint_path, _ in checkpoints[self.config.save_total_limit:]:
            import shutil
            shutil.rmtree(checkpoint_path)
            print(f"Removed old checkpoint: {checkpoint_path}")

    def train(self):
        """Run the full training loop."""
        if self.accelerator.is_main_process:
            print("=" * 60)
            if self.training_stage == 1:
                print("VyvoOmni Stage-1 Training: Audio Projector")
            elif self.training_stage == 2:
                print("VyvoOmni Stage-2 Training: LLM Fine-tuning")
            print("=" * 60)
            print(f"Number of GPUs: {self.accelerator.num_processes}")
            print(f"Total training steps: {self.total_steps}")
            print(f"Epochs: {self.config.num_train_epochs}")
            print(f"Per-device batch size: {self.config.per_device_train_batch_size}")
            print(f"Total batch size: {self.config.per_device_train_batch_size * self.accelerator.num_processes * self.config.gradient_accumulation_steps}")
            print(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
            print(f"Learning rate: {self.config.learning_rate}")
            print(f"Mixed precision: {self.accelerator.mixed_precision}")
            print("=" * 60)

            if self.training_stage == 1:
                print("\nTrainable: audio_projector ONLY")
                print(f"Projector params: {sum(p.numel() for p in self.model.audio_projector.parameters()):,}")
            elif self.training_stage == 2:
                print("\nTrainable: LLM ONLY")
                llm_params = sum(p.numel() for p in self.model.llm.parameters() if p.requires_grad)
                print(f"LLM trainable params: {llm_params:,}")

            print("=" * 60)

        # Set trainable component to train mode
        if self.training_stage == 1:
            self.model.audio_projector.train()
        elif self.training_stage == 2:
            self.model.llm.train()

            # Verify LLM parameters have gradients enabled
            if self.accelerator.is_main_process:
                # Unwrap LLM from DDP to access its methods
                unwrapped_llm = self.accelerator.unwrap_model(self.model.llm)

                llm_trainable = sum(p.numel() for p in unwrapped_llm.parameters() if p.requires_grad)
                llm_total = sum(p.numel() for p in unwrapped_llm.parameters())
                embed_requires_grad = unwrapped_llm.get_input_embeddings().weight.requires_grad

                print(f"\nLLM gradient check:")
                print(f"  Trainable params: {llm_trainable:,} / {llm_total:,}")
                print(f"  Embedding layer requires_grad: {embed_requires_grad}")

                if llm_trainable == 0:
                    raise RuntimeError(
                        "ERROR: LLM has no trainable parameters! "
                        "Check that training_stage=2 is set correctly in the model config."
                    )

        accumulation_loss = 0.0

        for epoch in range(self.config.num_train_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            num_steps = 0

            train_dataloader = self.train_dataloader
            if self.accelerator.is_main_process:
                train_dataloader = tqdm(
                    train_dataloader,
                    desc=f"Epoch {epoch + 1}/{self.config.num_train_epochs}",
                )

            for step, batch in enumerate(train_dataloader):
                # Accumulate gradients for the trainable component
                trainable_component = self.model.audio_projector if self.training_stage == 1 else self.model.llm

                with self.accelerator.accumulate(trainable_component):
                    loss = self._forward_pass(batch)
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        # Clip gradients for trainable parameters
                        if self.training_stage == 1:
                            self.accelerator.clip_grad_norm_(
                                self.model.audio_projector.parameters(),
                                self.config.max_grad_norm,
                            )
                        elif self.training_stage == 2:
                            self.accelerator.clip_grad_norm_(
                                self.model.llm.parameters(),
                                self.config.max_grad_norm,
                            )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                loss_value = loss.detach().item()
                accumulation_loss += loss_value
                epoch_loss += loss_value
                num_steps += 1

                if self.accelerator.sync_gradients:
                    self.global_step += 1

                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = accumulation_loss / self.config.gradient_accumulation_steps
                        lr = self.scheduler.get_last_lr()[0]
                        self._log({
                            "loss": avg_loss,
                            "learning_rate": lr,
                        })
                        accumulation_loss = 0.0

                    if (
                        self.eval_dataloader is not None
                        and self.global_step % self.config.eval_steps == 0
                    ):
                        eval_metrics = self.evaluate()
                        self._log(eval_metrics)

                        if eval_metrics.get("eval_loss", float("inf")) < self.best_eval_loss:
                            self.best_eval_loss = eval_metrics["eval_loss"]
                            self._save_checkpoint(is_best=True)

                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint()

                if self.accelerator.is_main_process and hasattr(train_dataloader, 'set_postfix'):
                    train_dataloader.set_postfix({
                        "loss": f"{loss_value:.4f}",
                        "step": self.global_step,
                    })

            avg_epoch_loss = epoch_loss / max(num_steps, 1)
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

            if self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                self._log({"epoch_eval": True, **eval_metrics})

        self._save_checkpoint()

        # End trackers (wandb/tensorboard)
        if self.config.report_to != "none":
            self.accelerator.end_training()

        if self.accelerator.is_main_process:
            print("Training completed!")

        return self.log_history

    def resume_from_checkpoint(self, checkpoint_dir: str):
        """Resume training from a checkpoint."""
        training_state = torch.load(
            os.path.join(checkpoint_dir, "training_state.pt"),
            weights_only=True,
            map_location="cpu",
        )

        self.global_step = training_state["global_step"]
        self.current_epoch = training_state["epoch"]
        self.best_eval_loss = training_state["best_eval_loss"]

        accelerator_state_dir = os.path.join(checkpoint_dir, "accelerator_state")
        if os.path.exists(accelerator_state_dir):
            self.accelerator.load_state(accelerator_state_dir)

        # Load the appropriate component based on training stage
        if self.training_stage == 1:
            projector_path = os.path.join(checkpoint_dir, "audio_projector.pt")
            if os.path.exists(projector_path):
                unwrapped_projector = self.accelerator.unwrap_model(self.model.audio_projector)
                unwrapped_projector.load_state_dict(
                    torch.load(projector_path, weights_only=True, map_location="cpu")
                )
        elif self.training_stage == 2:
            llm_path = os.path.join(checkpoint_dir, "llm_model")
            if os.path.exists(llm_path):
                from transformers import AutoModelForCausalLM
                unwrapped_llm = self.accelerator.unwrap_model(self.model.llm)
                loaded_llm = AutoModelForCausalLM.from_pretrained(llm_path)
                unwrapped_llm.load_state_dict(loaded_llm.state_dict())

        if self.accelerator.is_main_process:
            print(f"Resumed from checkpoint: {checkpoint_dir}")
            print(f"Training Stage: {self.training_stage}")
            print(f"Global step: {self.global_step}, Epoch: {self.current_epoch}")
