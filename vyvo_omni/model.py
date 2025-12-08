import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperModel,
    WhisperFeatureExtractor,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import VyvoOmniConfig


@dataclass
class VyvoOmniOutput:
    """Output from VyvoOmni model."""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[Tuple] = None
    hidden_states: Optional[Tuple] = None
    attentions: Optional[Tuple] = None


class AudioProjector(nn.Module):
    """
    Projects Whisper encoder outputs to LLM embedding space.

    This is the ONLY trainable component in Stage-1 training.
    """

    def __init__(
        self,
        whisper_hidden_size: int,
        llm_hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.whisper_hidden_size = whisper_hidden_size
        self.llm_hidden_size = llm_hidden_size

        if num_layers == 1:
            self.projection = nn.Linear(whisper_hidden_size, llm_hidden_size)
        else:
            layers = []
            in_size = whisper_hidden_size

            for i in range(num_layers):
                out_size = llm_hidden_size if i == num_layers - 1 else (whisper_hidden_size + llm_hidden_size) // 2
                layers.append(nn.Linear(in_size, out_size))
                if i < num_layers - 1:
                    layers.append(nn.GELU())
                    layers.append(nn.Dropout(dropout))
                in_size = out_size

            self.projection = nn.Sequential(*layers)

    def forward(self, whisper_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            whisper_outputs: (batch_size, seq_len, whisper_hidden_size)
        Returns:
            projected: (batch_size, seq_len, llm_hidden_size)
        """
        return self.projection(whisper_outputs)


class VyvoOmniModel(nn.Module):
    """
    Omni model combining Whisper encoder with an LLM for speech-to-text generation.

    Stage-1 Training Flow:
        1. Audio (mel spectrogram) -> Whisper Encoder (FROZEN) -> Audio hidden states
        2. Audio hidden states -> Audio Projector (TRAINABLE) -> Projected audio embeddings
        3. Text -> LLM Tokenizer -> Token IDs -> LLM Embedding Layer -> Text embeddings
        4. Concatenate: [text_before_audio, projected_audio, text_after_audio]
        5. Combined embeddings -> LLM (FROZEN) -> Output logits
        6. Compute loss on response tokens only
    """

    def __init__(self, config: VyvoOmniConfig):
        super().__init__()
        self.config = config

        # Load Whisper encoder (will be FROZEN)
        self.whisper = WhisperModel.from_pretrained(
            config.whisper_model_name,
            torch_dtype=torch.bfloat16,
        )
        self.whisper_encoder = self.whisper.encoder
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(config.whisper_model_name)

        # Load LLM (will be FROZEN)
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="kernels-community/flash-attn3" if config.llm_use_flash_attention else "sdpa",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.llm_model_name,
            trust_remote_code=True,
        )

        # Add special tokens for audio boundaries
        special_tokens = {
            "additional_special_tokens": [
                config.audio_start_token,
                config.audio_end_token,
            ]
        }
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            self.llm.resize_token_embeddings(len(self.tokenizer))

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get hidden sizes
        self.whisper_hidden_size = self.whisper_encoder.config.d_model
        self.llm_hidden_size = self.llm.config.hidden_size

        # Create audio projector (TRAINABLE)
        self.audio_projector = AudioProjector(
            whisper_hidden_size=self.whisper_hidden_size,
            llm_hidden_size=self.llm_hidden_size,
            num_layers=config.projection_num_layers,
            dropout=config.projection_dropout,
        )

        # Store special token IDs
        self.audio_start_token_id = self.tokenizer.convert_tokens_to_ids(config.audio_start_token)
        self.audio_end_token_id = self.tokenizer.convert_tokens_to_ids(config.audio_end_token)

        # Freeze components for Stage-1 training
        self._freeze_components()

    def _freeze_components(self):
        """Freeze Whisper and LLM, keep only projector trainable."""
        # Freeze Whisper encoder
        if self.config.freeze_whisper:
            self.whisper_encoder.eval()
            for param in self.whisper_encoder.parameters():
                param.requires_grad = False

        # Freeze LLM
        if self.config.freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

        # Audio projector remains trainable by default

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (only projector in Stage-1)."""
        return [p for p in self.audio_projector.parameters() if p.requires_grad]

    def print_trainable_parameters(self):
        """Print trainable parameter statistics."""
        trainable_params = 0
        all_params = 0

        print("\nTrainable parameters:")
        for name, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"  {name}: {param.numel():,}")

        print(f"\nTotal params: {all_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Trainable %: {100 * trainable_params / all_params:.4f}%")

    @torch.no_grad()
    def encode_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Encode audio using frozen Whisper encoder.

        Args:
            audio_features: Mel spectrogram (batch_size, n_mels, time_steps)

        Returns:
            audio_hidden: (batch_size, audio_seq_len, whisper_hidden_size)
        """
        # Whisper encoder is frozen, use no_grad
        outputs = self.whisper_encoder(audio_features)
        return outputs.last_hidden_state

    def project_audio(self, audio_hidden: torch.Tensor) -> torch.Tensor:
        """
        Project audio hidden states to LLM embedding space.

        Args:
            audio_hidden: (batch_size, audio_seq_len, whisper_hidden_size)

        Returns:
            audio_embeds: (batch_size, audio_seq_len, llm_hidden_size)
        """
        return self.audio_projector(audio_hidden)

    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get text embeddings from LLM embedding layer."""
        return self.llm.get_input_embeddings()(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_features: torch.Tensor,
        audio_positions: List[Tuple[int, int]],
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> VyvoOmniOutput:
        """
        Forward pass for Stage-1 training.

        Args:
            input_ids: Token IDs with audio placeholder (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            audio_features: Mel spectrogram (batch, n_mels, time_steps)
            audio_positions: List of (start, end) positions for audio placeholder per sample
            labels: Target token IDs for loss computation (batch, seq_len)

        Returns:
            VyvoOmniOutput with loss and logits
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        dtype = self.llm.dtype

        # Step 1: Encode audio with frozen Whisper
        audio_hidden = self.encode_audio(audio_features)  # (batch, audio_seq, whisper_dim)

        # Step 2: Project audio to LLM space (TRAINABLE)
        audio_embeds = self.project_audio(audio_hidden)  # (batch, audio_seq, llm_dim)
        audio_seq_len = audio_embeds.shape[1]

        # Step 3: Get text embeddings
        text_embeds = self.get_text_embeddings(input_ids)  # (batch, text_seq, llm_dim)

        # Step 4: Merge audio and text embeddings
        merged_embeds_list = []
        merged_labels_list = []
        merged_attention_list = []

        for i in range(batch_size):
            start_pos, end_pos = audio_positions[i]

            # Text embeddings: before audio_start, audio_start itself, after audio_end
            # We keep audio_start token and insert audio after it, then continue from audio_end
            pre_audio = text_embeds[i, :start_pos + 1]  # up to and including audio_start
            post_audio = text_embeds[i, end_pos:]  # from audio_end onwards

            # Merge: [pre_audio, audio_embeds, post_audio]
            merged = torch.cat([pre_audio, audio_embeds[i], post_audio], dim=0)
            merged_embeds_list.append(merged)

            # Handle labels if provided
            if labels is not None:
                pre_labels = labels[i, :start_pos + 1]
                post_labels = labels[i, end_pos:]
                # Audio tokens should be ignored in loss (-100)
                audio_labels = torch.full(
                    (audio_seq_len,), -100, dtype=labels.dtype, device=device
                )
                merged_label = torch.cat([pre_labels, audio_labels, post_labels], dim=0)
                merged_labels_list.append(merged_label)

            # Handle attention mask
            pre_attn = attention_mask[i, :start_pos + 1]
            post_attn = attention_mask[i, end_pos:]
            audio_attn = torch.ones(audio_seq_len, dtype=attention_mask.dtype, device=device)
            merged_attn = torch.cat([pre_attn, audio_attn, post_attn], dim=0)
            merged_attention_list.append(merged_attn)

        # Pad to same length
        max_len = max(e.shape[0] for e in merged_embeds_list)

        # Padded embeddings
        padded_embeds = torch.zeros(
            batch_size, max_len, self.llm_hidden_size,
            device=device, dtype=dtype
        )
        padded_attention = torch.zeros(
            batch_size, max_len,
            device=device, dtype=attention_mask.dtype
        )
        padded_labels = torch.full(
            (batch_size, max_len), -100,
            device=device, dtype=torch.long
        ) if labels is not None else None

        for i in range(batch_size):
            seq_len = merged_embeds_list[i].shape[0]
            padded_embeds[i, :seq_len] = merged_embeds_list[i]
            padded_attention[i, :seq_len] = merged_attention_list[i]
            if padded_labels is not None:
                padded_labels[i, :seq_len] = merged_labels_list[i]

        # Step 5: Forward through frozen LLM
        outputs = self.llm(
            inputs_embeds=padded_embeds,
            attention_mask=padded_attention,
            labels=padded_labels,
            return_dict=True,
        )

        return VyvoOmniOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )

    @torch.no_grad()
    def generate(
        self,
        audio_features: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_positions: List[Tuple[int, int]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Generate text from audio input."""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        dtype = self.llm.dtype

        # Encode and project audio
        audio_hidden = self.encode_audio(audio_features)
        audio_embeds = self.project_audio(audio_hidden)
        audio_seq_len = audio_embeds.shape[1]

        # Get text embeddings
        text_embeds = self.get_text_embeddings(input_ids)

        # Merge embeddings
        merged_embeds_list = []
        merged_attention_list = []

        for i in range(batch_size):
            start_pos, end_pos = audio_positions[i]

            pre_audio = text_embeds[i, :start_pos + 1]
            post_audio = text_embeds[i, end_pos:]

            merged = torch.cat([pre_audio, audio_embeds[i], post_audio], dim=0)
            merged_embeds_list.append(merged)

            pre_attn = attention_mask[i, :start_pos + 1]
            post_attn = attention_mask[i, end_pos:]
            audio_attn = torch.ones(audio_seq_len, dtype=attention_mask.dtype, device=device)
            merged_attn = torch.cat([pre_attn, audio_attn, post_attn], dim=0)
            merged_attention_list.append(merged_attn)

        # Pad
        max_len = max(e.shape[0] for e in merged_embeds_list)

        padded_embeds = torch.zeros(batch_size, max_len, self.llm_hidden_size, device=device, dtype=dtype)
        padded_attention = torch.zeros(batch_size, max_len, device=device, dtype=attention_mask.dtype)

        for i in range(batch_size):
            seq_len = merged_embeds_list[i].shape[0]
            padded_embeds[i, :seq_len] = merged_embeds_list[i]
            padded_attention[i, :seq_len] = merged_attention_list[i]

        # Generate
        outputs = self.llm.generate(
            inputs_embeds=padded_embeds,
            attention_mask=padded_attention,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        return outputs

    def save_pretrained(self, save_directory: str):
        """Save only the projector weights (the trainable part)."""
        import os
        import json

        os.makedirs(save_directory, exist_ok=True)

        # Save projector
        projector_path = os.path.join(save_directory, "audio_projector.pt")
        torch.save(self.audio_projector.state_dict(), projector_path)

        # Save config
        config_path = os.path.join(save_directory, "vyvo_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

        # Save tokenizer (includes special tokens)
        self.tokenizer.save_pretrained(save_directory)

        print(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, load_directory: str, config: Optional[VyvoOmniConfig] = None):
        """Load a pretrained VyvoOmni model."""
        import os
        import json

        if config is None:
            config_path = os.path.join(load_directory, "vyvo_config.json")
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            config = VyvoOmniConfig(**config_dict)

        model = cls(config)

        projector_path = os.path.join(load_directory, "audio_projector.pt")
        if os.path.exists(projector_path):
            model.audio_projector.load_state_dict(
                torch.load(projector_path, weights_only=True, map_location="cpu")
            )

        return model
