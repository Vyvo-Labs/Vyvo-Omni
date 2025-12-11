import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json
import os

from transformers import WhisperFeatureExtractor, PreTrainedTokenizer


class SpeechTextDataset(Dataset):
    """
    Dataset for speech-to-text training.

    Expects data in one of three formats:
    1. OmniAudio format (recommended):
       JSON: {"samples": [{"audio_path": str, "task_token": str, "instruction": str, "response": str}, ...]}
    2. Legacy format with system prompt:
       JSON: {"system_prompt": str, "samples": [{"audio_path": str, "text": str}, ...]}
    3. Directory with audio files and corresponding .txt files

    Args:
        data_path: Path to JSON file or directory containing audio files
        feature_extractor: Whisper feature extractor
        tokenizer: LLM tokenizer
        max_audio_length: Maximum audio length in seconds
        sample_rate: Audio sample rate
        audio_start_token: Token marking start of audio
        audio_end_token: Token marking end of audio
        max_text_length: Maximum text length in tokens
        system_prompt: Default system prompt for legacy format (default: "Transcribe the following audio:")
    """

    def __init__(
        self,
        data_path: str,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer: PreTrainedTokenizer,
        max_audio_length: float = 30.0,
        sample_rate: int = 16000,
        audio_start_token: str = "<|audio_start|>",
        audio_end_token: str = "<|audio_end|>",
        max_text_length: int = 512,
        system_prompt: Optional[str] = None,
    ):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_audio_length = max_audio_length
        self.sample_rate = sample_rate
        self.audio_start_token = audio_start_token
        self.audio_end_token = audio_end_token
        self.max_text_length = max_text_length
        self.system_prompt = system_prompt or "Transcribe the following audio:"

        # Store data_path directory for resolving relative audio paths
        self.data_dir = os.path.dirname(os.path.abspath(data_path)) if os.path.isfile(data_path) else os.path.abspath(data_path)

        self.samples = self._load_data(data_path)

    def _load_data(self, data_path: str) -> List[Dict[str, str]]:
        """Load data from JSON file or directory.

        JSON format can be either:
        1. List of samples: [{"audio_path": ..., "text": ...}, ...]
        2. Dict with metadata: {"system_prompt": "...", "samples": [...]}
        """
        samples = []

        if os.path.isfile(data_path) and data_path.endswith(".json"):
            with open(data_path, "r") as f:
                data = json.load(f)

            # Check if it's the new format with metadata
            if isinstance(data, dict) and "samples" in data:
                # Use global system_prompt if provided and not overridden in __init__
                if "system_prompt" in data and self.system_prompt == "Transcribe the following audio:":
                    self.system_prompt = data["system_prompt"]
                samples = data["samples"]
            else:
                # Old format: just a list of samples
                samples = data
        elif os.path.isdir(data_path):
            audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            for filename in os.listdir(data_path):
                name, ext = os.path.splitext(filename)
                if ext.lower() in audio_extensions:
                    audio_path = os.path.join(data_path, filename)
                    text_path = os.path.join(data_path, f"{name}.txt")
                    if os.path.exists(text_path):
                        with open(text_path, "r") as f:
                            text = f.read().strip()
                        samples.append({
                            "audio_path": audio_path,
                            "text": text,
                        })
        else:
            raise ValueError(f"Invalid data path: {data_path}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file."""
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Trim to max length
        max_samples = int(self.max_audio_length * self.sample_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

        return waveform.squeeze(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Load audio - handle both absolute and relative paths
        audio_path = sample["audio_path"]
        if not os.path.isabs(audio_path):
            # Relative path - resolve from JSON file directory
            audio_path = os.path.join(self.data_dir, audio_path)
        waveform = self._load_audio(audio_path)

        # Extract audio features (mel spectrogram)
        audio_features = self.feature_extractor(
            waveform.numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        audio_features = audio_features.input_features.squeeze(0)

        # Support both old (transcription) and new (conversational) formats
        if "instruction" in sample and "response" in sample:
            # New conversational format with task tokens (OmniAudio format)
            task_token = sample.get("task_token", "")
            instruction = sample["instruction"]
            response = sample["response"]

            # Format: <task_token> <instruction> <audio_start> [AUDIO] <audio_end> <response>
            # Handle empty instruction gracefully (no extra space)
            if instruction.strip():
                input_text = f"{task_token} {instruction}\n{self.audio_start_token}{self.audio_end_token}\n"
            else:
                # For transcription task with no instruction
                input_text = f"{task_token}\n{self.audio_start_token}{self.audio_end_token}\n"
            target_text = response
        else:
            # Old transcription format (backward compatible)
            text = sample["text"]
            prompt = sample.get("prompt", self.system_prompt)

            # Format: <prompt> <audio_start> [AUDIO] <audio_end> <response>
            input_text = f"{prompt}\n{self.audio_start_token}{self.audio_end_token}\n"
            target_text = text

        return {
            "audio_features": audio_features,
            "input_text": input_text,
            "target_text": target_text,
            "audio_path": audio_path,
        }


@dataclass
class SpeechTextCollator:
    """
    Data collator for speech-text training.

    Handles batching of audio features and text, creating appropriate
    attention masks and labels for causal language modeling.
    """

    tokenizer: PreTrainedTokenizer
    audio_start_token: str = "<|audio_start|>"
    audio_end_token: str = "<|audio_end|>"
    max_length: int = 2048
    padding: str = "longest"
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.

        Args:
            features: List of sample dicts from dataset

        Returns:
            Batched tensors for model input
        """
        batch_size = len(features)

        # Stack audio features
        audio_features = torch.stack([f["audio_features"] for f in features])

        # Tokenize input and target text
        input_texts = [f["input_text"] for f in features]
        target_texts = [f["target_text"] for f in features]

        # Combine input and target for causal LM training
        full_texts = [inp + tgt + self.tokenizer.eos_token for inp, tgt in zip(input_texts, target_texts)]

        # Tokenize
        encodings = self.tokenizer(
            full_texts,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # Create labels (mask input portion)
        labels = input_ids.clone()

        # Find positions of audio tokens and mask input portion
        audio_positions = []
        for i in range(batch_size):
            # Tokenize just the input to find where target starts
            input_encoding = self.tokenizer(
                input_texts[i],
                add_special_tokens=False,
                return_tensors="pt",
            )
            input_len = input_encoding["input_ids"].shape[1]

            # Mask input portion in labels
            labels[i, :input_len] = self.label_pad_token_id

            # Find audio token positions
            ids = input_ids[i].tolist()
            audio_start_id = self.tokenizer.convert_tokens_to_ids(self.audio_start_token)
            audio_end_id = self.tokenizer.convert_tokens_to_ids(self.audio_end_token)

            start_pos = None
            end_pos = None
            for j, tid in enumerate(ids):
                if tid == audio_start_id and start_pos is None:
                    start_pos = j
                if tid == audio_end_id and start_pos is not None:
                    end_pos = j
                    break

            if start_pos is not None and end_pos is not None:
                audio_positions.append((start_pos, end_pos))
            else:
                audio_positions.append((0, 0))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_features": audio_features,
            "audio_positions": audio_positions,
            "labels": labels,
        }


class DummySpeechTextDataset(Dataset):
    """
    Dummy dataset for testing and debugging.

    Generates random audio and text pairs.
    """

    def __init__(
        self,
        num_samples: int = 100,
        feature_extractor: Optional[WhisperFeatureExtractor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        audio_start_token: str = "<|audio_start|>",
        audio_end_token: str = "<|audio_end|>",
        sample_rate: int = 16000,
        audio_length: float = 5.0,
    ):
        self.num_samples = num_samples
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.audio_start_token = audio_start_token
        self.audio_end_token = audio_end_token
        self.sample_rate = sample_rate
        self.audio_length = audio_length

        # Generate dummy data
        self.samples = self._generate_samples()

    def _generate_samples(self) -> List[Dict[str, Any]]:
        """Generate dummy samples."""
        samples = []
        dummy_texts = [
            "Hello, how are you today?",
            "The weather is nice outside.",
            "I love programming in Python.",
            "Machine learning is fascinating.",
            "Speech recognition has improved significantly.",
            "Natural language processing enables many applications.",
            "Deep learning models can understand audio.",
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming industries.",
            "Voice assistants are becoming more common.",
        ]

        for i in range(self.num_samples):
            # Generate random audio
            num_samples = int(self.audio_length * self.sample_rate)
            waveform = torch.randn(num_samples) * 0.1

            # Get audio features
            if self.feature_extractor is not None:
                audio_features = self.feature_extractor(
                    waveform.numpy(),
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                )
                audio_features = audio_features.input_features.squeeze(0)
            else:
                # Dummy features matching Whisper output shape
                audio_features = torch.randn(80, 3000)

            text = dummy_texts[i % len(dummy_texts)]

            samples.append({
                "audio_features": audio_features,
                "input_text": f"Transcribe the audio:\n{self.audio_start_token}{self.audio_end_token}\n",
                "target_text": text,
                "audio_path": f"dummy_audio_{i}.wav",
            })

        return samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
