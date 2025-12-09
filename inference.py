#!/usr/bin/env python3
"""
Inference script for VyvoOmni model.

Usage:
    python inference.py

Edit the manual configuration section in main() to set your model path and audio file.
"""

import torch
import torchaudio
from typing import Optional

from vyvo_omni import VyvoOmniModel, VyvoOmniConfig


def load_audio(audio_path: str, sample_rate: int = 16000) -> torch.Tensor:
    """Load and preprocess audio file."""
    waveform, sr = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    return waveform.squeeze(0)


def transcribe(
    model: VyvoOmniModel,
    audio_path: str,
    prompt: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """
    Transcribe audio using the VyvoOmni model.

    Args:
        model: Loaded VyvoOmni model
        audio_path: Path to audio file
        prompt: Optional prompt to use
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        do_sample: Whether to use sampling

    Returns:
        Transcribed text
    """
    device = next(model.parameters()).device

    # Load audio
    waveform = load_audio(audio_path, model.config.audio_sample_rate)

    # Extract audio features
    audio_features = model.feature_extractor(
        waveform.numpy(),
        sampling_rate=model.config.audio_sample_rate,
        return_tensors="pt",
    )
    # Match model's dtype (e.g., bfloat16, float32)
    model_dtype = next(model.parameters()).dtype
    audio_features = audio_features.input_features.to(device=device, dtype=model_dtype)

    # Create input text with audio placeholder
    prompt = prompt or "Transcribe the following audio:"
    input_text = f"{prompt}\n{model.config.audio_start_token}{model.config.audio_end_token}\n"

    # Tokenize
    encoding = model.tokenizer(
        input_text,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Find audio token positions
    audio_start_id = model.audio_start_token_id
    audio_end_id = model.audio_end_token_id

    ids = input_ids[0].tolist()
    start_pos = ids.index(audio_start_id)
    end_pos = ids.index(audio_end_id)
    audio_positions = [(start_pos, end_pos)]

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            audio_features=audio_features,
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_positions=audio_positions,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

    # Decode output
    transcription = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Remove the prompt from the output if present
    if transcription.startswith(prompt):
        transcription = transcription[len(prompt):].strip()

    return transcription


def main():
    # Manual configuration
    model_path = "vyvo_omni_output_stage2/checkpoint-500"
    audio_path = "test.wav"  # Change this to your audio file
    prompt = "Transcribe the following audio:"  # Optional prompt in Turkish
    max_new_tokens = 512
    temperature = 0.9
    top_p = 0.9
    do_sample = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {model_path}...")
    model = VyvoOmniModel.from_pretrained(model_path)

    # Ensure all model components have the same dtype
    # Get dtype from whisper encoder (which is loaded in bfloat16)
    whisper_dtype = next(model.whisper_encoder.parameters()).dtype
    print(f"Converting model to {whisper_dtype}...")

    # Convert audio projector to match whisper dtype
    model.audio_projector = model.audio_projector.to(whisper_dtype)

    model.to(device)
    model.eval()

    print(f"Transcribing {audio_path}...")
    transcription = transcribe(
        model=model,
        audio_path=audio_path,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
    )

    print("\n" + "=" * 50)
    print("Transcription:")
    print("=" * 50)
    print(transcription)


if __name__ == "__main__":
    main()
