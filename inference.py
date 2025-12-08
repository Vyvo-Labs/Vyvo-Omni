#!/usr/bin/env python3
"""
Inference script for VyvoOmni model.

Usage:
    python inference.py --model_path ./vyvo_omni_output/best_model --audio_path audio.wav
    python inference.py --model_path ./vyvo_omni_output/best_model --audio_path audio.wav --prompt "Transcribe:"
"""

import argparse
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
    audio_features = audio_features.input_features.to(device)

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
    parser = argparse.ArgumentParser(description="VyvoOmni Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to audio file to transcribe",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Transcribe the following audio:",
        help="Prompt to use for transcription",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Disable sampling (use greedy decoding)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model = VyvoOmniModel.from_pretrained(args.model_path)
    model.to(args.device)
    model.eval()

    print(f"Transcribing {args.audio_path}...")
    transcription = transcribe(
        model=model,
        audio_path=args.audio_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.no_sample,
    )

    print("\n" + "=" * 50)
    print("Transcription:")
    print("=" * 50)
    print(transcription)


if __name__ == "__main__":
    main()
