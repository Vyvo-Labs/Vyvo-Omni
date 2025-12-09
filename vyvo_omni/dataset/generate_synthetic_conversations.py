#!/usr/bin/env python3
"""
Generate synthetic conversational data from transcriptions using LLM.

This script transforms pure transcription data into instruction-response pairs,
similar to OmniAudio's SFT stage.

Usage:
    1. Set your API key in environment: export DEEPSEEK_API_KEY=your-key
    2. Edit configuration below
    3. Run: python generate_synthetic_conversations.py
"""

import os
import json
import random
from tqdm import tqdm
from typing import List, Dict

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input/Output
INPUT_JSON = "data/train.json"  # Your transcription JSON
OUTPUT_JSON = "data/train_conversational.json"  # Output conversational JSON

# LLM API Settings
LLM_PROVIDER = "deepseek"  # Options: "deepseek", "anthropic", "openai"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Generation settings
MAX_SAMPLES = None  # Limit samples to process (None = all)
QA_PAIRS_PER_TRANSCRIPTION = 3  # Number of Q&A pairs per transcription
BATCH_SIZE = 10  # Process in batches to save progress

# Task distribution (should sum to 1.0)
TASK_DISTRIBUTION = {
    "transcribe": 0.2,  # 20% pure transcription
    "summarize": 0.2,   # 20% summarization
    "question": 0.3,    # 30% Q&A about content
    "describe": 0.15,   # 15% description/explanation
    "chat": 0.15,       # 15% conversational follow-up
}

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def get_llm_client():
    """Initialize LLM client."""
    if LLM_PROVIDER == "deepseek":
        try:
            from openai import OpenAI
            if not DEEPSEEK_API_KEY:
                raise ValueError("DEEPSEEK_API_KEY not set")
            return OpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com"
            )
        except ImportError:
            raise ImportError("Install: pip install openai")
    elif LLM_PROVIDER == "anthropic":
        try:
            import anthropic
            if not ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY not set")
            return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        except ImportError:
            raise ImportError("Install: pip install anthropic")
    elif LLM_PROVIDER == "openai":
        try:
            from openai import OpenAI
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set")
            return OpenAI(api_key=OPENAI_API_KEY)
        except ImportError:
            raise ImportError("Install: pip install openai")
    else:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")


def generate_qa_pairs(transcription: str, client, num_pairs: int = 3) -> List[Dict]:
    """Generate Q&A pairs from transcription using LLM."""

    prompt = f"""Given this transcription from an audio recording:

"{transcription}"

Generate {num_pairs} diverse question-answer pairs that someone might ask about this audio. Include different types:
1. Content questions (what is this about?)
2. Summarization requests (summarize this)
3. Specific detail questions
4. Interpretation/explanation requests

Make responses natural and conversational, as if talking about what you heard.

Return ONLY a JSON array (no markdown, no explanation):
[
  {{"task": "question", "question": "...", "answer": "..."}},
  {{"task": "summarize", "question": "...", "answer": "..."}},
  {{"task": "describe", "question": "...", "answer": "..."}}
]"""

    try:
        if LLM_PROVIDER == "anthropic":
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text
        elif LLM_PROVIDER in ["deepseek", "openai"]:
            # Both use OpenAI interface
            model = "deepseek-chat" if LLM_PROVIDER == "deepseek" else "gpt-4o-mini"
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates question-answer pairs from audio transcriptions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.7
            )
            content = response.choices[0].message.content
        else:
            raise ValueError(f"Unknown provider: {LLM_PROVIDER}")

        # Extract JSON from response
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        qa_pairs = json.loads(content)
        return qa_pairs

    except Exception as e:
        # Fallback: create simple pairs
        return [
            {
                "task": "transcribe",
                "question": "What is in this audio?",
                "answer": f"Transcription: {transcription}"
            }
        ]


def create_instruction_sample(audio_path: str, transcription: str,
                               task: str, question: str, answer: str) -> Dict:
    """Create a single instruction-response sample."""

    # Map task to task token
    task_token_map = {
        "transcribe": "<|transcribe|>",
        "summarize": "<|summarize|>",
        "question": "<|question|>",
        "describe": "<|describe|>",
        "chat": "<|chat|>",
    }

    task_token = task_token_map.get(task, "<|question|>")

    return {
        "audio_path": audio_path,
        "task": task,
        "task_token": task_token,
        "instruction": question,
        "response": answer,
    }


def process_transcriptions():
    """Main processing function."""

    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_json_abs = os.path.abspath(os.path.join(script_dir, INPUT_JSON))
    output_json_abs = os.path.abspath(os.path.join(script_dir, OUTPUT_JSON))

    print(f"Generating synthetic data: {input_json_abs} → {output_json_abs}")
    print(f"LLM: {LLM_PROVIDER}")

    # Load input data
    with open(input_json_abs, "r") as f:
        data = json.load(f)

    samples = data["samples"]
    if MAX_SAMPLES:
        samples = samples[:MAX_SAMPLES]

    print(f"Processing {len(samples)} samples...")

    # Initialize LLM client
    client = get_llm_client()

    # Process samples
    conversational_samples = []

    for i, sample in enumerate(tqdm(samples, desc="Processing")):
        audio_path = sample["audio_path"]
        transcription = sample.get("response") or sample.get("text", "")

        # Always include original transcription task
        conversational_samples.append(
            create_instruction_sample(
                audio_path=audio_path,
                transcription=transcription,
                task="transcribe",
                question="Transcribe this audio.",
                answer=transcription
            )
        )

        # Generate synthetic Q&A pairs
        qa_pairs = generate_qa_pairs(transcription, client, QA_PAIRS_PER_TRANSCRIPTION)

        for qa in qa_pairs:
            conversational_samples.append(
                create_instruction_sample(
                    audio_path=audio_path,
                    transcription=transcription,
                    task=qa.get("task", "question"),
                    question=qa["question"],
                    answer=qa["answer"]
                )
            )

        # Save progress every BATCH_SIZE samples
        if (i + 1) % BATCH_SIZE == 0:
            temp_output = output_json_abs.replace(".json", f"_temp_{i+1}.json")
            with open(temp_output, "w") as f:
                json.dump({"samples": conversational_samples}, f, indent=2)

    # Save final output
    output_data = {
        "samples": conversational_samples,
        "metadata": {
            "source": input_json_abs,
            "total_samples": len(conversational_samples),
            "original_transcriptions": len(samples),
            "avg_pairs_per_audio": len(conversational_samples) / len(samples),
        }
    }

    with open(output_json_abs, "w") as f:
        json.dump(output_data, f, indent=2)

    # Clean up temp files
    output_dir = os.path.dirname(output_json_abs)
    for temp_file in os.listdir(output_dir):
        if temp_file.startswith("train_conversational_temp_"):
            os.remove(os.path.join(output_dir, temp_file))

    print(f"Done: {len(conversational_samples)} samples → {output_json_abs}")


if __name__ == "__main__":
    process_transcriptions()
