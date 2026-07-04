# omni

A speech-to-speech LLM. It **listens** (audio in), **thinks** (a text inner
monologue), and **speaks** (audio out) — emotion-aware, multilingual, and able
to hold one speaker's voice across every language.

It's a single decoder-only transformer running at 12.5 Hz over parallel token
streams: one text stream carries the monologue and control tags, N audio streams
carry [Mimi](https://huggingface.co/kyutai/mimi) codec tokens. Every task — ASR,
TTS, speech continuation, text LM, spoken dialogue, full-duplex — is the same
format with different rows masked out. It runs fully offline on CPU for
development and scales to 8 GPUs (DDP / FSDP2) for real training.

Two backbones share every other module:
- **From scratch** — trained end to end (tiny CPU tests, ablations).
- **Pretrained backbone** — mounts a frozen Qwen3 / Llama 3 / Gemma as the
  transformer, so you skip text pretraining and train only the audio side.

## Install

```bash
uv venv .venv --python 3.12
uv pip install -p .venv/bin/python -e ".[dev]"
.venv/bin/pytest          # runs on CPU, fully offline
```

## Quickstart

No GPU, no downloads — a tiny model end to end on fake data:

```bash
# 1. make offline demo shards
.venv/bin/python scripts/prepare_data.py fake --n 256 --out data/shards/fake \
    --preset tiny model.n_codebooks=2 model.text_vocab_size=320

# 2. train + export a small model
.venv/bin/python scripts/train.py --preset tiny --data data/shards/fake \
    model.n_codebooks=2 model.text_vocab_size=320 data.num_workers=0 \
    train.max_steps=200 --export checkpoints/tiny

# 3. speak some text
.venv/bin/python scripts/chat.py --task tts --text "hello" --out hello.wav \
    --ckpt checkpoints/tiny --codec fake --tokenizer byte
```

Every script takes `--preset <name>` plus dotted overrides like
`train.max_steps=20`.

## Presets

| Preset | Size | Where |
|---|---|---|
| `tiny` | ~22M | CPU — tests, demos |
| `small` | ~0.3B | 1 GPU — ablations |
| `quality` | ~1.0B | 8 GPU — from-scratch flagship |
| `qwen3-1.7b` · `qwen3-8b` · `llama32-3b` · `gemma3-4b` | backbone | 1–8 GPU — pretrained backbone |

## Inference

```bash
# text -> speech
.venv/bin/python scripts/chat.py --task tts --text "hello" --out hello.wav \
    --ckpt <dir> --codec fake --tokenizer byte

# speech -> text
.venv/bin/python scripts/chat.py --task asr --in hello.wav \
    --ckpt <dir> --codec fake --tokenizer byte

# spoken reply, with an emotion and a cloned voice
.venv/bin/python scripts/chat.py --task s2s --in question.wav --out reply.wav \
    --ckpt <dir> --codec fake --tokenizer byte --emotion calm --lang en \
    --voice me.wav          # 10s reference wav pins the speaker voice
```

Use `--codec mimi` with real checkpoints.

## Streaming console

```bash
uv pip install -p .venv/bin/python -e ".[serve]"
.venv/bin/python scripts/serve.py --preset tiny      # -> http://127.0.0.1:7860
```

Pick a task, talk or type, and watch the reply stream live — audio plays as it's
generated and the inner monologue appears as the model speaks. Local only, no auth.

## Docs

Everything else lives in [`docs/`](docs/):

- [architecture](docs/DESIGN.md) — the grid format, delay pattern, training recipe
- [audio quality](docs/DESIGN_V3_AUDIO.md) · [emotion + multilingual](docs/DESIGN_V4_EMOTION_I18N.md) · [voice cloning](docs/DESIGN_V5_VOICE.md)
- [pretrained backbone](docs/DESIGN_V6_PRETRAINED_BACKBONE.md) — the v6 path in detail
- [module APIs](docs/INTERFACES.md) — the binding interface contract
