# omni

A speech-to-speech LLM. It **listens** (audio in), **thinks** (a text inner
monologue), and **speaks** (audio out) — emotion-aware, multilingual, and able
to hold a speaker's voice across every language.

The whole model is one decoder-only transformer running at **12.5 Hz** over
parallel token streams:

```
                one 80 ms frame ─┐
                                 ▼
  text   │ <s2s> <assistant> <lang_en> <calm>  w1   w2   w3  ...   ← the model thinks in text
  audio1 │  •     •     c c c  c c c  c c c  c c c  c c c  ...     ← Mimi codec tokens
  audio2 │  •     •     c c c  c c c  c c c  c c c  c c c  ...
   ...   │                    (N codebooks)
```

One text stream carries the inner monologue and control tags; N audio streams
carry [Mimi](https://huggingface.co/kyutai/mimi) codec tokens. **Every task —
ASR, TTS, speech continuation, text LM, spoken dialogue (s2s), full-duplex — is
the same grid** with different rows masked out of the loss. The only pretrained,
frozen piece is the audio codec.

Two backbones share every other module:

- **From scratch** (`OmniModel`) — trained end to end; used for tiny CPU tests and ablations.
- **Pretrained backbone** (`HFOmniModel`, *v6*) — mounts a frozen Qwen3 / Llama 3 / Gemma
  as the temporal transformer, so you skip text pretraining and train only the audio side.

Runs fully **offline on CPU** for development (a `FakeCodec` stands in for Mimi);
scales to **8 GPUs** with automatic DDP / FSDP2 for real training.

📄 Design docs: [architecture](docs/DESIGN.md) ·
[audio quality](docs/DESIGN_V3_AUDIO.md) ·
[emotion + multilingual](docs/DESIGN_V4_EMOTION_I18N.md) ·
[voice cloning](docs/DESIGN_V5_VOICE.md) ·
[pretrained backbone](docs/DESIGN_V6_PRETRAINED_BACKBONE.md) ·
[module APIs](docs/INTERFACES.md)

---

## Install

```bash
uv venv .venv --python 3.12
uv pip install -p .venv/bin/python -e ".[dev]"
.venv/bin/pytest          # 264 tests · CPU · fully offline, nothing downloads
```

## Try it in 3 commands

No GPU, no downloads — a tiny model end to end on fake data:

```bash
# 1. make offline demo shards (FakeCodec + a sine-wave TTS)
.venv/bin/python scripts/prepare_data.py fake --n 256 --out data/shards/fake \
    --preset tiny model.n_codebooks=2 model.text_vocab_size=320

# 2. train + export a ~22M model
.venv/bin/python scripts/train.py --preset tiny --data data/shards/fake \
    model.n_codebooks=2 model.text_vocab_size=320 data.num_workers=0 \
    train.max_steps=200 --export checkpoints/tiny

# 3. speak some text
.venv/bin/python scripts/chat.py --task tts --text "hello" --out hello.wav \
    --ckpt checkpoints/tiny --codec fake --tokenizer byte
```

Every script takes `--preset <name>` plus dotted overrides like
`model.n_codebooks=2 train.max_steps=20`.

---

## How it works

### The grid

A sample is a grid of `[S, T]` integers, where `S = 1 + n_codebooks` streams and
each column `T` is one 80 ms frame:

```
row 0        text     <bos> <s2s> <user> ..user speech.. <end_of_turn>
                      <assistant> <lang_en> <emo_pcv> <angry> <emo_rsp> <calm> w1 w2 ...
rows 1..N    audio    Mimi codebook tokens, one column = one frame
loss_mask             which positions are training targets (user speech is input-only)
channel               who owns the turn (user / assistant)
```

Logits at position `p` predict position `p+1`; the `loss_mask` decides which
targets count. Want ASR? Mask everything but the text. Want TTS? Mask everything
but the audio. Same model, same code path.

### The delay trick

Grids on disk are **undelayed** — every codebook of frame `t` sits in column `t`.
The MusicGen-style per-codebook delay (`streams.apply_delay`) is applied **at
batch time**, so you can change the delay pattern without re-tokenizing a single
shard. The model and generator always see delayed grids.

### Control tags live in the text

Emotion, language, and paralinguistics are just special text tokens (ids 0..63
are reserved; real text starts at 64). The model reads your tone
(`<emo_pcv> <angry>`), picks a register (`<emo_rsp> <calm>`), and names a
language (`<lang_en>`) *in its monologue, before it speaks* — all inspectable and
all overridable.

---

## Prepare data

You feed in plain dialogue dicts (JSONL-friendly); emotion + language are
optional per turn:

```json
{"turns": [
  {"user": "why is my order late again",
   "assistant": "I am sorry about that, let me check it right away",
   "user_emotion": "angry", "response_style": "calm", "lang": "en"}
]}
```

Prep writes binary shards (`shard-00000.bin` + `.idx.jsonl` + `meta.json`). Each
task has its own prep command:

```bash
# offline demo (no network) — all tasks incl. s2s
.venv/bin/python scripts/prepare_data.py fake --n 256 --out data/shards/fake \
    --preset tiny model.n_codebooks=2 model.text_vocab_size=320

# text pretraining rows (Stage A)
.venv/bin/python scripts/prepare_data.py textlm --dataset HuggingFaceFW/fineweb-edu \
    --name sample-10BT --max-samples 100000 --lang en \
    --tokenizer data/tokenizer/omni_bpe.json --out data/shards/textlm --preset quality

# ASR / TTS / speech-LM from a speech corpus (Stage B)
.venv/bin/python scripts/prepare_data.py asr --dataset openslr/librispeech_asr --name clean \
    --split train.100 --max-samples 20000 --codec mimi --lang en \
    --tokenizer data/tokenizer/omni_bpe.json --out data/shards/speech --preset quality

# spoken dialogues (Stage C): text dialogues -> TTS -> Mimi tokens
.venv/bin/python scripts/prepare_data.py s2s --dialogues soda-emotional --tts vibevoice \
    --codec mimi --max-samples 50000 --tokenizer data/tokenizer/omni_bpe.json \
    --out data/shards/s2s --preset quality
```

The 48k multilingual BPE tokenizer is a one-time build (skip it entirely if you
use a pretrained backbone — see below):

```bash
.venv/bin/python scripts/train_tokenizer.py \
    --dataset HuggingFaceFW/fineweb-edu:sample-10BT \
    --vocab-size 48000 --out data/tokenizer/omni_bpe.json
```

<details>
<summary>Same thing from Python</summary>

```python
from omni.audio.codec import FakeCodec
from omni.config import load_config
from omni.data.prepare import prepare_s2s
from omni.data.synthesize import SineTTS, fake_dialogues
from omni.text.tokenizer import ByteTokenizer

cfg = load_config("tiny", ["model.n_codebooks=2", "model.text_vocab_size=320",
                           "data.batch_size=2"])
dialogues = [{"turns": [{"user": "hello there", "assistant": "hi, how can I help",
                         "user_emotion": "happy", "response_style": "calm", "lang": "en"}]}]
dialogues += list(fake_dialogues(15, seed=0))
prepare_s2s("data/shards/demo", dialogues=dialogues, tts=SineTTS(),
            codec=FakeCodec(n_codebooks=2), tokenizer=ByteTokenizer(),
            cfg=cfg, max_samples=16)
```
</details>

---

## Train

```bash
# single process (CPU or 1 GPU)
.venv/bin/python scripts/train.py --preset tiny --data data/shards/fake \
    model.n_codebooks=2 model.text_vocab_size=320 data.num_workers=0 \
    train.max_steps=200 --export checkpoints/tiny

# 8 GPUs — same script. <300M params -> DDP, larger -> FSDP2, chosen automatically.
# --data DIR:WEIGHT mixes shard dirs; checkpoints resume on their own.
torchrun --standalone --nproc_per_node=8 scripts/train.py --preset quality \
    --data data/shards/s2s:0.6 --data data/shards/speech:0.25 --data data/shards/textlm:0.15 \
    --export checkpoints/quality
```

Add `train.wandb=true` to any train command to stream per-head losses, lr,
grad-norm, and throughput to Weights & Biases; resuming a checkpoint resumes the
*same* run (`pip install 'omni[wandb]'`).

### Presets

| Preset | Size | Where | Notes |
|---|---|---|---|
| `tiny` | ~22M | CPU | smoke tests, wiring demos |
| `small` | ~0.3B | 1 GPU | ablations (32 codebooks + depth transformer) |
| `quality` | ~1.0B | 8 GPU | from-scratch flagship |
| `base` | 0.76B | 8 GPU | legacy 8-codebook path |
| `qwen3-1.7b` `qwen3-8b` `llama32-3b` `gemma3-4b` | backbone | 1–8 GPU | **v6 pretrained backbone** |

<details>
<summary>Same thing from Python</summary>

```python
from omni.config import load_config
from omni.data.dataset import build_dataloader
from omni.model.omni import OmniModel
from omni.train.loop import Trainer

cfg = load_config("tiny", ["model.n_codebooks=2", "model.text_vocab_size=320",
                           "data.batch_size=2", "data.num_workers=0",
                           "train.max_steps=20", "train.ckpt_dir=checkpoints/demo"])
model = OmniModel(cfg.model)
model.init_weights()
trainer = Trainer(cfg, model, build_dataloader(cfg, ["data/shards/demo"]))
trainer.fit()                                # logs loss per head (text + each codebook)
trainer.export_model("checkpoints/demo-export")
```
</details>

### Pretrained backbone (v6) — skip text pretraining

Instead of pretraining a text LM, mount a pretrained decoder as the temporal
backbone. Its tokenizer rides the text stream shifted by +64
(`--tokenizer hf:<model_id>`); new audio embeddings and the depth transformer are
grafted on. Training becomes two cheap stages:

1. **Align** — backbone frozen (`model.freeze_backbone=true`, the default), train only the audio modules.
2. **Finetune** — unfreeze at a low LR (`train.backbone_lr`) or use LoRA (`model.lora_rank=32`, needs `pip install 'omni[lora]'`).

Exports write `adapters.safetensors` only — the backbone is referenced by id, not copied.

```bash
# prepare with the backbone tokenizer (downloads the model on first use)
.venv/bin/python scripts/prepare_data.py asr --dataset openslr/librispeech_asr \
    --name clean --split train.100 --codec mimi --lang en \
    --tokenizer hf:Qwen/Qwen3-1.7B-Base --out data/shards/speech --preset qwen3-1.7b

# Stage 1: frozen backbone (a single GPU is fine at 1.7B)
.venv/bin/python scripts/train.py --preset qwen3-1.7b --data data/shards/speech \
    --export checkpoints/qwen3-s1

# Stage 2: unfreeze at low LR on the s2s + speech mixture
torchrun --standalone --nproc_per_node=8 scripts/train.py --preset qwen3-8b \
    model.freeze_backbone=false --data data/shards/s2s:0.7 --data data/shards/speech:0.3 \
    --export checkpoints/qwen3-s2
```

Chat, serve, and benchmark take the exported dir via `--ckpt` unchanged.

---

## Inference

```bash
# text -> speech
.venv/bin/python scripts/chat.py --task tts --text "hello" --out hello.wav \
    --ckpt checkpoints/demo-export --codec fake --tokenizer byte

# speech -> text
.venv/bin/python scripts/chat.py --task asr --in hello.wav \
    --ckpt checkpoints/demo-export --codec fake --tokenizer byte

# spoken reply, with an emotion register and a cloned voice
.venv/bin/python scripts/chat.py --task s2s --in question.wav --out reply.wav \
    --ckpt checkpoints/demo-export --codec fake --tokenizer byte \
    --emotion calm --lang en \
    --voice me.wav          # 10s reference wav pins the speaker voice across ALL languages

# full-duplex (mic and model on the same clock)
.venv/bin/python scripts/chat.py --task duplex --in user.wav --out assistant.wav \
    --ckpt checkpoints/duplex-export --codec fake --tokenizer byte
```

Use `--codec mimi` with real checkpoints. Voice cloning is a one-time ~6 ms
reference-audio prefill; per-frame decode is unchanged.

<details>
<summary>Same thing from Python</summary>

```python
import torch
from omni.audio.codec import FakeCodec, save_wav      # build_codec("mimi") for real runs
from omni.config import load_config
from omni.infer.generate import OmniGenerator
from omni.model.omni import OmniModel
from omni.streams import turn_prefix
from omni.text.tokenizer import ByteTokenizer

model = OmniModel.from_pretrained("checkpoints/demo-export")
cfg = load_config("tiny")
cfg.model = model.cfg                                 # keep checks coherent with weights
codec = FakeCodec(n_codebooks=model.cfg.n_codebooks)
gen = OmniGenerator(model, cfg, device="cpu", tokenizer=ByteTokenizer())

gen.set_voice(torch.randn(codec.sample_rate * 5).clamp(-1, 1), codec)   # reference wav
wav = torch.sin(torch.linspace(0, 500, codec.samples_per_frame * 10))   # your mic audio
reply = gen.s2s(wav, codec, seed=0,
                prefix_ids=turn_prefix(lang="en", response_style="calm"))
print(reply.text)                                     # the inner monologue (specials stripped)
save_wav("reply.wav", codec.decode(reply.audio_codes), codec.sample_rate)
```
</details>

---

## Streaming console

```bash
uv pip install -p .venv/bin/python -e ".[serve]"
.venv/bin/python scripts/serve.py --ckpt checkpoints/quality-export --codec mimi \
    --tokenizer data/tokenizer/omni_bpe.json      # or just --preset tiny for a wiring demo
# -> http://127.0.0.1:7860
```

Pick a task, talk or type, and watch the reply **stream**: audio plays as frames
are generated, the inner monologue appears live (control tags render as chips —
you see the model choose a language and emotion before it speaks), and a
per-frame budget meter tracks latency against the 80 ms real-time line
(TTFA / ms-per-frame / realtime-× readouts). Reference-voice and duplex modes
included. Local tool — binds to `127.0.0.1`, no auth.

---

## Status

Code-complete and contract-tested on CPU (264 tests, fully offline). A GPU box is
needed for the 48k tokenizer run, per-language codec validation, and the
Stage A→D training campaign. Remaining engineering (depth-compute amortization,
CFG decode, eval battery, synthesis QC, DPO) is tracked in the implementation
queues of [DESIGN_V3](docs/DESIGN_V3_AUDIO.md) and
[DESIGN_V4](docs/DESIGN_V4_EMOTION_I18N.md).
