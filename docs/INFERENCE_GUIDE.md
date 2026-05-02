# 金庸微调模型推理指南
# Jinyong LoRA Fine-tune: Inference Guide

> Fine-tuned model: `Qwen2.5-7B-Instruct` + LoRA adapter (`outputs/jinyong-qlora/adapter`)  
> Covers two environments: **Local MacBook M3 Pro** and **Cloud AutoDL (China)**

---

## 1. Output File Structure

```
outputs/jinyong-qlora/adapter/
├── adapter_config.json         # LoRA hyperparams (rank, alpha, target modules)
├── adapter_model.safetensors   # Fine-tuned delta weights (~100–300MB)
├── added_tokens.json           # Any new tokens added during training
├── merges.txt                  # BPE merge rules (tokenizer)
├── README.md.gz                # Auto-generated model card
├── special_tokens_map.json     # EOS/BOS/PAD token mapping
├── tokenizer.json              # Full tokenizer (load from here, not HuggingFace)
├── tokenizer_config.json       # Chat template + tokenizer settings
└── vocab.json                  # Vocabulary
```

> ✅ The adapter folder contains the **complete tokenizer** — no need to download from HuggingFace for tokenization.

---

## 2. Environment Comparison

| Factor | MacBook M3 Pro (Canada) | AutoDL Cloud (China) |
|--------|------------------------|----------------------|
| Hardware | Apple Silicon MPS | NVIDIA GPU (CUDA) |
| 4-bit quantization | ❌ bitsandbytes not supported | ✅ Fully supported |
| HuggingFace access | ✅ Direct (VPN or mirror) | ❌ Blocked, use cache/mirror |
| ModelScope access | ✅ | ✅ Recommended |
| RAM / VRAM | Unified memory (16–96GB) | Dedicated VRAM (24GB typical) |
| Base model loading | float16 (~14GB RAM) | 4-bit quantized (~5GB VRAM) |
| Inference speed | Moderate (MPS) | Fast (CUDA) |
| Best use case | Development, testing | Training, production inference |

---

## 3. MacBook M3 Pro Setup (Local, China)

### 3.1 Prerequisites

```bash
# Check Python version (3.10+ recommended)
python --version

# Install dependencies
pip install torch torchvision torchaudio  # MPS backend included in PyTorch >= 2.0
pip install transformers peft accelerate
pip install sentencepiece protobuf
```

### 3.2 Environment Variables (Cell 1 — always first)

```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # suppress fork warning
# Optional: point to HF mirror if not using VPN
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

### 3.3 Device Detection

```python
import torch

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")  # Expected: mps
```

### 3.4 RAM Check Before Loading

Qwen2.5-7B in float16 requires ~14GB RAM. Check first:

```python
import subprocess
result = subprocess.run(["sysctl", "hw.memsize"], capture_output=True, text=True)
ram_gb = int(result.stdout.split(":")[1].strip()) / 1e9
print(f"Total RAM: {ram_gb:.0f} GB")
# 16GB: tight but possible | 32GB+: comfortable
```

### 3.5 Load Model + Adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

adapter_path   = "./outputs/jinyong-qlora/adapter"   # adjust to your local path
base_model_id  = "Qwen/Qwen2.5-7B-Instruct"          # downloads from HF (needs internet)

# ✅ Load tokenizer from adapter folder — no network needed
tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ✅ Load base model — NO quantization on Mac (bitsandbytes is CUDA-only)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,   # float16 works well on M3
    device_map="cpu",            # load to CPU first
    trust_remote_code=True,
)

# Attach LoRA adapter and move to MPS
model = PeftModel.from_pretrained(model, adapter_path)
model = model.to(device)
model.eval()
print(f"✅ Model loaded on {device}")
```

> **Tip:** If you already downloaded the base model previously, find its local cache path:
> ```python
> import os
> cache = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots")
> snapshot = os.listdir(cache)[0]
> base_model_local = os.path.join(cache, snapshot)
> # Then use base_model_local instead of "Qwen/Qwen2.5-7B-Instruct"
> ```

### 3.6 Inference

```python
messages = [
    {"role": "system", "content": "你是金庸小说里的郭靖，用郭靖的口吻和性格回答问题。"},
    {"role": "user",   "content": "你如何看待江湖恩怨？"}
]

text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)
print(response)
```

---

## 4. AutoDL Cloud Setup (China)

### 4.1 Network Strategy

AutoDL blocks HuggingFace. Use one of these approaches:

**Option A — Offline mode (base model already cached):**
```python
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

**Option B — HF Mirror (hf-mirror.com, accessible from AutoDL):**
```python
import os
os.environ["HF_ENDPOINT"]           = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

**Option C — ModelScope (most reliable, recommended):**
```python
# pip install modelscope
from modelscope import snapshot_download
model_dir = snapshot_download(
    'qwen/Qwen2.5-7B-Instruct',
    cache_dir='/root/autodl-tmp/models'
)
# Use model_dir as your base_model_id below
```

### 4.2 Check Cache Exists (Offline mode only)

```python
import os
cache = os.path.expanduser("~/.cache/huggingface/hub")
for d in os.listdir(cache):
    if "Qwen" in d:
        print(d)  # expect: models--Qwen--Qwen2.5-7B-Instruct
```

### 4.3 Load Model + Adapter (with 4-bit quantization)

```python
import os
os.environ["TRANSFORMERS_OFFLINE"]  = "1"
os.environ["HF_HUB_OFFLINE"]        = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

assert torch.cuda.is_available(), "No CUDA GPU detected!"

adapter_path   = "./outputs/jinyong-qlora/adapter"
base_model_id  = "Qwen/Qwen2.5-7B-Instruct"  # or local path from ModelScope

# ✅ 4-bit quantization — CUDA only, saves ~9GB VRAM vs float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()
print("✅ Model loaded with 4-bit quantization")
```

### 4.4 Inference (same as Mac, device handled by device_map="auto")

```python
messages = [
    {"role": "system", "content": "你是金庸小说里的郭靖，用郭靖的口吻和性格回答问题。"},
    {"role": "user",   "content": "你如何看待江湖恩怨？"}
]

text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(text, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)
print(response)
```

---

## 5. Optional: Merge Adapter into Base Model

If you want a single standalone model (no PEFT dependency, faster loading):

```python
# ⚠️ Requires enough RAM/VRAM to hold full float16 model
# Best done on AutoDL where VRAM is sufficient

merged_model = model.merge_and_unload()
merged_model.save_pretrained("./outputs/jinyong-merged")
tokenizer.save_pretrained("./outputs/jinyong-merged")
print("✅ Merged model saved")

# Load merged model later (no PeftModel needed):
# model = AutoModelForCausalLM.from_pretrained("./outputs/jinyong-merged", ...)
```

---

## 6. Workflow Summary

```
Training (AutoDL) ──► adapter/ folder ──┬──► AutoDL inference  (4-bit, CUDA)
                                         └──► Mac M3 inference   (float16, MPS)
                                         └──► Merge & export ──► llama.cpp / GGUF
```

### Recommended workflow by use case:

| Goal | Where | Method |
|------|-------|--------|
| Quick test after training | AutoDL | 4-bit + CUDA |
| Development & prompting | Mac M3 | float16 + MPS |
| Production serving | AutoDL or any Linux GPU | Merged model or vLLM |
| Mobile / CPU deployment | Mac M3 | Export to GGUF via llama.cpp |

---

## 7. Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `AttributeError: 'LoraConfig' has no attribute 'bnb_config'` | `PeftConfig` doesn't store BnB config | Define `BitsAndBytesConfig` manually |
| `AssertionError: torch.cuda.is_available()` | Running CUDA code on Mac | Use `mps` device, remove `assert cuda` |
| `Network is unreachable huggingface.co` | AutoDL blocks HF | Set `TRANSFORMERS_OFFLINE=1` or use ModelScope |
| `TOKENIZERS_PARALLELISM` warning | Jupyter fork after tokenizer init | Set `os.environ["TOKENIZERS_PARALLELISM"] = "false"` |
| OOM on Mac (16GB) | Model too large for RAM | Use float32 → actually worse; better: convert to GGUF |
| Slow inference on MPS | MPS not optimized for all ops | Expected; use AutoDL for speed |

---

## 8. Resources

- Source code: https://github.com/jxjwilliam/jinyong-finetune
- Dataset: https://www.kaggle.com/datasets/evilpsycho42/jinyong-wuxia
- Training platform: https://www.autodl.com
- HF Mirror (China): https://hf-mirror.com
- ModelScope (China): https://modelscope.cn/models/qwen/Qwen2.5-7B-Instruct
