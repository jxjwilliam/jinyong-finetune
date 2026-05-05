# 金庸风格 LoRA 微调应用 — 产品评估与视频生成指南
# Jin Yong LoRA Fine-Tune App — Review & Video Pipeline Guide

---

## 1. Pros and Cons

### Pros (优势)

| 优势 | 说明 |
|------|------|
| **高质量风格迁移** | 基于 Qwen2.5-7B-Instruct + QLoRA，能有效学习金庸典雅文风 |
| **完整的端到端流水线** | 从原始小说文本 → 清洗 → 指令数据集 → 训练 → 推理 → GGUF → Ollama，覆盖全流程 |
| **Typed pairs 多样性** | 支持 Claude / DeepSeek / Kimi / MiniMax / GLM 多 API 生成不同类型场景，避免单一风格 |
| **AutoDL 云训练优化** | 针对 RTX 4090 (24GB) 优化，`packing: false` 修复了 ChatML 格式问题 |
| **本地部署友好** | 支持 M1/M3 Mac 本地推理，GGUF 量化后仅需 ~5GB RAM |
| **视频管线文档完整** | `docs/JINYONG_VIDEO_PIPELINE.md` 提供了从段落 → 视频 prompt → NanoBanana 的完整流程 |
| **开源可扩展** | MIT 风格，脚本模块化，易于添加新功能 |

### Cons (劣势)

| 劣势 | 说明 |
|------|------|
| **需要 GPU 训练** | QLoRA 虽然降低要求，但仍需 24GB VRAM（4090）或云 GPU，本地 Mac 无法训练 |
| **视频管线脚本未实现** | `docs/JINYONG_VIDEO_PIPELINE.md` 中的 `generate_paragraph.py`、`pipeline.py` 等仅存在于文档中，未落地 |
| **数据集可能失衡** | sliding-window continuation 对可能远多于 typed pairs，导致模型倾向"续写"而非"指令跟随" |
| **无自动质量评估** | 没有脚本自动筛选高质量段落用于视频生成，需人工筛选 |
| **NanoBanana API 未验证** | 文档中 API 端点为占位符 (`nanobanana.io/v1/...`)，实际可用性未知 |
| **无 Web UI** | 纯 CLI 工具，非技术用户无法直接使用 |
| **模型仅支持中文** | 只能生成中文武侠内容，视频字幕/配音需额外处理 |

---

## 2. App Ranking / 应用评分

### 综合评分表

| 维度 | 评分 (1-5) | 说明 |
|------|------------|------|
| **技术架构** | 4.5 | QLoRA + TRL SFTTrainer + 模块化脚本，设计合理 |
| **易用性** | 3.0 | 需要命令行、Python、GPU 知识，非技术用户门槛高 |
| **输出质量** | 4.0 | 金庸风格迁移效果良好，但需更多 typed pairs 提升指令跟随 |
| **视频适配性** | 3.5 | 段落生成质量高，但视频转换脚本缺失 |
| **可扩展性** | 4.5 | 模块化设计，易于添加新生成器/新场景类型 |
| **文档完整性** | 4.0 | README + AGENTS.md + 多份 runbook，但视频脚本未落地 |
| **部署便捷性** | 4.0 | Ollama 部署简单，GGUF 量化成熟 |
| **成本效率** | 4.0 | AutoDL 4090 ~¥6/小时，训练 2 epoch 约 2-3 小时 |

### 总体评分：**3.9 / 5.0**

> **定位：** 适合开发者/技术用户的金庸风格内容生成工具，视频生成管线需进一步完善。

---

## 3. Comparison to Existing Similar Web Apps / 竞品对比

### 3.1 直接竞品

| 应用 | 类型 | 优势 | 劣势 | 对比本应用 |
|------|------|------|------|-----------|
| **NovelAI** | 商业 AI 写作平台 | 开箱即用 Web UI，多语言支持，图像生成集成 | 订阅制，无金庸专属微调，风格不可控 | 本应用更专注金庸风格，但 No-Code 体验差 |
| **Character.AI** | 角色对话 AI | 海量角色，对话体验好 | 不支持长文生成，无法导出用于视频 | 本应用生成的是可用于视频的段落，非对话 |
| **Claude / GPT-4** | 通用 LLM | 无需训练，直接 prompt 生成 | 风格不稳定，成本高，无专属微调 | 本应用风格更一致，推理成本极低（本地） |
| **文心一格 / 通义万相** | 中文 AI 图像生成 | 中文友好，操作简单 | 仅图像，无文字生成，无视频管线 | 本应用覆盖文字→视频全链路 |
| **即梦 AI (抖音)** | 中文 AI 视频生成 | 直接文生视频，中文友好 | 无金庸风格专属，无法定制模型 | 本应用可生成专属风格内容 |

### 3.2 差异化优势

本应用的核心差异化在于：
1. **专属金庸风格微调** — 不是通用 prompt，而是真正学习金庸文风的模型
2. **本地部署 + 零推理成本** — 训练完成后，本地 Ollama 推理无需 API 费用
3. **视频管线整合** — 从文字到视频 prompt 到 AI 视频生成的一站式流程（文档中）

---

## 4. New Features to Add & Areas for Improvement

### P0 — 必须实现

| 功能 | 说明 | 预估工作量 |
|------|------|-----------|
| **视频管线脚本落地** | 将 `docs/JINYONG_VIDEO_PIPELINE.md` 中的 `generate_paragraph.py`、`translate_to_prompt.py`、`pipeline.py` 实现为 `scripts/video/` 下的可运行脚本 | 2-3 天 |
| **段落质量评分器** | 用 Claude API 对生成段落评分（文风典雅、画面感、原创性），自动筛选 Top-K 用于视频 | 1 天 |
| **场景模板库** | 创建 `configs/video_scenes.json`，包含 50+  curated 场景描述，适配视频生成 | 0.5 天 |
| **NanoBanana API 验证** | 测试实际 API 端点，更新文档中的占位符 URL | 0.5 天 |

### P1 — 强烈建议

| 功能 | 说明 | 预估工作量 |
|------|------|-----------|
| **批量视频生成 CLI** | `python scripts/video/batch_generate.py --scenes configs/video_scenes.json --output-dir videos/` | 1-2 天 |
| **字幕生成 + 配音** | 集成 ElevenLabs 中文配音，生成 SRT 字幕文件 | 1 天 |
| **Web UI (Gradio)** | 简单的 Web 界面，输入场景描述 → 生成段落 → 一键转换视频 prompt | 2-3 天 |
| **数据集质量报告** | `python scripts/data/dataset_report.py --stats` 输出 continuation vs typed 比例、平均长度等 | 0.5 天 |
| **3 epoch 训练选项** | 当前 2 epoch 可能不足，添加 `--num-epochs` 参数到 CLI | 0.5 天（含测试） |

### P2 — 未来增强

| 功能 | 说明 |
|------|------|
| **多角色一致性** | 训练时加入角色描述，使生成段落中角色形象一致 |
| **RAG 检索增强** | 检索金庸原著相似场景作为 few-shot 示例 |
| **视频自动剪辑** | 集成 ffmpeg，自动拼接多个片段 + 添加转场 + BGM |
| **TikTok/YouTube 一键发布** | 集成平台 API，直接发布生成的视频 |
| **移动端 App** | React Native / Flutter App，调用云端推理 API |

---

## 5. Best Practices for Effective App Usage / 最佳实践

### 5.1 训练阶段

```bash
# 1. 始终保持 packing: false（已修复，不要改回）
#    packing=true 会导致模型忽略指令，只续写原文

# 2. 检查数据集比例
python scripts/data/build_instructions.py --dry-run --stats
# 确保 typed pairs 占比 > 30%，否则模型倾向续写而非指令跟随

# 3. 使用多个 API 生成 typed pairs，增加多样性
python scripts/gen/generate_typed_pairs.py claude --bucket claude --per-template 10
python scripts/gen/generate_typed_pairs.py openai --providers deepseek,kimi,minimax,glm --per-template 10

# 4. 训练后验证 —— 不要跳过
python scripts/infer/inference.py --config configs/qlora_config.yaml --prompt "以金庸风格描写一场雨夜对决，约200字"
# 检查是否：a) 不复制原著 b) 文风典雅 c) 有画面感
```

### 5.2 推理阶段（视频生成用）

```python
# 最佳实践：生成视频段落时的 system prompt
system_prompt = """
你是金庸风格的武侠小说作家。
每次只写一段，约150-200字。
文笔典雅，富有画面感，适合转化为视频场景。
不要写对话，只写景物、动作、氛围描写。
"""

# 场景描述要具体，包含：地点 + 时间 + 天气 + 动作
scene = "郭靖只身站在华山之巅，寒风呼啸，云海翻涌，手握弓箭，目光如炬。"
# ✅ 好：具体、视觉化
# ❌ 差："写一个武侠场景"（太抽象，生成内容不可控）
```

### 5.3 视频 Prompt 转换

```
中文段落 → 英文视频 prompt 的最佳转换模板：

[CHARACTER POSE/ACTION], [SETTING + TIME OF DAY], [LIGHTING],
[CAMERA MOVEMENT], [ATMOSPHERE], [STYLE: ancient China, wuxia aesthetic, cinematic, 4K, dramatic]
```

### 5.4 模型选择建议

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| 武打动作场景 | **Kling 3.0** | 动作连贯性最好，角色一致性高 |
| 山水/风景空镜 | **Veo 3.1 Quality** | 照片级真实感，光影效果出色 |
| 情感/剧情场景 | **Sora 2 Pro** | 电影级叙事感，细节丰富 |
| 快速预览/草稿 | **Veo 3.1 Fast** | 速度快，适合快速迭代 |

---

## 6. Recommended Tool Stack for Creating Short TikTok/YouTube Videos

### 6.1 完整工具栈

```
┌─────────────────────────────────────────────────────────────┐
│                    TikTok / YouTube Shorts                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  视频剪辑 & 发布                                          │
│  • CapCut（推荐）— 中文友好，自动字幕，TikTok 直发        │
│  • DaVinci Resolve — 专业调色，免费版够用                   │
│  • Adobe Premiere Pro — 行业标准，集成 After Effects       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  视频生成 AI                                              │
│  • NanoBanana (Kling 3.0 / Sora 2 Pro / Veo 3.1)        │
│  • Runway Gen-3 Alpha — 电影级质量，动作连贯                │
│  • Pika 2.0 — 快速迭代，成本低                            │
│  • Luma Dream Machine — 物理真实感强                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Prompt 翻译 & 优化                                        │
│  • Claude Sonnet 4 — 中文段落 → 英文视频 prompt           │
│  • GPT-4o — 备选方案                                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  段落生成 (本应用)                                         │
│  • Qwen2.5-7B + Jin Yong LoRA adapter                     │
│  • Ollama (本地) 或 AutoDL (云端推理)                      │
│  • scripts/infer/inference.py                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  音频：配音 & 背景音乐                                     │
│  • ElevenLabs — 中文配音，支持克隆声音                     │
│  • 剪映/CapCut 内置配音 — 免费，中文效果好                 │
│  • YouTube Audio Library — 免费 BGM                        │
│  • Epidemic Sound — 付费，版权清晰                         │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 推荐工作流（最低成本）

| 步骤 | 工具 | 成本 |
|------|------|------|
| 1. 生成段落 | 本地 Ollama (Qwen2.5-7B + LoRA) | **免费** |
| 2. 翻译 Prompt | Claude API (Sonnet 4) | ~$0.01/段 |
| 3. 生成视频 | Kling 3.0 via NanoBanana | ~$0.50/段 (5s) |
| 4. 添加字幕/配音 | CapCut (内置功能) | **免费** |
| 5. 剪辑/发布 | CapCut | **免费** |

**单条视频总成本：~$0.50-1.00**

### 6.3 推荐工作流（最佳质量）

| 步骤 | 工具 | 成本 |
|------|------|------|
| 1. 生成段落 | 本地 Ollama | 免费 |
| 2. 翻译 Prompt | Claude Sonnet 4 | ~$0.01/段 |
| 3. 生成视频 | Sora 2 Pro via NanoBanana | ~$2.00/段 (10s) |
| 4. 配音 | ElevenLabs (中文 Multi-lingual v2) | ~$0.30/段 |
| 5. 专业剪辑 | DaVinci Resolve + 手动调色 | 时间成本 |
| 6. BGM | Epidemic Sound | $15/月 |

**单条视频总成本：~$2.50 + 订阅费**

### 6.4 快速启动脚本（推荐）

```bash
# 安装依赖
pip install openai anthropic requests

# 一键生成视频（待实现，基于 docs/JINYONG_VIDEO_PIPELINE.md）
python scripts/video/pipeline.py \
  --scenes configs/video_scenes.json \
  --model kling_3 \
  --output-dir output_videos/ \
  --add-narration \
  --narration-voice "chinese_male"
```

---

## 附录：文档版本

- **创建日期：** 2026-05-05
- **基于仓库版本：** `5bfce79` (fix: tune learning rate and remove modules_to_save; add inference and GGUF conversion scripts)
- **相关文档：**
  - `docs/JINYONG_VIDEO_PIPELINE.md` — 视频生成管线详细文档
  - `docs/LORA_TO_GGUF_GUIDE.md` — GGUF 转换与 Ollama 部署
  - `docs/TYPED_PAIRS_PIPELINE.md` — Typed pairs 生成管线
  - `docs/autoDL.md` — AutoDL 云训练 runbook
