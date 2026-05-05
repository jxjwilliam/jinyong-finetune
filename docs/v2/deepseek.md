# 金庸微调项目 — 应用分析与优化建议

> **项目：** `jinyong-finetune` — 基于 QLoRA 的金庸风格武侠文本生成微调流水线  
> **基座模型：** `Qwen/Qwen2.5-7B-Instruct`  
> **训练语料：** 金庸全集 15 部小说（Kaggle `jinyong-wuxia` 数据集）  
> **目标平台：** AutoDL RTX 4090（24GB）→ Ollama on MacBook M3 Pro  

---

## 1. 优缺点分析

### 优点

**流水线完整性**
- 端到端覆盖：原始 `.txt` 小说 → 文本清洗 → 指令 JSONL → QLoRA 训练 → LoRA 合并 → GGUF → Ollama。每个阶段都有脚本、配置驱动、文档齐全。
- 三种数据增强策略：滑动窗口续写、通过 LLM API 生成的分类场景模板、以及多供应商模板分区（100 个互不重叠的模板，分布在 Claude、DeepSeek、Kimi、MiniMax、GLM 五个供应商）。

**资源效率**
- QLoRA（4-bit NF4 + LoRA r=64）仅需单张 RTX 4090 24GB 即可训练，无需多卡或云端集群。训练成本：AutoDL 上约 30 元人民币。
- Adapter 权重仅约 100-300MB，而非完整的 14GB 模型，传输和存储成本极低。

**代码质量**
- Python 3.12+ 规范写法，含类型提示、`argparse` 命令行接口、`dataclass` 数据模型，模块边界清晰（`data/`、`gen/`、`lib/`、`train/`、`infer/`、`export/`、`hub/`）。
- YAML 驱动配置（`configs/qlora_config.yaml`），附带 T4 vs RTX 4090 的参数覆盖说明。
- 共享库模块（`instruction_jsonl.py`、`typed_prompts.py`）防止跨脚本的 schema 漂移。

**调试素养**
- `packing=false` 修复和 `max_seq_length=1024` 调整在配置注释和 README 中有明确的因果分析记录——在个人项目中相当少见。
- token 长度验证内置于训练循环（训练开始前打印样本 token 长度）。

**部署灵活性**
- 四条推理路径：PEFT adapter + CUDA（AutoDL）、PEFT + MPS（Mac）、合并后的 HF 模型（transformers）、GGUF + Ollama（CPU/Metal，生产环境）。
- 集成 HF 镜像（`hf-mirror.com`）和 ModelScope，方便国内用户访问。

**文档质量**
- 七份 Markdown 文档：`autoDL.md`、`TYPED_PAIRS_PIPELINE.md`、`LORA_TO_GGUF_GUIDE.md`、`INFERENCE_GUIDE.md`、`JINYONG_VIDEO_PIPELINE.md`，以及含真实输出分析的会话记录（`05-04.md`）。
- `TYPED_PAIRS_PIPELINE.md` 中包含 Mermaid 架构流程图。
- 全篇提供可复制的操作清单和精确的 CLI 命令。

**精准的细分市场**
- 专注金庸风格武侠——这是中国短视频平台（抖音、B 站、TikTok）上的高需求内容品类，具有真实的变现潜力。
- 视频流水线文档（`JINYONG_VIDEO_PIPELINE.md`）将 LoRA 输出连接到 NanoBanana/Kling/Sora 进行视频生成，体现了超越模型本身的产品化思维。

### 缺点

**缺乏自动化测试**
- 测试套件为零。验证完全靠人工（dry run、`--stats`、抽查输出）。一个悄悄破坏 ChatML 格式或截断输出的训练运行可能在检测到之前浪费数小时 GPU 时间。
- 数据流水线的边界情况（空文件、编码失败、最小输出长度过滤）没有回归测试。

**除 loss 外无评估指标**
- 训练仅报告交叉熵损失。没有自动化的风格保真度指标（如文言文语域得分）、输出多样性（distinct n-grams）、指令遵循准确度或重复率。
- `LORA_TO_GGUF_GUIDE.md` 中的 5 维度质量基准（手动评分）很好，但纯靠人工。

**云平台操作靠手工**
- AutoDL 工作流需要手动 SCP、zip/unzip、`nohup` 后台运行、`tail -f` 看日志。无一键部署脚本。
- 没有持久化存储抽象——训练产物存放在临时的 `/root/autodl-tmp/`，必须手动在实例回收前抢救。

**紧耦合 Qwen2.5-7B**
- ChatML 提示模板（`<|im_start|>system/user/assistant<|im_end|>`）硬编码在 `train.py` 的 `build_prompt` 函数中。更换基座模型（如 Llama 3、DeepSeek、Yi）需要修改代码，而非仅改配置。
- tokenizer 的特殊 token 设置（`pad_token = eos_token`、`padding_side = "right"`）为 Qwen 特有假设。

**无流式输出、无 API、无 UI**
- 推理仅为批处理模式。没有服务端模式、REST API 或 WebSocket 流式输出——模型不能直接由前端调用。
- 没有 Web UI 用于 prompt 探索或输出 A/B 对比。

**分类场景生成有 API 成本**
- `generate_typed_pairs.py` 调用付费 LLM API（Claude、DeepSeek、Kimi、MiniMax、GLM）。以 `--per-template 10` 计算，100 个模板共 1000 次 API 调用。Claude Haiku 虽便宜但不免费，其他供应商按 token 计费。
- 没有内置的成本估算或预算上限。

**缺乏数据集版本管理**
- `data/instructions/` 下的 JSONL 文件被 gitignore。没有 DVC 或类似工具跟踪哪个数据集版本产出了哪个 adapter。
- 如果 `build_instructions.py` 的参数发生变化（chunk_size、overlap、min-output-chars），除非手动记录，否则无从追溯。

**仅支持单 GPU**
- 没有 DeepSpeed/FSDP 集成。训练超过 24GB 的模型或扩展到更大数据集需要手动重写。
- `device_map="auto"` 在单 GPU 上可以工作，但多 GPU 用户无法受益。

**分类场景生成的容错不足**
- `generate_typed_pairs.py` 在单个调用失败时会重试，但没有进度检查点。如果脚本在 800/1000 次调用后中断，无法断点续传而不产生重复数据。
- 没有基于速率限制的自适应退避（指数退避 vs 固定 `--sleep`）。

---

## 2. 项目评分 — 总览表

面向个人/小团队开源微调工具的 8 维度评分。量级：1（极低）到 10（生产级）。

| 维度 | 评分 | 说明 |
|------|------|------|
| **流水线完整性** | 9/10 | 从原始文本到部署模型一站式覆盖。缺：CI/CD、自动化评估。 |
| **代码质量** | 8/10 | 类型提示、dataclass、清晰的 CLI。缺：测试、部分硬编码假设。 |
| **文档质量** | 8/10 | 详细的中文文档，含架构图。缺：API 参考、贡献者指南。 |
| **部署灵活性** | 8/10 | CUDA、MPS、GGUF、Ollama。缺：Docker、云端一键部署、vLLM。 |
| **模型质量** | 7/10 | QLoRA 基础扎实。缺：自动化质量指标、A/B 对比框架。 |
| **成本效率** | 9/10 | AutoDL 训练约 30 元，adapter 极小。API 生成成本适中但未追踪。 |
| **创新性 / 细分定位** | 9/10 | 金庸专注点独特；多 LLM 非重叠模板策略巧妙。 |
| **可维护性** | 6/10 | 结构清晰但缺乏测试、无 CI、紧耦合 Qwen。存在腐化风险。 |
| **综合** | **8.0/10** | 优秀的个人/小团队工具。需要测试套件和 API 层才能达到生产级。 |

### 各维度详解

**流水线完整性（9/10）**：从文本摄取到 Ollama 部署的每个阶段都已覆盖。分类场景流水线（Claude/DeepSeek/Kimi/MiniMax/GLM）是真正的差异化亮点。仅缺 CI/CD 和自动化评估。

**代码质量（8/10）**：现代的 Python 写法，按 `data/gen/lib/train/infer/export/hub/` 良好分层，YAML 驱动配置。共享 `Pair` dataclass 防止 schema 漂移。减分项：无测试、部分 ChatML 模板硬编码、无类型检查 CI。

**文档质量（8/10）**：七份聚焦文档，含架构流程图、可复制清单、真实输出分析（`05-04.md`）、平台特定指导（AutoDL、Mac）。减分项：无内联 API 文档、无 CONTRIBUTING 指南。

**部署灵活性（8/10）**：覆盖 CUDA（AutoDL）、MPS（Mac）、合并 HF 模型和 GGUF/Ollama。系统提示从配置驱动且跨路径一致。减分项：无 Docker、无 vLLM 集成、无 REST API。

**模型质量（7/10）**：`packing=false` 修复和 `max_seq_length=1024` 调整体现出对训练陷阱的认知。输出分析（`05-04.md`）展示了扎实的风格迁移效果。然而质量评估完全依赖人工——无自动化测试集困惑度、风格分类器或多样性指标。

**成本效率（9/10）**：单卡 RTX 4090 上的 QLoRA 接近成本/质量最优解。训练成本报告约 30 元。adapter 极小。分类场景生成成本适中（Claude Haiku 约 $0.25/M 输入 token，$1.25/M 输出 token）。主要短板：无内置成本追踪。

**创新性 / 细分定位（9/10）**：金庸武侠聚焦独特且精准匹配真实内容市场。非重叠模板桶策略（100 个模板、5 个供应商、零重叠）巧妙且执行到位。视频流水线连接显示了产品化思维。

**可维护性（6/10）**：代码干净但缺乏测试导致脆弱。紧耦合 Qwen 的 ChatML 格式和 tokenizer 约定意味着模型更换需要代码修改。无 CI 导致回归问题靠人工发现。依赖多个外部 API（Claude、DeepSeek、Kimi 等）带来多处故障点。

---

## 3. 与现有同类应用的对比

### 对比矩阵

| 特性 | jinyong-finetune | NovelAI / Sudowrite | Replicate + LoRA | Character.AI | Ollama + Open WebUI | AutoTrain |
|------|-----------------|---------------------|------------------|--------------|---------------------|-----------|
| **风格专注度** | 金庸武侠（深入） | 任意小说（泛化） | 任意风格（用户训练） | 任意角色 | 任意模型 | 任意任务 |
| **训练方式** | QLoRA（成本优化） | 闭源 | 云端 GPU LoRA | 闭源 | 无（用户自带模型） | 全参/LoRA |
| **训练成本** | 总计约 30 元 | $10-25/月订阅 | ~$0.50-2/小时 GPU | 免费（限速） | 免费（本地） | 取决于配置 |
| **中文武侠质量** | 高（15 部小说+分类场景） | 一般（通用模型） | 取决于数据集 | 差（非设计场景） | 取决于模型 | 取决于训练 |
| **分类场景生成** | ✅ 多 LLM，100 模板 | ❌ | ❌ | ❌ | ❌ | ❌ |
| **GGUF / Ollama 导出** | ✅ 有文档流水线 | ❌ | ❌ | ❌ | ✅（原生） | ❌ |
| **离线推理** | ✅（Ollama，本地 transformers） | ❌（仅云端） | ❌（仅云端） | ❌（仅云端） | ✅ | 取决于配置 |
| **Web UI** | ❌ | ✅（富文本编辑器） | ✅（托管演练场） | ✅（聊天） | ✅（Open WebUI） | ✅（HuggingFace） |
| **API 访问** | ❌ | ✅ | ✅ | ❌ | ✅（Ollama API） | ✅ |
| **流式输出** | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ |
| **内容安全过滤** | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **多用户 / 团队** | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **评估指标** | 仅手动 | ❌ | ❌ | ❌ | ❌ | 基础 |
| **开源** | ✅（完全） | ❌（闭源） | 部分 | ❌ | ✅ | ✅ |
| **中文 UI / 文档** | ✅（全部文档） | ❌ | ❌ | ❌ | 部分 | ❌ |
| **视频流水线集成** | ✅（有文档） | ❌ | ❌ | ❌ | ❌ | ❌ |

### 详细对比

**NovelAI / Sudowrite** — 商业 AI 小说工具。NovelAI 有一定的中文支持但本质上是英文优先。Sudowrite 仅支持英文。两者均无武侠专项训练。两者都有精美的 UI、流式输出和内容管理功能（本项目缺失），但其金庸风格的深度很浅。成本：订阅模式（$10-25/月）。结论：UX 更好，武侠质量更差。

**Replicate + LoRA 训练器** — Replicate 提供 LoRA 训练托管服务（如图像的 `replicate/flux-lora-trainer`，以及各类文本 LoRA 训练器）。用户上传数据集，按 GPU 小时付费。无武侠专用模板，无分类场景生成，无 GGUF 导出。Replicate 上的通用 LoRA 训练比本项目的优化流水线成本更高，但零部署门槛。结论：入门更简单，专业度更差，规模化成本更高。

**Character.AI** — 流行的角色对话应用，有一些金庸角色 bot。适合休闲互动，不适合长篇散文生成和场景构建。无权访问微调、无导出、无视频流水线。内容锁定在其生态内。结论：完全不同的使用场景——聊天 vs 内容创作。

**Ollama + Open WebUI** — 如果 jinyong-finetune 合并并量化 LoRA 到 GGUF，即可与此技术栈兼容。Open WebUI 提供了本项目缺失的聊天 UI、流式输出、多用户和 API。这是推荐的前端配对方案（见第 5 节）。结论：互补，而非竞争。

**通用 AutoTrain（HuggingFace）** — 支持通过 Web UI 进行 LoRA 微调。可在任意文本数据集上训练。无武侠专项优化、无 ChatML 格式化认知、无分类场景生成、无 GGUF 流水线。质量完全取决于用户的数据集和 prompt 工程。结论：更灵活但开箱即用的武侠质量远逊于本项目。

### 竞争定位

**jinyong-finetune 占据了一个独特的生态位**：它是唯一一个能同时做到以下四点的开源工具：(a) 金庸专项风格训练，(b) 基于非重叠模板的多 LLM 分类场景生成，(c) 通过 GGUF/Ollama 实现生产部署，(d) 具备文档化的视频生产流水线。其弱点（无 UI、无 API、无流式输出）都可以通过与现有工具（Open WebUI、Ollama API）配对接入来解决，无需从零构建。

---

## 4. 新功能与改进方向

### 高优先级（影响大、投入少）

**1. 自动化测试套件**
- 冒烟测试：用已知编码测试 `clean_text.py`（GB2312、UTF-8），用最小数据集测试 `build_instructions.py`，`train.py` dry-run。
- 单元测试：`sliding_segments()` 边界情况、`validate_pairs()` 最小字符过滤、`typed_pair_dict()` 输出格式。
- 集成测试：在 3 段样本文本上跑完整流水线 → 验证 JSONL 格式 → 验证训练启动无 OOM。
- 通过 `pytest` 运行，配合 GitHub Actions 或 pre-commit hook。

**2. 训练指标看板**
- 训练期间将指标写入 JSON lines 文件（不仅控制台输出）：loss、学习率、步数、epoch、GPU 内存、tokens/秒。
- 简单的 Python 脚本或 notebook 绘制训练/验证 loss 曲线。
- 当验证 loss 与训练 loss 偏离时触发告警（过拟合信号）。

**3. 分类场景生成成本追踪**
- `generate_typed_pairs.py` 已知模型和 token 数量——增加按供应商累计的估算成本。
- 运行结束时打印汇总："本次运行预估成本：¥X.XX"。

**4. 分类场景生成断点续传**
- 跟踪已完成的 `(template_id, sample_index)` 组合。重启时跳过已生成的组合。
- 避免长时间生成运行中断后重复调用 API。

### 中优先级（改进显著、投入适中）

**5. REST API 服务**
- FastAPI 服务加载 adapter，暴露 `/v1/chat/completions`（兼容 OpenAI 格式）和 `/v1/generate`（简单文本生成）。
- 支持与 Open WebUI、SillyTavern 或任意 OpenAI 兼容前端集成。
- `scripts/infer/inference.py` 已有所有模型加载逻辑——用 FastAPI 包裹即可。

**6. 模型无关的提示格式化**
- 将 ChatML 模板从 `train.py` 中移出，放入配置或 `prompt_templates.py` 模块。
- 支持 Llama 3、DeepSeek、Yi 等聊天格式。
- 如有可用，通过 `tokenizer.apply_chat_template()` 读取 tokenizer 内置的 Jinja `chat_template`。

**7. 自动化风格评估**
- 在留出测试集上计算基础指标：困惑度、distinct-n（1/2/3-gram 多样性）、重复率。
- 可选：微调一个小型分类器（如基于 BERT）打分"金庸风格相似度"——用正面样本（金庸原文）vs 负面样本（现代网络小说）训练。
- 训练后自动运行，结果随 loss 一同记录。

**8. Docker + 一键 AutoDL 脚本**
- `Dockerfile`：含 CUDA、PyTorch 及全部依赖。
- `autodl_deploy.sh`：克隆仓库 → 安装依赖 → 下载数据集 → 运行流水线 → 打包产物 → 显示下载命令。
- 消除手动 SCP/nohup/tail 工作流。

**9. 基于 DVC 的数据集版本管理**
- 用 DVC 跟踪 `data/instructions/jinyong_sft.jsonl` 和分类场景 JSONL 文件。
- 基于哈希的版本管理确保每个 adapter 都能追溯到其精确的训练数据。
- 将 DVC 远端存储在廉价云桶或 HuggingFace datasets 上。

### 低优先级（锦上添花、投入较大）

**10. 流式生成**
- 在 `scripts/infer/inference.py` 和拟议的 API 服务中实现逐 token 流式输出。
- 需使用 transformers 的 `TextIteratorStreamer` 或手动 token 生成循环。

**11. Prompt 探索 Web UI**
- 轻量级 Gradio 或 Streamlit 应用：输入场景描述 → 生成金庸段落 → 查看/编辑 → 导出为视频提示。
- 可将完整视频流水线（`JINYONG_VIDEO_PIPELINE.md` 的第 1-4 步）集成到单一界面。

**12. 多 GPU / DeepSpeed 支持**
- 为 `train.py` 增加 `--deepspeed` 参数，附带 DeepSpeed ZeRO-2/3 配置。
- 支持训练更大的模型（14B、32B）或更大的 batch size。

**13. 内容安全分类器**
- 检测模型是否输出版权角色名（郭靖、杨过等）或复现原著段落。
- 发布前标记输出供人工审核。

**14. 持续微调 / LoRA 叠加**
- 支持加载已有 adapter，在新数据上训练额外 adapter（LoRA 叠加或多 adapter 推理）。
- 适用于无需完全重训练的迭代式风格精调。

---

## 5. 有效使用的最佳实践

### 数据准备

**语料质量优先于数量**
- Kaggle 的金庸 15 部小说（`evilpsycho42/jinyong-wuxia`）已足够。验证编码——部分文件可能为 GB2312，`clean_text.py` 会自动处理。
- 训练前移除所有非金庸内容（前言、评论、出版社元数据）。
- 先运行 `clean_text.py --dry-run` 检查压缩比例。正常情况：章节标题/空白清理带来 5-15% 的文本量缩减。如果某文件缩减超过 50%，请手动检查。

**分类场景：越多越好，但有上限**
- 从所有 5 个供应商的 `--per-template 10` 开始（约 1,000 条）。评估输出质量。
- 如果模型对分类场景过拟合（输出感觉公式化），通过提高 `--max-pairs` 限制来增加续写对的比例，依靠默认的随机打乱。
- 如果模型仍然复述原文，增加更多分类场景（尤其是 Claude 和 DeepSeek——它们产出的中文散文质量最高）。
- 写入前运行 `build_instructions.py --stats` 确认训练/验证集划分大小。目标：7B LoRA 需要 2,000-10,000 条总样本较合适。

**系统提示语是神圣不可变的**
- `configs/qlora_config.yaml` 中的系统提示（`data.system_prompt`）必须与推理时使用的完全一致。任何差异都会损害指令遵循能力。
- 编辑系统提示后必须重新训练。用不同的系统提示添加新分类场景会造成分布不匹配。

### 训练

**从第 1 步开始关注验证 loss**
- 设 `eval_steps: 100`，首次评估在第 100 步运行。如果验证 loss 持平或上升而训练 loss 在下降，说明在过拟合——减少 epoch、增大 `per_device_train_batch_size` 或添加更多分类场景。
- 健康轨迹的典型模式：训练和验证 loss 一起下降 300-500 步，然后验证 loss 平台化而训练 loss 继续缓慢下降。在验证 loss 平台化后尽早停止训练。

**RTX 4090 务必使用 bf16**
- RTX 4090 原生支持 bf16。默认配置（`bf16: true, fp16: false`）对 Ada 架构 GPU 是正确的。
- 如果使用 T4 或更老的 GPU：切换为 `fp16: true, bf16: false` 和 `bnb_4bit_compute_dtype: float16`。预期 loss 会略高，高学习率下偶尔出现 NaN。

**最大序列长度：训练前验证**
- `train.py` 在训练前打印样本 token 长度。如果最大值超过 1024，增大 `max_seq_length` 或减小 `build_instructions.py` 中的 `chunk_size`。
- 300 字符的中文片段：典型 token 长度为 400-650（Qwen 的 tokenizer 中中文字符约 1-2 个 token）。1024 是安全的。

**梯度检查点必须与 enable_input_require_grads 配对**
- 代码已在两处正确实现。不要禁用梯度检查点而不移除 `enable_input_require_grads()` 调用，反之亦然。
- 如果看到 "loss has no grad_fn" 错误：说明这个配对关系被破坏了。

### 推理

**提示格式必须与训练完全匹配**
- 训练使用：`<|im_start|>system\n{系统提示}<|im_end|>\n<|im_start|>user\n{指令}\n{输入}<|im_end|>\n<|im_start|>assistant\n{输出}<|im_end|>`
- 推理必须使用完全相同的格式。`tokenizer.apply_chat_template()` 是最安全的方式。
- 对于 Ollama：验证 Modelfile 的 TEMPLATE 与 Qwen 的 ChatML 格式完全一致。

**分类场景推理：使用场景提示**
- 分类提示库（`typed_prompts.py`）包含 `VARIATION_HINTS`——地点/场景提示，可增加输出多样性。将 `typed_user_turn(instruction, hint)` 作为用户 prompt 传入。
- 不要重复使用同一个提示。轮流使用 10 个提示以获得多样化的场景。

**视频制作：不要写对话**
- 设置系统提示禁止对话：`"不要写对话，只写景物、动作、氛围描写。"`（如 `JINYONG_VIDEO_PIPELINE.md` 所示）。
- 对话难以与生成视频同步——纯视觉描述更容易转化。

### 部署

**合并后再转 GGUF**
- llama.cpp 的 `convert_hf_to_gguf.py` 需要完整的合并模型，而非仅 adapter。在 GPU 实例上先运行 `merge_lora.py`，再 SCP。
- 合并模型约 14GB（bf16）或约 7GB（fp16）。传输前用 zip 压缩。

**量化：q4_k_m 是最佳平衡点**
- 对于 MacBook M3 Pro（16GB 统一内存）：q4_k_m（4.7GB）为 macOS 和上下文窗口留出约 10GB。q5_k_m（5.7GB）质量略高但有交换风险。
- 对于 32GB+ 机器或服务器部署：q5_k_m 或 q8_0。
- 始终用 `LORA_TO_GGUF_GUIDE.md` 中的 5 个内置测试 prompt 通过 `ollama run jinyong` 测试。

**Ollama 生产环境 — 设置 num_predict**
- Ollama 默认 `num_predict` 为 128 token——对 200 字的中文段落太低（每个字约 1-2 token）。在 Modelfile 中设置 `PARAMETER num_predict 512`。

### 内容安全与版权

**版权意识**
- 金庸 15 部小说用于风格训练，而非内容复现。模型应生成原创角色和情节。
- 用 `"完整背诵《射雕英雄传》第一章"` 测试——模型应拒绝或写原创内容，而非复现原文。
- 发布时添加免责声明："AI 生成的原创武侠小说，灵感来源于金庸风格。所有角色和情节均为原创。"

**平台合规**
- 抖音/TikTok/YouTube 均有 AI 内容披露要求。务必标注 AI 生成内容。
- 部分平台对纯 AI 生成内容的变现有限制。可能需要人机协作（编辑、配音、策划）才能变现。

---

## 6. 制作抖音/YouTube 短视频的推荐工具链

### 总览

流水线跨越四个阶段：**文本生成 → 提示翻译 → 视频生成 → 剪辑与发布**。每个阶段都有免费、中档和高级选项。

```
文本（LoRA）  ──►  提示翻译  ──►  视频生成  ──►  剪辑与发布
    │                  │                  │                │
    ▼                  ▼                  ▼                ▼
jinyong GGUF      Claude / GPT       Kling / Sora      剪映 / CapCut
(Ollama, 本地)    (API, 付费)        (NanoBanana, 付费) (桌面端, 免费)
```

### 第一阶段：文本生成 — 金庸风格散文

| 工具 | 成本 | 优点 | 缺点 |
|------|------|------|------|
| **jinyong LoRA + Ollama**（本地） | 免费 | 完全控制，离线可用，无速率限制 | 需要部署，需要 Mac/GPU |
| **jinyong LoRA on AutoDL** | ~3 元/小时 | 快速 CUDA 推理，无需本地硬件 | 实例临时的，需 SCP 导出 |
| **Qwen2.5-7B 基座**（无 LoRA） | 免费（本地）或 API | 部署更简单 | 武侠风格弱很多（见 05-01.md 分析） |

**推荐**：日常生成使用本地 Ollama 运行 jinyong GGUF。批量生成（一次 100+ 场景）使用 AutoDL。

### 第二阶段：提示翻译 — 中文散文 → 英文视频提示

| 工具 | 成本 | 质量 | 速度 |
|------|------|------|------|
| **Claude API**（Sonnet/Haiku） | ~$0.25-3/M 输入 token | 优秀 — 细腻翻译最出色 | 快 |
| **GPT-4o** | ~$2.5-10/M 输入 token | 优秀 — 与 Claude 相当 | 快 |
| **DeepSeek API** | ~¥1/M 输入 token | 中译英非常好 | 快 |
| **人工翻译** | 免费（时间成本） | 因人而异 | 慢 |

**推荐**：预算用 Claude Haiku，质量用 Claude Sonnet。中文团队性价比最高是 DeepSeek。`JINYONG_VIDEO_PIPELINE.md`（第 3 节第 2 步）中的提示模板可直接用于生产。

### 第三阶段：视频生成 — 文本 → 视觉画面

| 工具 | 模型 | 成本 | 最适合 | 备注 |
|------|------|------|--------|------|
| **NanoBanana**（nanobanana.art） | Kling 3.0、Sora 2 Pro、Veo 3.1 | 付费，按次计费 | 聚合器 — 最佳模型选择 | 项目文档推荐 |
| **Kling / 可灵**（kling.kuaishou.com） | Kling 1.6 / 2.0 | 付费，点数制 | 角色一致性、打斗场景 | 武侠动作最佳 |
| **Runway Gen-3/4** | Gen-3 Alpha | $15-95/月 | 电影质感、风格控制 | Sora 的好替代 |
| **Pika** | Pika 2.0 | 免费额度 + 付费 | 快速草稿、社交媒体风格 | 迭代最快 |
| **MiniMax Hailuo / 海螺**（hailuoai.com） | Hailuo | 付费 | 中文优化、亚洲面孔效果好 | 武侠强有力竞争者 |
| **即梦**（jimeng.jianying.com） | 字节跳动模型 | 免费/付费 | 与剪映生态集成 | 抖音创作者最方便 |
| **Stable Video Diffusion** | SVD | 免费（需本地 GPU） | 开源、无审查 | 需高性能 GPU |

**武侠场景专项推荐**（综合项目文档和社区经验）：

| 场景类型 | 推荐工具 | 原因 |
|----------|----------|------|
| 打斗动作 | Kling 3.0 | 运动连贯性最佳 |
| 山上/自然风光 | Veo 3.1 Quality | 风景最逼真 |
| 人物特写 | Kling 3.0 | 跨镜头面部一致性 |
| 史诗定场镜头 | Sora 2 Pro | 电影级叙事 |
| 快速草稿/迭代 | Pika 或即梦 | 速度快、成本低 |

### 第四阶段：剪辑与发布

#### 视频剪辑

| 工具 | 平台 | 成本 | 最适合 |
|------|------|------|--------|
| **剪映 / CapCut** | 桌面 + 移动端 | 免费（基础）/ ¥79-179/年（专业版） | **首推** — 中文 UI、AI 功能、直接发布抖音 |
| **DaVinci Resolve** | 桌面 | 免费 | 专业调色、多轨剪辑 |
| **Premiere Pro** | 桌面 | ¥150/月 | 行业标准、学习曲线陡峭 |
| **Final Cut Pro** | Mac | ¥1,998 买断 | Mac 优化、渲染快 |
| **Canva** | Web | 免费/Pro | 快速模板、社交媒体优化 |

#### AI 配音（中文旁白）

| 工具 | 成本 | 质量 | 备注 |
|------|------|------|------|
| **剪映 AI 配音** | 免费（内置） | 好 — 多种古风男声 | 编辑器内置，最便捷 |
| **讯飞配音** | 付费 | 优秀 — 最自然的中文语音 | 行业标准 |
| **ElevenLabs** | $5-99/月 | 优秀 — 多语言、声音克隆 | 品牌一致声音最佳 |
| **Azure TTS** | 免费额度 + 付费 | 很好 | 声音选择广泛 |

#### 背景音乐（BGM）

| 来源 | 成本 | 风格覆盖 |
|------|------|----------|
| **剪映音乐库** | 免费（内置） | 国风器乐选择好 |
| **Epidemic Sound** | $15/月 | 曲库大、版权安全 |
| **Artlist** | $15/月 | 电影级、高质量 |
| **网易云音乐 / QQ音乐** | 个人使用免费 | 古筝/二胡国风最佳 |
| **Uppbeat** | 免费额度 | 选择不错、需署名 |

#### 字幕

| 工具 | 成本 | 备注 |
|------|------|------|
| **剪映自动字幕** | 免费 | 中文优化、从配音自动生成 |
| **CapCut 自动字幕** | 免费 | 多语言、准确度高 |
| **Subtitle Edit** | 免费（开源） | 手动精细调整 |
| **Descript** | $24/月 | AI 驱动、基于转录的编辑 |

#### 发布平台

| 平台 | 视频格式 | 最佳时长 | 内容风格 |
|------|----------|----------|----------|
| **抖音** | 9:16 竖屏 | 推荐 1-3 分钟 | 快节奏、前 3 秒强钩子 |
| **TikTok** | 9:16 竖屏 | 推荐 1-3 分钟 | 与抖音类似、更国际化 |
| **B 站** | 16:9 或 9:16 | 不限 | 长视频可接受、社区驱动 |
| **YouTube Shorts** | 9:16 竖屏 | ≤ 60 秒 | 快节奏、全球受众 |
| **YouTube**（常规） | 16:9 横屏 | 不限 | 长视频、制作水准要求更高 |
| **小红书** | 9:16 竖屏 | ≤ 15 分钟 | 生活美学、年轻女性受众 |

### 推荐制作流程

**抖音/TikTok 短视频（1-3 分钟）：**

```
1. jinyong Ollama（本地）            → 生成 3-5 段场景描述
2. Claude API                        → 翻译为英文视频提示
3. Kling 或即梦                      → 每场景生成 5-10 秒片段
4. 剪映桌面版                         → 拼接片段，添加：
                                           - AI 配音（古风男声）
                                           - 自动字幕（大字古风）
                                           - BGM（古筝/箫 纯音乐）
                                           - 转场（叠化/淡入淡出）
5. 导出 1080p 60fps                 → 发布至 抖音 + TikTok + B 站
```

**预估每视频耗时**：30-60 分钟（文本生成约 5 分钟，视频生成约 15-30 分钟，剪辑约 15-20 分钟）。

**预估每视频成本**：¥5-15（Claude API 约 ¥1，视频生成约 ¥3-10，其他工具免费）。

### 规模化 — 批量生产

对于日更 3-5 条视频的短视频账号：

- **批量文本生成**：在 AutoDL 上运行 jinyong LoRA，一次生成 50+ 场景 → SCP 到本地。
- **批量提示翻译**：使用 Claude batch API（半价优惠）或 DeepSeek（最便宜）。
- **批量视频生成**：NanoBanana API 或 Kling 批量生成。队列提交任务，收集输出。
- **模板化剪辑**：创建剪映模板，统一片头/片尾、BGM 和字幕风格。拖入新片段即可。
- **定时发布**：剪映支持直接定时发布到抖音。跨平台用 Buffer 或 Later。

### 工具成本汇总（月度，中等产量创作者）

| 类别 | 工具 | 月成本 |
|------|------|--------|
| 文本生成 | jinyong Ollama（本地） | ¥0 |
| 提示翻译 | DeepSeek API（约 200 次翻译） | ~¥20 |
| 视频生成 | Kling/即梦（约 100 个片段） | ~¥100-300 |
| 配音 | 剪映 AI 配音（内置） | ¥0 |
| BGM | 剪映音乐库（内置） | ¥0 |
| 剪辑 | 剪映专业版 | ¥79/年 = ~¥7/月 |
| **合计** | | **~¥130-330/月** |

对于一个每月可生产 30-60 条武侠视频的内容流水线来说，成本相当低。

---

## 附录：项目关键指标

| 指标 | 数值 |
|------|------|
| 基座模型 | Qwen2.5-7B-Instruct |
| 训练方法 | QLoRA（4-bit NF4） |
| LoRA 参数 | r=64, alpha=128, 7 个目标模块 |
| 可训练参数量 | 总量的 ~2-3%（约 1.6 亿 / 70 亿） |
| Adapter 大小 | ~100-300MB |
| 训练数据集 | 金庸 15 部小说 + 5 个 LLM 生成的分类场景 |
| 训练时间（RTX 4090） | 2 epoch 约 2-3 小时 |
| 训练成本（AutoDL） | 约 30 元 |
| 合并模型大小（bf16） | ~14GB |
| GGUF q4_k_m 大小 | ~4.7GB |
| 推理速度（M3 Pro, Ollama） | ~15-25 tokens/秒 |
| 目标输出长度 | 150-300 中文字符 |
| 脚本数量 | 14 个 Python 脚本 |
| 文档页数 | 7 份 Markdown 文档 |
| 模板数量（分类场景） | 100 个模板，22 个类别 |
| LLM 供应商（分类场景生成） | 5 个（Claude、DeepSeek、Kimi、MiniMax、GLM） |
