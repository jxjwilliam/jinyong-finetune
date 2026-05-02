## STEP 1: Funetune

- 源代码：https://github.com/jxjwilliam/jinyong-finetune
- 数据集：https://www.kaggle.com/datasets/evilpsycho42/jinyong-wuxia
- 训练平台：https://www.autodl.com/console/instance/list
- jupyter训练过程：https://a989599-bbd8-b3d6860d.cqa1.seetacloud.com:8443/jupyter/lab/tree/autodl-tmp/jinyong-finetune

**这个 Kaggle 数据集 jinyong-wuxia 包含金庸全部 15 部武侠小说**。
对应“飞雪连天射白鹿，笑书神侠倚碧鸳”14部+短篇《越女剑》，完整清单如下：
1. 飞狐外传
2. 雪山飞狐
3. 连城诀
4. 天龙八部
5. 射雕英雄传
6. 白马啸西风
7. 鹿鼎记
8. 笑傲江湖
9. 书剑恩仇录
10. 神雕侠侣
11. 侠客行
12. 倚天屠龙记
13. 碧血剑
14. 鸳鸯刀
15. 越女剑

### (1) Kaggle Notebook

### (2) Google Colab

- Out of memory

### (3) AutoDL

```bash
accelerate launch train.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --load_in_4bit True \
  --lora_r 64 \
  --lora_alpha 128 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --bf16 True
```

- RTX 4090
- `nohup python train.py > train.log 2>&1 &`
- 训练结束，花费30元人民币：saved adapter to: outputs/jinyong-qlora/adapter
- 训练输出目录 outputs/jinyong-qlora/adapter 里，关键文件包括：
```
adapter_config.json：LoRA 配置文件
adapter_model.bin/adapter_model.safetensors：LoRA 权重文件
tokenizer/：分词器配置（如果微调了分词器）
```

- ssh -p 46840 root@connect.cqa1.seetacloud.com
- scp -P 46840 root@connect.cqa1.seetacloud.com:/root/autodl-tmp/jinyong-lora-adapter.zip ~/Desktop/
- 6aiDauArdhbL

### (4) Macbook Pro M3

- No GPU
- GGUF

```bash
python ~/my-tools/llama.cpp/convert_hf_to_gguf.py ./outputs/jinyong-merged --outfile ./jinyong-q4.gguf --outtype q4_k_m
```

## STEP 2: How to use

```bash
❯ tree outputs
outputs
└── jinyong-qlora
    └── adapter
        ├── README.md.gz
        ├── adapter_config.json
        ├── adapter_model.safetensors
        ├── added_tokens.json
        ├── merges.txt
        ├── special_tokens_map.json
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── vocab.json
```