Here's the breakdown behind each tab:

**Pros & Cons** — the config quality (packing fix, seq_len fix) is genuinely above average for a hobby project. The main structural problem is data imbalance: 3,142 continuation pairs dominate 412 typed pairs, which means the model leans toward raw continuation rather than instruction following. That's fixable cheaply with more `generate_typed_pairs.py` runs (bump `--per-template` to 50+).

**Ranking** — inference/deployment is the strongest layer (full GGUF+Ollama path with a correct Modelfile is rare). Evaluation is the weakest — the rubric exists but there's no script to actually run it, so you can't tell if a retrain is better or worse.

**Comparison** — this project's real moat vs Chinese-LLaMA-Alpaca and Firefly is the video pipeline. No comparable open-source project pairs wuxia domain fine-tuning with AI video generation end-to-end.

**New features** — priority order I'd recommend: (1) deduplication (free quality gain), (2) automated eval harness (gates every retrain), (3) DPO pass (sharpest quality jump), (4) FastAPI streaming server (unlocks the video pipeline).

**Video stack** — for TikTok/YouTube Shorts, use Kling 3.0 for anything with a character in frame (it holds face/costume consistency best), Sora 2 Pro for landscape establishing shots, and Fish Audio instead of ElevenLabs if you're targeting a Chinese-speaking audience (much cheaper, native Mandarin voices). Target ¥2–6 / clip all-in at current pricing.