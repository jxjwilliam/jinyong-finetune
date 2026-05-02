# 金庸风格段落 → AI 视频生成 完整流程
# Jin Yong Paragraph → AI Video Pipeline

> **Stack:** Fine-tuned Qwen2.5-7B (LoRA) → Claude/GPT prompt translator → NanoBanana (Kling / Sora / Veo 3.1)

---

## 1. Pipeline Overview

```
┌─────────────────────┐
│  Your Story Bible   │  (characters, scene, plot beat)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Fine-tuned LoRA    │  Qwen2.5-7B + jinyong adapter
│  (local or AutoDL)  │  → Chinese wuxia paragraph
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Prompt Translator  │  Claude API / GPT-4
│                     │  Chinese prose → English cinematic prompt
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐  ┌──────────────┐
│ Image  │  │ Text-to-Video│
│  Gen   │  │   directly   │
│(NanoBn)│  │  (NanoBn)    │
└───┬────┘  └──────┬───────┘
    │               │
    ▼               │
┌────────────┐      │
│ Image-to-  │      │
│   Video    │      │
│ (NanoBn)   │      │
└───┬────────┘      │
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │  Final Clip   │  5–10 sec cinematic wuxia video
    │  + narration  │  (optional: ElevenLabs voice-over)
    └───────────────┘
```

---

## 2. NanoBanana: Which Model to Use

NanoBanana brings together Kling 2.6, Kling 3.0, Sora 2 Pro, Veo 3.1 Fast, and Veo 3.1 Quality — each with its own strengths: Kling excels at natural motion and character consistency, Sora delivers cinematic storytelling with rich detail, and Veo produces photorealistic scenes with stunning clarity.

| Model | Best for | Wuxia use case |
|-------|----------|----------------|
| **Kling 3.0** | Character consistency, natural motion | Fight scenes, character dialogue |
| **Sora 2 Pro** | Cinematic storytelling, rich detail | Landscape shots, emotional scenes |
| **Veo 3.1 Quality** | Photorealism, clarity | Mountain scenery, dramatic lighting |
| **Veo 3.1 Fast** | Quick iteration | Draft/preview before final render |

**Recommendation for wuxia:** Start with **Kling 3.0** (character consistency matters most), use **Sora 2 Pro** for epic landscape scenes.

---

## 3. Step-by-Step Python Pipeline

### Step 1: Generate Jin Yong Paragraph (LoRA)

```python
# generate_paragraph.py
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_model(adapter_path="./outputs/jinyong-qlora/adapter"):
    device = "mps" if torch.backends.mps.is_available() else "cuda"
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        adapter_path if os.path.exists(f"{adapter_path}/config.json") 
                     else "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_paragraph(model, tokenizer, device, scene_setup: str) -> str:
    """Generate one Jin Yong-style paragraph from a scene setup."""
    messages = [
        {
            "role": "system",
            "content": (
                "你是金庸风格的武侠小说作家。"
                "每次只写一段，约150-200字。"
                "文笔典雅，富有画面感，适合转化为视频场景。"
                "不要写对话，只写景物、动作、氛围描写。"
            )
        },
        {"role": "user", "content": scene_setup}
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.75,
            top_p=0.9,
            repetition_penalty=1.15,
        )

    paragraph = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return paragraph.strip()


# Example usage
if __name__ == "__main__":
    model, tokenizer, device = load_model()
    
    scene = "郭靖只身站在华山之巅，寒风呼啸，远处云海翻涌，他手握弓箭，目光如炬。"
    paragraph = generate_paragraph(model, tokenizer, device, scene)
    print("=== Generated Paragraph ===")
    print(paragraph)
```

---

### Step 2: Translate Paragraph → Cinematic Video Prompt

The hardest part: Chinese literary prose → English visual prompt that NanoBanana understands.

```python
# translate_to_prompt.py
import anthropic  # or openai

def paragraph_to_video_prompt(chinese_paragraph: str) -> dict:
    """
    Use Claude to convert a Chinese wuxia paragraph into:
    - A cinematic English video prompt
    - A suggested NanoBanana model
    - Camera direction
    """
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""You are a cinematic prompt engineer specializing in wuxia (Chinese martial arts) visual scenes.

Convert this Chinese wuxia paragraph into a detailed English video generation prompt for NanoBanana AI.

Chinese paragraph:
{chinese_paragraph}

Return a JSON object with:
{{
  "video_prompt": "detailed English cinematic prompt, 2-3 sentences, include: setting, lighting, camera movement, mood, visual style",
  "image_prompt": "shorter English image prompt for still frame reference",
  "recommended_model": "kling_3" | "sora_2_pro" | "veo_3_quality",
  "camera": "e.g. slow pan, aerial shot, close-up, tracking shot",
  "mood": "e.g. epic, melancholic, tense, serene",
  "duration": 5 | 10
}}

Style guidelines:
- Ancient China aesthetic, Song/Ming dynasty setting
- Cinematic quality, dramatic lighting
- Wuxia visual language: misty mountains, flowing robes, martial arts poses
- No text or subtitles in the scene
- Return only valid JSON, no other text."""
        }]
    )
    
    import json
    result = json.loads(response.content[0].text)
    return result


# Example
if __name__ == "__main__":
    paragraph = """
    郭靖立于华山绝顶，长风裂石，衣袂猎猎作响。
    脚下云海如棉，翻涌无际，远山如黛，隐没于苍茫之中。
    他双目微阖，内力运转，周身隐有金光流动，
    仿佛与这天地山川融为一体，忘却了江湖恩怨，
    只余那一片浩然正气，充塞天地之间。
    """
    
    result = paragraph_to_video_prompt(paragraph)
    print("=== Video Prompt ===")
    print(f"Prompt: {result['video_prompt']}")
    print(f"Model: {result['recommended_model']}")
    print(f"Camera: {result['camera']}")
```

---

### Step 3: Send to NanoBanana API

```python
# nanobanana_generate.py
import requests
import time
import os

NANO_API_KEY = os.environ["NANOBANANA_API_KEY"]  # set in your env

def generate_video(prompt_data: dict, output_path: str = "output.mp4") -> str:
    """Send prompt to NanoBanana and download the resulting video."""
    
    headers = {
        "Authorization": f"Bearer {NANO_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Text-to-video request
    payload = {
        "prompt": prompt_data["video_prompt"],
        "model": prompt_data.get("recommended_model", "kling_3"),
        "duration": prompt_data.get("duration", 5),
        "resolution": "1080p",
        "camera_movement": prompt_data.get("camera", "slow pan"),
    }
    
    # Submit job
    response = requests.post(
        "https://api.nanobanana.io/v1/video/generate",  # check actual API docs
        json=payload,
        headers=headers
    )
    job = response.json()
    job_id = job["id"]
    print(f"Job submitted: {job_id}")
    
    # Poll for completion
    while True:
        status_resp = requests.get(
            f"https://api.nanobanana.io/v1/jobs/{job_id}",
            headers=headers
        )
        status = status_resp.json()
        
        if status["status"] == "completed":
            video_url = status["output"]["url"]
            print(f"✅ Video ready: {video_url}")
            
            # Download
            video_data = requests.get(video_url).content
            with open(output_path, "wb") as f:
                f.write(video_data)
            print(f"✅ Saved to {output_path}")
            return output_path
            
        elif status["status"] == "failed":
            raise Exception(f"Generation failed: {status.get('error')}")
        
        print(f"  Status: {status['status']} — waiting...")
        time.sleep(10)
```

---

### Step 4: Full Pipeline — One Function

```python
# pipeline.py
from generate_paragraph import load_model, generate_paragraph
from translate_to_prompt import paragraph_to_video_prompt
from nanobanana_generate import generate_video

def generate_wuxia_video_clip(
    scene_setup: str,
    output_path: str = "wuxia_clip.mp4",
    model=None, tokenizer=None, device=None
):
    """
    Full pipeline: scene description → Jin Yong paragraph → video clip
    """
    print("📖 Step 1: Generating Jin Yong paragraph...")
    paragraph = generate_paragraph(model, tokenizer, device, scene_setup)
    print(paragraph)
    print()

    print("🎬 Step 2: Translating to cinematic prompt...")
    prompt_data = paragraph_to_video_prompt(paragraph)
    print(f"  Prompt: {prompt_data['video_prompt']}")
    print(f"  Model:  {prompt_data['recommended_model']}")
    print()

    print("🎥 Step 3: Generating video with NanoBanana...")
    video_path = generate_video(prompt_data, output_path)
    
    return {
        "paragraph": paragraph,
        "prompt_data": prompt_data,
        "video_path": video_path
    }


# ===== MAIN =====
if __name__ == "__main__":
    # Load model once
    model, tokenizer, device = load_model()
    
    # Define your scenes
    scenes = [
        "郭靖只身站在华山之巅，寒风呼啸，云海翻涌，手握弓箭，目光如炬。",
        "黄蓉乔装打扮，混入敌营，月色下悄然穿行于营帐之间。",
        "洪七公与欧阳锋在绝壁之上激战，招招致命，各不相让。",
    ]
    
    for i, scene in enumerate(scenes):
        print(f"\n{'='*50}")
        print(f"Scene {i+1}: {scene}")
        print('='*50)
        
        result = generate_wuxia_video_clip(
            scene_setup=scene,
            output_path=f"clip_{i+1:02d}.mp4",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        print(f"✅ Clip {i+1} done: {result['video_path']}")
```

---

## 4. Prompt Engineering Tips for Wuxia Videos

### What makes a good wuxia video prompt

```
✅ GOOD:
"A lone warrior in ancient Chinese robes stands at the peak of a misty mountain,
wind sweeping through his garments, dramatic clouds below, golden sunset light
casting long shadows. Slow cinematic pan, Song dynasty aesthetic, epic scale."

❌ BAD:
"郭靖站在山上"  (too short, Chinese text, no visual detail)
"A man on a mountain"  (no style, no atmosphere)
```

### Prompt template

```
[CHARACTER POSE/ACTION], [SETTING + TIME OF DAY], [LIGHTING], 
[CAMERA MOVEMENT], [ATMOSPHERE], [STYLE: ancient China, wuxia aesthetic, 
cinematic, 4K, dramatic]
```

### Scene type → Best model

| Scene Type | Recommended Model | Why |
|------------|-------------------|-----|
| Fight choreography | Kling 3.0 | Best motion consistency |
| Mountain/nature | Veo 3.1 Quality | Photorealistic landscapes |
| Character emotion close-up | Kling 3.0 | Face consistency |
| Epic establishing shot | Sora 2 Pro | Cinematic storytelling |
| Quick draft/test | Veo 3.1 Fast | Speed |

---

## 5. Alternative: NanoBanana Web UI (No Code)

If you don't want to use the API yet:

1. Go to **nanobanana.io** or **nanobanana.art**
2. Run your LoRA pipeline locally → copy the generated paragraph
3. Paste into Claude.ai: *"Convert this Chinese wuxia paragraph to a cinematic English video prompt for NanoBanana"*
4. Copy the English prompt → paste into NanoBanana web UI
5. Choose model (Kling 3.0 for wuxia), generate

---

## 6. Optional Enhancements

### Add Voice-over Narration

```python
# After generating video, add Chinese narration with ElevenLabs
import elevenlabs

client = elevenlabs.ElevenLabs(api_key=os.environ["ELEVEN_API_KEY"])

audio = client.generate(
    text=paragraph,           # the original Chinese paragraph
    voice="Chinese Male",     # or clone your own voice
    model="eleven_multilingual_v2"
)
elevenlabs.save(audio, "narration.mp3")

# Merge with video using ffmpeg
os.system(f"ffmpeg -i wuxia_clip.mp4 -i narration.mp3 -c:v copy -c:a aac final.mp4")
```

### Stitch Multiple Clips into a Scene

```python
# ffmpeg concat multiple clips
with open("clips.txt", "w") as f:
    for i in range(len(scenes)):
        f.write(f"file 'clip_{i+1:02d}.mp4'\n")

os.system("ffmpeg -f concat -safe 0 -i clips.txt -c copy full_scene.mp4")
```

---

## 7. Full Stack Summary

```
LOCAL MAC M3                    CLOUD / API
────────────────                ─────────────────────────
Fine-tuned LoRA          ───►  Claude API (prompt translation)
generates paragraph             │
                                ▼
                         NanoBanana API
                         (Kling / Sora / Veo)
                                │
                                ▼
                         ElevenLabs (narration)
                                │
                                ▼
                         ffmpeg (stitch + merge)
                                │
                                ▼
                         Final wuxia video clip 🎬
```

---

## 8. Resources

| Tool | URL | Use |
|------|-----|-----|
| NanoBanana Video | nanobanana.art | Kling + Sora + Veo aggregator |
| NanoBanana Image | nanobananas.ai (ViNano AI) | Image generation |
| NanoBanana API | nanobananavideo.com | Developer API + SDKs |
| ElevenLabs | elevenlabs.io | Chinese voice-over |
| ffmpeg | ffmpeg.org | Video stitching |
| Your LoRA | outputs/jinyong-qlora/adapter | Jin Yong style engine |
